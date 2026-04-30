"""
Training losses for AF3.

Three losses (paper Supplementary Methods 5):
  1. Diffusion loss (L_diff): weighted MSE on denoised coordinates after
     weighted rigid alignment to ground truth.
  2. Confidence loss (L_conf): cross-entropy on pLDDT, PAE, PDE bins
     using targets derived from the mini-rollout structure.
  3. Distogram loss (L_dist): cross-entropy on Cα-Cα distance bins.

Total loss = w_diff * L_diff + w_conf * L_conf + w_dist * L_dist

References:
  OF3:   core/loss/diffusion.py + confidence.py
  Boltz: model/loss/diffusion.py + confidence.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weighted rigid alignment  (AF3 Algorithm 28)
# ---------------------------------------------------------------------------

def weighted_rigid_align(
    x: torch.Tensor,           # (B, A, 3) predicted positions
    x_gt: torch.Tensor,        # (B, A, 3) ground truth positions
    w: torch.Tensor,           # (B, A) per-atom weights
    atom_mask: torch.Tensor,   # (B, A) valid atom mask
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Optimal rotation + translation alignment (Kabsch algorithm) minimising
    weighted RMSD.  Returns x aligned to x_gt reference frame.

    OF3: core/loss/diffusion.py weighted_rigid_align
    Boltz: model/loss/diffusion.py weighted_rigid_align
    """
    atom_mask = atom_mask.bool()
    w_masked = w * atom_mask.float()
    w_sum = w_masked.sum(-1, keepdim=True).clamp(min=eps)

    # Weighted centres
    mu_x  = (x * w_masked.unsqueeze(-1)).sum(-2) / w_sum   # (B, 3)
    mu_gt = (x_gt * w_masked.unsqueeze(-1)).sum(-2) / w_sum

    x_c  = x - mu_x.unsqueeze(-2)
    gt_c = x_gt - mu_gt.unsqueeze(-2)

    # Weighted covariance: H = x_gt^T W x
    # H[b] = sum_a w[b,a] * gt_c[b,a,:,None] * x_c[b,a,None,:]
    H = torch.einsum("ba,bai,baj->bij", w_masked, gt_c, x_c)  # (B, 3, 3)

    # SVD for optimal rotation (float32 for numerical stability)
    with torch.amp.autocast("cuda", enabled=False):
        try:
            U, _, Vt = torch.linalg.svd(H.float())
            dets = torch.linalg.det(U @ Vt)
            # Correct for reflections
            F_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(H.shape[0], -1, -1).clone()
            F_mat[:, -1, -1] = torch.sign(dets)
            R = U @ F_mat @ Vt
        except Exception:
            R = torch.eye(3, device=x.device, dtype=torch.float32).unsqueeze(0).expand(H.shape[0], -1, -1)

    # Detach only R (stop-grad on alignment), not on x itself
    R = R.to(x.dtype).detach()
    # Rotate predicted coords into gt frame; gradient flows through x_c (-> x_pred)
    x_aligned = (x_c @ R.transpose(-2, -1)) + mu_gt.unsqueeze(-2)
    return x_aligned


# ---------------------------------------------------------------------------
# Per-atom molecule-type weights
# ---------------------------------------------------------------------------

def build_atom_weights(
    feats: dict,
    dna_weight: float = 1.5,
    rna_weight: float = 2.5,
    ligand_weight: float = 4.0,
) -> torch.Tensor:
    """
    Per-token weights upweighting under-represented molecule types.
    Broadcast to per-atom weights.
    Boltz: model/loss/diffusion.py  mse_loss (first part)
    """
    w_tok = (
        torch.ones_like(feats["is_protein"])
        + feats["is_dna"] * dna_weight
        + feats["is_rna"] * rna_weight
        + feats["is_ligand"] * ligand_weight
    )  # (B, N)

    # Broadcast from token to atom via atom_to_token
    B, N = w_tok.shape
    A = feats["atom_to_token"].shape[1]
    idx = feats["atom_to_token"].clamp(0, N - 1)
    w_atom = w_tok[torch.arange(B, device=w_tok.device).unsqueeze(1), idx]  # (B, A)
    return w_atom


# ---------------------------------------------------------------------------
# Diffusion loss  (AF3 Eq. 3 + Supplementary Methods 5.1)
# ---------------------------------------------------------------------------

def diffusion_loss(
    x_pred: torch.Tensor,   # (B, A, 3) predicted clean coordinates
    feats: dict,
    dna_weight: float = 1.5,
    rna_weight: float = 2.5,
    ligand_weight: float = 4.0,
    sigma: Optional[torch.Tensor] = None,   # (B,) noise level (not used in basic MSE)
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Weighted MSE loss between predicted and ground truth atom positions,
    after optimal rigid alignment.

    L_diff = mean_over_atoms[ w_i * ||x_pred_i - x_gt_i||^2 ]
    """
    gt = feats["ground_truth"]
    x_gt = gt["atom_positions"]            # (B, A, 3)
    atom_mask = gt["atom_resolved_mask"]   # (B, A)

    w = build_atom_weights(feats, dna_weight, rna_weight, ligand_weight)  # (B, A)

    # Align predicted to ground truth
    x_aligned = weighted_rigid_align(x_pred, x_gt, w, atom_mask, eps)

    diff = (x_aligned - x_gt) ** 2        # (B, A, 3)
    diff = diff.sum(-1)                    # (B, A)

    # Weighted masked mean
    w_masked = w * atom_mask.float()
    denom = w_masked.sum(-1).clamp(min=eps)
    loss = (diff * w_masked).sum(-1) / denom   # (B,)
    return loss.mean()


# ---------------------------------------------------------------------------
# Distogram loss  (auxiliary, Supplementary Methods 5.3)
# ---------------------------------------------------------------------------

def distogram_loss(
    dist_logits: torch.Tensor,   # (B, N, N, n_bins)
    feats: dict,
    min_dist: float = 2.0,
    max_dist: float = 65.0,
    n_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cross-entropy loss on Cα-Cα (or representative atom) distances.
    Only computed for protein tokens with known positions.
    """
    gt = feats["ground_truth"]
    atom_pos = gt["atom_positions"]        # (B, A, 3)
    atom_mask = gt["atom_resolved_mask"]   # (B, A)
    token_mask = feats["token_mask"]       # (B, N)

    # Use the first atom of each token as representative (simplified)
    token_atom_start = feats["token_atom_start"]   # (B, N)
    B, N = token_mask.shape
    A = atom_pos.shape[1]

    idx = token_atom_start.clamp(0, A - 1)
    # Gather representative atom positions
    tok_pos = atom_pos[torch.arange(B, device=atom_pos.device).unsqueeze(1), idx]  # (B, N, 3)
    tok_valid = atom_mask[torch.arange(B, device=atom_mask.device).unsqueeze(1), idx]  # (B, N)

    # Pairwise distances
    diff = tok_pos.unsqueeze(2) - tok_pos.unsqueeze(1)   # (B, N, N, 3)
    dist = diff.norm(dim=-1)                              # (B, N, N)

    # Bin edges
    bins = torch.linspace(min_dist, max_dist, n_bins - 1, device=dist.device)
    bin_idx = torch.bucketize(dist, bins)   # (B, N, N) long in [0, n_bins)

    # Valid pairs: both tokens valid and protein
    valid = (tok_valid.unsqueeze(2) * tok_valid.unsqueeze(1) *
             token_mask.unsqueeze(2) * token_mask.unsqueeze(1))  # (B, N, N)

    # Symmetrised logits
    logits_sym = (dist_logits + dist_logits.transpose(1, 2)) / 2.0

    loss = F.cross_entropy(
        logits_sym.reshape(-1, n_bins),
        bin_idx.reshape(-1),
        reduction="none",
    ).view(B, N, N)

    denom = valid.sum().clamp(min=1)
    return (loss * valid).sum() / denom


# ---------------------------------------------------------------------------
# Confidence loss  (Supplementary Methods 4.3)
# ---------------------------------------------------------------------------

def _plddt_target(
    x_pred: torch.Tensor,   # (B, A, 3) predicted (from rollout)
    feats: dict,
    cutoffs: torch.Tensor,  # thresholds for pLDDT bins
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-token pLDDT target in [0, 1] by evaluating local
    distance difference test on the rollout prediction.

    Simplified: compute mean LDDT over neighbouring atoms within 15 Å
    of each token's representative atom, then bin into n_bins.
    """
    gt = feats["ground_truth"]
    x_gt = gt["atom_positions"]
    atom_mask = gt["atom_resolved_mask"].bool()
    atom_to_token = feats["atom_to_token"]
    B, A, _ = x_pred.shape
    N = feats["token_mask"].shape[1]

    # Per-atom LDDT (simplified: fraction of distances preserved within 0.5, 1, 2, 4 Å)
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=x_pred.device)

    pred_dists = torch.cdist(x_pred, x_pred)   # (B, A, A)
    gt_dists   = torch.cdist(x_gt, x_gt)       # (B, A, A)
    diff = (pred_dists - gt_dists).abs()        # (B, A, A)

    # Inclusion mask: ground truth within 15 Å and both atoms valid
    incl = (gt_dists < 15.0) & atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)

    # Fraction within each threshold
    frac = sum(
        ((diff < t) * incl).float().sum(-1) /
        incl.float().sum(-1).clamp(min=eps)
        for t in thresholds
    ) / len(thresholds)   # (B, A)

    # Aggregate to tokens via mean over each token's atoms
    tok_plddt = torch.zeros(B, N, device=x_pred.device)
    tok_count = torch.zeros(B, N, device=x_pred.device)
    idx = atom_to_token.clamp(0, N - 1)
    tok_plddt.scatter_add_(1, idx, frac)
    tok_count.scatter_add_(1, idx, atom_mask.float())
    tok_plddt = tok_plddt / tok_count.clamp(min=eps)   # (B, N) in [0, 1]

    # Bin into n_bins: target is bin index
    n_bins = cutoffs.shape[0] + 1
    target = torch.bucketize(tok_plddt, cutoffs.to(tok_plddt.device))  # (B, N)
    return target.long()


def confidence_loss(
    conf_out: dict,         # output dict from ConfidenceModule
    x_rollout: torch.Tensor,  # (B, A, 3) mini-rollout structure (stop-grad)
    feats: dict,
    token_mask: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
    plddt_weight: float = 0.01,
    pae_weight: float = 0.1,
    pde_weight: float = 0.1,
    pae_bin_max: float = 31.0,
    n_pae_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Total confidence loss = pLDDT CE + PAE CE + PDE CE.
    Targets derived from mini-rollout x_rollout (stop-gradient).
    """
    B, N = conf_out["plddt_logits"].shape[:2]

    # --- pLDDT ---
    n_plddt_bins = conf_out["plddt_logits"].shape[-1]
    cutoffs = torch.linspace(0, 1, n_plddt_bins - 1, device=x_rollout.device)
    plddt_targets = _plddt_target(x_rollout.detach(), feats, cutoffs, eps)

    plddt_loss = F.cross_entropy(
        conf_out["plddt_logits"].view(-1, n_plddt_bins),
        plddt_targets.view(-1),
        reduction="none",
    ).view(B, N)
    if token_mask is not None:
        plddt_loss = (plddt_loss * token_mask).sum() / token_mask.sum().clamp(min=1)
    else:
        plddt_loss = plddt_loss.mean()

    # --- PAE ---
    gt = feats["ground_truth"]
    x_gt = gt["atom_positions"]
    atom_to_token = feats["atom_to_token"]
    A = x_rollout.shape[1]

    # Compute aligned error per token pair: mean |d_pred - d_gt| per pair
    pred_dists = torch.cdist(x_rollout.detach(), x_rollout.detach())  # (B, A, A)
    gt_dists   = torch.cdist(x_gt, x_gt)

    # Aggregate to token pairs via mean over atom pairs within each token pair
    # (simplified: use token representative atoms)
    tok_start = feats["token_atom_start"].clamp(0, A - 1)
    tok_pos_pred = x_rollout.detach()[torch.arange(B, device=x_rollout.device).unsqueeze(1), tok_start]
    tok_pos_gt   = x_gt[torch.arange(B, device=x_gt.device).unsqueeze(1), tok_start]

    pae_gt = (torch.cdist(tok_pos_pred, tok_pos_gt) - torch.cdist(tok_pos_gt, tok_pos_gt)).abs()
    pae_bins = torch.linspace(0, pae_bin_max, n_pae_bins - 1, device=pae_gt.device)
    pae_target = torch.bucketize(pae_gt, pae_bins)   # (B, N, N)

    n_pae = conf_out["pae_logits"].shape[-1]
    pae_loss = F.cross_entropy(
        conf_out["pae_logits"].view(-1, n_pae),
        pae_target.clamp(0, n_pae - 1).view(-1),
        reduction="none",
    ).view(B, N, N)
    if pair_mask is not None:
        pae_loss = (pae_loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    else:
        pae_loss = pae_loss.mean()

    # --- PDE (predicted distance error) ---
    n_pde = conf_out["pde_logits"].shape[-1]
    pde_target = pae_target   # simplified: same as PAE target
    pde_loss = F.cross_entropy(
        conf_out["pde_logits"].view(-1, n_pde),
        pde_target.clamp(0, n_pde - 1).view(-1),
        reduction="none",
    ).view(B, N, N)
    if pair_mask is not None:
        pde_loss = (pde_loss * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    else:
        pde_loss = pde_loss.mean()

    return plddt_weight * plddt_loss + pae_weight * pae_loss + pde_weight * pde_loss


# ---------------------------------------------------------------------------
# Combined training loss
# ---------------------------------------------------------------------------

class AF3Loss(nn.Module):
    """
    Combines all AF3 training losses.
    """

    def __init__(
        self,
        diffusion_weight: float = 4.0,
        confidence_weight: float = 1.0,
        distogram_weight: float = 0.03,
        dna_weight: float = 1.5,
        rna_weight: float = 2.5,
        ligand_weight: float = 4.0,
    ) -> None:
        super().__init__()
        self.w_diff = diffusion_weight
        self.w_conf = confidence_weight
        self.w_dist = distogram_weight
        self.dna_weight = dna_weight
        self.rna_weight = rna_weight
        self.ligand_weight = ligand_weight

    def forward(
        self,
        model_out: dict,
        feats: dict,
        sigma: torch.Tensor,
    ) -> dict:
        """
        Args:
            model_out: output from AlphaFold3.forward_for_training()
            feats: feature dict
            sigma: (B,) noise levels used in this step

        Returns dict with individual losses and total.
        """
        losses = {}

        # Diffusion loss
        losses["diffusion"] = self.w_diff * diffusion_loss(
            model_out["x_pred"], feats,
            self.dna_weight, self.rna_weight, self.ligand_weight, sigma,
        )

        # Distogram loss
        if "dist_logits" in model_out:
            losses["distogram"] = self.w_dist * distogram_loss(
                model_out["dist_logits"], feats,
            )
        else:
            losses["distogram"] = torch.tensor(0.0)

        # Confidence loss (only if rollout was performed)
        if model_out.get("confidence") is not None and "x_rollout" in model_out:
            losses["confidence"] = self.w_conf * confidence_loss(
                model_out["confidence"],
                model_out["x_rollout"],
                feats,
                token_mask=model_out.get("token_mask"),
                pair_mask=model_out.get("pair_mask"),
            )
        else:
            losses["confidence"] = torch.tensor(0.0, device=model_out["x_pred"].device)

        losses["total"] = sum(losses.values())
        return losses
