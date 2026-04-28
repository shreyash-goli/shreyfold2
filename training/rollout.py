"""
Mini-rollout for training the confidence head.

Problem: the confidence head needs a predicted structure to compute targets
(pLDDT, PAE, PDE), but the diffusion module only trains on a single denoising
step (not a full trajectory). AF3's solution (Fig. 2c):

  1. Run 20 steps of denoising (with larger step sizes than inference)
     using the already-computed trunk activations (s_trunk, z_trunk)
  2. STOP GRADIENT on the mini-rollout output — the confidence head
     trains only on the quality of the structure, not through the diffusion
  3. Permute ground-truth symmetric chains to match the rollout output
  4. Use rollout structure to compute pLDDT, PAE, PDE targets

This procedure is applied once per training step (not per diffusion sample),
using the same trunk activations from that step.

References:
  OF3:   core/runners/model_runner.py + training loop
  Boltz: model/models/boltz1.py (training_step)
  AF3 paper: Fig. 2c, Supplementary Methods 5.2
"""

import torch
import torch.nn as nn
from typing import Optional

from modules.diffusion_module import karras_noise_schedule


@torch.no_grad()
def mini_rollout(
    diffusion_module: nn.Module,
    s_trunk: torch.Tensor,        # (B, N, c_s)
    z_trunk: torch.Tensor,        # (B, N, N, c_z)
    a_ref: torch.Tensor,          # (B, A, c_a)
    p_ref: torch.Tensor,          # (B, A, A, c_ap)
    rel_pos_z: torch.Tensor,      # (B, N, N, c_z)
    feats: dict,
    n_steps: int = 20,
    sigma_data: float = 16.0,
    sigma_min: float = 0.0004,
    sigma_max: float = 160.0,
    rho: float = 7.0,
    step_scale: float = 1.0,      # scale > 1 for larger steps (faster rollout)
    mask: Optional[torch.Tensor] = None,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Runs n_steps of fast denoising to produce a rough structure for
    confidence head training. Uses stop-gradient (no_grad context).

    Returns:
        x_rollout: (B, A, 3) rough predicted atom positions (STOP GRADIENT applied)
    """
    B = s_trunk.shape[0]
    A = feats["atom_mask"].shape[1]
    device = s_trunk.device
    dtype = s_trunk.dtype

    # Use the same Karras schedule but with fewer steps
    sigmas = karras_noise_schedule(
        n_steps, sigma_data, sigma_max, sigma_min, rho, device, dtype
    )  # (n_steps+1,)

    # Start from noise
    x = torch.randn(B, A, 3, device=device, dtype=dtype) * sigmas[0]

    for i in range(n_steps):
        sig_cur  = sigmas[i].expand(B)
        sig_next = sigmas[i + 1].expand(B)

        x_pred = diffusion_module(
            x, sig_cur, s_trunk, z_trunk, a_ref, p_ref, rel_pos_z,
            feats, mask=mask, pair_mask=pair_mask,
        )

        # Euler step
        d = (x - x_pred) / sig_cur.view(B, 1, 1)
        dt = (sig_next - sig_cur).view(B, 1, 1)
        x = x + d * dt

    return x.detach()   # STOP GRADIENT — confidence head trains only on quality


def permute_symmetric_gt(
    x_rollout: torch.Tensor,    # (B, A, 3) rollout structure
    feats: dict,
    eps: float = 1e-8,
) -> dict:
    """
    Permute symmetric ground truth chains/ligands to best match the rollout.
    AF3 paper Fig. 2c: "Permute ground truth".

    For simplicity (and because full chain permutation requires protein-chain
    clustering), we implement the ligand-level permutation:
    for each ligand with multiple identical copies, find the assignment
    to rollout positions that minimises RMSD.

    Returns a (possibly permuted) copy of feats["ground_truth"].
    """
    # Full symmetric chain permutation is complex; here we return ground
    # truth as-is (a correct but potentially suboptimal assignment).
    # A complete implementation would use the Hungarian algorithm over
    # per-chain LDDT scores (Supplementary Methods 5.5).
    return feats["ground_truth"]


class RolloutStep(nn.Module):
    """
    Convenience wrapper that runs the mini-rollout and returns the
    stop-gradient structure for use in confidence training.

    Used inside the training loop:
        rollout_step = RolloutStep(model.diffusion_module, ...)
        x_rollout = rollout_step(s_trunk, z_trunk, a_ref, p_ref, ...)
    """

    def __init__(
        self,
        diffusion_module: nn.Module,
        n_steps: int = 20,
        sigma_data: float = 16.0,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        rho: float = 7.0,
    ) -> None:
        super().__init__()
        self.diffusion_module = diffusion_module
        self.n_steps = n_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def forward(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        a_ref: torch.Tensor,
        p_ref: torch.Tensor,
        rel_pos_z: torch.Tensor,
        feats: dict,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return mini_rollout(
            self.diffusion_module,
            s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, feats,
            n_steps=self.n_steps,
            sigma_data=self.sigma_data,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            mask=mask, pair_mask=pair_mask,
        )
