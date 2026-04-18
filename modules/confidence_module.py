"""
Confidence Module — AF3 Supplementary Methods 4.3.

4 pairformer-no-seq blocks, then three prediction heads:
  pLDDT:  per-token local distance difference test score (n_plddt_bins)
  PAE:    predicted aligned error matrix (n_pae_bins per token pair)
  PDE:    predicted distance error matrix (n_pde_bins per token pair)
  pTM / ipTM are derived from PAE logits at inference (no separate head)

At training time the module is trained on the output of a "mini-rollout"
of the diffusion module (20 faster denoising steps). The STOP gradient
is applied to the mini-rollout output before feeding into this module.

References:
  OF3:   model/heads/prediction_heads.py + model/structure/diffusion_module.py
  Boltz: model/modules/confidence.py
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import LinearNoBias, Transition, final_init_
from .template_module import PairformerNoSeqBlock


# ---------------------------------------------------------------------------
# Helper: compute pTM / ipTM from PAE logits
# ---------------------------------------------------------------------------

def pae_to_ptm(
    pae_logits: torch.Tensor,   # (B, N, N, n_bins)
    bin_edges: torch.Tensor,    # (n_bins,) midpoints of PAE bins in Angstrom
    max_bin: float = 31.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes per-residue TM-score contribution from PAE distribution.
    TM-score uses d0 = 1.24 * (N - 15)^(1/3) - 1.8 for N > 21.
    """
    B, N, _, n_bins = pae_logits.shape
    probs = pae_logits.softmax(dim=-1)                        # (B, N, N, n_bins)
    # Expected PAE for each pair
    pae = (probs * bin_edges.to(pae_logits.device)).sum(-1)   # (B, N, N)

    # d0 per sequence length
    d0 = max(1.24 * max(N - 15, 0) ** (1.0 / 3.0) - 1.8, 1e-3)
    tm = 1.0 / (1.0 + (pae / d0) ** 2)                       # (B, N, N)
    return tm.mean(dim=-1)   # (B, N) per-query mean TM


def compute_iptm(
    pae_logits: torch.Tensor,   # (B, N, N, n_bins)
    bin_edges: torch.Tensor,    # (n_bins,)
    interface_mask: torch.Tensor,  # (B, N, N) 1 at cross-chain interface pairs
) -> torch.Tensor:
    """Interface pTM: TM-score restricted to interface pairs."""
    probs = pae_logits.softmax(dim=-1)
    pae = (probs * bin_edges.to(pae_logits.device)).sum(-1)   # (B, N, N)
    B, N, _ = pae.shape
    d0 = max(1.24 * max(N - 15, 0) ** (1.0 / 3.0) - 1.8, 1e-3)
    tm = 1.0 / (1.0 + (pae / d0) ** 2)
    denom = interface_mask.sum(dim=-1).clamp(min=1)
    return (tm * interface_mask).sum(dim=-1) / denom   # (B, N)


# ---------------------------------------------------------------------------
# Confidence Module
# ---------------------------------------------------------------------------

class ConfidenceModule(nn.Module):
    """
    4 pairformer-no-seq blocks then three output heads.

    The input pair representation z comes from the pairformer trunk
    (possibly updated with the mini-rollout predicted structure via
    distance features — see training/rollout.py).

    pLDDT head: z -> pool over j -> MLP -> n_plddt_bins logits per token
    PAE head:   z -> linear -> n_pae_bins logits per token pair
    PDE head:   z + z^T -> linear -> n_pde_bins logits per token pair
    """

    def __init__(
        self,
        c_z: int = 128,
        c_s: int = 384,
        n_blocks: int = 4,
        n_heads_tri: int = 4,
        tri_head_width: int = 32,
        dropout: float = 0.25,
        transition_factor: int = 4,
        n_plddt_bins: int = 50,
        n_pae_bins: int = 64,
        n_pde_bins: int = 64,
        pae_bin_max: float = 31.0,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        self.n_pae_bins = n_pae_bins
        self.pae_bin_max = pae_bin_max

        # 4-block pair-only pairformer
        self.blocks = nn.ModuleList([
            PairformerNoSeqBlock(
                c_z=c_z,
                n_heads_tri=n_heads_tri,
                tri_head_width=tri_head_width,
                dropout=dropout,
                transition_factor=transition_factor,
                inf=inf,
            )
            for _ in range(n_blocks)
        ])

        # Norm before heads
        self.norm_z = nn.LayerNorm(c_z)

        # pLDDT head: pools pair along j, then MLP
        self.plddt_head = nn.Sequential(
            LinearNoBias(c_z, c_z),
            nn.ReLU(),
            nn.Linear(c_z, n_plddt_bins),
        )

        # PAE head: directly from pair
        self.pae_head = nn.Linear(c_z, n_pae_bins)

        # PDE head: symmetric pair (z[i,j] + z[j,i]) / 2
        self.pde_head = nn.Linear(c_z, n_pde_bins)

        # Register PAE bin midpoints for pTM computation
        bin_edges = torch.linspace(0, pae_bin_max, n_pae_bins)
        self.register_buffer("pae_bin_edges", bin_edges)

        # Zero-init output projections
        final_init_(self.plddt_head[-1].weight)
        final_init_(self.pae_head.weight)
        final_init_(self.pde_head.weight)

    def forward(
        self,
        z: torch.Tensor,                            # (B, N, N, c_z)
        pair_mask: Optional[torch.Tensor] = None,   # (B, N, N)
        token_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> dict:
        """
        Returns a dict with keys:
          plddt_logits: (B, N, n_plddt_bins)
          pae_logits:   (B, N, N, n_pae_bins)
          pde_logits:   (B, N, N, n_pde_bins)
          plddt:        (B, N)  predicted per-token confidence [0,1]
          ptm:          (B, N)  per-query TM contribution
        """
        # Run 4 pairformer-no-seq blocks
        for block in self.blocks:
            z = block(z, pair_mask=pair_mask)

        z = self.norm_z(z)

        # --- pLDDT head ---
        # Pool pair along the key dimension (j) masked-mean, then predict
        if token_mask is not None:
            m = token_mask.unsqueeze(1).float()     # (B, 1, N)
            z_pooled = (z * m.unsqueeze(-1)).sum(2) / m.sum(2, keepdim=True).clamp(min=1)
        else:
            z_pooled = z.mean(2)                    # (B, N, c_z)
        plddt_logits = self.plddt_head(z_pooled)    # (B, N, n_plddt_bins)

        # Expected pLDDT: bin-weighted expected value (bins span [0, 1])
        n_bins = plddt_logits.shape[-1]
        bin_centers = torch.linspace(0.0, 1.0, n_bins, device=z.device)
        plddt = (plddt_logits.softmax(-1) * bin_centers).sum(-1)  # (B, N)

        # --- PAE head ---
        pae_logits = self.pae_head(z)               # (B, N, N, n_pae_bins)

        # --- PDE head (symmetric) ---
        z_sym = (z + z.transpose(1, 2)) / 2.0
        pde_logits = self.pde_head(z_sym)           # (B, N, N, n_pde_bins)

        # --- pTM ---
        ptm = pae_to_ptm(pae_logits, self.pae_bin_edges, self.pae_bin_max)

        return {
            "plddt_logits": plddt_logits,
            "pae_logits":   pae_logits,
            "pde_logits":   pde_logits,
            "plddt":        plddt,
            "ptm":          ptm,
        }
