"""
Pairformer — AF3 Supplementary Methods 3.6 / Fig. 2a.

The pairformer replaces AF2's evoformer as the dominant processing block.
Key difference from AF2 evoformer:
  - MSA representation is NOT retained (only pair + single pass through)
  - Same triangle operations as AF2
  - Single attention uses pair as bias (AttentionPairBias)
  - 48 blocks total

Each PairformerBlock (Fig. 2a):
  Pair track:
    z += DropoutRowwise(TriangleMultiplicationOutgoing(z))
    z += DropoutRowwise(TriangleMultiplicationIncoming(z))
    z += DropoutRowwise(TriangleAttentionStartingNode(z))
    z += DropoutColumnwise(TriangleAttentionEndingNode(z))
    z += Transition(z)
  Single track:
    s += AttentionPairBias(s, z)
    s += Transition(s)

References:
  OF3:   model/latent/pairformer.py
  Boltz: model/layers/pairformer.py  PairformerLayer / PairformerModule
"""

from typing import Optional

import torch
import torch.nn as nn

from .primitives import (
    Transition, AttentionPairBias,
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode, TriangleAttentionEndingNode,
)
from .template_module import PairformerNoSeqBlock


def _row_dropout_mask(p: float, z: torch.Tensor, training: bool) -> torch.Tensor:
    """Shared dropout mask across the column dimension (row-wise dropout)."""
    if not training or p == 0.0:
        return torch.ones(1, device=z.device)
    B, N, _, C = z.shape
    mask = torch.bernoulli(torch.full((B, N, 1, 1), 1.0 - p, device=z.device)) / (1.0 - p)
    return mask


def _col_dropout_mask(p: float, z: torch.Tensor, training: bool) -> torch.Tensor:
    """Shared dropout mask across the row dimension (column-wise dropout)."""
    if not training or p == 0.0:
        return torch.ones(1, device=z.device)
    B, _, N, C = z.shape
    mask = torch.bernoulli(torch.full((B, 1, N, 1), 1.0 - p, device=z.device)) / (1.0 - p)
    return mask


class PairformerBlock(nn.Module):
    """
    Single pairformer block: pair track + single track.

    Boltz: PairformerLayer
    OF3:   pairformer.py PairformerBlock
    """

    def __init__(
        self,
        c_z: int,
        c_s: int,
        n_heads_tri: int = 4,
        tri_head_width: int = 32,
        n_heads_single: int = 16,
        single_head_width: int = 24,
        dropout: float = 0.25,
        transition_factor: int = 4,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        # Pair track
        self.tri_mul_out   = TriangleMultiplicationOutgoing(c_z)
        self.tri_mul_in    = TriangleMultiplicationIncoming(c_z)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, tri_head_width, n_heads_tri, inf=inf)
        self.tri_att_end   = TriangleAttentionEndingNode(c_z, tri_head_width, n_heads_tri, inf=inf)
        self.transition_z  = Transition(c_z, hidden_factor=transition_factor)
        # Single track
        self.attn_pair_bias = AttentionPairBias(
            c_s=c_s, c_z=c_z,
            n_heads=n_heads_single,
            head_width=single_head_width,
            inf=inf,
        )
        self.transition_s  = Transition(c_s, hidden_factor=transition_factor)
        self.dropout = dropout

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,       # (B, N) token mask
        pair_mask: Optional[torch.Tensor] = None,  # (B, N, N)
    ) -> tuple:
        # --- Pair track ---
        z = z + _row_dropout_mask(self.dropout, z, self.training) * self.tri_mul_out(z, mask=pair_mask)
        z = z + _row_dropout_mask(self.dropout, z, self.training) * self.tri_mul_in(z, mask=pair_mask)
        z = z + _row_dropout_mask(self.dropout, z, self.training) * self.tri_att_start(z, mask=pair_mask)
        z = z + _col_dropout_mask(self.dropout, z, self.training) * self.tri_att_end(z, mask=pair_mask)
        z = z + self.transition_z(z)

        # --- Single track ---
        s = s + self.attn_pair_bias(s, z, mask=mask)
        s = s + self.transition_s(s)

        return s, z


class Pairformer(nn.Module):
    """
    48-block pairformer stack (AF3 main trunk).

    Input:
      s:  (B, N, c_s)    single representation (from input embedder)
      z:  (B, N, N, c_z) pair representation (from input embedder + MSA + templates)

    Output:
      s:  (B, N, c_s)    updated single representation
      z:  (B, N, N, c_z) updated pair representation

    Both are passed to the diffusion module and confidence module.
    """

    def __init__(
        self,
        c_z: int = 128,
        c_s: int = 384,
        n_blocks: int = 48,
        n_heads_tri: int = 4,
        tri_head_width: int = 32,
        n_heads_single: int = 16,
        single_head_width: int = 24,
        dropout: float = 0.25,
        transition_factor: int = 4,
        inf: float = 1e9,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.blocks = nn.ModuleList([
            PairformerBlock(
                c_z=c_z, c_s=c_s,
                n_heads_tri=n_heads_tri, tri_head_width=tri_head_width,
                n_heads_single=n_heads_single, single_head_width=single_head_width,
                dropout=dropout, transition_factor=transition_factor, inf=inf,
            )
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                s, z = torch.utils.checkpoint.checkpoint(block, s, z, mask, pair_mask)
            else:
                s, z = block(s, z, mask, pair_mask)
        return s, z
