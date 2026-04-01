"""
MSA Module — AF3 Supplementary Methods 3.3 / Algorithm 8.

AF3 de-emphasises MSA processing compared to AF2:
  - Only 4 blocks (vs. 48 Evoformer blocks in AF2)
  - MSA row attention replaced by cheap pair-weighted averaging
  - No column attention
  - MSA representation is NOT retained after this module;
    only the pair representation z carries forward

Each MSA block:
  1. MSAPairWeightedAveraging: update m using z as weights
  2. Transition on m
  3. OuterProductMean: update z from m
  4. (optional) triangular updates on z within this block — in practice
     AF3 uses a simpler pair update; we include it for completeness

References:
  OF3:   model/latent/msa_module.py  MSAModuleBlock
  Boltz: model/modules/trunk.py      MSAModule (via pair_averaging.py)
"""

from typing import Optional

import torch
import torch.nn as nn

from .primitives import (
    Transition, OuterProductMean, MSAPairWeightedAveraging,
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode, TriangleAttentionEndingNode,
    final_init_,
)


class MSABlock(nn.Module):
    """
    Single AF3 MSA module block.

    MSA track:
      m -> PairWeightedAveraging(m, z) -> m'
      m -> Transition -> m''

    Pair track (outer product mean feeds into z):
      m -> OuterProductMean -> delta_z
      z -> delta_z -> z'
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden_msa: int = 8,
        n_heads_msa: int = 8,
        c_hidden_opm: int = 32,
        dropout_msa: float = 0.15,
        transition_factor: int = 4,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        self.msa_avg = MSAPairWeightedAveraging(
            c_m=c_m, c_z=c_z, c_hidden=c_hidden_msa,
            n_heads=n_heads_msa, inf=inf,
        )
        self.msa_transition = Transition(c_m, hidden_factor=transition_factor)
        self.opm = OuterProductMean(c_m=c_m, c_z=c_z, c_hidden=c_hidden_opm)
        self.dropout_msa = nn.Dropout(p=dropout_msa)

    def forward(
        self,
        m: torch.Tensor,                      # (B, S, N, c_m)
        z: torch.Tensor,                      # (B, N, N, c_z)
        msa_mask: Optional[torch.Tensor] = None,   # (B, S, N)
        pair_mask: Optional[torch.Tensor] = None,  # (B, N, N)
    ) -> tuple:
        # 1. MSA pair-weighted averaging
        m = m + self.dropout_msa(
            self.msa_avg(m, z, msa_mask=msa_mask, pair_mask=pair_mask)
        )
        # 2. MSA transition
        m = m + self.msa_transition(m)
        # 3. Outer product mean -> pair
        z = z + self.opm(m, mask=msa_mask)
        return m, z


class MSAModule(nn.Module):
    """
    4-block MSA module stack (AF3 paper: 4 MSA module blocks).

    After the 4 blocks the MSA representation m is discarded;
    only z (and the derived s_inputs from input embedder) continue.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        n_blocks: int = 4,
        c_hidden_msa: int = 8,
        n_heads_msa: int = 8,
        c_hidden_opm: int = 32,
        dropout_msa: float = 0.15,
        transition_factor: int = 4,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            MSABlock(
                c_m=c_m, c_z=c_z,
                c_hidden_msa=c_hidden_msa, n_heads_msa=n_heads_msa,
                c_hidden_opm=c_hidden_opm,
                dropout_msa=dropout_msa,
                transition_factor=transition_factor,
                inf=inf,
            )
            for _ in range(n_blocks)
        ])

        # Project raw MSA one-hot -> c_m
        # MSA input is integer-encoded; we embed then project
        self.msa_embed = nn.Sequential(
            nn.LayerNorm(c_m),
        )

    def forward(
        self,
        m: torch.Tensor,                      # (B, S, N, c_m)  already embedded
        z: torch.Tensor,                      # (B, N, N, c_z)
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            z: (B, N, N, c_z) updated pair representation
        """
        for block in self.blocks:
            m, z = block(m, z, msa_mask=msa_mask, pair_mask=pair_mask)
        return z   # m is discarded after the MSA module


class MSAEmbedder(nn.Module):
    """
    Projects integer-encoded MSA sequences into the MSA representation.

    Input msa: (B, S, N) long tensor of residue indices.
    Output: (B, S, N, c_m)
    """

    def __init__(self, n_vocab: int, c_m: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_vocab, c_m, padding_idx=0)
        # Extra: embed whether each position in the MSA is a gap
        self.gap_embed = nn.Embedding(2, c_m)
        nn.init.zeros_(self.gap_embed.weight)

    def forward(self, msa: torch.Tensor, gap_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # msa: (B, S, N) long
        m = self.embed(msa)
        if gap_mask is not None:
            m = m + self.gap_embed(gap_mask.long())
        return m
