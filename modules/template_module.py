"""
Template Module — AF3 Supplementary Methods 3.5 / Algorithm 16-17.

2 blocks of pairformer WITHOUT the single (sequence) track.
Template pair features are projected and processed, then the resulting
pair representation is added (with a learned weight) to z.

Pipeline:
  1. Embed raw template pair features (backbone distances, angles, masks)
     into a template pair representation of shape (T, N, N, c_z)
  2. Run 2 PairformerNoSeq blocks to contextualise the template pairs
  3. Mean-pool over templates (weighted by template validity mask)
  4. Project to c_z and add to trunk pair z

References:
  OF3:   model/latent/template_module.py
  Boltz: model/modules/trunk.py TemplateModule
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    LinearNoBias, Transition,
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode, TriangleAttentionEndingNode,
    gating_init_, final_init_,
)


# ---------------------------------------------------------------------------
# Pairformer block without sequence track (shared with TemplateModule)
# ---------------------------------------------------------------------------

class PairformerNoSeqBlock(nn.Module):
    """
    Single pairformer block operating on the pair representation only
    (no single/sequence track).  Used in template module and confidence module.

    Boltz: PairformerNoSeqLayer
    OF3:   MSABlock pair-only path
    """

    def __init__(
        self,
        c_z: int,
        n_heads_tri: int = 4,
        tri_head_width: int = 32,
        dropout: float = 0.25,
        transition_factor: int = 4,
        inf: float = 1e9,
    ) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z)
        self.tri_mul_in  = TriangleMultiplicationIncoming(c_z)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, tri_head_width, n_heads_tri, inf=inf)
        self.tri_att_end   = TriangleAttentionEndingNode(c_z, tri_head_width, n_heads_tri, inf=inf)
        self.transition    = Transition(c_z, hidden_factor=transition_factor)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = z + self.dropout(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.dropout(self.tri_att_start(z, mask=pair_mask))
        z = z + self.dropout(self.tri_att_end(z, mask=pair_mask))
        z = z + self.transition(z)
        return z


# ---------------------------------------------------------------------------
# Raw template pair feature embedder
# ---------------------------------------------------------------------------

class TemplatePairEmbedder(nn.Module):
    """
    Embeds raw template pair features into c_z-dimensional vectors.

    Raw template pair features (AF3 Supplementary Table 5):
      - distogram one-hot (39 bins)               dim 39
      - backbone unit-vector (3)                  dim  3
      - CA-CA distance (1)                        dim  1
      - template residue types (2 x n_aa)         dim 44
      - template mask (1)                         dim  1
    Total: ~88 features

    Boltz: encoders.py TemplatePairEmbedder  (similar)
    OF3:   feature_embedders/template_embedders.py
    """

    def __init__(self, c_template: int = 88, c_z: int = 128) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(c_template),
            LinearNoBias(c_template, c_z),
            nn.ReLU(),
            LinearNoBias(c_z, c_z),
        )

    def forward(self, template_pair: torch.Tensor) -> torch.Tensor:
        # template_pair: (B, T, N, N, c_template)
        return self.proj(template_pair)   # (B, T, N, N, c_z)


# ---------------------------------------------------------------------------
# Full Template Module
# ---------------------------------------------------------------------------

class TemplateModule(nn.Module):
    """
    AF3 Template Module (2 pairformer-no-seq blocks per template, then pool).

    Args:
        c_template: raw template pair feature dimension
        c_z:        trunk pair representation dimension
        n_blocks:   number of pairformer-no-seq blocks (paper: 2)
    """

    def __init__(
        self,
        c_template: int = 88,
        c_z: int = 128,
        n_blocks: int = 2,
        n_heads_tri: int = 4,
        tri_head_width: int = 32,
        dropout: float = 0.25,
        transition_factor: int = 4,
        inf: float = 1e9,
    ) -> None:
        super().__init__()

        self.template_pair_embedder = TemplatePairEmbedder(c_template, c_z)

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

        # Final projection before adding to trunk z
        self.proj_out = LinearNoBias(c_z, c_z)
        final_init_(self.proj_out.weight)

        # Learned scalar gate for the template contribution
        self.gate = nn.Linear(1, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)   # start mostly gated off

    def forward(
        self,
        template_pair: torch.Tensor,             # (B, T, N, N, c_template)
        template_mask: Optional[torch.Tensor] = None,  # (B, T) float, 1=valid
        pair_mask: Optional[torch.Tensor] = None,      # (B, N, N)
    ) -> torch.Tensor:
        """
        Returns:
            delta_z: (B, N, N, c_z) to be added to trunk pair representation
        """
        B, T, N, _, _ = template_pair.shape

        # Embed raw features
        z_t = self.template_pair_embedder(template_pair)  # (B, T, N, N, c_z)

        # Flatten T into batch for pairformer blocks
        z_t = z_t.view(B * T, N, N, -1)
        if pair_mask is not None:
            pm = pair_mask.unsqueeze(1).expand(B, T, N, N).reshape(B * T, N, N)
        else:
            pm = None

        for block in self.blocks:
            z_t = block(z_t, pair_mask=pm)

        z_t = z_t.view(B, T, N, N, -1)   # (B, T, N, N, c_z)

        # Pool over templates (mask-weighted mean)
        if template_mask is not None:
            w = template_mask[:, :, None, None, None].float()   # (B, T, 1, 1, 1)
            z_agg = (z_t * w).sum(1) / (w.sum(1).clamp(min=1))  # (B, N, N, c_z)
        else:
            z_agg = z_t.mean(1)                                  # (B, N, N, c_z)

        # Gate and project
        gate = torch.sigmoid(self.gate(torch.ones(1, 1, device=z_agg.device)))
        return gate * self.proj_out(z_agg)
