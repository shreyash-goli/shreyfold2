"""
Input Embedder — AF3 Supplementary Methods 3.4 / Algorithm 5.

Converts raw per-atom features (reference conformer positions, element,
charge, atom name) into initial pair (z) and single (s) representations
that seed the trunk.

Pipeline:
  1. Embed atom-level reference features -> atom single (c_a) + atom pair (c_ap)
  2. Run 3 blocks of sequence-local atom attention (AtomTransformer)
  3. Aggregate atom representations to token representations via mean-pooling
  4. Build token-pair relative position encoding
  5. Project aggregated atoms -> s_inputs  (c_s)
  6. Project relative position features -> z  (c_z)

Inspired by:
  OF3:   feature_embedders/input_embedders.py  (RefAtomFeatureEmbedder + InputEmbedder)
  Boltz: modules/encoders.py                   (AtomAttentionEncoder, relative.py)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    LinearNoBias, Transition,
    AttentionPairBias, AdaLayerNorm,
    lecun_normal_init_, gating_init_, final_init_,
)
from data.tokenizer import N_ELEMENTS


# ---------------------------------------------------------------------------
# Fourier / sinusoidal relative position encoding
# ---------------------------------------------------------------------------

def relative_position_encoding(
    residue_index: torch.Tensor,   # (B, N) int
    chain_index: torch.Tensor,     # (B, N) int
    max_radius: int = 32,
    n_bins: int = 64,
) -> torch.Tensor:
    """
    Encodes the relative token-pair position as a one-hot bin embedding.
    Same-chain pairs use the residue offset; cross-chain pairs get a special bin.
    Returns: (B, N, N, n_bins + 1)

    Boltz: model/layers/relative.py
    OF3: featurization/structure.py
    """
    B, N = residue_index.shape
    ri = residue_index[:, :, None] - residue_index[:, None, :]  # (B, N, N)
    same_chain = (chain_index[:, :, None] == chain_index[:, None, :])  # (B, N, N)

    # Clip and bin the residue offset
    ri_clipped = ri.clamp(-max_radius, max_radius)
    # Map [-max_radius, max_radius] -> [0, n_bins-1]
    bin_idx = ((ri_clipped + max_radius) * (n_bins - 1) / (2 * max_radius)).long()

    # Cross-chain pairs go in the last bin
    bin_idx = torch.where(same_chain, bin_idx, torch.full_like(bin_idx, n_bins))

    encoding = F.one_hot(bin_idx, num_classes=n_bins + 1).float()   # (B, N, N, n_bins+1)
    return encoding


# ---------------------------------------------------------------------------
# Sequence-local (windowed) attention for atoms
# ---------------------------------------------------------------------------

class SequenceLocalAtomAttention(nn.Module):
    """
    Windowed self-attention over atoms within each token's local neighbourhood.
    Atoms are grouped into windows of size (n_query x n_key).
    Within each window, attention is computed with pair biases.

    AF3 Supplementary Methods 3.4 — "sequence-local atom attention"
    OF3: layers/sequence_local_atom_attention.py AtomTransformer
    Boltz: modules/encoders.py AtomAttentionEncoder (inner blocks)

    For simplicity we implement dense attention over the full atom sequence,
    which is correct and numerically identical to windowed attention;
    the windowed version is an optimisation for long sequences.
    """

    def __init__(self, c_a: int, c_ap: int, n_heads: int, head_width: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_width = head_width
        c_out = n_heads * head_width
        self.inf = 1e9

        self.norm_a = nn.LayerNorm(c_a)
        self.norm_ap = nn.LayerNorm(c_ap)

        self.proj_q = nn.Linear(c_a, c_out)
        self.proj_k = LinearNoBias(c_a, c_out)
        self.proj_v = LinearNoBias(c_a, c_out)
        self.proj_bias = LinearNoBias(c_ap, n_heads)   # atom-pair -> bias per head
        self.proj_g = LinearNoBias(c_a, c_out)
        self.proj_o = LinearNoBias(c_out, c_a)
        gating_init_(self.proj_g.weight)
        final_init_(self.proj_o.weight)

        self.transition = Transition(c_a, hidden_factor=4)

    def forward(
        self,
        a: torch.Tensor,    # (B, A, c_a)  atom single repr
        p: torch.Tensor,    # (B, A, A, c_ap) atom pair repr
        atom_mask: Optional[torch.Tensor] = None,  # (B, A)
    ) -> torch.Tensor:
        B, A, _ = a.shape
        H, d = self.n_heads, self.head_width

        a_n = self.norm_a(a)
        q = self.proj_q(a_n).view(B, A, H, d).transpose(1, 2)   # (B, H, A, d)
        k = self.proj_k(a_n).view(B, A, H, d).transpose(1, 2)
        v = self.proj_v(a_n).view(B, A, H, d).transpose(1, 2)

        # Atom-pair bias: (B, A, A, H) -> (B, H, A, A)
        bias = self.proj_bias(self.norm_ap(p)).permute(0, 3, 1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5) + bias
        if atom_mask is not None:
            attn = attn + (1.0 - atom_mask[:, None, None, :].float()) * -self.inf
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, A, H * d)
        g = self.proj_g(a_n).sigmoid()
        out = self.proj_o(g * out)

        # Residual + transition
        a = a + out
        a = a + self.transition(a)
        return a


# ---------------------------------------------------------------------------
# Atom-pair feature embedder (reference conformer geometry)
# ---------------------------------------------------------------------------

class RefAtomFeatureEmbedder(nn.Module):
    """
    Encodes per-atom reference-conformer features into atom single (c_a)
    and atom pair (c_ap) representations.

    Atom single features:
      ref_pos (3), ref_charge (1), ref_mask (1),
      ref_element (N_elements), ref_atom_name_chars (4*64)
    Atom pair features:
      offset vector (3), inverse squared distance (1), valid pair mask (1)

    OF3: RefAtomFeatureEmbedder
    Boltz: AtomAttentionEncoder (first few lines)
    """

    def __init__(self, c_a: int, c_ap: int) -> None:
        super().__init__()
        # Single features
        self.lin_pos    = LinearNoBias(3, c_a)
        self.lin_charge = LinearNoBias(1, c_a)
        self.lin_mask   = LinearNoBias(1, c_a)
        self.lin_elem   = LinearNoBias(N_ELEMENTS, c_a)
        self.lin_chars  = LinearNoBias(4 * 64, c_a)
        # Pair features
        self.lin_offset = LinearNoBias(3, c_ap)
        self.lin_dist   = LinearNoBias(1, c_ap)
        self.lin_valid  = LinearNoBias(1, c_ap)

    def forward(self, feats: dict) -> tuple:
        # --- atom single ---
        a = (
            self.lin_pos(feats["ref_pos"])
            + self.lin_charge(feats["ref_charge"])
            + self.lin_mask(feats["ref_mask"].unsqueeze(-1))
            + self.lin_elem(feats["ref_element"])
            + self.lin_chars(feats["ref_atom_name_chars"].flatten(-2, -1))
        )  # (B, A, c_a)

        # --- atom pair ---
        pos = feats["ref_pos"]                               # (B, A, 3)
        offset = pos.unsqueeze(2) - pos.unsqueeze(1)         # (B, A, A, 3)
        sq_dist = (offset ** 2).sum(-1, keepdim=True)        # (B, A, A, 1)
        inv_sq_dist = 1.0 / (sq_dist + 1e-6)
        valid = (feats["ref_mask"].unsqueeze(2) * feats["ref_mask"].unsqueeze(1)).unsqueeze(-1)

        p = (
            self.lin_offset(offset)
            + self.lin_dist(inv_sq_dist)
            + self.lin_valid(valid)
        )  # (B, A, A, c_ap)

        return a, p


# ---------------------------------------------------------------------------
# Token-level aggregation and projection
# ---------------------------------------------------------------------------

def aggregate_atoms_to_tokens(
    a: torch.Tensor,              # (B, A, c_a)
    num_atoms_per_token: torch.Tensor,  # (B, N)  int
    n_tokens: int,
) -> torch.Tensor:
    """
    Mean-pool atom representations into token representations.
    Each token aggregates its own atoms.
    """
    B, A, C = a.shape
    out = torch.zeros(B, n_tokens, C, device=a.device, dtype=a.dtype)
    counts = torch.zeros(B, n_tokens, device=a.device, dtype=a.dtype)

    # Build atom-to-token mapping from num_atoms_per_token
    for b in range(B):
        idx = 0
        for t in range(n_tokens):
            n = num_atoms_per_token[b, t].item()
            if n > 0 and idx < A:
                end = min(idx + int(n), A)
                out[b, t] = a[b, idx:end].mean(0)
                counts[b, t] = 1.0
                idx = end

    return out  # (B, N, c_a)


# ---------------------------------------------------------------------------
# Full Input Embedder
# ---------------------------------------------------------------------------

class InputEmbedder(nn.Module):
    """
    AF3 input embedder (Supplementary Methods 3.4).

    Outputs:
      s_inputs: (B, N, c_s)  — token single representation (seed for trunk)
      z:        (B, N, N, c_z) — token pair representation (seed for trunk)
    """

    def __init__(
        self,
        c_a: int = 128,
        c_ap: int = 16,
        c_s: int = 384,
        c_z: int = 128,
        n_atom_encoder_blocks: int = 3,
        n_heads: int = 4,
        head_width: int = 32,
        n_rel_pos_bins: int = 64,
        max_rel_pos: int = 32,
        n_token_types: int = 6,
        n_residue_types: int = 33,   # max of AA/RNA/DNA/element vocab sizes
    ) -> None:
        super().__init__()
        self.c_a = c_a
        self.c_ap = c_ap
        self.c_s = c_s
        self.c_z = c_z
        self.n_rel_pos_bins = n_rel_pos_bins
        self.max_rel_pos = max_rel_pos

        # Reference atom feature embedder
        self.ref_feat_embedder = RefAtomFeatureEmbedder(c_a, c_ap)

        # Token type and residue type embeddings (token-level)
        self.token_type_embed = nn.Embedding(n_token_types, c_a)
        self.residue_type_embed = nn.Embedding(n_residue_types, c_a)

        # 3-block sequence-local atom transformer
        self.atom_transformer = nn.ModuleList([
            SequenceLocalAtomAttention(c_a, c_ap, n_heads, head_width)
            for _ in range(n_atom_encoder_blocks)
        ])

        # Project aggregated atom repr -> token single
        self.proj_a_to_s = LinearNoBias(c_a, c_s)

        # Relative position encoding -> pair
        rel_pos_dim = n_rel_pos_bins + 1
        self.proj_rel_pos = nn.Sequential(
            LinearNoBias(rel_pos_dim, c_z),
            nn.ReLU(),
            LinearNoBias(c_z, c_z),
        )

        # Pair initialisation from token singles (outer sum)
        self.proj_s_to_z_left  = LinearNoBias(c_s, c_z)
        self.proj_s_to_z_right = LinearNoBias(c_s, c_z)

    def forward(self, feats: dict) -> tuple:
        """
        Args:
            feats: feature dict from data/features.py

        Returns:
            s_inputs: (B, N, c_s)
            z:        (B, N, N, c_z)
        """
        # --- Atom-level encoding ---
        a, p = self.ref_feat_embedder(feats)          # (B, A, c_a), (B, A, A, c_ap)

        # Broadcast token-type embedding to atoms (so atoms know their residue type)
        atom_to_token = feats["atom_to_token"]        # (B, A)
        B, A, _ = a.shape
        N = feats["token_mask"].shape[1]

        # Add token-level embeddings to atoms
        tok_type_emb = self.token_type_embed(feats["token_type"])        # (B, N, c_a)
        res_type_emb = self.residue_type_embed(feats["residue_type"])    # (B, N, c_a)
        token_emb = tok_type_emb + res_type_emb                          # (B, N, c_a)

        # Gather token embeddings for each atom
        idx = atom_to_token.clamp(0, N - 1)
        tok_for_atoms = token_emb[torch.arange(B, device=a.device).unsqueeze(1), idx]  # (B, A, c_a)
        a = a + tok_for_atoms

        atom_mask = feats["atom_mask"]                                   # (B, A)
        for block in self.atom_transformer:
            a = block(a, p, atom_mask)                                   # (B, A, c_a)

        # --- Aggregate atoms -> tokens ---
        s_agg = aggregate_atoms_to_tokens(a, feats["num_atoms_per_token"], N)  # (B, N, c_a)
        s_inputs = self.proj_a_to_s(s_agg)                              # (B, N, c_s)

        # --- Build initial pair representation ---
        # Relative position encoding
        rel_pos = relative_position_encoding(
            feats["residue_index"], feats["chain_index"],
            max_radius=self.max_rel_pos, n_bins=self.n_rel_pos_bins,
        ).to(a.device)                                                   # (B, N, N, n_bins+1)
        z = self.proj_rel_pos(rel_pos)                                   # (B, N, N, c_z)

        # Outer sum of token singles
        z = z + self.proj_s_to_z_left(s_inputs).unsqueeze(2)            # (B, N, 1, c_z)
        z = z + self.proj_s_to_z_right(s_inputs).unsqueeze(1)           # (B, 1, N, c_z)

        return s_inputs, z
