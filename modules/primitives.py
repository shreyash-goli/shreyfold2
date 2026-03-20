"""
Core primitive operations shared across all AF3 modules.

Sources:
  - AF3 paper Fig. 2a (pairformer), Supplementary Methods 3.6
  - AF2 Supplementary Algorithms 11-14 (triangle ops, same in AF3 pairformer)
  - Boltz: triangular_mult.py, triangular_attention/, attention.py, transition.py
  - OpenFold3: triangular_multiplicative_update.py, triangular_attention.py,
               attention_pair_bias.py, outer_product_mean.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Weight initialization helpers (Boltz convention)
# ---------------------------------------------------------------------------

def lecun_normal_init_(w: torch.Tensor) -> None:
    """LeCun normal: std = 1/sqrt(fan_in). Good default for projection weights."""
    fan_in = w.shape[1] if w.dim() > 1 else w.shape[0]
    nn.init.trunc_normal_(w, std=math.sqrt(1.0 / fan_in))


def gating_init_(w: torch.Tensor) -> None:
    """Zero-init for gate weights: gates start closed, output initially zero."""
    nn.init.zeros_(w)


def final_init_(w: torch.Tensor) -> None:
    """Zero-init for final output projections: residual starts near zero."""
    nn.init.zeros_(w)


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class LinearNoBias(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)
        lecun_normal_init_(self.weight)


class Transition(nn.Module):
    """
    SwiGLU MLP used throughout AF3 (pairformer, MSA, diffusion).
    out = fc3(silu(fc1(norm(x))) * fc2(norm(x)))
    Matches Boltz's Transition and OF3's Transition.
    """

    def __init__(self, dim: int, hidden_factor: int = 4, out_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = dim * hidden_factor
        if out_dim is None:
            out_dim = dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = LinearNoBias(dim, hidden)
        self.fc2 = LinearNoBias(dim, hidden)   # gate branch
        self.fc3 = LinearNoBias(hidden, out_dim)
        final_init_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Triangle Multiplicative Updates  (AF2 Algorithms 11 & 12, unchanged in AF3)
# ---------------------------------------------------------------------------

class _TriangleMultiplication(nn.Module):
    """
    Shared base for outgoing/incoming triangle multiplicative updates.
    Boltz's clean factorisation: project to 2*c, sigmoid-gate, split,
    einsum, LayerNorm, output-gate.
    """

    def __init__(self, c_z: int, outgoing: bool) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.norm_in = nn.LayerNorm(c_z)
        # Projects to a and b simultaneously; gate for gating both
        self.p_in = LinearNoBias(c_z, 2 * c_z)
        self.g_in = LinearNoBias(c_z, 2 * c_z)   # gate; zero-init
        self.norm_out = nn.LayerNorm(c_z)
        self.p_out = LinearNoBias(c_z, c_z)
        self.g_out = LinearNoBias(c_z, c_z)       # gate; zero-init
        gating_init_(self.g_in.weight)
        gating_init_(self.g_out.weight)
        final_init_(self.p_out.weight)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # z: (B, N, N, c_z)
        z_norm = self.norm_in(z)
        # Fused projection + gating
        ab = self.p_in(z_norm) * self.g_in(z_norm).sigmoid()  # (B, N, N, 2c)
        if mask is not None:
            ab = ab * mask.unsqueeze(-1)
        a, b = ab.float().chunk(2, dim=-1)                     # each (B, N, N, c)
        if self.outgoing:
            # sum_k  a[i,k] * b[j,k]  => pair[i,j]
            x = torch.einsum("...ikc,...jkc->...ijc", a, b)
        else:
            # sum_k  a[k,i] * b[k,j]  => pair[i,j]
            x = torch.einsum("...kic,...kjc->...ijc", a, b)
        x = self.p_out(self.norm_out(x.to(z.dtype))) * self.g_out(z_norm).sigmoid()
        return x


class TriangleMultiplicationOutgoing(_TriangleMultiplication):
    """AF2 Alg. 11 — pair[i,j] aggregates over shared k via a[i,k]*b[j,k]."""
    def __init__(self, c_z: int) -> None:
        super().__init__(c_z, outgoing=True)


class TriangleMultiplicationIncoming(_TriangleMultiplication):
    """AF2 Alg. 12 — pair[i,j] aggregates over shared k via a[k,i]*b[k,j]."""
    def __init__(self, c_z: int) -> None:
        super().__init__(c_z, outgoing=False)


# ---------------------------------------------------------------------------
# Triangle Self-Attention  (AF2 Algorithms 13 & 14, unchanged in AF3)
# ---------------------------------------------------------------------------

class _TriangleSelfAttention(nn.Module):
    """
    Shared base.  For starting node (row-wise): each row i independently
    attends over its j positions, with pair bias from z[i, k].
    For ending node (column-wise): same after transposing.

    Implementation strategy (Boltz/OF3): treat each row as a batch element.
      q/k/v: (B*N_row, N_col, H, d)
      bias:  (B*N_row, 1, H, N_col)  [key-position bias]
      Output gated by sigmoid(linear_g(z)).
    """

    def __init__(self, c_z: int, head_width: int, n_heads: int, starting: bool,
                 inf: float = 1e9) -> None:
        super().__init__()
        self.starting = starting
        self.n_heads = n_heads
        self.head_width = head_width
        self.inf = inf

        self.norm = nn.LayerNorm(c_z)
        self.linear_q = LinearNoBias(c_z, n_heads * head_width)
        self.linear_k = LinearNoBias(c_z, n_heads * head_width)
        self.linear_v = LinearNoBias(c_z, n_heads * head_width)
        # Project pair to n_heads scalar biases for each (row, col) key position
        self.linear_b = LinearNoBias(c_z, n_heads)
        self.linear_g = LinearNoBias(c_z, n_heads * head_width)
        self.linear_out = LinearNoBias(n_heads * head_width, c_z)
        gating_init_(self.linear_g.weight)
        final_init_(self.linear_out.weight)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # z: (B, N, N, c_z); mask: (B, N, N)
        if not self.starting:
            z = z.transpose(-2, -3)   # (B, N, N, c) -> columns become rows
            if mask is not None:
                mask = mask.transpose(-1, -2)

        B, N, M, _ = z.shape   # N = row ("starting node"), M = col (attended over)
        H, d = self.n_heads, self.head_width

        z_norm = self.norm(z)

        # For each starting-node row i, q/k/v come from z[i, j, :]
        q = self.linear_q(z_norm).view(B, N, M, H, d)  # (B, N, M, H, d)
        k = self.linear_k(z_norm).view(B, N, M, H, d)
        v = self.linear_v(z_norm).view(B, N, M, H, d)

        # Pair bias: for key position k at row i, bias = linear_b(z[i, k])
        # shape: (B, N, M, H) -> (B*N, H, 1, M) for broadcasting over query
        pair_bias = self.linear_b(z_norm)              # (B, N, M, H)
        pair_bias = pair_bias.view(B * N, M, H).permute(0, 2, 1).unsqueeze(2)  # (B*N, H, 1, M)

        # Flatten row dimension into batch for efficient attention
        q = q.view(B * N, M, H, d).permute(0, 2, 1, 3)  # (B*N, H, M, d)
        k = k.view(B * N, M, H, d).permute(0, 2, 1, 3)
        v = v.view(B * N, M, H, d).permute(0, 2, 1, 3)

        scale = d ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*N, H, M, M)
        attn = attn + pair_bias

        if mask is not None:
            # mask: (B, N, M) — valid positions; attending j to k, mask on k
            # (B, N, M) -> (B*N, 1, 1, M)
            m = mask.reshape(B * N, M).unsqueeze(1).unsqueeze(2)
            attn = attn + (1.0 - m.float()) * -self.inf

        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)           # (B*N, H, M, d)
        out = out.permute(0, 2, 1, 3).reshape(B, N, M, H * d)  # (B, N, M, H*d)

        # Gating: g = sigmoid(linear_g(z_norm))
        g = self.linear_g(z_norm).sigmoid()   # (B, N, M, H*d)
        out = self.linear_out(g * out)        # (B, N, M, c_z)

        if not self.starting:
            out = out.transpose(-2, -3)
        return out


class TriangleAttentionStartingNode(_TriangleSelfAttention):
    """AF2 Alg. 13 — for each starting node i, attend over j (row-wise)."""
    def __init__(self, c_z: int, head_width: int = 32, n_heads: int = 4,
                 inf: float = 1e9) -> None:
        super().__init__(c_z, head_width, n_heads, starting=True, inf=inf)


class TriangleAttentionEndingNode(_TriangleSelfAttention):
    """AF2 Alg. 14 — for each ending node j, attend over i (column-wise)."""
    def __init__(self, c_z: int, head_width: int = 32, n_heads: int = 4,
                 inf: float = 1e9) -> None:
        super().__init__(c_z, head_width, n_heads, starting=False, inf=inf)


# ---------------------------------------------------------------------------
# Attention with Pair Bias  (single track in pairformer; AF3 Alg. 24 / AF2 Alg. 6)
# ---------------------------------------------------------------------------

class AttentionPairBias(nn.Module):
    """
    Standard multi-head attention with an additive pair bias from z.
    Used for the single representation (s) track in pairformer blocks.
    Also used (with AdaLN) in the diffusion transformer.

    Boltz: attention.py AttentionPairBias
    OF3:   attention_pair_bias.py AttentionPairBias
    """

    def __init__(self, c_s: int, c_z: int, n_heads: int,
                 head_width: Optional[int] = None, inf: float = 1e6,
                 gating: bool = True) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_width = head_width if head_width is not None else c_s // n_heads
        assert c_s % n_heads == 0 or head_width is not None
        self.inf = inf
        self.gating = gating
        c_out = n_heads * self.head_width

        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_q = nn.Linear(c_s, c_out)
        self.proj_k = LinearNoBias(c_s, c_out)
        self.proj_v = LinearNoBias(c_s, c_out)
        # Project pair dim → n_heads scalar biases; bias shape (B, N, N, H) -> (B, H, N, N)
        self.proj_z = LinearNoBias(c_z, n_heads)
        if gating:
            self.proj_g = LinearNoBias(c_s, c_out)
            gating_init_(self.proj_g.weight)
        self.proj_o = LinearNoBias(c_out, c_s)
        final_init_(self.proj_o.weight)

    def forward(self, s: torch.Tensor, z: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # s: (B, N, c_s); z: (B, N, N, c_z); mask: (B, N)
        B, N, _ = s.shape
        H, d = self.n_heads, self.head_width

        s_n = self.norm_s(s)
        q = self.proj_q(s_n).view(B, N, H, d).transpose(1, 2)   # (B, H, N, d)
        k = self.proj_k(s_n).view(B, N, H, d).transpose(1, 2)
        v = self.proj_v(s_n).view(B, N, H, d).transpose(1, 2)

        # Pair bias: (B, N, N, H) -> (B, H, N, N)
        pair_bias = self.proj_z(self.norm_z(z)).permute(0, 3, 1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)  # (B, H, N, N)
        attn = attn + pair_bias

        if mask is not None:
            attn = attn + (1.0 - mask[:, None, None, :].float()) * -self.inf

        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)           # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, H * d)

        if self.gating:
            g = self.proj_g(s_n).sigmoid()
            out = g * out
        return self.proj_o(out)


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN-Zero) — used in diffusion transformer
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm conditioned on single representation s.
    Implements AdaLN-Zero: scale + shift initialized to identity/zero
    so the module acts as identity at init.

    OF3: primitives/AdaLN
    """

    def __init__(self, c_a: int, c_s: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_a, elementwise_affine=False)
        # Project single representation to scale (γ) and shift (β)
        self.proj = nn.Linear(c_s, 2 * c_a)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # a: (B, N, c_a); s: (B, N, c_s) or (B, 1, c_s)
        a_n = self.norm(a)
        gamma, beta = self.proj(s).chunk(2, dim=-1)
        return a_n * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Outer Product Mean  (MSA → pair update; AF2 Alg. 10, unchanged in AF3)
# ---------------------------------------------------------------------------

class OuterProductMean(nn.Module):
    """
    Computes mean outer product over sequences:
      z[i,j] += linear_out(mean_s(a[s,i] ⊗ b[s,j]))

    Boltz: outer_product_mean.py  |  OF3: outer_product_mean.py
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 32) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_m)
        self.proj_a = LinearNoBias(c_m, c_hidden)
        self.proj_b = LinearNoBias(c_m, c_hidden)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_z)
        final_init_(self.proj_o.weight)
        final_init_(self.proj_o.bias)

    def forward(self, m: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # m: (B, S, N, c_m); mask: (B, S, N)
        m = self.norm(m)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()   # (B, S, N, 1)
            a = self.proj_a(m) * mask_f           # (B, S, N, c_h)
            b = self.proj_b(m) * mask_f
            num = (mask[:, :, None, :] * mask[:, :, :, None]).float().sum(1).clamp(min=1)
        else:
            a = self.proj_a(m)
            b = self.proj_b(m)
            num = m.shape[1]  # scalar int

        # Outer product summed over sequences
        z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())  # (B, N, N, ch, ch)
        z = z / (num if isinstance(num, int) else num.unsqueeze(-1).unsqueeze(-1))
        z = z.reshape(*z.shape[:3], self.c_hidden ** 2)             # (B, N, N, ch²)
        return self.proj_o(z.to(m.dtype))


# ---------------------------------------------------------------------------
# MSA Pair-Weighted Averaging  (replaces AF2 row attention in AF3 MSA module)
# ---------------------------------------------------------------------------

class MSAPairWeightedAveraging(nn.Module):
    """
    AF3 Supplementary Methods 3.3 / OF3 msa.py MSAPairWeightedAveraging.
    For each residue position j and each MSA sequence s:
      w[s, i→j] = softmax_s( linear_w(z[i, j]) )
      m'[s, i] += sum_j( w[s, i→j] * linear_v(m[s, j]) )
    This is a cheap pair-biased aggregation replacing expensive row attention.
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, n_heads: int,
                 inf: float = 1e9) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.c_hidden = c_hidden
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_v = LinearNoBias(c_m, n_heads * c_hidden)
        # Pair bias projected to n_heads; shape (B, N, N, H)
        self.proj_bias = LinearNoBias(c_z, n_heads)
        self.proj_g = LinearNoBias(c_m, n_heads * c_hidden)
        self.proj_o = LinearNoBias(n_heads * c_hidden, c_m)
        gating_init_(self.proj_g.weight)
        final_init_(self.proj_o.weight)

    def forward(self, m: torch.Tensor, z: torch.Tensor,
                msa_mask: Optional[torch.Tensor] = None,
                pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # m: (B, S, N, c_m); z: (B, N, N, c_z)
        B, S, N, _ = m.shape
        H, d = self.n_heads, self.c_hidden

        m_n = self.norm_m(m)
        v = self.proj_v(m_n).view(B, S, N, H, d)      # (B, S, N, H, d)
        g = self.proj_g(m_n).sigmoid()                  # (B, S, N, H*d)

        # Pair bias: (B, N, N, H) — for each (i, j) pair
        bias = self.proj_bias(self.norm_z(z))           # (B, N, N, H)

        # For each sequence s and position i, compute weighted average over j
        # weight[i, j, h] = softmax_j(bias[i, j, h])
        # out[s, i, h, d] = sum_j weight[i, j, h] * v[s, j, h, d]

        # bias: (B, N, N, H) -> (B, 1, H, N, N) to broadcast over S
        w = bias.permute(0, 3, 1, 2).unsqueeze(1)      # (B, 1, H, N, N)

        if pair_mask is not None:
            w = w + (1.0 - pair_mask[:, None, None].float()) * -self.inf

        w = w.softmax(dim=-1)                           # (B, 1, H, N, N)

        # v: (B, S, N, H, d) -> (B, S, H, N, d)
        v_t = v.permute(0, 1, 3, 2, 4)                 # (B, S, H, N, d)

        # weighted sum: out[s, h, i, d] = sum_j w[h, i, j] * v[s, h, j, d]
        out = torch.matmul(w, v_t)                      # (B, S, H, N, d)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, S, N, H * d)  # (B, S, N, H*d)
        out = self.proj_o(g * out)                      # (B, S, N, c_m)

        if msa_mask is not None:
            out = out * msa_mask.unsqueeze(-1)
        return out
