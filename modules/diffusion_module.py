"""
Diffusion Module — AF3 Supplementary Methods 3.7 / Fig. 2b / Algorithm 20.

This is the core new component in AF3 vs AF2. It replaces the AF2 structure
module (which used SE(3)-equivariant frames + torsion angles) with a standard
denoising diffusion model operating directly on raw atom coordinates.

Architecture (Fig. 2b, left to right):
  Conditioning:
    SingleConditioning:   s_trunk + Fourier(sigma) -> s  (per-token noise-level cond.)
    PairwiseConditioning: z_trunk + rel_pos_encoding -> z
  AtomAttentionEncoder (3 seq-local blocks):
    ref atom features + noisy positions -> per-atom repr q
    -> aggregate to token-level a via weighted pooling
  Project s_trunk -> a (add trunk single to token repr)
  DiffusionTransformer (24 global blocks):
    AttentionPairBias (with AdaLN noise conditioning) + ConditionedTransition
  AtomAttentionDecoder (3 seq-local blocks):
    a -> per-atom repr -> coordinate updates

Noise schedule (Karras et al. 2022, AF3 p.24):
  sigma(t) = sigma_data * (s_max^(1/rho) + t*(s_min^(1/rho) - s_max^(1/rho)))^rho
  for t in [0, 1] (t=0 -> sigma_min, t=1 -> sigma_max)

Training:
  x_noisy = x_0 + sigma * epsilon,  epsilon ~ N(0, I)
  Denoiser D predicts x_0 from x_noisy conditioned on sigma and trunk activations
  Loss = (c_skip * x_noisy + c_out * D(x_noisy, sigma, ...)) - x_0)^2 with weighting

References:
  OF3:   model/structure/diffusion_module.py
  Boltz: model/modules/diffusion.py + encoders.py + transformers.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import (
    LinearNoBias, Transition, AttentionPairBias, AdaLayerNorm,
    lecun_normal_init_, gating_init_, final_init_,
)


# ---------------------------------------------------------------------------
# Karras noise schedule (AF3 Supplementary p.24)
# ---------------------------------------------------------------------------

def karras_noise_schedule(
    n_steps: int,
    sigma_data: float = 16.0,
    s_max: float = 160.0,
    s_min: float = 0.0004,
    rho: float = 7.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns sigma values for n_steps+1 timesteps (index 0 = sigma_min, n_steps = sigma_max).
    AF3 Supplementary Methods p.24 / OF3 diffusion_module.py create_noise_schedule.
    """
    t = torch.linspace(0, 1, n_steps + 1, device=device, dtype=dtype)
    sigma = sigma_data * (
        s_max ** (1.0 / rho) + t * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))
    ) ** rho
    return sigma   # (n_steps+1,)


def sample_train_sigmas(
    batch_size: int,
    sigma_data: float = 16.0,
    p_mean: float = -1.2,
    p_std: float = 1.5,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample noise levels at training time from a log-normal distribution.
    Boltz / Karras eq.: log(sigma) ~ N(p_mean, p_std^2)
    """
    log_sigma = torch.randn(batch_size, device=device) * p_std + p_mean
    return torch.exp(log_sigma) * sigma_data


# ---------------------------------------------------------------------------
# Fourier embedding for noise level (sigma) conditioning
# ---------------------------------------------------------------------------

class FourierEmbedding(nn.Module):
    """
    Random Fourier features embedding for log(sigma/sigma_data).
    Boltz: modules/encoders.py FourierEmbedding
    """

    def __init__(self, dim: int = 256, sigma_data: float = 16.0) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        # Fixed random frequencies
        self.register_buffer("freqs", torch.randn(dim // 2))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,)
        log_sigma = torch.log(sigma / self.sigma_data)             # (B,)
        x = log_sigma[:, None] * self.freqs[None, :] * 2 * math.pi  # (B, dim//2)
        return torch.cat([x.cos(), x.sin()], dim=-1)                # (B, dim)


# ---------------------------------------------------------------------------
# Single (token) conditioning on noise level  (AF3 Algorithm 21)
# ---------------------------------------------------------------------------

class SingleConditioning(nn.Module):
    """
    Combines trunk single representation with noise-level Fourier embedding
    to produce the single conditioning used in the diffusion transformer.

    s_cond = LayerNorm(s_trunk) + proj(Fourier(sigma))

    Boltz: modules/encoders.py SingleConditioning
    """

    def __init__(
        self,
        c_s: int = 384,
        dim_fourier: int = 256,
        sigma_data: float = 16.0,
        n_transitions: int = 2,
        transition_factor: int = 2,
    ) -> None:
        super().__init__()
        self.fourier = FourierEmbedding(dim_fourier, sigma_data)
        self.norm = nn.LayerNorm(c_s)
        self.proj_fourier = nn.Linear(dim_fourier, c_s)
        self.transitions = nn.ModuleList([
            Transition(c_s, hidden_factor=transition_factor) for _ in range(n_transitions)
        ])

    def forward(self, s_trunk: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # s_trunk: (B, N, c_s); sigma: (B,)
        fourier = self.fourier(sigma)                              # (B, dim_fourier)
        s_noise = self.proj_fourier(fourier).unsqueeze(1)         # (B, 1, c_s)
        s = self.norm(s_trunk) + s_noise                          # (B, N, c_s)
        for t in self.transitions:
            s = s + t(s)
        return s


# ---------------------------------------------------------------------------
# Pairwise conditioning  (AF3 Algorithm 22)
# ---------------------------------------------------------------------------

class PairwiseConditioning(nn.Module):
    """
    Combines trunk pair representation with token relative-position encoding.
    Boltz: modules/encoders.py PairwiseConditioning
    """

    def __init__(
        self,
        c_z: int = 128,
        n_transitions: int = 2,
        transition_factor: int = 2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_z)
        self.transitions = nn.ModuleList([
            Transition(c_z, hidden_factor=transition_factor) for _ in range(n_transitions)
        ])

    def forward(self, z_trunk: torch.Tensor, rel_pos: torch.Tensor) -> torch.Tensor:
        # z_trunk: (B, N, N, c_z); rel_pos: (B, N, N, c_z) (already projected in embedder)
        z = self.norm(z_trunk) + rel_pos
        for t in self.transitions:
            z = z + t(z)
        return z


# ---------------------------------------------------------------------------
# Conditioned transition block (used in diffusion transformer)
# ---------------------------------------------------------------------------

class ConditionedTransition(nn.Module):
    """
    SwiGLU transition conditioned on s via AdaLN (AdaLN-Zero).
    Used in the diffusion transformer blocks.

    OF3: layers/transition.py ConditionedTransitionBlock
    Boltz: modules/transformers.py ConditionedTransitionBlock
    """

    def __init__(self, c_a: int, c_s: int, hidden_factor: int = 2) -> None:
        super().__init__()
        hidden = c_a * hidden_factor
        self.ada_norm = AdaLayerNorm(c_a, c_s)
        self.fc1 = LinearNoBias(c_a, hidden)
        self.fc2 = LinearNoBias(c_a, hidden)   # gate
        self.fc3 = LinearNoBias(hidden, c_a)
        # Zero-init fc3 so this block starts as identity
        final_init_(self.fc3.weight)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        a_n = self.ada_norm(a, s)
        return self.fc3(F.silu(self.fc1(a_n)) * self.fc2(a_n))


# ---------------------------------------------------------------------------
# Diffusion Transformer block (AF3 Algorithm 23)
# ---------------------------------------------------------------------------

class DiffusionTransformerBlock(nn.Module):
    """
    Single block of the diffusion token transformer.
    Uses AdaLN attention (conditioning on s) + conditioned transition.

    OF3: layers/diffusion_transformer.py DiffusionTransformerBlock
    Boltz: modules/transformers.py DiffusionTransformerLayer
    """

    def __init__(
        self,
        c_a: int,      # token repr dim (2*c_s in boltz/AF3)
        c_s: int,      # single conditioning dim
        c_z: int,      # pair conditioning dim
        n_heads: int,
        head_width: int,
        transition_factor: int = 2,
        inf: float = 1e6,
    ) -> None:
        super().__init__()
        self.ada_norm = AdaLayerNorm(c_a, c_s)
        self.attn = AttentionPairBias(
            c_s=c_a, c_z=c_z, n_heads=n_heads, head_width=head_width, inf=inf,
        )
        self.cond_transition = ConditionedTransition(c_a, c_s, hidden_factor=transition_factor)

    def forward(
        self,
        a: torch.Tensor,   # (B, N, c_a) token-level activations
        s: torch.Tensor,   # (B, N, c_s) noise-conditioned single
        z: torch.Tensor,   # (B, N, N, c_z) pair
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # AdaLN-conditioned attention
        a_n = self.ada_norm(a, s)
        a = a + self.attn(a_n, z, mask=mask)
        # Conditioned transition
        a = a + self.cond_transition(a, s)
        return a


class DiffusionTransformer(nn.Module):
    """24-block global token transformer for diffusion."""

    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        n_blocks: int = 24,
        n_heads: int = 16,
        head_width: int = 24,
        transition_factor: int = 2,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(c_a, c_s, c_z, n_heads, head_width, transition_factor)
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                a = torch.utils.checkpoint.checkpoint(block, a, s, z, mask)
            else:
                a = block(a, s, z, mask)
        return a


# ---------------------------------------------------------------------------
# Atom Attention Encoder (seq-local, 3 blocks)  AF3 Algorithm 6
# ---------------------------------------------------------------------------

class AtomTransformerBlock(nn.Module):
    """
    One block of the atom-level transformer (inside the encoder/decoder).
    Same structure as SequenceLocalAtomAttention in input_embedder.py but
    additionally conditioned on the single representation s via AdaLN.
    """

    def __init__(self, c_a: int, c_ap: int, c_s: int, n_heads: int, head_width: int) -> None:
        super().__init__()
        c_out = n_heads * head_width
        self.inf = 1e9
        self.ada_norm = AdaLayerNorm(c_a, c_s)
        self.proj_q = nn.Linear(c_a, c_out)
        self.proj_k = LinearNoBias(c_a, c_out)
        self.proj_v = LinearNoBias(c_a, c_out)
        self.proj_bias = LinearNoBias(c_ap, n_heads)
        self.proj_g = LinearNoBias(c_a, c_out)
        self.proj_o = LinearNoBias(c_out, c_a)
        gating_init_(self.proj_g.weight)
        final_init_(self.proj_o.weight)
        self.cond_transition = ConditionedTransition(c_a, c_s)
        self.n_heads = n_heads
        self.head_width = head_width

    def forward(
        self,
        a: torch.Tensor,    # (B, A, c_a)
        p: torch.Tensor,    # (B, A, A, c_ap)
        s: torch.Tensor,    # (B, A, c_s) — broadcast from token-level to atoms
        atom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, A, _ = a.shape
        H, d = self.n_heads, self.head_width

        a_n = self.ada_norm(a, s)
        q = self.proj_q(a_n).view(B, A, H, d).transpose(1, 2)
        k = self.proj_k(a_n).view(B, A, H, d).transpose(1, 2)
        v = self.proj_v(a_n).view(B, A, H, d).transpose(1, 2)

        bias = self.proj_bias(p).permute(0, 3, 1, 2)   # (B, H, A, A)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5) + bias
        if atom_mask is not None:
            attn = attn + (1.0 - atom_mask[:, None, None, :].float()) * -self.inf
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, A, H * d)
        g = self.proj_g(a_n).sigmoid()
        a = a + self.proj_o(g * out)
        a = a + self.cond_transition(a, s)
        return a


class AtomAttentionEncoder(nn.Module):
    """
    3-block atom-level encoder (seq-local) that processes noisy atom positions.
    Outputs per-atom representation q and token-level aggregated representation a.

    OF3: layers/sequence_local_atom_attention.py AtomAttentionEncoder
    Boltz: modules/encoders.py AtomAttentionEncoder
    """

    def __init__(
        self,
        c_a: int,     # atom single dim
        c_ap: int,    # atom pair dim
        c_s: int,     # single (token) dim for conditioning
        c_token: int, # output token dim (= 2*c_s in Boltz)
        n_blocks: int = 3,
        n_heads: int = 4,
        head_width: int = 32,
    ) -> None:
        super().__init__()
        # Embed noisy positions (3 -> c_a)
        self.proj_pos = LinearNoBias(3, c_a)

        # Atom-level pair embedding for encoder blocks
        self.proj_ap_offset = LinearNoBias(3, c_ap)

        self.blocks = nn.ModuleList([
            AtomTransformerBlock(c_a, c_ap, c_s, n_heads, head_width)
            for _ in range(n_blocks)
        ])

        # Project atom single -> token single via mean-pooling then linear
        self.proj_out = LinearNoBias(c_a, c_token)

    def forward(
        self,
        a_ref: torch.Tensor,           # (B, A, c_a) from input embedder
        p_ref: torch.Tensor,           # (B, A, A, c_ap) from input embedder
        r_noisy: torch.Tensor,         # (B, A, 3) noisy atom positions
        s_broadcast: torch.Tensor,     # (B, A, c_s) s_trunk broadcast to atoms
        atom_mask: Optional[torch.Tensor] = None,
        num_atoms_per_token: Optional[torch.Tensor] = None,
        n_tokens: Optional[int] = None,
    ) -> tuple:
        # Add noisy position info to atom repr
        a = a_ref + self.proj_pos(r_noisy)

        # Atom pair: offset between noisy positions
        offset = r_noisy.unsqueeze(2) - r_noisy.unsqueeze(1)   # (B, A, A, 3)
        p = p_ref + self.proj_ap_offset(offset)

        for block in self.blocks:
            a = block(a, p, s_broadcast, atom_mask)

        q = a   # per-atom repr before aggregation (used as skip connection in decoder)

        # Aggregate to tokens
        if num_atoms_per_token is not None and n_tokens is not None:
            from .input_embedder import aggregate_atoms_to_tokens
            a_tok = aggregate_atoms_to_tokens(a, num_atoms_per_token, n_tokens)
        else:
            a_tok = a.mean(1, keepdim=True)   # fallback

        return q, self.proj_out(a_tok)  # (B, A, c_a), (B, N, c_token)


# ---------------------------------------------------------------------------
# Atom Attention Decoder (seq-local, 3 blocks)  AF3 Algorithm 7
# ---------------------------------------------------------------------------

class AtomAttentionDecoder(nn.Module):
    """
    3-block atom-level decoder that maps updated token activations back to
    per-atom coordinate updates.

    OF3: layers/sequence_local_atom_attention.py AtomAttentionDecoder
    Boltz: modules/encoders.py AtomAttentionDecoder
    """

    def __init__(
        self,
        c_a: int,
        c_ap: int,
        c_s: int,
        c_token: int,
        n_blocks: int = 3,
        n_heads: int = 4,
        head_width: int = 32,
    ) -> None:
        super().__init__()
        # Project token activations -> atom single
        self.proj_token_to_atom = LinearNoBias(c_token, c_a)

        self.blocks = nn.ModuleList([
            AtomTransformerBlock(c_a, c_ap, c_s, n_heads, head_width)
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(c_a)
        # Output 3D coordinate updates
        self.proj_coord = LinearNoBias(c_a, 3)
        final_init_(self.proj_coord.weight)

    def forward(
        self,
        a_token: torch.Tensor,          # (B, N, c_token) updated from transformer
        q_skip: torch.Tensor,           # (B, A, c_a) skip from encoder
        p_ref: torch.Tensor,            # (B, A, A, c_ap)
        s_broadcast: torch.Tensor,      # (B, A, c_s) single broadcast to atoms
        atom_to_token: torch.Tensor,    # (B, A) long
        atom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, A, _ = q_skip.shape
        N = a_token.shape[1]

        # Broadcast token activations to atoms
        idx = atom_to_token.clamp(0, N - 1)
        a_from_tokens = a_token[torch.arange(B, device=a_token.device).unsqueeze(1), idx]
        a = q_skip + self.proj_token_to_atom(a_from_tokens)

        for block in self.blocks:
            a = block(a, p_ref, s_broadcast, atom_mask)

        r_update = self.proj_coord(self.norm(a))   # (B, A, 3)
        return r_update


# ---------------------------------------------------------------------------
# Full Diffusion Module  (AF3 Algorithm 20)
# ---------------------------------------------------------------------------

class DiffusionModule(nn.Module):
    """
    AF3 diffusion module.

    At training time, receives noisy coordinates x_noisy = x_0 + sigma * eps
    and predicts the clean coordinates x_0.

    At inference time, iteratively denoises from random noise using the
    Karras noise schedule.
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_a: int = 128,
        c_ap: int = 16,
        sigma_data: float = 16.0,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        atom_encoder_head_width: int = 32,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 16,
        token_transformer_head_width: int = 24,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_decoder_head_width: int = 32,
        n_conditioning_transitions: int = 2,
        transition_factor: int = 2,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data

        # Token dim in the transformer = 2*c_s (Boltz convention)
        c_token = 2 * c_s

        self.single_cond = SingleConditioning(
            c_s=c_s, dim_fourier=dim_fourier, sigma_data=sigma_data,
            n_transitions=n_conditioning_transitions, transition_factor=transition_factor,
        )
        self.pair_cond = PairwiseConditioning(
            c_z=c_z, n_transitions=n_conditioning_transitions,
            transition_factor=transition_factor,
        )

        self.atom_encoder = AtomAttentionEncoder(
            c_a=c_a, c_ap=c_ap, c_s=c_s, c_token=c_token,
            n_blocks=atom_encoder_depth,
            n_heads=atom_encoder_heads, head_width=atom_encoder_head_width,
        )

        # Linear projection of s_trunk into the token activation space
        self.proj_s_trunk = nn.Sequential(
            nn.LayerNorm(c_s),
            LinearNoBias(c_s, c_token),
        )
        final_init_(self.proj_s_trunk[1].weight)

        self.token_transformer = DiffusionTransformer(
            c_a=c_token, c_s=c_s, c_z=c_z,
            n_blocks=token_transformer_depth,
            n_heads=token_transformer_heads, head_width=token_transformer_head_width,
            transition_factor=transition_factor,
            activation_checkpointing=activation_checkpointing,
        )

        self.norm_a = nn.LayerNorm(c_token)

        self.atom_decoder = AtomAttentionDecoder(
            c_a=c_a, c_ap=c_ap, c_s=c_s, c_token=c_token,
            n_blocks=atom_decoder_depth,
            n_heads=atom_decoder_heads, head_width=atom_decoder_head_width,
        )

    # ------------------------------------------------------------------
    # Karras preconditioning scalars  (Karras et al. 2022 eq. 7)
    # ------------------------------------------------------------------

    def _preconditioning(self, sigma: torch.Tensor) -> tuple:
        """
        Returns (c_skip, c_out, c_in, c_noise) for a given sigma.
        c_skip * x_noisy + c_out * F(c_in * x_noisy, c_noise) = D(x_noisy)
        """
        sd = self.sigma_data
        c_skip  = sd**2 / (sigma**2 + sd**2)
        c_out   = sigma * sd / (sigma**2 + sd**2).sqrt()
        c_in    = 1.0 / (sigma**2 + sd**2).sqrt()
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    # ------------------------------------------------------------------
    # Single forward pass of the denoiser network
    # ------------------------------------------------------------------

    def _denoiser_network(
        self,
        r_noisy: torch.Tensor,    # (B, A, 3) noisy coordinates * c_in
        sigma: torch.Tensor,      # (B,)
        s_trunk: torch.Tensor,    # (B, N, c_s) from pairformer
        z_trunk: torch.Tensor,    # (B, N, N, c_z) from pairformer
        a_ref: torch.Tensor,      # (B, A, c_a) from input embedder
        p_ref: torch.Tensor,      # (B, A, A, c_ap) from input embedder
        rel_pos_z: torch.Tensor,  # (B, N, N, c_z) relative position encoding
        feats: dict,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, A, _ = r_noisy.shape
        N = s_trunk.shape[1]

        # 1. Conditioning
        s = self.single_cond(s_trunk, sigma)                     # (B, N, c_s)
        z = self.pair_cond(z_trunk, rel_pos_z)                   # (B, N, N, c_z)

        # Broadcast s to atom level via atom_to_token
        atom_to_token = feats["atom_to_token"]                   # (B, A)
        idx = atom_to_token.clamp(0, N - 1)
        s_atoms = s[torch.arange(B, device=s.device).unsqueeze(1), idx]  # (B, A, c_s)

        atom_mask = feats["atom_mask"]

        # 2. Atom encoder (3 seq-local blocks)
        q_skip, a_tok = self.atom_encoder(
            a_ref, p_ref, r_noisy, s_atoms, atom_mask,
            feats["num_atoms_per_token"], N,
        )  # q_skip: (B, A, c_a), a_tok: (B, N, c_token)

        # 3. Add trunk single to token activations
        a = a_tok + self.proj_s_trunk(s_trunk)                   # (B, N, c_token)

        # 4. Token transformer (24 global blocks)
        a = self.token_transformer(a, s, z, mask=mask)           # (B, N, c_token)
        a = self.norm_a(a)

        # 5. Atom decoder (3 seq-local blocks)
        r_update = self.atom_decoder(
            a, q_skip, p_ref, s_atoms, atom_to_token, atom_mask,
        )   # (B, A, 3)

        return r_update

    # ------------------------------------------------------------------
    # Training forward  (predict clean coords from noisy input)
    # ------------------------------------------------------------------

    def forward(
        self,
        r_noisy: torch.Tensor,    # (B, A, 3) noisy atom positions
        sigma: torch.Tensor,      # (B,) noise level
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        a_ref: torch.Tensor,
        p_ref: torch.Tensor,
        rel_pos_z: torch.Tensor,
        feats: dict,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns predicted clean coordinates x_0_pred: (B, A, 3).
        Uses Karras preconditioning: D(x) = c_skip*x + c_out*F(c_in*x, c_noise).
        """
        c_skip, c_out, c_in, _ = self._preconditioning(sigma)

        # Shape: (B, 1, 1) for broadcasting over atoms
        def bcast(t):
            return t.view(-1, 1, 1)

        x_in = bcast(c_in) * r_noisy
        r_update = self._denoiser_network(
            x_in, sigma, s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, feats, mask, pair_mask,
        )
        x_pred = bcast(c_skip) * r_noisy + bcast(c_out) * r_update
        return x_pred

    # ------------------------------------------------------------------
    # Inference: iterative denoising
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        a_ref: torch.Tensor,
        p_ref: torch.Tensor,
        rel_pos_z: torch.Tensor,
        feats: dict,
        n_steps: int = 200,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        rho: float = 7.0,
        mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        gamma: float = 0.8,   # stochastic parameter (0 = deterministic DDIM-like)
    ) -> torch.Tensor:
        """
        Generates atom positions by iterative denoising.
        Returns: (B, A, 3) predicted clean atom coordinates.
        """
        B = s_trunk.shape[0]
        A = feats["atom_mask"].shape[1]
        device = s_trunk.device
        dtype = s_trunk.dtype

        sigmas = karras_noise_schedule(
            n_steps, self.sigma_data, sigma_max, sigma_min, rho, device, dtype
        )  # (n_steps+1,), sigmas[0] = sigma_max, sigmas[-1] = sigma_min

        # Start from pure noise
        x = torch.randn(B, A, 3, device=device, dtype=dtype) * sigmas[0]

        for i in range(n_steps):
            sig_cur = sigmas[i].expand(B)
            sig_next = sigmas[i + 1].expand(B)

            # Optionally add extra noise for stochastic sampling
            if gamma > 0 and i < n_steps - 1:
                sig_hat = sig_cur * (1.0 + gamma)
                noise = torch.randn_like(x) * (sig_hat**2 - sig_cur**2).sqrt()
                x = x + noise[:, None] if noise.dim() == 1 else (
                    x + noise.view(B, 1, 1) * (sig_hat**2 - sig_cur**2).sqrt().view(B, 1, 1)
                )
                x = x + torch.randn_like(x) * ((sig_hat.view(B,1,1)**2 - sig_cur.view(B,1,1)**2).clamp(min=0).sqrt())
                sig_cur = sig_hat

            # Denoise
            x_pred = self(
                x, sig_cur, s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, feats, mask, pair_mask,
            )

            # Euler step
            d = (x - x_pred) / sig_cur.view(B, 1, 1)
            dt = sig_next.view(B, 1, 1) - sig_cur.view(B, 1, 1)
            x = x + d * dt

        return x
