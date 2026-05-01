"""
Unit tests for modules/primitives.py

Tests cover:
- Output shapes
- Weight initialization (gating zeros, final proj zeros)
- Residual-safe: output near zero at init for gated modules
- Gradient flow
- Masking behaviour
"""

import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.primitives import (
    Transition,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
    AttentionPairBias,
    AdaLayerNorm,
    OuterProductMean,
    MSAPairWeightedAveraging,
    gating_init_, final_init_, lecun_normal_init_,
)

B, N, C_Z, C_S, C_M = 2, 8, 128, 384, 64


def _randomize(module):
    """Replace all zero-init weights with random values so gated outputs are non-zero."""
    with torch.no_grad():
        for p in module.parameters():
            if p.abs().max() == 0:
                torch.nn.init.normal_(p, std=0.02)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pair():
    return torch.randn(B, N, N, C_Z)

@pytest.fixture
def single():
    return torch.randn(B, N, C_S)

@pytest.fixture
def msa():
    return torch.randn(B, 4, N, C_M)

@pytest.fixture
def pair_mask():
    return torch.ones(B, N, N)

@pytest.fixture
def token_mask():
    return torch.ones(B, N)

@pytest.fixture
def msa_mask():
    return torch.ones(B, 4, N)


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

class TestTransition:
    def test_shape(self, pair):
        out = Transition(C_Z)(pair)
        assert out.shape == pair.shape

    def test_shape_single(self, single):
        out = Transition(C_S)(single)
        assert out.shape == single.shape

    def test_custom_out_dim(self, pair):
        out = Transition(C_Z, out_dim=64)(pair)
        assert out.shape == (B, N, N, 64)

    def test_gradients_flow(self, pair):
        pair = pair.requires_grad_(True)
        out = Transition(C_Z)(pair)
        out.sum().backward()
        assert pair.grad is not None
        assert not torch.isnan(pair.grad).any()

    def test_fc3_zero_init(self):
        t = Transition(C_Z)
        assert t.fc3.weight.abs().max().item() == 0.0, "fc3 should be zero-init (final_init_)"


# ---------------------------------------------------------------------------
# Triangle Multiplicative Update
# ---------------------------------------------------------------------------

class TestTriangleMult:
    @pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
    def test_shape(self, cls, pair, pair_mask):
        out = cls(C_Z)(pair, pair_mask)
        assert out.shape == pair.shape

    @pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
    def test_shape_no_mask(self, cls, pair):
        out = cls(C_Z)(pair)
        assert out.shape == pair.shape

    def test_output_near_zero_at_init(self, pair, pair_mask):
        """g_out is zero-init => output should be near zero before any training."""
        mod = TriangleMultiplicationOutgoing(C_Z)
        with torch.no_grad():
            out = mod(pair, pair_mask)
        assert out.abs().mean().item() < 0.01, f"Expected near-zero at init, got {out.abs().mean():.4f}"

    def test_gating_weight_zero_init(self):
        mod = TriangleMultiplicationOutgoing(C_Z)
        assert mod.g_in.weight.abs().max().item() == 0.0
        assert mod.g_out.weight.abs().max().item() == 0.0

    def test_mask_zeros_out_masked_positions(self):
        """Positions where mask=0 should not contribute to the output."""
        mod = _randomize(TriangleMultiplicationOutgoing(C_Z))
        z = torch.randn(1, 4, 4, C_Z)
        mask_full = torch.ones(1, 4, 4)
        mask_half = torch.zeros(1, 4, 4)
        mask_half[:, :2, :2] = 1.0  # only top-left quadrant

        out_full = mod(z, mask_full)
        out_half = mod(z, mask_half)
        # They must differ (mask actually does something)
        assert not torch.allclose(out_full, out_half)

    @pytest.mark.parametrize("cls", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming])
    def test_gradients_flow(self, cls, pair, pair_mask):
        pair = pair.requires_grad_(True)
        out = cls(C_Z)(pair, pair_mask)
        out.sum().backward()
        assert pair.grad is not None and not torch.isnan(pair.grad).any()

    def test_outgoing_vs_incoming_differ(self, pair, pair_mask):
        """Outgoing and incoming should produce different results on asymmetric input."""
        # Randomize weights so gated outputs are non-zero
        out_out = _randomize(TriangleMultiplicationOutgoing(C_Z))(pair, pair_mask)
        out_in  = _randomize(TriangleMultiplicationIncoming(C_Z))(pair, pair_mask)
        assert not torch.allclose(out_out, out_in)


# ---------------------------------------------------------------------------
# Triangle Self-Attention
# ---------------------------------------------------------------------------

class TestTriangleAttention:
    @pytest.mark.parametrize("cls", [TriangleAttentionStartingNode, TriangleAttentionEndingNode])
    def test_shape(self, cls, pair, pair_mask):
        out = cls(C_Z, head_width=32, n_heads=4)(pair, pair_mask)
        assert out.shape == pair.shape

    @pytest.mark.parametrize("cls", [TriangleAttentionStartingNode, TriangleAttentionEndingNode])
    def test_shape_no_mask(self, cls, pair):
        out = cls(C_Z, head_width=32, n_heads=4)(pair)
        assert out.shape == pair.shape

    def test_gating_zero_init(self):
        mod = TriangleAttentionStartingNode(C_Z)
        assert mod.linear_g.weight.abs().max().item() == 0.0
        assert mod.linear_out.weight.abs().max().item() == 0.0

    def test_output_near_zero_at_init(self, pair):
        """linear_g zero-init => gated output near zero."""
        mod = TriangleAttentionStartingNode(C_Z)
        with torch.no_grad():
            out = mod(pair)
        assert out.abs().mean().item() < 0.01

    def test_starting_vs_ending_differ(self, pair, pair_mask):
        # Randomize weights so gated outputs are non-zero
        out_s = _randomize(TriangleAttentionStartingNode(C_Z))(pair, pair_mask)
        out_e = _randomize(TriangleAttentionEndingNode(C_Z))(pair, pair_mask)
        assert not torch.allclose(out_s, out_e)

    @pytest.mark.parametrize("cls", [TriangleAttentionStartingNode, TriangleAttentionEndingNode])
    def test_gradients_flow(self, cls, pair, pair_mask):
        pair = pair.requires_grad_(True)
        out = cls(C_Z)(pair, pair_mask)
        out.sum().backward()
        assert pair.grad is not None and not torch.isnan(pair.grad).any()


# ---------------------------------------------------------------------------
# AttentionPairBias
# ---------------------------------------------------------------------------

class TestAttentionPairBias:
    def test_shape(self, single, pair, token_mask):
        out = AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24)(single, pair, token_mask)
        assert out.shape == single.shape

    def test_shape_no_mask(self, single, pair):
        out = AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24)(single, pair)
        assert out.shape == single.shape

    def test_proj_o_zero_init(self):
        mod = AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24)
        assert mod.proj_o.weight.abs().max().item() == 0.0

    def test_gating_zero_init(self):
        mod = AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24)
        assert mod.proj_g.weight.abs().max().item() == 0.0

    def test_output_near_zero_at_init(self, single, pair):
        mod = AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24)
        with torch.no_grad():
            out = mod(single, pair)
        assert out.abs().mean().item() < 0.01

    def test_mask_changes_output(self, single, pair):
        mod = _randomize(AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24))
        mask_full = torch.ones(B, N)
        mask_half = torch.zeros(B, N); mask_half[:, :N//2] = 1.0
        out_full = mod(single, pair, mask_full)
        out_half = mod(single, pair, mask_half)
        assert not torch.allclose(out_full, out_half)

    def test_gradients_flow(self, single, pair):
        single = single.requires_grad_(True)
        out = AttentionPairBias(C_S, C_Z, n_heads=16, head_width=24)(single, pair)
        out.sum().backward()
        assert single.grad is not None and not torch.isnan(single.grad).any()


# ---------------------------------------------------------------------------
# AdaLayerNorm
# ---------------------------------------------------------------------------

class TestAdaLayerNorm:
    def test_shape(self, single):
        a = torch.randn(B, N, 256)
        s = single
        mod = AdaLayerNorm(c_a=256, c_s=C_S)
        out = mod(a, s)
        assert out.shape == a.shape

    def test_identity_at_init(self):
        """proj is zero-init => at init, gamma=0, beta=0 => output = LayerNorm(a)."""
        a = torch.randn(2, 6, 64)
        s = torch.randn(2, 6, 128)
        mod = AdaLayerNorm(c_a=64, c_s=128)
        with torch.no_grad():
            out = mod(a, s)
            expected = nn.LayerNorm(64, elementwise_affine=False)(a)
        assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# OuterProductMean
# ---------------------------------------------------------------------------

class TestOuterProductMean:
    def test_shape(self, msa, msa_mask):
        out = OuterProductMean(C_M, C_Z, c_hidden=32)(msa, msa_mask)
        assert out.shape == (B, N, N, C_Z)

    def test_shape_no_mask(self, msa):
        out = OuterProductMean(C_M, C_Z, c_hidden=32)(msa)
        assert out.shape == (B, N, N, C_Z)

    def test_proj_o_zero_init(self):
        mod = OuterProductMean(C_M, C_Z, c_hidden=32)
        assert mod.proj_o.weight.abs().max().item() == 0.0
        assert mod.proj_o.bias.abs().max().item() == 0.0

    def test_symmetric_on_symmetric_input(self):
        """If m is constant across positions, outer product should be symmetric."""
        mod = OuterProductMean(C_M, C_Z, c_hidden=32)
        # Make m identical at all positions
        m_val = torch.randn(1, C_M)
        m = m_val.unsqueeze(0).unsqueeze(0).expand(1, 4, N, C_M).clone()
        with torch.no_grad():
            out = mod(m)
        # z[i,j] should equal z[j,i] up to floating point
        assert torch.allclose(out, out.transpose(1, 2), atol=1e-5)

    def test_gradients_flow(self, msa, msa_mask):
        msa = msa.requires_grad_(True)
        out = OuterProductMean(C_M, C_Z)(msa, msa_mask)
        out.sum().backward()
        assert msa.grad is not None and not torch.isnan(msa.grad).any()


# ---------------------------------------------------------------------------
# MSAPairWeightedAveraging
# ---------------------------------------------------------------------------

class TestMSAPairWeightedAveraging:
    def test_shape(self, msa, pair, msa_mask, pair_mask):
        mod = MSAPairWeightedAveraging(C_M, C_Z, c_hidden=8, n_heads=8)
        out = mod(msa, pair, msa_mask=msa_mask, pair_mask=pair_mask)
        assert out.shape == msa.shape

    def test_shape_no_mask(self, msa, pair):
        mod = MSAPairWeightedAveraging(C_M, C_Z, c_hidden=8, n_heads=8)
        out = mod(msa, pair)
        assert out.shape == msa.shape

    def test_gating_zero_init(self):
        mod = MSAPairWeightedAveraging(C_M, C_Z, c_hidden=8, n_heads=8)
        assert mod.proj_g.weight.abs().max().item() == 0.0
        assert mod.proj_o.weight.abs().max().item() == 0.0

    def test_output_near_zero_at_init(self, msa, pair):
        mod = MSAPairWeightedAveraging(C_M, C_Z, c_hidden=8, n_heads=8)
        with torch.no_grad():
            out = mod(msa, pair)
        assert out.abs().mean().item() < 0.01

    def test_msa_mask_zeros_output(self, msa, pair):
        """Masked MSA rows should produce zero output."""
        mod = MSAPairWeightedAveraging(C_M, C_Z, c_hidden=8, n_heads=8)
        msa_mask = torch.ones(B, 4, N)
        msa_mask[:, 2:, :] = 0.0   # mask out last two sequences
        out = mod(msa, pair, msa_mask=msa_mask)
        # Rows 2 and 3 should be zeroed out
        assert out[:, 2:, :, :].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Weight initialization helpers
# ---------------------------------------------------------------------------

class TestWeightInit:
    def test_gating_init_zeros(self):
        w = torch.randn(64, 128)
        gating_init_(w)
        assert w.abs().max().item() == 0.0

    def test_final_init_zeros(self):
        w = torch.randn(64, 128)
        final_init_(w)
        assert w.abs().max().item() == 0.0

    def test_lecun_normal_std(self):
        """LeCun normal: std ≈ 1/sqrt(fan_in)."""
        fan_in = 256
        w = torch.empty(64, fan_in)
        lecun_normal_init_(w)
        expected_std = (1.0 / fan_in) ** 0.5
        actual_std = w.std().item()
        assert abs(actual_std - expected_std) < 0.02, \
            f"Expected std~{expected_std:.4f}, got {actual_std:.4f}"
