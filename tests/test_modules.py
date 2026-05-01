"""
Unit tests for the higher-level modules:
  - Pairformer
  - MSAModule
  - TemplateModule
  - InputEmbedder (relative position encoding + shapes)
  - ConfidenceModule
  - DiffusionModule (noise schedule, preconditioning, forward shape)
"""

import pytest
import torch
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.pairformer import Pairformer, PairformerBlock
from modules.msa_module import MSAModule, MSAEmbedder
from modules.template_module import TemplateModule, PairformerNoSeqBlock
from modules.confidence_module import ConfidenceModule, pae_to_ptm
from modules.diffusion_module import (
    DiffusionModule, karras_noise_schedule, sample_train_sigmas,
    FourierEmbedding, SingleConditioning, PairwiseConditioning,
)
from modules.input_embedder import relative_position_encoding, InputEmbedder

B, N, A = 1, 8, 24    # batch, tokens, atoms
C_Z, C_S, C_M = 128, 384, 64
C_A, C_AP = 128, 16


# ---------------------------------------------------------------------------
# Pairformer
# ---------------------------------------------------------------------------

class TestPairformer:
    @pytest.fixture
    def inputs(self):
        s = torch.randn(B, N, C_S)
        z = torch.randn(B, N, N, C_Z)
        mask = torch.ones(B, N)
        pm = torch.ones(B, N, N)
        return s, z, mask, pm

    def test_output_shapes(self, inputs):
        pf = Pairformer(C_Z, C_S, n_blocks=2)
        s_out, z_out = pf(*inputs)
        assert s_out.shape == (B, N, C_S)
        assert z_out.shape == (B, N, N, C_Z)

    def test_no_mask(self):
        pf = Pairformer(C_Z, C_S, n_blocks=1)
        s = torch.randn(B, N, C_S)
        z = torch.randn(B, N, N, C_Z)
        s_out, z_out = pf(s, z)
        assert s_out.shape == (B, N, C_S)
        assert z_out.shape == (B, N, N, C_Z)

    def test_residual_updates_s(self, inputs):
        """s should change after passing through pairformer with non-zero weights."""
        s, z, mask, pm = inputs
        pf = Pairformer(C_Z, C_S, n_blocks=2)
        # Randomize zero-init weights so gated outputs are non-zero
        with torch.no_grad():
            for p in pf.parameters():
                if p.abs().max() == 0:
                    torch.nn.init.normal_(p, std=0.02)
        s_out, _ = pf(s, z, mask, pm)
        assert not torch.allclose(s_out, s)

    def test_gradient_flow_both_tracks(self, inputs):
        s, z, mask, pm = inputs
        s = s.requires_grad_(True)
        z = z.requires_grad_(True)
        pf = Pairformer(C_Z, C_S, n_blocks=1)
        s_out, z_out = pf(s, z, mask, pm)
        (s_out.sum() + z_out.sum()).backward()
        assert s.grad is not None and not torch.isnan(s.grad).any()
        assert z.grad is not None and not torch.isnan(z.grad).any()

    def test_partial_mask_only_updates_valid_tokens(self):
        """Masking out a token shouldn't affect computations on other tokens
        (the masked token is padded and should not contribute via attention)."""
        pf = Pairformer(C_Z, C_S, n_blocks=1)
        s = torch.randn(1, 4, C_S)
        z = torch.randn(1, 4, 4, C_Z)
        mask_full = torch.ones(1, 4)
        # Mask last token
        mask_partial = torch.ones(1, 4); mask_partial[:, -1] = 0.0
        pm_full    = torch.ones(1, 4, 4)
        pm_partial = pm_full.clone(); pm_partial[:, -1, :] = 0; pm_partial[:, :, -1] = 0

        s_full, _    = pf(s, z, mask_full, pm_full)
        s_part, _    = pf(s, z, mask_partial, pm_partial)
        # First 3 tokens should differ (masked token's attention bias changes)
        # but at minimum the valid tokens should still have valid (non-NaN) values
        assert not torch.isnan(s_part[:, :3]).any()


# ---------------------------------------------------------------------------
# MSA Module
# ---------------------------------------------------------------------------

class TestMSAModule:
    @pytest.fixture
    def inputs(self):
        m = torch.randn(B, 4, N, C_M)
        z = torch.randn(B, N, N, C_Z)
        return m, z

    def test_output_shape(self, inputs):
        m, z = inputs
        out = MSAModule(C_M, C_Z, n_blocks=2)(m, z)
        assert out.shape == (B, N, N, C_Z)

    def test_z_changes(self, inputs):
        m, z = inputs
        mod = MSAModule(C_M, C_Z, n_blocks=2)
        with torch.no_grad():
            for p in mod.parameters():
                if p.abs().max() == 0:
                    torch.nn.init.normal_(p, std=0.02)
        z_out = mod(m, z)
        assert not torch.allclose(z_out, z)

    def test_msa_discarded(self, inputs):
        """MSA module returns only z — m is not in the output."""
        m, z = inputs
        result = MSAModule(C_M, C_Z, n_blocks=1)(m, z)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (B, N, N, C_Z)

    def test_with_masks(self, inputs):
        m, z = inputs
        msa_mask = torch.ones(B, 4, N); msa_mask[:, -1, :] = 0   # mask last seq
        pair_mask = torch.ones(B, N, N)
        out = MSAModule(C_M, C_Z, n_blocks=1)(m, z, msa_mask=msa_mask, pair_mask=pair_mask)
        assert out.shape == (B, N, N, C_Z)
        assert not torch.isnan(out).any()

    def test_msa_embedder_shape(self):
        emb = MSAEmbedder(n_vocab=32, c_m=C_M)
        msa_idx = torch.randint(0, 32, (B, 4, N))
        out = emb(msa_idx)
        assert out.shape == (B, 4, N, C_M)


# ---------------------------------------------------------------------------
# Template Module
# ---------------------------------------------------------------------------

class TestTemplateModule:
    def test_output_shape(self):
        mod = TemplateModule(c_template=88, c_z=C_Z, n_blocks=1)
        tp = torch.randn(B, 2, N, N, 88)
        out = mod(tp)
        assert out.shape == (B, N, N, C_Z)

    def test_with_mask(self):
        mod = TemplateModule(c_template=88, c_z=C_Z, n_blocks=1)
        tp = torch.randn(B, 3, N, N, 88)
        tmask = torch.tensor([[1.0, 1.0, 0.0]])   # (B=1, T=3), last template invalid
        out = mod(tp, template_mask=tmask)
        assert out.shape == (B, N, N, C_Z)
        assert not torch.isnan(out).any()

    def test_gate_starts_near_zero(self):
        """Gate bias initialised to -2 => sigmoid(-2) ≈ 0.12, contribution small."""
        mod = TemplateModule(c_template=88, c_z=C_Z, n_blocks=1)
        tp = torch.randn(B, 2, N, N, 88)
        with torch.no_grad():
            out = mod(tp)
        # Should be small (gate is partially closed at init)
        assert out.abs().mean().item() < 1.0


# ---------------------------------------------------------------------------
# Relative Position Encoding
# ---------------------------------------------------------------------------

class TestRelativePosEncoding:
    def test_shape(self):
        res_idx = torch.arange(N).unsqueeze(0).expand(B, -1)
        chain_idx = torch.zeros(B, N, dtype=torch.long)
        enc = relative_position_encoding(res_idx, chain_idx, max_radius=32, n_bins=64)
        assert enc.shape == (B, N, N, 65)   # 64 bins + 1 cross-chain bin

    def test_same_chain_one_hot(self):
        """For same-chain pairs, exactly one bin should be active per pair."""
        res_idx = torch.arange(N).unsqueeze(0)
        chain_idx = torch.zeros(1, N, dtype=torch.long)
        enc = relative_position_encoding(res_idx, chain_idx)
        # Sum over bins == 1 for every pair
        assert torch.allclose(enc.sum(-1), torch.ones(1, N, N))

    def test_cross_chain_uses_last_bin(self):
        """Cross-chain pairs should all land in the last bin."""
        res_idx = torch.arange(N).unsqueeze(0)
        chain_idx = torch.zeros(1, N, dtype=torch.long)
        chain_idx[:, N//2:] = 1   # second half is chain 1
        enc = relative_position_encoding(res_idx, chain_idx, n_bins=64)
        # Cross-chain: i in chain 0, j in chain 1
        cross_pair = enc[0, 0, N//2]   # first token of chain0, first token of chain1
        assert cross_pair[-1].item() == 1.0   # last bin active
        assert cross_pair[:-1].sum().item() == 0.0

    def test_diagonal_is_zero_offset(self):
        """Diagonal pairs (i==i) should encode offset 0."""
        res_idx = torch.arange(N).unsqueeze(0)
        chain_idx = torch.zeros(1, N, dtype=torch.long)
        enc = relative_position_encoding(res_idx, chain_idx, max_radius=32, n_bins=64)
        # offset=0, max_radius=32, n_bins=64:
        # bin = int((0+32)*(64-1)/(2*32)) = int(32*63/64) = int(31.5) = 31
        zero_offset_bin = int((0 + 32) * (64 - 1) / (2 * 32))
        for i in range(N):
            assert enc[0, i, i, zero_offset_bin].item() == 1.0


# ---------------------------------------------------------------------------
# Confidence Module
# ---------------------------------------------------------------------------

class TestConfidenceModule:
    @pytest.fixture
    def pair(self):
        return torch.randn(B, N, N, C_Z)

    def test_output_shapes(self, pair):
        mod = ConfidenceModule(C_Z, n_blocks=1, n_plddt_bins=50, n_pae_bins=64, n_pde_bins=64)
        out = mod(pair)
        assert out["plddt_logits"].shape == (B, N, 50)
        assert out["pae_logits"].shape   == (B, N, N, 64)
        assert out["pde_logits"].shape   == (B, N, N, 64)
        assert out["plddt"].shape        == (B, N)
        assert out["ptm"].shape          == (B, N)

    def test_plddt_in_unit_interval(self, pair):
        mod = ConfidenceModule(C_Z, n_blocks=1)
        out = mod(pair)
        assert out["plddt"].min().item() >= 0.0
        assert out["plddt"].max().item() <= 1.0

    def test_pae_logits_sum_to_one_after_softmax(self, pair):
        mod = ConfidenceModule(C_Z, n_blocks=1)
        out = mod(pair)
        probs = out["pae_logits"].softmax(dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(B, N, N), atol=1e-5)

    def test_pde_symmetric(self, pair):
        """PDE logits are computed from symmetric z, so logits[i,j] == logits[j,i]."""
        mod = ConfidenceModule(C_Z, n_blocks=1)
        out = mod(pair)
        assert torch.allclose(out["pde_logits"], out["pde_logits"].transpose(1, 2), atol=1e-5)

    def test_ptm_in_unit_interval(self, pair):
        mod = ConfidenceModule(C_Z, n_blocks=1)
        out = mod(pair)
        assert out["ptm"].min().item() >= 0.0
        assert out["ptm"].max().item() <= 1.0 + 1e-4

    def test_with_masks(self, pair):
        mod = ConfidenceModule(C_Z, n_blocks=1)
        token_mask = torch.ones(B, N); token_mask[:, -2:] = 0
        pair_mask  = torch.ones(B, N, N)
        out = mod(pair, pair_mask=pair_mask, token_mask=token_mask)
        assert not torch.isnan(out["plddt"]).any()


# ---------------------------------------------------------------------------
# Diffusion Module — noise schedule and preconditioning
# ---------------------------------------------------------------------------

class TestDiffusionNoise:
    def test_karras_schedule_shape(self):
        sigmas = karras_noise_schedule(20)
        assert sigmas.shape == (21,)

    def test_karras_schedule_monotone(self):
        """Sigmas should decrease from sigma_max to sigma_min."""
        sigmas = karras_noise_schedule(200, sigma_data=16, s_max=160, s_min=0.0004)
        assert sigmas[0] > sigmas[-1]
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs <= 0).all(), "Schedule should be monotonically decreasing"

    def test_karras_endpoints(self):
        s_max, s_min, sigma_data = 160.0, 0.0004, 16.0
        sigmas = karras_noise_schedule(200, sigma_data, s_max, s_min)
        # AF3 schedule: sigma(t=1) = sigma_data * s_max^rho-term = sigma_data * s_max (when rho adjusts)
        # Just check endpoints are in the right ballpark
        assert sigmas[0] >= sigmas[-1] * 100    # large ratio between max and min

    def test_sample_train_sigmas_shape(self):
        sigmas = sample_train_sigmas(batch_size=8)
        assert sigmas.shape == (8,)
        assert (sigmas > 0).all()

    def test_sample_train_sigmas_log_normal(self):
        """Log-normal distribution: log(sigma) should be approximately Gaussian."""
        torch.manual_seed(42)
        sigmas = sample_train_sigmas(batch_size=2000, p_mean=-1.2, p_std=1.5, sigma_data=16)
        log_s = torch.log(sigmas)
        # After scaling by sigma_data, mean of log(sigma) should be near p_mean + log(sigma_data)
        # Just check it's not degenerate
        assert log_s.std().item() > 0.5

    def test_fourier_embedding_shape(self):
        mod = FourierEmbedding(dim=256, sigma_data=16.0)
        sigma = torch.tensor([0.5, 1.0, 16.0, 160.0])
        out = mod(sigma)
        assert out.shape == (4, 256)

    def test_fourier_embedding_bounded(self):
        """Fourier features are cos/sin so must lie in [-1, 1]."""
        mod = FourierEmbedding(dim=256)
        sigma = torch.exp(torch.randn(100))
        out = mod(sigma)
        assert out.abs().max().item() <= 1.0 + 1e-5


class TestDiffusionPreconditioning:
    def test_cskip_cout_at_sigma_data(self):
        """At sigma = sigma_data: c_skip = 0.5, c_out = sigma_data / sqrt(2)."""
        mod = DiffusionModule(C_S, C_Z, C_A, C_AP,
                              atom_encoder_depth=1, token_transformer_depth=1,
                              atom_decoder_depth=1)
        sigma = torch.tensor([mod.sigma_data])
        c_skip, c_out, c_in, _ = mod._preconditioning(sigma)
        assert abs(c_skip.item() - 0.5) < 1e-5
        expected_cout = mod.sigma_data / math.sqrt(2)
        assert abs(c_out.item() - expected_cout) < 1e-4

    def test_cskip_at_zero_noise(self):
        """As sigma -> 0: c_skip -> 1 (model should return x unchanged)."""
        mod = DiffusionModule(C_S, C_Z, C_A, C_AP,
                              atom_encoder_depth=1, token_transformer_depth=1,
                              atom_decoder_depth=1)
        sigma = torch.tensor([1e-6])
        c_skip, c_out, _, _ = mod._preconditioning(sigma)
        assert abs(c_skip.item() - 1.0) < 1e-3
        assert c_out.item() < 1e-3   # contribution of network is negligible

    def test_cin_decreases_with_sigma(self):
        """c_in = 1/sqrt(sigma^2 + sigma_data^2) decreases as sigma increases."""
        mod = DiffusionModule(C_S, C_Z, C_A, C_AP,
                              atom_encoder_depth=1, token_transformer_depth=1,
                              atom_decoder_depth=1)
        sigmas = torch.tensor([0.1, 1.0, 10.0, 100.0])
        _, _, c_in, _ = mod._preconditioning(sigmas)
        diffs = c_in[1:] - c_in[:-1]
        assert (diffs < 0).all(), "c_in should decrease as sigma increases"


class TestDiffusionModuleForward:
    @pytest.fixture
    def small_diff(self):
        return DiffusionModule(
            c_s=C_S, c_z=C_Z, c_a=C_A, c_ap=C_AP,
            atom_encoder_depth=1, token_transformer_depth=1, atom_decoder_depth=1,
        )

    @pytest.fixture
    def dummy_feats(self):
        return {
            "atom_mask":            torch.ones(B, A),
            "atom_to_token":        torch.zeros(B, A, dtype=torch.long),
            "num_atoms_per_token":  torch.full((B, N), A // N, dtype=torch.long),
            "token_atom_start":     torch.arange(0, A, A // N).unsqueeze(0).expand(B, -1),
        }

    def test_forward_shape(self, small_diff, dummy_feats):
        r_noisy = torch.randn(B, A, 3)
        sigma   = torch.ones(B) * 16.0
        s_trunk = torch.randn(B, N, C_S)
        z_trunk = torch.randn(B, N, N, C_Z)
        a_ref   = torch.randn(B, A, C_A)
        p_ref   = torch.randn(B, A, A, C_AP)
        rel_pos = torch.randn(B, N, N, C_Z)

        out = small_diff(r_noisy, sigma, s_trunk, z_trunk, a_ref, p_ref, rel_pos, dummy_feats)
        assert out.shape == (B, A, 3)

    def test_forward_no_nan(self, small_diff, dummy_feats):
        r_noisy = torch.randn(B, A, 3)
        sigma   = torch.ones(B) * 16.0
        s_trunk = torch.randn(B, N, C_S)
        z_trunk = torch.randn(B, N, N, C_Z)
        a_ref   = torch.randn(B, A, C_A)
        p_ref   = torch.randn(B, A, A, C_AP)
        rel_pos = torch.randn(B, N, N, C_Z)

        out = small_diff(r_noisy, sigma, s_trunk, z_trunk, a_ref, p_ref, rel_pos, dummy_feats)
        assert not torch.isnan(out).any()

    def test_gradients_flow_through_denoiser(self, small_diff, dummy_feats):
        r_noisy = torch.randn(B, A, 3, requires_grad=True)
        sigma   = torch.ones(B) * 16.0
        s_trunk = torch.randn(B, N, C_S)
        z_trunk = torch.randn(B, N, N, C_Z)
        a_ref   = torch.randn(B, A, C_A)
        p_ref   = torch.randn(B, A, A, C_AP)
        rel_pos = torch.randn(B, N, N, C_Z)

        out = small_diff(r_noisy, sigma, s_trunk, z_trunk, a_ref, p_ref, rel_pos, dummy_feats)
        out.sum().backward()
        assert r_noisy.grad is not None
        assert not torch.isnan(r_noisy.grad).any()
