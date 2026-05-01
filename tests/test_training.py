"""
Unit tests for training components:
  - diffusion_loss (Kabsch alignment, molecule type weights, masked mean)
  - distogram_loss (binning, cross-entropy shape)
  - AF3Loss (combined, zero gradients where expected)
  - mini-rollout (shape, stop-gradient)
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.loss import (
    weighted_rigid_align, build_atom_weights,
    diffusion_loss, distogram_loss, AF3Loss,
)
from training.rollout import mini_rollout
from modules.diffusion_module import DiffusionModule

B, N, A = 1, 6, 12
C_Z, C_S, C_A, C_AP = 128, 384, 128, 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feats(b=B, n=N, a=A):
    """Minimal feats dict with ground truth, needed for losses."""
    tok_per_token = a // n
    return {
        "is_protein": torch.ones(b, n),
        "is_rna":     torch.zeros(b, n),
        "is_dna":     torch.zeros(b, n),
        "is_ligand":  torch.zeros(b, n),
        "token_mask": torch.ones(b, n),
        "chain_index":   torch.zeros(b, n, dtype=torch.long),
        "residue_index": torch.arange(n).unsqueeze(0).expand(b, -1),
        "atom_to_token": torch.arange(a, dtype=torch.long).unsqueeze(0).expand(b, -1) // tok_per_token,
        "num_atoms_per_token": torch.full((b, n), tok_per_token, dtype=torch.long),
        "token_atom_start": torch.arange(0, a, tok_per_token, dtype=torch.long).unsqueeze(0).expand(b, -1),
        "ground_truth": {
            "atom_positions":    torch.randn(b, a, 3),
            "atom_resolved_mask": torch.ones(b, a),
        },
    }


# ---------------------------------------------------------------------------
# Weighted rigid alignment
# ---------------------------------------------------------------------------

class TestWeightedRigidAlign:
    def test_identity_when_already_aligned(self):
        x = torch.randn(B, A, 3)
        w = torch.ones(B, A)
        mask = torch.ones(B, A)
        out = weighted_rigid_align(x, x, w, mask)
        # Aligned to itself: should be the same
        assert torch.allclose(out, x, atol=1e-4)

    def test_pure_translation(self):
        """Translation-only misalignment should be corrected perfectly."""
        x    = torch.randn(1, 6, 3)
        x_gt = x + torch.tensor([[[5.0, -3.0, 2.0]]])   # uniform shift
        w    = torch.ones(1, 6)
        mask = torch.ones(1, 6)
        out = weighted_rigid_align(x, x_gt, w, mask)
        assert torch.allclose(out, x_gt, atol=1e-3)

    def test_output_shape(self):
        x    = torch.randn(B, A, 3)
        x_gt = torch.randn(B, A, 3)
        w    = torch.ones(B, A)
        mask = torch.ones(B, A)
        out  = weighted_rigid_align(x, x_gt, w, mask)
        assert out.shape == (B, A, 3)

    def test_gradient_flows_through_x(self):
        """Gradient should flow through x (predicted coords), not through R."""
        x    = torch.randn(B, A, 3, requires_grad=True)
        x_gt = torch.randn(B, A, 3)
        w    = torch.ones(B, A)
        mask = torch.ones(B, A)
        out  = weighted_rigid_align(x, x_gt, w, mask)
        out.sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()

    def test_masked_atoms_ignored(self):
        """Atoms with mask=0 should not affect alignment."""
        x    = torch.randn(1, 6, 3)
        x_gt = x.clone()
        # Corrupt last 3 atoms in x — but mask them out
        x_corrupt = x.clone(); x_corrupt[:, 3:] = 999.0
        mask_all  = torch.ones(1, 6)
        mask_half = torch.zeros(1, 6); mask_half[:, :3] = 1.0
        w = torch.ones(1, 6)

        out_all  = weighted_rigid_align(x, x_gt, w, mask_all)
        out_half = weighted_rigid_align(x_corrupt, x_gt, w, mask_half)
        # First 3 atoms (unmasked) should still align well
        assert torch.allclose(out_half[:, :3], x_gt[:, :3], atol=0.1)


# ---------------------------------------------------------------------------
# Atom weights
# ---------------------------------------------------------------------------

class TestBuildAtomWeights:
    def test_protein_weight_is_one(self):
        feats = make_feats()
        w = build_atom_weights(feats, dna_weight=1.5, rna_weight=2.5, ligand_weight=4.0)
        assert torch.allclose(w, torch.ones(B, A))

    def test_ligand_upweighted(self):
        feats = make_feats()
        feats["is_protein"] = torch.zeros(B, N)
        feats["is_ligand"]  = torch.ones(B, N)
        w = build_atom_weights(feats, ligand_weight=4.0)
        # Weight should be 1 + 4 = 5 for all ligand atoms
        assert torch.allclose(w, torch.full((B, A), 5.0))

    def test_mixed_weights(self):
        feats = make_feats()
        # First half protein (w=1), second half RNA (w=1+2.5=3.5)
        feats["is_protein"] = torch.zeros(B, N); feats["is_protein"][:, :N//2] = 1.0
        feats["is_rna"]     = torch.zeros(B, N); feats["is_rna"][:, N//2:] = 1.0
        w = build_atom_weights(feats, rna_weight=2.5)
        # Atoms from first half -> w=1, second half -> w=3.5
        atoms_per_tok = A // N
        first_half_atoms = (N // 2) * atoms_per_tok
        assert torch.allclose(w[:, :first_half_atoms],   torch.ones(B, first_half_atoms))
        assert torch.allclose(w[:, first_half_atoms:],   torch.full((B, A - first_half_atoms), 3.5))


# ---------------------------------------------------------------------------
# Diffusion loss
# ---------------------------------------------------------------------------

class TestDiffusionLoss:
    def test_scalar_output(self):
        feats  = make_feats()
        x_pred = torch.randn(B, A, 3)
        loss   = diffusion_loss(x_pred, feats)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_zero_loss_on_perfect_prediction(self):
        """If pred == gt (up to rigid transform), loss should be ~0."""
        feats   = make_feats()
        x_gt    = feats["ground_truth"]["atom_positions"]
        # Perfect prediction (same as gt)
        x_pred  = x_gt.clone()
        loss = diffusion_loss(x_pred, feats)
        assert loss.item() < 1e-4

    def test_loss_increases_with_noise(self):
        """Adding more noise to predictions should increase loss."""
        feats = make_feats()
        x_gt  = feats["ground_truth"]["atom_positions"]
        torch.manual_seed(0)
        loss_small = diffusion_loss(x_gt + torch.randn_like(x_gt) * 0.1, feats).item()
        loss_large = diffusion_loss(x_gt + torch.randn_like(x_gt) * 5.0, feats).item()
        assert loss_large > loss_small

    def test_gradients_flow(self):
        feats  = make_feats()
        x_pred = torch.randn(B, A, 3, requires_grad=True)
        loss   = diffusion_loss(x_pred, feats)
        loss.backward()
        assert x_pred.grad is not None
        assert not torch.isnan(x_pred.grad).any()

    def test_all_masked_is_nan_safe(self):
        """All atoms masked out -> loss should handle gracefully (no crash)."""
        feats = make_feats()
        feats["ground_truth"]["atom_resolved_mask"] = torch.zeros(B, A)
        x_pred = torch.randn(B, A, 3)
        # Should not raise
        loss = diffusion_loss(x_pred, feats)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Distogram loss
# ---------------------------------------------------------------------------

class TestDistogramLoss:
    def test_scalar_output(self):
        feats = make_feats()
        dist_logits = torch.randn(B, N, N, 64)
        loss = distogram_loss(dist_logits, feats, n_bins=64)
        assert loss.shape == ()

    def test_non_negative(self):
        feats = make_feats()
        dist_logits = torch.randn(B, N, N, 64)
        loss = distogram_loss(dist_logits, feats)
        assert loss.item() >= 0.0

    def test_gradients_flow(self):
        feats = make_feats()
        dist_logits = torch.randn(B, N, N, 64, requires_grad=True)
        loss = distogram_loss(dist_logits, feats)
        loss.backward()
        assert dist_logits.grad is not None


# ---------------------------------------------------------------------------
# AF3Loss combined
# ---------------------------------------------------------------------------

class TestAF3Loss:
    def test_returns_dict_with_all_keys(self):
        loss_fn = AF3Loss()
        feats   = make_feats()
        x_pred  = torch.randn(B, A, 3)
        sigma   = torch.ones(B) * 16.0
        model_out = {
            "x_pred":      x_pred,
            "dist_logits": torch.randn(B, N, N, 64),
            "token_mask":  torch.ones(B, N),
            "pair_mask":   torch.ones(B, N, N),
        }
        losses = loss_fn(model_out, feats, sigma)
        assert "diffusion"  in losses
        assert "distogram"  in losses
        assert "confidence" in losses
        assert "total"      in losses

    def test_total_is_sum_of_parts(self):
        loss_fn = AF3Loss(diffusion_weight=4.0, distogram_weight=0.03)
        feats   = make_feats()
        sigma   = torch.ones(B) * 16.0
        model_out = {
            "x_pred":      torch.randn(B, A, 3),
            "dist_logits": torch.randn(B, N, N, 64),
        }
        losses = loss_fn(model_out, feats, sigma)
        expected_total = losses["diffusion"] + losses["distogram"] + losses["confidence"]
        assert torch.allclose(losses["total"], expected_total, atol=1e-5)

    def test_total_backward(self):
        loss_fn = AF3Loss()
        feats   = make_feats()
        sigma   = torch.ones(B) * 16.0
        x_pred  = torch.randn(B, A, 3, requires_grad=True)
        model_out = {"x_pred": x_pred, "dist_logits": torch.randn(B, N, N, 64)}
        losses = loss_fn(model_out, feats, sigma)
        losses["total"].backward()
        assert x_pred.grad is not None and not torch.isnan(x_pred.grad).any()


# ---------------------------------------------------------------------------
# Mini-rollout
# ---------------------------------------------------------------------------

class TestMiniRollout:
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

    def test_output_shape(self, small_diff, dummy_feats):
        s_trunk = torch.randn(B, N, C_S)
        z_trunk = torch.randn(B, N, N, C_Z)
        a_ref   = torch.randn(B, A, C_A)
        p_ref   = torch.randn(B, A, A, C_AP)
        rel_pos = torch.randn(B, N, N, C_Z)

        out = mini_rollout(
            small_diff, s_trunk, z_trunk, a_ref, p_ref, rel_pos, dummy_feats,
            n_steps=3,
        )
        assert out.shape == (B, A, 3)

    def test_output_has_no_gradient(self, small_diff, dummy_feats):
        """Stop-gradient: rollout output must not require grad."""
        s_trunk = torch.randn(B, N, C_S, requires_grad=True)
        z_trunk = torch.randn(B, N, N, C_Z)
        a_ref   = torch.randn(B, A, C_A)
        p_ref   = torch.randn(B, A, A, C_AP)
        rel_pos = torch.randn(B, N, N, C_Z)

        out = mini_rollout(
            small_diff, s_trunk, z_trunk, a_ref, p_ref, rel_pos, dummy_feats,
            n_steps=2,
        )
        assert not out.requires_grad

    def test_no_nan_output(self, small_diff, dummy_feats):
        s_trunk = torch.randn(B, N, C_S)
        z_trunk = torch.randn(B, N, N, C_Z)
        a_ref   = torch.randn(B, A, C_A)
        p_ref   = torch.randn(B, A, A, C_AP)
        rel_pos = torch.randn(B, N, N, C_Z)

        out = mini_rollout(
            small_diff, s_trunk, z_trunk, a_ref, p_ref, rel_pos, dummy_feats,
            n_steps=3,
        )
        assert not torch.isnan(out).any()

    def test_different_seeds_give_different_outputs(self, small_diff, dummy_feats):
        """Two rollouts from different noise seeds should differ."""
        kwargs = dict(
            diffusion_module=small_diff,
            s_trunk=torch.randn(B, N, C_S),
            z_trunk=torch.randn(B, N, N, C_Z),
            a_ref=torch.randn(B, A, C_A),
            p_ref=torch.randn(B, A, A, C_AP),
            rel_pos_z=torch.randn(B, N, N, C_Z),
            feats=dummy_feats,
            n_steps=2,
        )
        torch.manual_seed(0); out1 = mini_rollout(**kwargs)
        torch.manual_seed(1); out2 = mini_rollout(**kwargs)
        assert not torch.allclose(out1, out2)
