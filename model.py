"""
AlphaFold 3 — full model assembly.

Forward pass (inference / training trunk):
  1. InputEmbedder:    feats -> s_inputs (B,N,c_s), z (B,N,N,c_z)
  2. MSAEmbedder:      msa -> m (B,S,N,c_m)
  3. MSAModule:        m, z -> z  (MSA updates pair, then m discarded)
  4. TemplateModule:   template_pair, z -> delta_z  (added to z)
  5. Pairformer:       s_inputs, z -> s_trunk (B,N,c_s), z_trunk (B,N,N,c_z)
  6. DiffusionModule:  noisy coords + trunk -> x_pred (B,A,3)
     (training: single forward; inference: iterative sampling)
  7. ConfidenceModule: z_trunk + rollout structure -> pLDDT, PAE, PDE
     (training only, gated by stop-gradient on mini-rollout output)

References:
  OF3:   projects/of3_all_atom/model.py
  Boltz: model/models/boltz1.py
"""

from typing import Optional

import torch
import torch.nn as nn

from config import ModelConfig, get_default_config
from modules.input_embedder import InputEmbedder, relative_position_encoding
from modules.msa_module import MSAModule, MSAEmbedder
from modules.template_module import TemplateModule
from modules.pairformer import Pairformer
from modules.diffusion_module import DiffusionModule
from modules.confidence_module import ConfidenceModule
from data.tokenizer import (
    AA_VOCAB, RNA_VOCAB, DNA_VOCAB, N_ELEMENTS,
)

# Max vocab sizes for embedding tables
_N_AA = len(AA_VOCAB)        # 22
_N_RNA = len(RNA_VOCAB)      # 6
_N_DNA = len(DNA_VOCAB)      # 6
_N_RES_TYPES = max(_N_AA, _N_RNA, _N_DNA, N_ELEMENTS) + 1   # shared embedding table


class AlphaFold3(nn.Module):
    """
    Full AF3 model.

    Typical usage:
        model = AlphaFold3()
        # During training (diffusion step):
        x_pred = model.diffusion_step(feats, x_noisy, sigma)
        # During inference:
        x_pred = model.sample(feats, n_diffusion_seeds=5, n_diffusion_samples=5)
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = get_default_config()
        self.config = config
        d = config.dims
        pf = config.pairformer
        msa = config.msa
        tmpl = config.template
        diff = config.diffusion
        conf = config.confidence

        # ------------------------------------------------------------------ #
        # 1. Input Embedder                                                   #
        # ------------------------------------------------------------------ #
        self.input_embedder = InputEmbedder(
            c_a=d.c_a, c_ap=d.c_ap,
            c_s=d.c_s, c_z=d.c_z,
            n_atom_encoder_blocks=3,
            n_heads=4, head_width=32,
            n_rel_pos_bins=64, max_rel_pos=32,
            n_token_types=6,
            n_residue_types=_N_RES_TYPES,
        )

        # ------------------------------------------------------------------ #
        # 2. MSA                                                              #
        # ------------------------------------------------------------------ #
        self.msa_embedder = MSAEmbedder(n_vocab=_N_RES_TYPES, c_m=d.c_m)
        self.msa_module = MSAModule(
            c_m=d.c_m, c_z=d.c_z,
            n_blocks=msa.n_blocks,
            c_hidden_msa=msa.c_hidden_msa,
            n_heads_msa=msa.n_heads,
            c_hidden_opm=msa.c_hidden_opm,
            dropout_msa=msa.dropout,
            transition_factor=msa.transition_factor,
        )

        # ------------------------------------------------------------------ #
        # 3. Templates                                                        #
        # ------------------------------------------------------------------ #
        self.template_module = TemplateModule(
            c_template=tmpl.c_template_pair,
            c_z=d.c_z,
            n_blocks=tmpl.n_blocks,
            n_heads_tri=tmpl.n_heads_tri,
            tri_head_width=tmpl.tri_head_width,
            dropout=tmpl.dropout,
            transition_factor=tmpl.transition_factor,
        )

        # ------------------------------------------------------------------ #
        # 4. Pairformer trunk                                                 #
        # ------------------------------------------------------------------ #
        # Project s_inputs (c_s) into a learned initial single representation
        self.s_initial_proj = nn.Sequential(
            nn.LayerNorm(d.c_s),
            nn.Linear(d.c_s, d.c_s),
        )

        self.pairformer = Pairformer(
            c_z=d.c_z, c_s=d.c_s,
            n_blocks=pf.n_blocks,
            n_heads_tri=pf.n_heads_tri,
            tri_head_width=pf.tri_head_width,
            n_heads_single=pf.n_heads_single,
            single_head_width=pf.single_head_width,
            dropout=pf.dropout,
            transition_factor=pf.transition_factor,
        )

        # ------------------------------------------------------------------ #
        # 5. Diffusion                                                        #
        # ------------------------------------------------------------------ #
        self.diffusion_module = DiffusionModule(
            c_s=d.c_s, c_z=d.c_z,
            c_a=d.c_a, c_ap=d.c_ap,
            sigma_data=diff.sigma_data,
            dim_fourier=diff.dim_fourier,
            atom_encoder_depth=diff.atom_encoder_depth,
            atom_encoder_heads=diff.atom_encoder_heads,
            atom_encoder_head_width=diff.atom_encoder_head_width,
            token_transformer_depth=diff.token_transformer_depth,
            token_transformer_heads=diff.token_transformer_heads,
            token_transformer_head_width=diff.token_transformer_head_width,
            atom_decoder_depth=diff.atom_decoder_depth,
            atom_decoder_heads=diff.atom_decoder_heads,
            atom_decoder_head_width=diff.atom_decoder_head_width,
            n_conditioning_transitions=diff.n_conditioning_transitions,
            transition_factor=diff.transition_factor,
        )

        # ------------------------------------------------------------------ #
        # 6. Confidence                                                       #
        # ------------------------------------------------------------------ #
        self.confidence_module = ConfidenceModule(
            c_z=d.c_z, c_s=d.c_s,
            n_blocks=conf.n_blocks,
            n_heads_tri=conf.n_heads_tri,
            tri_head_width=conf.tri_head_width,
            dropout=conf.dropout,
            transition_factor=conf.transition_factor,
            n_plddt_bins=conf.n_plddt_bins,
            n_pae_bins=conf.n_pae_bins,
            n_pde_bins=conf.n_pde_bins,
        )

        # ------------------------------------------------------------------ #
        # 7. Distogram head (auxiliary loss, pair -> distance bins)          #
        # ------------------------------------------------------------------ #
        self.distogram_head = nn.Linear(d.c_z, 64)   # 64 distance bins

    # ---------------------------------------------------------------------- #
    # Core trunk (shared between training and inference)                     #
    # ---------------------------------------------------------------------- #

    def run_trunk(
        self,
        feats: dict,
        n_recycling_iters: int = 3,
    ) -> tuple:
        """
        Runs input embedder + MSA + templates + pairformer.

        Returns:
            s_trunk:    (B, N, c_s)
            z_trunk:    (B, N, N, c_z)
            a_ref:      (B, A, c_a)  atom single from input embedder
            p_ref:      (B, A, A, c_ap) atom pair from input embedder
            rel_pos_z:  (B, N, N, c_z) relative position encoding
        """
        device = feats["token_mask"].device
        B = feats["token_mask"].shape[0]
        N = feats["token_mask"].shape[1]

        # Build masks
        token_mask = feats["token_mask"]                  # (B, N)
        pair_mask = token_mask.unsqueeze(2) * token_mask.unsqueeze(1)  # (B, N, N)

        # Input embedder
        s_inputs, z = self.input_embedder(feats)          # (B,N,c_s), (B,N,N,c_z)

        # Extract reference atom representations (for diffusion)
        # These are computed inside input_embedder; we re-expose them here
        a_ref, p_ref = self.input_embedder.ref_feat_embedder(feats)

        # Relative position encoding for diffusion module pairwise conditioning
        rel_pos_z = z.clone()   # z already contains the relative position contribution

        # MSA
        if "msa" in feats:
            m = self.msa_embedder(feats["msa"])           # (B, S, N, c_m)
            msa_mask = feats.get("msa_mask", None)
            z = self.msa_module(m, z, msa_mask=msa_mask, pair_mask=pair_mask)

        # Templates
        if "template_pair" in feats:
            delta_z = self.template_module(
                feats["template_pair"],
                template_mask=feats.get("template_mask"),
                pair_mask=pair_mask,
            )
            z = z + delta_z

        # Pairformer (with recycling)
        s = self.s_initial_proj(s_inputs)                 # (B, N, c_s)
        for _ in range(n_recycling_iters):
            s, z = self.pairformer(s, z, mask=token_mask, pair_mask=pair_mask)

        return s, z, a_ref, p_ref, rel_pos_z, token_mask, pair_mask

    # ---------------------------------------------------------------------- #
    # Training: single diffusion denoising step                              #
    # ---------------------------------------------------------------------- #

    def diffusion_step(
        self,
        feats: dict,
        x_noisy: torch.Tensor,   # (B, A, 3) noisy coordinates
        sigma: torch.Tensor,     # (B,) noise level
        n_recycling_iters: int = 3,
    ) -> torch.Tensor:
        """
        Single denoising step for training.
        Returns predicted clean coordinates (B, A, 3).
        """
        s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, token_mask, pair_mask = (
            self.run_trunk(feats, n_recycling_iters)
        )
        x_pred = self.diffusion_module(
            x_noisy, sigma, s_trunk, z_trunk, a_ref, p_ref, rel_pos_z,
            feats, mask=token_mask, pair_mask=pair_mask,
        )
        return x_pred

    # ---------------------------------------------------------------------- #
    # Inference: full structure prediction                                    #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict_structure(
        self,
        feats: dict,
        n_diffusion_steps: int = 200,
        n_recycling_iters: int = 3,
    ) -> dict:
        """
        Run full inference: trunk + diffusion sampling + confidence.

        Returns a dict with:
          atom_positions: (B, A, 3) predicted atom coordinates
          plddt:          (B, N)    per-token confidence
          pae_logits:     (B, N, N, n_pae_bins)
          pde_logits:     (B, N, N, n_pde_bins)
          ptm:            (B, N)
        """
        s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, token_mask, pair_mask = (
            self.run_trunk(feats, n_recycling_iters)
        )

        # Sample atom positions
        diff_cfg = self.config.diffusion
        atom_positions = self.diffusion_module.sample(
            s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, feats,
            n_steps=n_diffusion_steps,
            sigma_min=diff_cfg.sigma_min,
            sigma_max=diff_cfg.sigma_max,
            rho=diff_cfg.rho,
            mask=token_mask, pair_mask=pair_mask,
        )

        # Confidence (stop-gradient on structure)
        conf_out = self.confidence_module(
            z_trunk.detach(), pair_mask=pair_mask, token_mask=token_mask,
        )

        # Distogram (auxiliary, pair of Cα-Cα distances)
        dist_logits = self.distogram_head(z_trunk)   # (B, N, N, 64)

        return {
            "atom_positions": atom_positions,
            "plddt":          conf_out["plddt"],
            "pae_logits":     conf_out["pae_logits"],
            "pde_logits":     conf_out["pde_logits"],
            "ptm":            conf_out["ptm"],
            "plddt_logits":   conf_out["plddt_logits"],
            "dist_logits":    dist_logits,
        }

    # ---------------------------------------------------------------------- #
    # Auxiliary head outputs for loss computation during training            #
    # ---------------------------------------------------------------------- #

    def forward_for_training(
        self,
        feats: dict,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        n_recycling_iters: int = 3,
        rollout_output: Optional[torch.Tensor] = None,  # stop-grad structure for confidence
    ) -> dict:
        """
        Full training forward pass. Returns all quantities needed for loss.
        """
        s_trunk, z_trunk, a_ref, p_ref, rel_pos_z, token_mask, pair_mask = (
            self.run_trunk(feats, n_recycling_iters)
        )

        x_pred = self.diffusion_module(
            x_noisy, sigma, s_trunk, z_trunk, a_ref, p_ref, rel_pos_z,
            feats, mask=token_mask, pair_mask=pair_mask,
        )

        # Confidence from stop-gradient rollout structure
        if rollout_output is not None:
            conf_out = self.confidence_module(
                z_trunk.detach(), pair_mask=pair_mask, token_mask=token_mask,
            )
        else:
            conf_out = None

        dist_logits = self.distogram_head(z_trunk)

        return {
            "x_pred":        x_pred,
            "confidence":    conf_out,
            "dist_logits":   dist_logits,
            "s_trunk":       s_trunk,
            "z_trunk":       z_trunk,
            "token_mask":    token_mask,
            "pair_mask":     pair_mask,
        }
