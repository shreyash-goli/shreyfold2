"""
AF3 model hyperparameters.
All dimensions match the paper (Fig. 2a, 2b, Supplementary Methods 3).
"""
from dataclasses import dataclass, field


@dataclass
class ModelDims:
    # Paper: pair (n,n,128), single (n,384)
    c_z: int = 128   # pair representation
    c_s: int = 384   # single representation
    c_m: int = 64    # MSA representation
    # Diffusion atom-level representations (not in paper directly, from Boltz)
    c_a: int = 128   # atom single representation
    c_ap: int = 16   # atom pair representation


@dataclass
class PairformerConfig:
    # Paper: 48 pairformer blocks; pair head-width=32, 4 heads => 128 pair dim
    n_blocks: int = 48
    n_heads_tri: int = 4          # triangle attention heads
    tri_head_width: int = 32      # per-head dim in triangle attention
    n_heads_single: int = 16      # single attention heads (c_s/head_width=384/24)
    single_head_width: int = 24
    dropout: float = 0.25
    transition_factor: int = 4    # MLP hidden = c * factor


@dataclass
class MSAConfig:
    # Paper: 4 MSA module blocks; pair-weighted averaging replaces row attention
    n_blocks: int = 4
    n_heads: int = 8
    c_hidden_msa: int = 8         # per-head dim in MSA pair-weighted averaging
    c_hidden_opm: int = 32        # outer product mean hidden
    dropout: float = 0.15
    transition_factor: int = 4


@dataclass
class TemplateConfig:
    # Paper: 2 template module blocks (pairformer without sequence track)
    n_blocks: int = 2
    n_heads_tri: int = 4
    tri_head_width: int = 32
    dropout: float = 0.25
    transition_factor: int = 4
    # Template feature dims
    c_template_pair: int = 88     # raw template pair features before projection


@dataclass
class DiffusionConfig:
    # Karras et al. 2022 noise schedule parameters (AF3 Supplementary p.24)
    sigma_data: float = 16.0
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    rho: float = 7.0              # controls density of noise levels
    # Architecture depths (paper: 3 + 24 + 3 blocks)
    atom_encoder_depth: int = 3
    atom_encoder_heads: int = 4
    atom_encoder_head_width: int = 32
    token_transformer_depth: int = 24
    token_transformer_heads: int = 16
    token_transformer_head_width: int = 24
    atom_decoder_depth: int = 3
    atom_decoder_heads: int = 4
    atom_decoder_head_width: int = 32
    # Fourier embedding for noise level conditioning
    dim_fourier: int = 256
    # Seq-local attention window sizes (Boltz: 32 queries, 128 keys)
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    n_conditioning_transitions: int = 2
    transition_factor: int = 2


@dataclass
class ConfidenceConfig:
    # 4 pairformer blocks (pair track only) then heads
    n_blocks: int = 4
    n_heads_tri: int = 4
    tri_head_width: int = 32
    dropout: float = 0.25
    transition_factor: int = 4
    # Output head bins
    n_plddt_bins: int = 50
    n_pde_bins: int = 64
    n_pae_bins: int = 64
    # pTM/ipTM are scalars computed from PAE logits


@dataclass
class TrainingConfig:
    # Training crop sizes (paper: 384 → 640 → 768)
    crop_size: int = 384
    max_msa_seqs: int = 512
    # Diffusion samples per step
    n_diffusion_samples: int = 48
    # Mini-rollout for confidence head (paper: 20 steps)
    n_rollout_steps: int = 20
    rollout_step_scale: float = 5.0   # larger steps than inference
    # Noise schedule for training: log-normal P(σ)
    p_mean: float = -1.2
    p_std: float = 1.5
    # Loss weights (paper: distogram 0.03, rest ~1)
    diffusion_loss_weight: float = 4.0
    confidence_loss_weight: float = 1.0
    distogram_loss_weight: float = 0.03
    # Molecule-type upweighting in diffusion loss
    dna_weight: float = 1.5
    rna_weight: float = 2.5
    ligand_weight: float = 4.0


@dataclass
class ModelConfig:
    dims: ModelDims = field(default_factory=ModelDims)
    pairformer: PairformerConfig = field(default_factory=PairformerConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config() -> ModelConfig:
    return ModelConfig()
