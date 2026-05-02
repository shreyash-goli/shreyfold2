# Shreyfold2

A ground-up reimplementation of **AlphaFold 3** built for learning purposes, drawing from three sources:

- **AlphaFold 3 paper** — Abramson et al., *Nature* 630, 493–500 (2024). The primary reference for all architectural decisions, hyperparameters, and training procedures.
- **Boltz** — MIT/Prescient Design open-source AF3 alternative. Referenced for clean PyTorch idioms: triangle multiplicative update factorisation, SwiGLU transitions, gating/final weight initialisation, and the diffusion conditioning pipeline.
- **OpenFold3** — AlQuraishi Lab open-source AF3 reimplementation. Referenced for the most paper-faithful implementations: MSA pair-weighted averaging, diffusion transformer with AdaLN, sequence-local atom attention, Kabsch alignment in the diffusion loss, and the confidence module.

---

## What is AlphaFold 3

AF3 predicts the 3D structure of biomolecular complexes — proteins, RNA, DNA, small molecules, ions, and modified residues — all within a single unified model. The two key architectural shifts from AF2:

1. **Unified token scheme**: polymer residues AND individual ligand atoms are all "tokens", enabling one model to handle arbitrary chemistry without special-casing.
2. **Diffusion replaces the structure module**: instead of SE(3)-equivariant frames + torsion angles, AF3 uses standard denoising diffusion directly on raw Cartesian atom coordinates. This eliminates the need for equivariance and handles general ligands naturally.

---

## Architecture

```
Sequences / ligands / bonds
          │
          ▼
   ┌─────────────────┐
   │  Input Embedder │  ref conformer → atom single (c_a=128) + pair (c_ap=16)
   │  (3 atom blocks)│  → aggregate atoms → tokens → s_inputs, z
   └────────┬────────┘
            │  s_inputs (B, N, 384)   z (B, N, N, 128)
            ▼
   ┌─────────────────┐
   │   MSA Module    │  4 blocks: pair-weighted averaging + outer product mean
   │   (4 blocks)    │  → z updated; MSA representation discarded
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │Template Module  │  2 pairformer-no-seq blocks per template, pool → Δz
   │   (2 blocks)    │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   Pairformer    │  48 blocks, each:
   │   (48 blocks)   │    pair: tri-mul-out → tri-mul-in → tri-att-start
   │                 │          → tri-att-end → transition
   │                 │    single: AttentionPairBias → transition
   └────────┬────────┘
            │  s_trunk (B, N, 384)   z_trunk (B, N, N, 128)
            ├──────────────────────────────────┐
            ▼                                  ▼
   ┌─────────────────┐              ┌──────────────────┐
   │ Diffusion Module│              │Confidence Module │
   │                 │              │   (4 blocks)     │
   │ SingleCond(σ)   │              │                  │
   │ PairwiseCond    │              │ pLDDT  (50 bins) │
   │ AtomEncoder × 3 │              │ PAE    (64 bins) │
   │ TokenTfmr  × 24 │              │ PDE    (64 bins) │
   │ AtomDecoder × 3 │              │ pTM / ipTM       │
   └────────┬────────┘              └──────────────────┘
            │
            ▼
     atom positions (B, A, 3)
```

### Key components

| File | What it implements |
|---|---|
| `modules/primitives.py` | Triangle mult (Alg 11/12), triangle attn (Alg 13/14), AttentionPairBias, AdaLayerNorm, OuterProductMean, MSAPairWeightedAveraging, SwiGLU Transition |
| `modules/input_embedder.py` | Ref conformer features → atom repr, 3-block seq-local atom transformer, token aggregation, relative position encoding |
| `modules/msa_module.py` | 4-block MSA processing; pair-weighted averaging replaces AF2 row attention |
| `modules/template_module.py` | 2-block pairformer-no-seq per template + gated pool |
| `modules/pairformer.py` | 48-block trunk; pair track + single track with structured dropout |
| `modules/diffusion_module.py` | Karras noise schedule, Fourier σ-embedding, AtomAttentionEncoder, DiffusionTransformer (24 blocks, AdaLN), AtomAttentionDecoder, Euler sampler |
| `modules/confidence_module.py` | 4 pairformer-no-seq blocks → pLDDT / PAE / PDE heads + pTM |
| `model.py` | Full model assembly with recycling |
| `training/loss.py` | Kabsch-aligned diffusion MSE, distogram CE, confidence CE, combined AF3Loss |
| `training/rollout.py` | 20-step mini-rollout with stop-gradient for confidence head training |
| `data/tokenizer.py` | Unified token scheme: residues + ligand atoms as tokens |
| `data/features.py` | Feature tensor construction from tokenised complex |
| `data/conformer.py` | RDKit ETKDGv3 reference conformer generation for ligands |

---

## What AF3 changes vs AF2

| Component | AlphaFold 2 | AlphaFold 3 |
|---|---|---|
| Token type | Residues only | Residues + ligand atoms |
| MSA processing | Evoformer (48 blocks, full MSA track) | 4 blocks, pair-weighted averaging only |
| Pair processing | Evoformer (MSA-coupled) | Pairformer (pair + single only) |
| Structure output | IPA + backbone frames + torsion angles | Diffusion on raw atom coordinates |
| Equivariance | SE(3)-equivariant IPA | None — random rotation augmentation instead |
| Chemical scope | Proteins only | Proteins, RNA, DNA, ligands, ions, modifications |
| Generative | No | Yes — distribution over structures |

---

## Project structure

```
shreyfold2/
├── config.py           # all hyperparameters
├── model.py            # full AlphaFold3 class
├── data/
│   ├── tokenizer.py    # unified token scheme
│   ├── features.py     # tensor feature construction
│   └── conformer.py    # RDKit conformer generation
├── modules/
│   ├── primitives.py   # all shared ops
│   ├── input_embedder.py
│   ├── msa_module.py
│   ├── template_module.py
│   ├── pairformer.py
│   ├── diffusion_module.py
│   └── confidence_module.py
├── training/
│   ├── loss.py         # diffusion + distogram + confidence losses
│   └── rollout.py      # mini-rollout for confidence training
├── tests/
│   ├── test_primitives.py   # 40 tests
│   ├── test_modules.py      # 48 tests
│   ├── test_data.py         # 22 tests
│   └── test_training.py     # 17 tests
└── requirements.txt
```

---

## Running the tests

```bash
pip install torch einops
pip install rdkit          # optional, only needed for conformer generation
python -m pytest tests/ -v
```

All 127 tests pass on CPU. No GPU required for tests.

---
