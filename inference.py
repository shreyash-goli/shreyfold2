"""
End-to-end inference smoke test.

Takes a short protein sequence, tokenizes it, builds model features,
runs the full AlphaFold3 forward pass (untrained, random-init weights),
and writes the predicted Cα trace to a PDB file.

The model has never been trained, so the output coordinates are
meaningless as a structure prediction -- this script exists to verify
that every module's input/output shapes line up across the full
pipeline (tokenizer -> features -> trunk -> diffusion sampler ->
confidence heads -> PDB writer).

Usage:
    python inference.py
"""

import torch

from config import get_default_config
from data.tokenizer import AF3Tokenizer, MolType, AA_VOCAB
from data.features import build_features, collate_features
from model import AlphaFold3

# First 10 residues of ubiquitin (1UBQ): MQIFVKTLTG
SEQUENCE = "MQIFVKTLTG"

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def build_chain(sequence: str) -> dict:
    """One Cα 'atom' per residue -- simplest possible all-atom-scheme input."""
    res_names = [ONE_TO_THREE[c] for c in sequence]
    return {
        "chain_id": "A",
        "mol_type": MolType.PROTEIN,
        "sequence": res_names,
        "atom_names": [["CA"] for _ in sequence],
        "elements": [["C"] for _ in sequence],
        "charges": [[0.0] for _ in sequence],
        "positions": None,  # no reference conformer -> ref_mask = False
    }


def write_pdb(path: str, sequence: str, coords: torch.Tensor, plddt: torch.Tensor) -> None:
    """coords: (N, 3), plddt: (N,) in [0, 1]."""
    lines = []
    for i, (aa, xyz, conf) in enumerate(zip(sequence, coords, plddt), start=1):
        x, y, z = xyz.tolist()
        lines.append(
            f"ATOM  {i:5d}  CA  {ONE_TO_THREE[aa]} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{conf * 100:6.2f}           C"
        )
    lines.append("TER")
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    torch.manual_seed(0)

    print(f"Sequence ({len(SEQUENCE)} residues): {SEQUENCE}")

    # 1. Tokenize
    tokenizer = AF3Tokenizer()
    complex_obj = tokenizer.tokenize([build_chain(SEQUENCE)])
    print(f"Tokenized: {complex_obj.n_tokens} tokens, {complex_obj.n_atoms} atoms")

    # 2. Build features and add a batch dimension
    feats = build_features(complex_obj)
    feats = collate_features([feats])

    # 3. Build model (random init -- no checkpoint)
    config = get_default_config()
    model = AlphaFold3(config)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # 4. Run full inference: trunk + diffusion sampling + confidence
    print("Running trunk + diffusion sampler (this may take a minute)...")
    with torch.no_grad():
        out = model.predict_structure(
            feats, n_diffusion_steps=20, n_recycling_iters=1,
        )

    coords = out["atom_positions"][0]   # (A, 3)
    plddt = out["plddt"][0].mean(dim=-1) if out["plddt"].dim() == 3 else out["plddt"][0]
    plddt = plddt.clamp(0, 1)

    print(f"Predicted coordinates shape: {tuple(coords.shape)}")
    print(f"Mean pLDDT (untrained, expect ~uninformative): {plddt.mean().item():.3f}")

    # 5. Write a PDB file with the Cα trace
    out_path = "inference_output.pdb"
    write_pdb(out_path, SEQUENCE, coords, plddt)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
