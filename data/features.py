"""
Feature construction from a tokenised Complex.

Builds the tensor dictionary consumed by the model's input embedder.
Feature names and shapes follow AF3 Supplementary Table 5 / Boltz featurizer.py.

Key features:
  Token-level (n_tokens):
    - token_type: mol type one-hot (6)
    - residue_type: residue/element index
    - token_mask: valid token mask

  Atom-level (n_atoms):
    - ref_pos:         (N_atom, 3)  reference conformer positions (Ångström)
    - ref_mask:        (N_atom,)    1 if atom position is known
    - ref_element:     (N_atom, N_elements)  element one-hot
    - ref_charge:      (N_atom, 1)  formal charge
    - ref_atom_name_chars: (N_atom, 4, 64)  atom name as unicode char one-hot
    - atom_to_token:   (N_atom,)    token index for each atom
    - atom_mask:       (N_atom,)    valid atom mask
    - num_atoms_per_token: (n_tokens,)

  Ground truth (when available):
    - atom_positions:  (N_atom, 3)
    - atom_resolved_mask: (N_atom,)

  MSA (S, N_tokens):
    - msa:             (S, N, n_res_types)  one-hot MSA
    - msa_mask:        (S, N)

  Template (T, N_tokens, N_tokens):
    - template_pair:   (T, N, N, C_t) template pair features
    - template_mask:   (T,) per-template validity mask
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .tokenizer import (
    Complex, MolType,
    AA_VOCAB, RNA_VOCAB, DNA_VOCAB,
    ELEMENT_TO_IDX, N_ELEMENTS,
)

# Atom name character vocabulary (unicode chars 0-127 → 64-dim one-hot of 4 chars)
_ATOM_NAME_LEN = 4
_ATOM_CHAR_DIM = 64


def _encode_atom_name(name: str) -> torch.Tensor:
    """Encode up to 4 chars of atom name as one-hot; (4, 64)."""
    out = torch.zeros(_ATOM_NAME_LEN, _ATOM_CHAR_DIM)
    for i, ch in enumerate(name[:_ATOM_NAME_LEN]):
        idx = min(ord(ch), _ATOM_CHAR_DIM - 1)
        out[i, idx] = 1.0
    return out


def build_features(
    complex_obj: Complex,
    msa: Optional[torch.Tensor] = None,        # (S, N) int residue indices
    msa_mask: Optional[torch.Tensor] = None,   # (S, N)
    template_pair: Optional[torch.Tensor] = None,   # (T, N, N, C_t)
    template_mask: Optional[torch.Tensor] = None,   # (T,)
    atom_positions_gt: Optional[torch.Tensor] = None,  # (N_atom, 3) ground truth
    atom_resolved_mask: Optional[torch.Tensor] = None,  # (N_atom,)
) -> dict:
    """Convert a Complex into the feature tensor dict for the model."""
    tokens = complex_obj.tokens
    N = complex_obj.n_tokens
    A = complex_obj.n_atoms

    # ------------------------------------------------------------------ #
    # Token-level features                                                 #
    # ------------------------------------------------------------------ #
    token_type = torch.zeros(N, dtype=torch.long)
    residue_type = torch.zeros(N, dtype=torch.long)
    is_protein = torch.zeros(N, dtype=torch.bool)
    is_rna = torch.zeros(N, dtype=torch.bool)
    is_dna = torch.zeros(N, dtype=torch.bool)
    is_ligand = torch.zeros(N, dtype=torch.bool)
    chain_index = torch.zeros(N, dtype=torch.long)
    residue_index = torch.zeros(N, dtype=torch.long)
    token_mask = torch.ones(N, dtype=torch.bool)

    chain_id_map: dict = {}
    for tok in tokens:
        i = tok.token_index
        token_type[i] = int(tok.token_type)
        residue_type[i] = tok.residue_type
        is_protein[i] = tok.token_type == MolType.PROTEIN
        is_rna[i] = tok.token_type == MolType.RNA
        is_dna[i] = tok.token_type == MolType.DNA
        is_ligand[i] = tok.token_type == MolType.LIGAND
        if tok.chain_id not in chain_id_map:
            chain_id_map[tok.chain_id] = len(chain_id_map)
        chain_index[i] = chain_id_map[tok.chain_id]
        residue_index[i] = tok.residue_index

    # ------------------------------------------------------------------ #
    # Atom-level features                                                  #
    # ------------------------------------------------------------------ #
    ref_pos = torch.zeros(A, 3)
    ref_mask = torch.zeros(A, dtype=torch.bool)
    ref_element = torch.zeros(A, N_ELEMENTS)
    ref_charge = torch.zeros(A, 1)
    ref_atom_name_chars = torch.zeros(A, _ATOM_NAME_LEN, _ATOM_CHAR_DIM)
    atom_to_token = torch.tensor(complex_obj.atom_to_token, dtype=torch.long)

    atom_idx = 0
    for tok in tokens:
        for atom in tok.atoms:
            if atom.position is not None:
                ref_pos[atom_idx] = torch.tensor(atom.position)
                ref_mask[atom_idx] = True
            elem_idx = ELEMENT_TO_IDX.get(atom.element, ELEMENT_TO_IDX.get("<UNK>", 0))
            ref_element[atom_idx, elem_idx] = 1.0
            ref_charge[atom_idx, 0] = atom.charge
            ref_atom_name_chars[atom_idx] = _encode_atom_name(atom.atom_name)
            atom_idx += 1

    num_atoms_per_token = torch.tensor(
        [tok.n_atoms for tok in tokens], dtype=torch.long
    )

    # ------------------------------------------------------------------ #
    # Relative position encoding (token-level)                            #
    # Will be further processed in input_embedder.                        #
    # ------------------------------------------------------------------ #
    # residue_index: (N,) for same-chain offset; chain_index: (N,)
    # actual encoding built in the embedder

    # ------------------------------------------------------------------ #
    # Assemble feature dict                                                #
    # ------------------------------------------------------------------ #
    feats: dict = {
        # Token-level
        "token_type": token_type,
        "residue_type": residue_type,
        "is_protein": is_protein.float(),
        "is_rna": is_rna.float(),
        "is_dna": is_dna.float(),
        "is_ligand": is_ligand.float(),
        "chain_index": chain_index,
        "residue_index": residue_index,
        "token_mask": token_mask,
        # Atom-level (reference conformer)
        "ref_pos": ref_pos,
        "ref_mask": ref_mask.float(),
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "atom_to_token": atom_to_token,
        "atom_mask": ref_mask.float(),
        "num_atoms_per_token": num_atoms_per_token,
        "token_atom_start": torch.tensor(complex_obj.token_atom_start, dtype=torch.long),
    }

    if msa is not None:
        feats["msa"] = msa
        feats["msa_mask"] = msa_mask if msa_mask is not None else torch.ones_like(msa, dtype=torch.bool)
    if template_pair is not None:
        feats["template_pair"] = template_pair
        feats["template_mask"] = template_mask if template_mask is not None else torch.ones(template_pair.shape[0])
    if atom_positions_gt is not None:
        feats["ground_truth"] = {
            "atom_positions": atom_positions_gt,
            "atom_resolved_mask": (atom_resolved_mask if atom_resolved_mask is not None
                                   else torch.ones(A, dtype=torch.bool)),
        }

    return feats


def collate_features(batch: list) -> dict:
    """Batch a list of feature dicts, padding to the maximum size."""
    max_tokens = max(f["token_mask"].shape[0] for f in batch)
    max_atoms = max(f["atom_mask"].shape[0] for f in batch)
    B = len(batch)

    def pad1d(t, length, val=0):
        n = t.shape[0]
        if n < length:
            t = F.pad(t, (0, length - n), value=val)
        return t

    def pad_nd(t, sizes, val=0):
        """Pad the first len(sizes) dimensions of t to given sizes."""
        pads = []
        for d, s in reversed(list(enumerate(sizes))):
            diff = s - t.shape[d]
            pads += [0, max(0, diff)]
        return F.pad(t, pads, value=val)

    out = {}
    keys_1d_token = ["token_type", "residue_type", "chain_index", "residue_index",
                      "is_protein", "is_rna", "is_dna", "is_ligand"]
    for k in keys_1d_token:
        if k in batch[0]:
            out[k] = torch.stack([pad1d(f[k], max_tokens) for f in batch])

    out["token_mask"] = torch.stack([pad1d(f["token_mask"].float(), max_tokens) for f in batch])

    for k in ["ref_pos", "ref_mask", "ref_charge", "atom_mask"]:
        if k in batch[0]:
            out[k] = torch.stack([pad_nd(f[k], [max_atoms]) for f in batch])
    for k in ["ref_element", "ref_atom_name_chars"]:
        if k in batch[0]:
            out[k] = torch.stack([pad_nd(f[k], [max_atoms]) for f in batch])

    out["atom_to_token"] = torch.stack([pad1d(f["atom_to_token"], max_atoms) for f in batch])
    out["num_atoms_per_token"] = torch.stack([pad1d(f["num_atoms_per_token"], max_tokens) for f in batch])
    out["token_atom_start"] = torch.stack([pad1d(f["token_atom_start"], max_tokens) for f in batch])

    return out
