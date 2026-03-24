"""
Unified token scheme for AF3.

AF3 key insight: polymer residues AND individual ligand atoms are ALL tokens.
This enables a single architecture to handle proteins, RNA, DNA, and small molecules.

Token types:
  - Polymer residues (proteins, RNA, DNA): one token per residue
    → atom positions aggregated from all heavy atoms in residue
  - Ligand atoms: one token per heavy atom
  - Modified residues: treated as polymer residues

Reference: AF3 paper Methods, Boltz data/const.py
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Residue / molecule type constants
# ---------------------------------------------------------------------------

class MolType(IntEnum):
    PROTEIN = 0
    RNA = 1
    DNA = 2
    LIGAND = 3
    ION = 4
    UNKNOWN = 5


# Standard 20 amino acids + special tokens
AA_VOCAB = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",  # unknown amino acid
    "<PAD>",
]

# RNA nucleotides
RNA_VOCAB = ["A", "C", "G", "U", "N", "<PAD>"]

# DNA nucleotides
DNA_VOCAB = ["DA", "DC", "DG", "DT", "DN", "<PAD>"]

AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
RNA_TO_IDX = {nt: i for i, nt in enumerate(RNA_VOCAB)}
DNA_TO_IDX = {nt: i for i, nt in enumerate(DNA_VOCAB)}

# Elements present in biomolecules (128 most common, one-hot encoded)
ELEMENT_VOCAB = [
    "H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "Fe", "Zn", "Cu", "Mg", "Ca", "Na", "K", "Mn", "Co", "Ni",
    "Se", "Si", "B", "As", "V", "Mo", "W", "<UNK>",
]
N_ELEMENTS = len(ELEMENT_VOCAB)
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}


# ---------------------------------------------------------------------------
# Token data structures
# ---------------------------------------------------------------------------

@dataclass
class AtomData:
    """Per-atom data for a single atom in the complex."""
    element: str                      # element symbol
    atom_name: str                    # IUPAC atom name (e.g. "CA", "N")
    charge: float                     # formal charge
    position: Optional[list] = None   # [x, y, z] in Å; None if unknown
    is_present: bool = True


@dataclass
class Token:
    """
    A single token in the AF3 token scheme.

    For polymer residues: token_type = MolType.{PROTEIN, RNA, DNA},
      residue_type encodes the residue, atoms contains all heavy atoms.
    For ligand atoms: token_type = MolType.LIGAND, atoms has one element.
    """
    token_type: MolType
    residue_type: int                 # index into per-type vocabulary
    chain_id: str
    residue_index: int
    token_index: int                  # global token index in the complex
    atoms: list                       # list of AtomData
    # Number of atoms in this token (1 for ligands, ≥1 for residues)
    n_atoms: int = field(init=False)

    def __post_init__(self):
        self.n_atoms = len(self.atoms)

    @property
    def is_polymer(self) -> bool:
        return self.token_type in (MolType.PROTEIN, MolType.RNA, MolType.DNA)

    @property
    def is_ligand(self) -> bool:
        return self.token_type == MolType.LIGAND


@dataclass
class Complex:
    """Full molecular complex after tokenisation."""
    tokens: list          # list[Token] ordered by token_index
    n_tokens: int
    n_atoms: int
    # Mapping from atom index to token index
    atom_to_token: list   # list[int], length n_atoms
    # Atom offsets: token_atom_start[i] = first atom index for token i
    token_atom_start: list  # list[int], length n_tokens


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class AF3Tokenizer:
    """
    Converts a list of chains (polymer or ligand) into an AF3 Complex.

    Input format (dict per chain):
      {
        "chain_id": str,
        "mol_type": MolType,
        "sequence": list[str],      # residue names or element symbols
        "atom_names": list[list[str]],  # per-residue atom names
        "elements": list[list[str]],    # per-residue elements
        "charges": list[list[float]],   # per-residue charges
        "positions": list[list[list[float]]] | None,  # per-residue atom positions
      }
    """

    def tokenize(self, chains: list) -> Complex:
        tokens = []
        atom_to_token = []
        atom_idx = 0

        for chain in chains:
            mol_type = chain["mol_type"]
            chain_id = chain["chain_id"]

            for res_idx, res_name in enumerate(chain["sequence"]):
                atom_names = chain["atom_names"][res_idx]
                elements = chain["elements"][res_idx]
                charges = chain["charges"][res_idx]
                positions = (chain["positions"][res_idx]
                             if chain.get("positions") is not None else
                             [None] * len(atom_names))

                atoms = [
                    AtomData(
                        element=elements[i],
                        atom_name=atom_names[i],
                        charge=charges[i],
                        position=positions[i],
                    )
                    for i in range(len(atom_names))
                ]

                if mol_type == MolType.PROTEIN:
                    res_type = AA_TO_IDX.get(res_name, AA_TO_IDX["UNK"])
                    token = Token(
                        token_type=mol_type, residue_type=res_type,
                        chain_id=chain_id, residue_index=res_idx,
                        token_index=len(tokens), atoms=atoms,
                    )
                    tokens.append(token)
                    for _ in atoms:
                        atom_to_token.append(len(tokens) - 1)
                        atom_idx += 1

                elif mol_type == MolType.RNA:
                    res_type = RNA_TO_IDX.get(res_name, RNA_TO_IDX["N"])
                    token = Token(
                        token_type=mol_type, residue_type=res_type,
                        chain_id=chain_id, residue_index=res_idx,
                        token_index=len(tokens), atoms=atoms,
                    )
                    tokens.append(token)
                    for _ in atoms:
                        atom_to_token.append(len(tokens) - 1)
                        atom_idx += 1

                elif mol_type == MolType.DNA:
                    res_type = DNA_TO_IDX.get(res_name, DNA_TO_IDX["DN"])
                    token = Token(
                        token_type=mol_type, residue_type=res_type,
                        chain_id=chain_id, residue_index=res_idx,
                        token_index=len(tokens), atoms=atoms,
                    )
                    tokens.append(token)
                    for _ in atoms:
                        atom_to_token.append(len(tokens) - 1)
                        atom_idx += 1

                else:
                    # Ligand / ion: one token per atom
                    for atom in atoms:
                        res_type = ELEMENT_TO_IDX.get(atom.element, ELEMENT_TO_IDX["<UNK>"])
                        token = Token(
                            token_type=mol_type, residue_type=res_type,
                            chain_id=chain_id, residue_index=res_idx,
                            token_index=len(tokens), atoms=[atom],
                        )
                        tokens.append(token)
                        atom_to_token.append(len(tokens) - 1)
                        atom_idx += 1

        # Build token_atom_start
        token_atom_start = []
        ptr = 0
        for tok in tokens:
            token_atom_start.append(ptr)
            ptr += tok.n_atoms

        return Complex(
            tokens=tokens,
            n_tokens=len(tokens),
            n_atoms=atom_idx,
            atom_to_token=atom_to_token,
            token_atom_start=token_atom_start,
        )
