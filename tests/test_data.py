"""
Unit tests for data layer:
  - Tokenizer (token scheme, atom-to-token mapping)
  - Feature builder (tensor shapes, correctness)
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.tokenizer import AF3Tokenizer, MolType, Complex
from data.features import build_features


# ---------------------------------------------------------------------------
# Helpers: minimal chain definitions
# ---------------------------------------------------------------------------

def make_protein_chain(n_residues=4):
    return {
        "chain_id": "A",
        "mol_type": MolType.PROTEIN,
        "sequence": ["ALA", "GLY", "SER", "VAL"][:n_residues],
        "atom_names": [["N", "CA", "C", "O"]] * n_residues,
        "elements":   [["N", "C",  "C", "O"]] * n_residues,
        "charges":    [[0.0, 0.0, 0.0, 0.0]]  * n_residues,
        "positions":  [[[0.0, 0.0, i * 1.5], [1.0, 0.0, i * 1.5],
                        [2.0, 0.0, i * 1.5], [2.5, 0.0, i * 1.5]]
                       for i in range(n_residues)],
    }

def make_ligand_chain(n_atoms=3):
    return {
        "chain_id": "B",
        "mol_type": MolType.LIGAND,
        "sequence": ["C"] * n_atoms,
        "atom_names": [[f"C{i}"] for i in range(n_atoms)],
        "elements":   [["C"]] * n_atoms,
        "charges":    [[0.0]] * n_atoms,
        "positions":  [[[float(i), 0.0, 0.0]] for i in range(n_atoms)],
    }

def make_rna_chain(n_residues=3):
    return {
        "chain_id": "C",
        "mol_type": MolType.RNA,
        "sequence": ["A", "C", "G"][:n_residues],
        "atom_names": [["P", "O5'", "C5'", "C4'"]] * n_residues,
        "elements":   [["P", "O",   "C",   "C"  ]] * n_residues,
        "charges":    [[0.0, 0.0, 0.0, 0.0]] * n_residues,
        "positions":  None,
    }


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_protein_token_count(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(4)])
        # 4 residues -> 4 tokens
        assert c.n_tokens == 4

    def test_protein_atom_count(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(4)])
        # 4 residues × 4 atoms = 16 atoms
        assert c.n_atoms == 16

    def test_ligand_token_per_atom(self):
        """Ligands: each atom is its own token."""
        tok = AF3Tokenizer()
        c = tok.tokenize([make_ligand_chain(5)])
        assert c.n_tokens == 5
        assert c.n_atoms == 5

    def test_mixed_complex_tokens(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(3), make_ligand_chain(4)])
        # 3 protein residues + 4 ligand atoms = 7 tokens
        assert c.n_tokens == 7

    def test_atom_to_token_mapping_protein(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(2)])
        # First 4 atoms -> token 0, next 4 -> token 1
        assert c.atom_to_token[:4] == [0, 0, 0, 0]
        assert c.atom_to_token[4:8] == [1, 1, 1, 1]

    def test_atom_to_token_mapping_ligand(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_ligand_chain(3)])
        # Each atom maps to its own token
        assert c.atom_to_token == [0, 1, 2]

    def test_token_atom_start_offsets(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(3)])
        # 4 atoms per residue
        assert c.token_atom_start == [0, 4, 8]

    def test_rna_no_positions(self):
        """RNA chain with no positions should still tokenise without error."""
        tok = AF3Tokenizer()
        c = tok.tokenize([make_rna_chain(3)])
        assert c.n_tokens == 3

    def test_token_index_sequential(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(3), make_ligand_chain(2)])
        indices = [t.token_index for t in c.tokens]
        assert indices == list(range(c.n_tokens))

    def test_chain_ids_correct(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(2), make_ligand_chain(2)])
        chain_ids = [t.chain_id for t in c.tokens]
        assert chain_ids[:2] == ["A", "A"]
        assert chain_ids[2:] == ["B", "B"]

    def test_mol_type_correct(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([make_protein_chain(2), make_ligand_chain(2)])
        types = [t.token_type for t in c.tokens]
        assert types[:2] == [MolType.PROTEIN, MolType.PROTEIN]
        assert types[2:] == [MolType.LIGAND, MolType.LIGAND]

    def test_empty_complex(self):
        tok = AF3Tokenizer()
        c = tok.tokenize([])
        assert c.n_tokens == 0
        assert c.n_atoms == 0


# ---------------------------------------------------------------------------
# Feature builder tests
# ---------------------------------------------------------------------------

class TestFeatureBuilder:
    @pytest.fixture
    def complex_obj(self):
        tok = AF3Tokenizer()
        return tok.tokenize([make_protein_chain(4), make_ligand_chain(3)])

    def test_token_feature_shapes(self, complex_obj):
        feats = build_features(complex_obj)
        N = complex_obj.n_tokens   # 7
        A = complex_obj.n_atoms    # 4*4 + 3 = 19

        assert feats["token_mask"].shape  == (N,)
        assert feats["token_type"].shape  == (N,)
        assert feats["residue_type"].shape== (N,)
        assert feats["chain_index"].shape == (N,)
        assert feats["residue_index"].shape == (N,)

    def test_atom_feature_shapes(self, complex_obj):
        feats = build_features(complex_obj)
        N = complex_obj.n_tokens
        A = complex_obj.n_atoms

        assert feats["ref_pos"].shape    == (A, 3)
        assert feats["ref_mask"].shape   == (A,)
        assert feats["ref_element"].shape== (A, 28)   # N_ELEMENTS
        assert feats["ref_charge"].shape == (A, 1)
        assert feats["ref_atom_name_chars"].shape == (A, 4, 64)
        assert feats["atom_to_token"].shape == (A,)
        assert feats["num_atoms_per_token"].shape == (N,)

    def test_ref_mask_is_one_when_positions_given(self, complex_obj):
        feats = build_features(complex_obj)
        # protein chain has positions -> ref_mask should be 1 for those atoms
        assert feats["ref_mask"][:16].all()

    def test_is_protein_flags(self, complex_obj):
        feats = build_features(complex_obj)
        # first 4 tokens are protein
        assert feats["is_protein"][:4].all()
        assert not feats["is_protein"][4:].any()

    def test_is_ligand_flags(self, complex_obj):
        feats = build_features(complex_obj)
        assert not feats["is_ligand"][:4].any()
        assert feats["is_ligand"][4:].all()

    def test_token_mask_all_ones(self, complex_obj):
        feats = build_features(complex_obj)
        assert feats["token_mask"].all()

    def test_element_one_hot_sums_to_one(self, complex_obj):
        feats = build_features(complex_obj)
        # Each atom has exactly one element set
        row_sums = feats["ref_element"].sum(-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    def test_ref_pos_values_match_input(self):
        """Positions in feats should match the positions given in the chain."""
        tok = AF3Tokenizer()
        chain = make_protein_chain(1)   # 1 residue, 4 atoms
        chain["positions"] = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        c = tok.tokenize([chain])
        feats = build_features(c)
        expected = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=torch.float)
        assert torch.allclose(feats["ref_pos"], expected)

    def test_ground_truth_included(self, complex_obj):
        A = complex_obj.n_atoms
        gt_pos  = torch.randn(A, 3)
        gt_mask = torch.ones(A, dtype=torch.bool)
        feats = build_features(complex_obj, atom_positions_gt=gt_pos, atom_resolved_mask=gt_mask)
        assert "ground_truth" in feats
        assert feats["ground_truth"]["atom_positions"].shape == (A, 3)
        assert feats["ground_truth"]["atom_resolved_mask"].shape == (A,)

    def test_msa_included(self, complex_obj):
        N = complex_obj.n_tokens
        msa = torch.randint(0, 20, (8, N))
        feats = build_features(complex_obj, msa=msa)
        assert "msa" in feats
        assert feats["msa"].shape == (8, N)
        assert "msa_mask" in feats
