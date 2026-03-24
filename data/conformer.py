"""
Reference conformer generation for ligands using RDKit.

AF3 Supplementary Methods 2.6: for each ligand the model receives a
reference 3-D conformer generated from SMILES. These positions encode
bond topology and geometry but are randomly rotated/translated at inference,
so the model only uses them to infer local chemistry, not global placement.

OF3: data/pipelines/featurization/conformer.py
Boltz: data/mol.py
"""

from typing import Optional

import numpy as np


def generate_conformer(smiles: str, seed: int = 0) -> Optional[np.ndarray]:
    """
    Generate a 3-D reference conformer from a SMILES string.

    Returns:
        positions: (N_atoms, 3) array of atom positions in Angstrom,
                   or None if generation fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit is required: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=seed)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if result != 0:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    return positions


def get_mol_features(smiles: str, seed: int = 0) -> Optional[dict]:
    """
    Extract atom-level features from a SMILES string for the input embedder.

    Returns a dict with:
      - positions:  (N, 3) reference conformer positions
      - elements:   list[str] element symbols (heavy atoms only)
      - charges:    list[float] formal charges
      - atom_names: list[str] generated atom names (element + index)
    Or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise ImportError("RDKit is required: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=seed)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if result != 0:
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()

    positions, elements, charges, atom_names = [], [], [], []
    element_counts: dict = {}

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        element_counts[sym] = element_counts.get(sym, 0) + 1
        positions.append(list(conf.GetAtomPosition(atom.GetIdx())))
        elements.append(sym)
        charges.append(float(atom.GetFormalCharge()))
        atom_names.append(f"{sym}{element_counts[sym]}")

    return {
        "positions": positions,
        "elements": elements,
        "charges": charges,
        "atom_names": atom_names,
    }
