from pathlib import Path
import pytest
import numpy as np

from src.data import ROOT
from src.structure import compute_rsa_plddt, STRUCT_DIR


def test_structure_rsa_plddt_parsing():
    """Verify parsing of AlphaFold PDB structure into RSA and pLDDT dictionaries."""
    pdb_file = "BLAT_ECOLX.pdb"
    full_path = STRUCT_DIR / pdb_file
    if not full_path.exists():
        pytest.skip(f"AlphaFold PDB structure {pdb_file} not found locally")

    rsa, plddt = compute_rsa_plddt(pdb_file)

    assert isinstance(rsa, dict), "RSA should be a dictionary"
    assert isinstance(plddt, dict), "pLDDT should be a dictionary"
    assert len(rsa) > 200, f"Expected >200 residues with RSA, got {len(rsa)}"
    assert len(plddt) > 200, f"Expected >200 residues with pLDDT, got {len(plddt)}"

    # Check RSA range (most surface residues are between 0.0 and 1.5)
    for res_id, rsa_val in rsa.items():
        assert 0.0 <= rsa_val <= 2.5, f"Residue {res_id} has out-of-range RSA: {rsa_val}"

    # Check pLDDT range ([0, 100])
    for res_id, plddt_val in plddt.items():
        assert 0.0 <= plddt_val <= 100.0, f"Residue {res_id} has out-of-range pLDDT: {plddt_val}"

    # Catalytic serine 68 should be in a well-ordered region (pLDDT >= 80)
    assert 68 in plddt, "Active site Serine 68 should be present"
    assert plddt[68] >= 80.0, f"Catalytic Serine 68 should have high pLDDT confidence, got {plddt[68]}"

    # Signal peptide residues (e.g. pos 15) typically have lower pLDDT than the catalytic core
    if 15 in plddt:
        assert plddt[15] < plddt[68], f"Signal peptide pos 15 ({plddt[15]}) should be less ordered than catalytic core pos 68 ({plddt[68]})"

    # Verify parsing from raw PDB text string
    raw_pdb = full_path.read_text()
    rsa_str, plddt_str = compute_rsa_plddt(raw_pdb)
    assert len(rsa_str) == len(rsa)
    assert np.isclose(rsa_str[68], rsa[68], atol=1e-4)
    assert np.isclose(plddt_str[68], plddt[68], atol=1e-4)


def test_ca_coordinates_and_distance():
    """Verify C-alpha coordinate extraction and 3D Euclidean distance calculation."""
    from src.structure import extract_ca_coords, compute_ca_distance

    pdb_file = "BLAT_ECOLX.pdb"
    full_path = STRUCT_DIR / pdb_file
    if not full_path.exists():
        pytest.skip(f"AlphaFold PDB structure {pdb_file} not found locally")

    coords = extract_ca_coords(pdb_file)
    assert len(coords) == 286, f"Expected 286 CA coordinates for TEM-1, got {len(coords)}"
    assert 68 in coords, "Residue 68 should have CA coordinates"
    assert 180 in coords, "Residue 180 should have CA coordinates"

    # Consecutive residues along the backbone have C-alpha distance ~3.8 Angstroms
    d_adjacent = compute_ca_distance(coords[68], coords[69])
    assert 3.5 <= d_adjacent <= 4.2, f"Adjacent CA distance should be ~3.8A, got {d_adjacent:.2f}A"

    # Distant residues across the protein structure
    d_distant = compute_ca_distance(coords[68], coords[180])
    assert d_distant > 10.0, f"Residues 68 and 180 should be distant (>10A), got {d_distant:.2f}A"

    # Test raw PDB string input
    raw_pdb = full_path.read_text()
    coords_from_str = extract_ca_coords(raw_pdb)
    assert len(coords_from_str) == 286
    np.testing.assert_allclose(coords[68], coords_from_str[68], atol=1e-4)

