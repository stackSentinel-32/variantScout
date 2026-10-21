import numpy as np
import pandas as pd
import pytest

from src.structure import extract_ca_coords, compute_ca_distance
from app.app import render_3d_structure, generate_combinatorial_mutants


def test_render_3d_structure_generation():
    """Verify py3Dmol viewer HTML generation with highlights and color modes."""
    # Synthetic PDB model
    pdb_mock = (
        "ATOM      1  CA  MET A   1      20.000  15.000  30.000  1.00 95.00           C\n"
        "ATOM      2  CA  ALA A   2      23.800  15.000  30.000  1.00 75.00           C\n"
        "ATOM      3  CA  SER A   3      27.600  15.000  30.000  1.00 55.00           C\n"
        "ATOM      4  CA  LEU A   4      31.400  15.000  30.000  1.00 45.00           C\n"
    )

    # 1. Candidate highlight mode
    html_high = render_3d_structure(
        pdb_text=pdb_mock,
        top_positions=[1],
        catalytic_positions=[3],
        custom_positions=[2],
        color_mode="Highlight Candidates & Catalytic Sites",
    )
    assert isinstance(html_high, str)
    assert "3dmolviewer_" in html_high
    assert "#EC4899" in html_high  # Hot pink top candidate
    assert "#FACC15" in html_high  # Yellow catalytic residue
    assert "#06B6D4" in html_high  # Cyan custom highlight

    # 2. pLDDT confidence mode
    html_plddt = render_3d_structure(
        pdb_text=pdb_mock,
        plddt_dict={1: 95.0, 2: 75.0, 3: 55.0, 4: 45.0},
        color_mode="AlphaFold pLDDT Confidence",
    )
    assert "#0053D6" in html_plddt  # Deep blue (>90)
    assert "#65CBF3" in html_plddt  # Cyan (70-90)
    assert "#FFDB13" in html_plddt  # Yellow (50-70)
    assert "#FF7D45" in html_plddt  # Orange (<50)


def test_generate_combinatorial_mutants():
    """Verify combinatorial double mutant generation, additive scoring, and spatial clash filtering."""
    # Mock singles dataframe
    mock_df = pd.DataFrame([
        {"mutant": "M180T", "pos": 180, "wt": "M", "mt": "T", "score": 0.820, "esm": 1.5, "Tier": "🟢 Tier 1 (High)"},
        {"mutant": "R162H", "pos": 162, "wt": "R", "mt": "H", "score": 0.810, "esm": 1.2, "Tier": "🟢 Tier 1 (High)"},
        {"mutant": "Q37R", "pos": 37, "wt": "Q", "mt": "R", "score": 0.800, "esm": 1.1, "Tier": "🟢 Tier 1 (High)"},
        {"mutant": "S68A", "pos": 68, "wt": "S", "mt": "A", "score": 0.400, "esm": -3.5, "Tier": "Low (< 85%)"},
    ])

    mock_ca_coords = {
        37: np.array([10.0, 10.0, 10.0], dtype=np.float32),
        162: np.array([25.0, 25.0, 25.0], dtype=np.float32),
        180: np.array([28.0, 25.0, 25.0], dtype=np.float32),  # 3.0A from 162 (< 5A close contact)
        68: np.array([50.0, 50.0, 50.0], dtype=np.float32),
    }

    doubles_df = generate_combinatorial_mutants(
        single_df=mock_df,
        score_col="score",
        ca_coords=mock_ca_coords,
        pool_size=10,
        only_tolerant=True,  # excludes S68A
    )

    # With 3 singles (M180T, R162H, Q37R), combinations = 3
    assert len(doubles_df) == 3
    assert "Tier" in doubles_df.columns
    assert "ca_dist" in doubles_df.columns
    assert "contact_risk" in doubles_df.columns

    # Check additive score: M180T (0.820) + R162H (0.810) = 1.630
    pair_180_162 = doubles_df[doubles_df["pair"] == "R162H + M180T"].iloc[0]
    assert np.isclose(pair_180_162["score_double"], 1.630, atol=1e-3)
    assert pair_180_162["contact_risk"] == True
    assert "⚠️ Contact" in pair_180_162["status"]

    # Check distant pair: Q37R + R162H (~25.9A)
    pair_37_162 = doubles_df[doubles_df["pair"] == "Q37R + R162H"].iloc[0]
    assert pair_37_162["ca_dist"] > 20.0
    assert pair_37_162["contact_risk"] == False
    assert "✅ Distant" in pair_37_162["status"]
