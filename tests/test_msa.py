from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data import ROOT, load_reference
from src.msa import load_msa, frequencies, compute_neff


def test_msa_loading_and_frequencies():
    """Verify MSA parsing and regularized frequency computation."""
    msa_path = ROOT / "data" / "raw" / "msa" / "DMS_msa_files" / "BLAT_ECOLX_full_11-26-2021_b02.a2m"
    if not msa_path.exists():
        pytest.skip("TEM-1 MSA file not found locally")

    query, msa = load_msa(msa_path)
    assert len(query) == 286, f"TEM-1 query should be 286 residues, got {len(query)}"
    assert msa.ndim == 2, "MSA array should be 2D"
    assert msa.shape[1] == 286, "MSA width should match query length"

    freq = frequencies(msa)
    assert freq.shape == (286, 20), f"Frequency matrix should be (286, 20), got {freq.shape}"
    col_sums = freq.sum(axis=1)
    # Pseudocount regularization sums to close to 1.0
    assert np.all(col_sums > 0.85) and np.all(col_sums < 1.05), "Frequencies should sum near 1.0 per column"


def test_compute_neff_reweighting():
    """Verify sequence-reweighting logic on controlled synthetic sequences."""
    # 2 identical sequences + 1 completely different sequence
    # Identical pair get weight 0.5 each (sum = 1.0), distinct sequence gets weight 1.0 -> total Neff = 2.0
    seqs = np.array([list("ACDEF"), list("ACDEF"), list("VWYYY")], dtype="S1")
    neff = compute_neff(seqs, theta=0.8)
    assert np.isclose(neff, 2.0), f"Expected Neff=2.0 for 2 identical + 1 distinct seqs, got {neff}"

    # 4 completely distinct sequences
    distinct_seqs = np.array([
        list("AAAAA"),
        list("CCCCC"),
        list("EEEEE"),
        list("GGGGG"),
    ], dtype="S1")
    neff_distinct = compute_neff(distinct_seqs, theta=0.8)
    assert np.isclose(neff_distinct, 4.0), f"Expected Neff=4.0 for 4 distinct seqs, got {neff_distinct}"


def test_reference_neff_l():
    """Verify that reference Neff/L for BLAT_ECOLX matches ProteinGym benchmark within tolerance."""
    ref = load_reference()
    if "BLAT_ECOLX_Stiffler_2015" in ref.index:
        ref_neff_l = float(ref.loc["BLAT_ECOLX_Stiffler_2015"].MSA_Neff_L)
        expected = 177.98
        # Tolerance within ±5%
        assert abs(ref_neff_l - expected) / expected < 0.05, f"Reference Neff/L {ref_neff_l} deviated from {expected}"
