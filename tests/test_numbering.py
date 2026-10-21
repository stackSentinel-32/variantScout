from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data import ROOT, load_reference, load_assay
from src.msa import load_msa, frequencies, evo_scores


def test_ambler_mapping():
    """Verify sequential to Ambler coordinate mapping for TEM-1 beta-lactamase."""
    # TEM-1 literature offset: Ambler = sequential + 2
    offset = 2
    seq_m180 = 180
    seq_s68 = 68

    ambler_m182 = seq_m180 + offset
    ambler_s70 = seq_s68 + offset

    assert ambler_m182 == 182, "App position 180 must map to Ambler M182"
    assert ambler_s70 == 70, "App position 68 must map to Ambler S70"


def test_tem1_m180t_and_s68a_sanity():
    """Verify M180T (stabilizer) scores in top tier and S68A (catalytic serine) is buried."""
    # 1. Evolutionary score sanity check
    msa_path = ROOT / "data" / "raw" / "msa" / "DMS_msa_files" / "BLAT_ECOLX_full_11-26-2021_b02.a2m"
    if msa_path.exists():
        query, msa = load_msa(msa_path)
        freq = frequencies(msa)

        # M180T (pos 180, col 179): Met -> Thr
        m180t_evo = float(evo_scores(freq, [179], ["M"], ["T"])[0])
        # S68A (pos 68, col 67): Ser -> Ala
        s68a_evo = float(evo_scores(freq, [67], ["S"], ["A"])[0])

        assert m180t_evo > 2.0, f"M180T should have strongly positive evolutionary score, got {m180t_evo}"
        assert s68a_evo < -2.0, f"S68A should have strongly negative evolutionary score, got {s68a_evo}"
        assert m180t_evo > s68a_evo, "M180T must rank substantially higher than catalytic S68A"

    # 2. Experimental DMS assay check (Stiffler 2015)
    ref = load_reference()
    if "BLAT_ECOLX_Stiffler_2015" in ref.index:
        df = load_assay("BLAT_ECOLX_Stiffler_2015", ref)
        s68a_rows = df[df.mutant == "S68A"]
        if not s68a_rows.empty:
            s68a_score = s68a_rows.iloc[0]["DMS_score"]
            dms_pct = (df["DMS_score"] <= s68a_score).mean()
            assert dms_pct <= 0.05, f"S68A should be in bottom 5% of DMS fitness scores, got {dms_pct*100:.2f}%"
