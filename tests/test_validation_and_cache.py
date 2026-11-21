from pathlib import Path
import pytest
import pandas as pd

from app.app import (
    TEM1_DEMO_SEQUENCE,
    validate_sequence,
    is_valid_feature_cache,
    parse_fasta,
)
from src.data import ROOT


def test_tem1_sequence_integrity():
    """Verify canonical TEM-1 sequence is exactly 286 aa and standard."""
    assert len(TEM1_DEMO_SEQUENCE) == 286
    valid, msg = validate_sequence(TEM1_DEMO_SEQUENCE)
    assert valid, f"Canonical TEM-1 sequence should be valid, got: {msg}"
    # Verify expected key motifs
    assert "LLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATT" in TEM1_DEMO_SEQUENCE
    assert TEM1_DEMO_SEQUENCE.startswith("MSIQHFRVALIPFFAAFCLPVFA")
    assert TEM1_DEMO_SEQUENCE.endswith("QATMDERNRQIAEIGASLIKHW")


def test_sequence_validation_rejects_non_standard_amino_acids():
    """Ensure non-standard characters (like B, Z, X, numbers) are blocked."""
    # B at the end
    seq_with_b = TEM1_DEMO_SEQUENCE[:120] + "GALANRVELRKDGETVIMSTYNQTSLDHVLELFWKDPGIATHRVVMKDAERQQLVBPQ"
    valid, msg = validate_sequence(seq_with_b)
    assert not valid
    assert "B" in msg
    assert "Only the 20 standard amino acids" in msg

    # Empty
    valid_empty, _ = validate_sequence("")
    assert not valid_empty

    # Too short
    valid_short, msg_short = validate_sequence("MKTAY")
    assert not valid_short
    assert "too short" in msg_short


def test_cache_validation_prevents_stale_data_leakage(tmp_path: Path):
    """Ensure feature cache is strictly keyed and verified against the query sequence."""
    test_seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 aa
    parquet_path = tmp_path / "test_features.parquet"

    # Create dummy dataframe matching test_seq
    rows = []
    for pos, wt in enumerate(test_seq, start=1):
        for mt in test_seq:
            if mt != wt:
                rows.append(dict(pos=pos, wt=wt, mt=mt, mutant=f"{wt}{pos}{mt}", blosum=1.,
                                 esm=0., rsa=0.5, plddt=90.))
    df_match = pd.DataFrame(rows)
    for invalid in [df_match.iloc[:-1], df_match.drop(columns="esm"),
                    pd.concat([df_match.iloc[1:], df_match.iloc[[1]]])]:
        invalid.to_parquet(parquet_path, index=False)
        assert not is_valid_feature_cache(parquet_path, test_seq)

    df_match.to_parquet(parquet_path, index=False)
    assert is_valid_feature_cache(parquet_path, test_seq)
    assert not is_valid_feature_cache(parquet_path, test_seq, with_alignment=True)

    # 2. Sequence with mismatch should be rejected
    mutated_seq = "ACDEFGHIKLMNPQRSTVWA"
    assert is_valid_feature_cache(parquet_path, mutated_seq) is False

    # 3. Truncated sequence should be rejected
    truncated_seq = test_seq[:10]
    assert is_valid_feature_cache(parquet_path, truncated_seq) is False
