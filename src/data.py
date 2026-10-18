from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
MSA_DIR = RAW / "msa" / "DMS_msa_files"
DMS_DIR = RAW / "dms" / "DMS_ProteinGym_substitutions"

STARTER_SET = [
    "BLAT_ECOLX_Stiffler_2015",
    "DYR_ECOLI_Thompson_2019",
    "KKA2_KLEPN_Melnikov_2014",
    "AMIE_PSEAE_Wrenbeck_2017",
    "PTEN_HUMAN_Mighell_2018",
    "TPMT_HUMAN_Matreyek_2018",
]


def load_reference():
    return pd.read_csv(RAW / "DMS_substitutions.csv").set_index("DMS_id", drop=False)


def load_assay(dms_id, ref):
    row = ref.loc[dms_id]
    df = pd.read_csv(DMS_DIR / row.DMS_filename)
    df = df[~df.mutant.str.contains(":")].copy()
    df["wt"] = df.mutant.str[0]
    df["pos"] = df.mutant.str[1:-1].astype(int)
    df["mt"] = df.mutant.str[-1]
    return df.reset_index(drop=True)


def check_numbering(row, df, msa_query):
    seq = row.target_seq
    wrong = sum(seq[p - 1] != a for p, a in zip(df.pos, df.wt))
    assert wrong == 0, f"{wrong} mutations don't match the target sequence"
    start, end = int(row.MSA_start), int(row.MSA_end)
    assert msa_query == seq[start - 1:end], "MSA query doesn't match the target sequence"
    assert df.pos.between(start, end).all(), "some mutations fall outside the MSA"
