from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
MSA_DIR = RAW / "msa" / "DMS_msa_files"
DMS_DIR = RAW / "dms" / "DMS_ProteinGym_substitutions"

STARTER_SET = [
    # --- original 10 ---
    "BLAT_ECOLX_Stiffler_2015",
    "DYR_ECOLI_Thompson_2019",
    "KKA2_KLEPN_Melnikov_2014",
    "AMIE_PSEAE_Wrenbeck_2017",
    "PTEN_HUMAN_Mighell_2018",
    "TPMT_HUMAN_Matreyek_2018",
    "R1AB_SARS2_Flynn_2022",
    "TPK1_HUMAN_Weile_2017",
    "RNC_ECOLI_Weeks_2023",
    "OXDA_RHOTO_Vanella_2023_activity",
    # --- 10 new (sorted by Neff/L) ---
    "GFP_AEQVI_Sarkisyan_2016",         # 0.06
    "P53_HUMAN_Kotler_2018",             # 0.37
    "MK01_HUMAN_Brenan_2016",            # 25.98
    "UBC9_HUMAN_Weile_2017",             # 55.33
    "ADRB2_HUMAN_Jones_2020",            # 67.99
    "RASH_HUMAN_Bandaru_2017",           # 125.77
    "SUMO1_HUMAN_Weile_2017",            # 140.90
    "CCDB_ECOLI_Tripathi_2016",          # 168.82
    "CALM1_HUMAN_Weile_2017",            # 190.77
    "IF1_ECOLI_Kelsic_2016",             # 536.24
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
