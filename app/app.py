"""Mutation Picker — Streamlit app.

Usage
-----
    streamlit run app/app.py

The user provides:
  • A protein sequence in FASTA format (required)
  • An optional .a2m MSA file
  • Their UniProt ID (for fetching the AlphaFold structure)

The app outputs a ranked top-20 table, a position×amino-acid heatmap, and
flags positions with shallow MSA depth or low pLDDT confidence.

Model choice
------------
  full    — Ridge trained on evo + entropy + blosum + rsa + esm   (needs MSA)
  fallback — Ridge trained on blosum + rsa + esm only             (no MSA needed)

Both models are re-fit here on the baseline data every time the app starts,
which is fast (~1 s) since all features are pre-cached.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.msa import AA, frequencies, evo_scores, site_entropy, load_msa
from src.structure import compute_rsa_plddt, STRUCT_DIR
from src.esm import score_sequence, esm_mutation_scores, _load_model

B62 = substitution_matrices.load("BLOSUM62")

FEATURES_FULL = ["evo", "entropy", "blosum", "rsa", "esm"]
FEATURES_FALLBACK = ["blosum", "rsa", "esm"]


# ---------------------------------------------------------------------------
# helper: train a ridge model on baseline data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_baseline_models():
    """Fit Ridge models on the baseline data (cached across sessions)."""
    from src.data import RESULTS, load_reference, STARTER_SET, MSA_DIR
    from src.baseline import score_assay

    ref = load_reference()
    frames = []
    for dms_id in STARTER_SET:
        msa_path = MSA_DIR / ref.loc[dms_id].MSA_filename
        if not msa_path.exists():
            continue
        df, _ = score_assay(dms_id, ref)
        df["assay"] = dms_id
        frames.append(df[["assay", "DMS_score", *FEATURES_FULL]])

    all_df = pd.concat(frames, ignore_index=True)

    def rank_within(df, cols):
        out = df.copy()
        out["y"] = out.groupby("assay")["DMS_score"].transform(lambda s: s.rank(pct=True))
        for c in cols:
            out[c] = out.groupby("assay")[c].transform(lambda s: s.rank(pct=True))
        return out

    full_df = rank_within(all_df, FEATURES_FULL)
    fallback_df = rank_within(all_df, FEATURES_FALLBACK)

    ridge_full = Ridge(alpha=1.0).fit(full_df[FEATURES_FULL], full_df["y"])
    ridge_fallback = Ridge(alpha=1.0).fit(fallback_df[FEATURES_FALLBACK], fallback_df["y"])
    return ridge_full, ridge_fallback


# ---------------------------------------------------------------------------
# helpers: parse FASTA, fetch AlphaFold structure
# ---------------------------------------------------------------------------

def parse_fasta(text):
    lines = text.strip().splitlines()
    seq_lines = [l for l in lines if not l.startswith(">")]
    return "".join(seq_lines).upper().replace(" ", "")


def fetch_alphafold_pdb(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    return r.text


def rsa_plddt_from_text(pdb_text):
    """Parse RSA and pLDDT from PDB text (in-memory, no file needed)."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.SeqUtils import seq1
    from src.structure import MAXASA

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_text))
    model = struct[0]
    ShrakeRupley().compute(model, level="R")

    chain = next(model.get_chains())
    rsa, plddt = {}, {}
    for r in chain:
        if r.id[0] != " ":
            continue
        aa = seq1(r.resname)
        if aa not in MAXASA:
            continue
        rsa[r.id[1]] = r.sasa / MAXASA[aa]
        plddt[r.id[1]] = r["CA"].get_bfactor()
    return rsa, plddt


# ---------------------------------------------------------------------------
# score all single mutants for a given sequence
# ---------------------------------------------------------------------------

def score_all_mutations(seq, freq, rsa_dict, plddt_dict, logprobs, tok):
    """Return a DataFrame with every single-mutant score."""
    rows = []
    for pos1 in range(1, len(seq) + 1):
        wt = seq[pos1 - 1]
        if wt not in AA:
            continue
        for mt in AA:
            if mt == wt:
                continue
            row = {
                "mutant": f"{wt}{pos1}{mt}",
                "pos": pos1,
                "wt": wt,
                "mt": mt,
                "blosum": B62[wt, mt],
                "rsa": rsa_dict.get(pos1, np.nan),
                "plddt": plddt_dict.get(pos1, np.nan),
            }
            if freq is not None:
                col = pos1 - 1
                row["evo"] = float(evo_scores(freq, [col], [wt], [mt])[0])
                row["entropy"] = float(site_entropy(freq)[col])
            rows.append(row)

    df = pd.DataFrame(rows)
    # ESM scores
    pos = df["pos"].to_numpy()
    wt_arr = df["wt"].values
    mt_arr = df["mt"].values
    df["esm"] = esm_mutation_scores(logprobs, tok, pos, wt_arr, mt_arr)
    return df


# ---------------------------------------------------------------------------
# heatmap figure
# ---------------------------------------------------------------------------

def make_heatmap(df, score_col, title):
    grid = df.pivot_table(index="mt", columns="pos", values=score_col).reindex(list(AA))
    fig, ax = plt.subplots(figsize=(max(12, len(grid.columns) // 3), 5))
    sns.heatmap(grid, cmap="viridis", ax=ax, cbar_kws={"label": score_col})
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Mutation Picker", layout="wide")
st.title("🧬 Mutation Picker")
st.caption("Ranks every single amino-acid substitution by predicted fitness impact.")

# sidebar inputs
with st.sidebar:
    st.header("Input")
    fasta_text = st.text_area("Protein sequence (FASTA or raw)", height=180,
                               placeholder=">MyProtein\nMKTAYIAKQRQ...")
    uniprot_id = st.text_input("UniProt ID (for AlphaFold structure)",
                                placeholder="e.g. P62593")
    msa_file = st.file_uploader("MSA file (.a2m, optional)", type=["a2m", "fasta", "fa"])
    run_btn = st.button("Score mutations", type="primary")

if not run_btn:
    st.info("Fill in the sequence and UniProt ID in the sidebar, then click **Score mutations**.")
    st.stop()

# parse inputs
if not fasta_text.strip():
    st.error("Please provide a protein sequence.")
    st.stop()

seq = parse_fasta(fasta_text)
if len(seq) > 1022:
    st.error(f"Sequence length {len(seq)} exceeds ESM-2 limit of 1022. Truncate or window first.")
    st.stop()

if not uniprot_id.strip():
    st.error("UniProt ID is required to fetch the AlphaFold structure.")
    st.stop()

with st.spinner("Loading models (one-time, ~10 s) …"):
    ridge_full, ridge_fallback = load_baseline_models()

# fetch structure
with st.spinner("Fetching AlphaFold structure …"):
    pdb_text = fetch_alphafold_pdb(uniprot_id.strip())

if pdb_text is None:
    st.warning("Could not fetch AlphaFold structure. RSA and pLDDT will be missing.")
    rsa_dict, plddt_dict = {}, {}
else:
    rsa_dict, plddt_dict = rsa_plddt_from_text(pdb_text)

# confidence flags
low_plddt_positions = {p for p, v in plddt_dict.items() if v < 70}

# MSA
freq = None
neff_L = None
if msa_file is not None:
    msa_bytes = msa_file.read().decode()
    tmp_path = ROOT / "data" / "processed" / "user_msa_tmp.a2m"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(msa_bytes)
    _, msa_mat = load_msa(tmp_path)
    freq = frequencies(msa_mat)
    n_seqs = msa_mat.shape[0]
    neff_L = n_seqs / len(seq)

# ESM
with st.spinner("Running ESM-2 (cached after first run) …"):
    tok, _ = _load_model()
    logprobs = score_sequence(seq, uniprot_id.strip())

# score all mutations
with st.spinner("Scoring all single mutations …"):
    df = score_all_mutations(seq, freq, rsa_dict, plddt_dict, logprobs, tok)

# choose model and score
use_full = freq is not None
if use_full:
    feats = FEATURES_FULL
    model = ridge_full
else:
    feats = FEATURES_FALLBACK
    model = ridge_fallback

feat_df = df[feats].copy()
# rank-normalise (approximate: within this protein's mutations)
for c in feats:
    feat_df[c] = feat_df[c].rank(pct=True)

df["score"] = model.predict(feat_df.values)

# ---------------------------------------------------------------------------
# display warnings
# ---------------------------------------------------------------------------

st.subheader("Confidence flags")
col1, col2, col3 = st.columns(3)
with col1:
    if neff_L is not None and neff_L < 10:
        st.warning(f"⚠️ Shallow MSA — Neff/L = {neff_L:.1f} (< 10). Evolutionary signal is unreliable.")
    elif neff_L is not None:
        st.success(f"✅ MSA depth OK — Neff/L = {neff_L:.1f}")
    else:
        st.info("No MSA uploaded — using fallback model (ESM-2 + BLOSUM + RSA).")
with col2:
    if low_plddt_positions:
        st.warning(f"⚠️ {len(low_plddt_positions)} positions have pLDDT < 70 (unreliable structure).")
    elif pdb_text:
        st.success("✅ All positions have pLDDT ≥ 70.")
with col3:
    model_label = "Full model (evo + entropy + BLOSUM + RSA + ESM-2)" if use_full \
                  else "Fallback model (BLOSUM + RSA + ESM-2)"
    st.info(f"Model: {model_label}")

# ---------------------------------------------------------------------------
# top-20 table
# ---------------------------------------------------------------------------

st.subheader("Top-20 predicted mutations")
top20 = df.nlargest(20, "score")[["mutant", "pos", "wt", "mt",
                                   "score", "esm", "blosum", "rsa", "plddt"]].copy()
top20["low_plddt"] = top20["pos"].isin(low_plddt_positions)
top20 = top20.round(3)
st.dataframe(top20, use_container_width=True)

csv_bytes = top20.to_csv(index=False).encode()
st.download_button("Download top-20 CSV", csv_bytes, "top20_mutations.csv", "text/csv")

# ---------------------------------------------------------------------------
# heatmap
# ---------------------------------------------------------------------------

st.subheader("Position × amino-acid heatmap (model score)")
fig = make_heatmap(df, "score", f"Predicted mutation scores — {uniprot_id}")
st.pyplot(fig)
plt.close(fig)
