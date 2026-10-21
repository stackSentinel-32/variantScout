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
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib
for mod_name in ["src.msa", "src.structure", "src.esm", "src.baseline", "src.data"]:
    if mod_name in sys.modules:
        try:
            importlib.reload(sys.modules[mod_name])
        except Exception:
            pass

from src.msa import AA, frequencies, evo_scores, site_entropy, load_msa, compute_neff
from src.structure import compute_rsa_plddt, STRUCT_DIR, extract_ca_coords, compute_ca_distance
from src.esm import score_sequence, esm_mutation_scores, _load_model, AVAILABLE_MODELS

B62 = substitution_matrices.load("BLOSUM62")

FEATURES_FULL = ["evo", "entropy", "blosum", "rsa", "esm"]
FEATURES_FALLBACK = ["blosum", "rsa", "esm"]

_MODEL_CACHE = ROOT / "data" / "processed" / "baseline_models.pkl"


# ---------------------------------------------------------------------------
# helpers: reference Neff lookup and training set detection
# ---------------------------------------------------------------------------

@st.cache_data
def get_reference_neff_dict():
    """Load reference sequence-reweighted MSA_Neff_L values from ProteinGym."""
    ref_path = ROOT / "data" / "raw" / "DMS_substitutions.csv"
    if not ref_path.exists():
        return {}
    df = pd.read_csv(ref_path)
    mapping = {}
    for _, row in df.iterrows():
        neff_l = row.get("MSA_Neff_L")
        if pd.notna(neff_l):
            val = float(neff_l)
            if pd.notna(row.get("UniProt_ID")):
                mapping[str(row["UniProt_ID"]).strip().upper()] = val
            if pd.notna(row.get("DMS_id")):
                mapping[str(row["DMS_id"]).strip().upper()] = val
    # Add alias mappings for TEM-1
    if "BLAT_ECOLX" in mapping:
        mapping["P62593"] = mapping["BLAT_ECOLX"]
        mapping["TEM-1"] = mapping["BLAT_ECOLX"]
        mapping["TEM1"] = mapping["BLAT_ECOLX"]
        mapping["TEM_1"] = mapping["BLAT_ECOLX"]
    return mapping


@st.cache_data
def get_training_set_uniprots():
    """Return set of UniProt IDs included in the baseline training set."""
    from src.data import load_reference, STARTER_SET
    try:
        ref = load_reference()
        uniprots = set()
        for dms_id in STARTER_SET:
            if dms_id in ref.index:
                u = ref.loc[dms_id].UniProt_ID
                if pd.notna(u):
                    uniprots.add(str(u).strip().upper())
            uniprots.add(dms_id.strip().upper())
        if "BLAT_ECOLX" in uniprots:
            uniprots.add("P62593")
            uniprots.add("TEM-1")
            uniprots.add("TEM1")
            uniprots.add("TEM_1")
        return uniprots
    except Exception:
        return {"BLAT_ECOLX", "P62593", "TEM-1"}



# ---------------------------------------------------------------------------
# helper: train a ridge model on baseline data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_baseline_models():
    """Load pre-fitted Ridge models from disk cache (instant).

    Falls back to full recompute only if the pickle file is missing.
    The cache is created automatically on the first (slow) run.
    """
    # Fast path: load from disk (~0.01 s)
    if _MODEL_CACHE.exists():
        with open(_MODEL_CACHE, "rb") as f:
            return pickle.load(f)

    # Slow path: recompute from MSA + RSA (~3-8 min on first run)
    from src.data import load_reference, STARTER_SET, MSA_DIR
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

    # Save for next time (instant load)
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_CACHE, "wb") as f:
        pickle.dump((ridge_full, ridge_fallback), f)

    return ridge_full, ridge_fallback


# ---------------------------------------------------------------------------
# Canonical demo sequence (TEM-1 beta-lactamase, 286 aa)
# ---------------------------------------------------------------------------

TEM1_DEMO_SEQUENCE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHY"
    "SQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRK"
    "LLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)


# ---------------------------------------------------------------------------
# helpers: parse FASTA, validate sequence, fetch AlphaFold structure
# ---------------------------------------------------------------------------

def parse_fasta(text: str) -> str:
    lines = text.strip().splitlines()
    seq_lines = [l for l in lines if not l.startswith(">")]
    return "".join(seq_lines).upper().replace(" ", "").replace("\t", "").replace("\r", "")


def validate_sequence(seq: str) -> tuple[bool, str]:
    """Validate that input sequence only contains standard 20 amino acids and meets length bounds."""
    if not seq:
        return False, "Please provide a protein sequence."
    invalid_chars = set(seq) - set(AA)
    if invalid_chars:
        bad_chars_str = ", ".join(sorted(invalid_chars))
        return False, (
            f"Invalid character(s) detected in sequence: {bad_chars_str}. "
            "Only the 20 standard amino acids (ACDEFGHIKLMNPQRSTVWY) are accepted."
        )
    if len(seq) < 10:
        return False, f"Sequence length ({len(seq)}) is too short. Minimum length is 10 residues."
    if len(seq) > 1022:
        return False, f"Sequence length ({len(seq)}) exceeds ESM-2 limit of 1022. Truncate or window first."
    return True, ""


def is_valid_feature_cache(parquet_path: Path, seq: str) -> bool:
    """Verify that a cached parquet file matches the input sequence exactly in length and wild-type residues."""
    if not parquet_path.exists():
        return False
    try:
        df_cached = pd.read_parquet(parquet_path)
        if df_cached.empty or "pos" not in df_cached.columns or "wt" not in df_cached.columns:
            return False
        # Reconstruct WT sequence from cached positions
        wt_pos = df_cached.drop_duplicates(subset=["pos"]).sort_values("pos")
        if len(wt_pos) != len(seq):
            return False
        if int(wt_pos["pos"].min()) != 1 or int(wt_pos["pos"].max()) != len(seq):
            return False
        reconstructed = "".join(wt_pos["wt"])
        return reconstructed == seq
    except Exception:
        return False


@st.cache_data
def resolve_uniprot_id(query_id: str):
    """Resolve UniProt entry name and primary accession.

    Supports both Accession (e.g. P62593) and Entry Name (e.g. BLAT_ECOLX).
    Queries UniProt REST API if not found in local aliases.
    """
    clean = query_id.strip()
    if not clean:
        return "", ""
    aliases = {
        "P62593": ("P62593", "BLAT_ECOLX"),
        "BLAT_ECOLX": ("P62593", "BLAT_ECOLX"),
        "TEM-1": ("P62593", "BLAT_ECOLX"),
        "TEM1": ("P62593", "BLAT_ECOLX"),
        "TEM_1": ("P62593", "BLAT_ECOLX"),
    }
    if clean.upper() in aliases:
        return aliases[clean.upper()]

    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{clean}.json", timeout=8)
        if r.status_code == 200:
            data = r.json()
            return data.get("primaryAccession", clean), data.get("uniProtkbId", clean)
    except Exception:
        pass
    return clean, clean


def fetch_alphafold_pdb(uniprot_id):
    """Fetch AlphaFold PDB structure supporting both UniProt Entry Names and Accessions."""
    acc, entry = resolve_uniprot_id(uniprot_id)
    candidates = list(dict.fromkeys([
        uniprot_id,
        entry,
        acc,
        uniprot_id.upper(),
        entry.upper() if entry else "",
        acc.upper() if acc else "",
    ]))
    candidates = [c for c in candidates if c]

    # 1. Check local ProteinGym AF2 cache first
    for candidate in candidates:
        p = STRUCT_DIR / f"{candidate}.pdb"
        if p.exists():
            return p.read_text()

    # 2. Check AlphaFold API for latest structure
    for candidate in [acc, uniprot_id, entry]:
        if not candidate:
            continue
        try:
            r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{candidate}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data and "pdbUrl" in data[0]:
                    pdb_r = requests.get(data[0]["pdbUrl"], timeout=20)
                    if pdb_r.status_code == 200:
                        return pdb_r.text
        except Exception:
            pass
        # 3. Fallback direct model_v4 URL
        try:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{candidate}-F1-model_v4.pdb"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
    return None



def rsa_plddt_from_text(pdb_text):
    """Parse RSA and pLDDT from PDB text (in-memory, no file needed)."""
    return compute_rsa_plddt(pdb_text)


# ---------------------------------------------------------------------------
# score all single mutants for a given sequence
# ---------------------------------------------------------------------------

def score_all_mutations(seq, freq, rsa_dict, plddt_dict, logprobs, tok):
    """Return a DataFrame with every single-mutant score (vectorized)."""
    rows = []
    for pos1 in range(1, len(seq) + 1):
        wt = seq[pos1 - 1]
        if wt not in AA:
            continue
        for mt in AA:
            if mt == wt:
                continue
            rows.append({
                "mutant": f"{wt}{pos1}{mt}",
                "pos": pos1,
                "wt": wt,
                "mt": mt,
                "blosum": B62[wt, mt],
                "rsa": rsa_dict.get(pos1, np.nan),
                "plddt": plddt_dict.get(pos1, np.nan),
            })

    if not rows:
        return pd.DataFrame(columns=["mutant", "pos", "wt", "mt", "blosum", "rsa", "plddt", "esm"])

    df = pd.DataFrame(rows)
    if freq is not None:
        cols = df["pos"].to_numpy() - 1
        df["evo"] = evo_scores(freq, cols, df["wt"], df["mt"])
        df["entropy"] = site_entropy(freq)[cols]

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


def make_interactive_heatmap(df, seq, score_col, title, has_evo=True):
    """Generate an interactive, full-width Altair heatmap with WT indicators and tooltips."""
    aa_list = list(AA)
    L = len(seq)

    lookup = {}
    for _, row in df.iterrows():
        lookup[(int(row["pos"]), row["mt"])] = row

    heatmap_rows = []
    for p in range(1, L + 1):
        wt = seq[p - 1]
        for mt in aa_list:
            is_wt = (wt == mt)
            item = {
                "pos": p,
                "wt": wt,
                "mt": mt,
                "is_wt": is_wt,
            }
            if is_wt:
                item["mutant"] = f"{wt}{p} (WT)"
                item["score"] = None
                item["esm"] = None
                item["rsa"] = None
                item["evo"] = None
            else:
                row_val = lookup.get((p, mt))
                if row_val is not None:
                    item["mutant"] = f"{wt}{p}{mt}"
                    item["score"] = float(row_val.get(score_col, 0.0))
                    item["esm"] = float(row_val["esm"]) if pd.notna(row_val.get("esm")) else None
                    item["rsa"] = float(row_val["rsa"]) if pd.notna(row_val.get("rsa")) else None
                    item["evo"] = float(row_val["evo"]) if has_evo and pd.notna(row_val.get("evo")) else None
                else:
                    item["mutant"] = f"{wt}{p}{mt}"
                    item["score"] = None
                    item["esm"] = None
                    item["rsa"] = None
                    item["evo"] = None
            heatmap_rows.append(item)

    h_df = pd.DataFrame(heatmap_rows)

    tooltips = [
        alt.Tooltip("mutant:N", title="Substitution"),
        alt.Tooltip("pos:O", title="Residue Position"),
        alt.Tooltip("wt:N", title="Wild-Type AA"),
        alt.Tooltip("mt:N", title="Target AA"),
        alt.Tooltip("score:Q", title="Model Score", format=".3f"),
        alt.Tooltip("esm:Q", title="ESM-2 Likelihood", format=".3f"),
        alt.Tooltip("rsa:Q", title="Surface RSA", format=".3f"),
    ]
    if has_evo:
        tooltips.append(alt.Tooltip("evo:Q", title="Evolutionary Score", format=".3f"))

    step_val = max(10, min(50, L // 10))
    axis_ticks = list(range(1, L + 1, step_val))
    if L not in axis_ticks:
        axis_ticks.append(L)

    base = (
        alt.Chart(h_df[~h_df["is_wt"]])
        .mark_rect()
        .encode(
            x=alt.X("pos:O", title="Residue Position", axis=alt.Axis(values=axis_ticks, labelAngle=0)),
            y=alt.Y("mt:N", sort=aa_list, title="Target Amino Acid"),
            color=alt.Color("score:Q", scale=alt.Scale(scheme="viridis"), title="Fitness Score"),
            tooltip=tooltips,
        )
    )

    wt_rect = (
        alt.Chart(h_df[h_df["is_wt"]])
        .mark_rect(color="#2B2D42")
        .encode(
            x=alt.X("pos:O"),
            y=alt.Y("mt:N", sort=aa_list),
            tooltip=[
                alt.Tooltip("mutant:N", title="Wild-Type Residue"),
                alt.Tooltip("pos:O", title="Position"),
            ],
        )
    )
    wt_dot = (
        alt.Chart(h_df[h_df["is_wt"]])
        .mark_text(text="•", color="white", fontSize=13, baseline="middle")
        .encode(
            x=alt.X("pos:O"),
            y=alt.Y("mt:N", sort=aa_list),
        )
    )

    chart = (base + wt_rect + wt_dot).properties(
        width="container",
        height=320,
        title=title,
    )
    return chart



# ---------------------------------------------------------------------------
# helper functions: confidence tiers, diversity cap, feature attribution
# ---------------------------------------------------------------------------

def assign_confidence_tiers(df, score_col):
    """Group candidates into 3 confidence tiers based on predicted score percentile:
    🟢 Tier 1: High Confidence (Top 1%)
    🟡 Tier 2: Moderate Confidence (Top 1% - 5%)
    ⚪ Tier 3: Tolerated / Low Confidence (Top 5% - 15%)
    """
    pct = df[score_col].rank(pct=True)
    tiers = []
    for p in pct:
        if p >= 0.99:
            tiers.append("🟢 Tier 1 (High)")
        elif p >= 0.95:
            tiers.append("🟡 Tier 2 (Moderate)")
        elif p >= 0.85:
            tiers.append("⚪ Tier 3 (Tolerated)")
        else:
            tiers.append("Low (< 85%)")
    return tiers, pct


def apply_diversity_cap(df_sorted, max_per_pos=2, top_n=20):
    """Cap recommended top variants at max_per_pos mutations per residue position."""
    pos_counts = {}
    picked_indices = []
    for idx, row in df_sorted.iterrows():
        p = row["pos"]
        if pos_counts.get(p, 0) < max_per_pos:
            picked_indices.append(idx)
            pos_counts[p] = pos_counts.get(p, 0) + 1
            if len(picked_indices) >= top_n:
                break
    return df_sorted.loc[picked_indices].copy()


def make_feature_attribution_chart(model, feats, feat_row, mutant_name):
    """Render an Altair horizontal bar chart of feature contributions for a mutation."""
    feature_labels = {
        "rsa": "RSA (Surface Exposure)",
        "evo": "Evolutionary Conservation",
        "blosum": "BLOSUM62 Prior",
        "esm": "ESM-2 Likelihood",
        "entropy": "Site Variability (Entropy)",
    }
    rows = []
    for feat_name, coef in zip(feats, model.coef_):
        pct_val = feat_row.get(feat_name, 0.5)
        contrib = coef * pct_val
        rows.append({
            "Signal": feature_labels.get(feat_name, feat_name),
            "Contribution": round(contrib, 4),
            "Weight": round(coef, 4),
            "Percentile": f"{pct_val * 100:.1f}%",
        })
    chart_df = pd.DataFrame(rows)
    total_pos = chart_df["Contribution"].clip(lower=0).sum()
    if total_pos > 0:
        chart_df["Relative %"] = chart_df["Contribution"].apply(lambda v: f"{max(0, v) / total_pos * 100:.1f}%")
    else:
        chart_df["Relative %"] = "N/A"

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            x=alt.X("Contribution:Q", title="Signal Contribution (Model Weight × Rank-Percentile)"),
            y=alt.Y("Signal:N", sort="-x", title=""),
            color=alt.Color("Signal:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=["Signal", "Contribution", "Weight", "Percentile", "Relative %"],
        )
        .properties(
            title=f"Feature Attribution for {mutant_name}",
            height=max(180, len(feats) * 35),
        )
    )
    return chart, chart_df


def format_candidates_table(candidate_df, score_col, tier_col, feat_cols, max_per_pos=2, top_n=20):
    """Format and sort top candidate recommendations with diversity cap and confidence tiers."""
    df_sorted = candidate_df.sort_values(by=score_col, ascending=False)
    picked = apply_diversity_cap(df_sorted, max_per_pos=max_per_pos, top_n=top_n)

    display_cols = ["mutant", "pos", "wt", "mt", score_col] + feat_cols
    for extra in ["plddt", "low_plddt"]:
        if extra in picked.columns:
            display_cols.append(extra)
    out = picked[display_cols].copy()
    out.insert(0, "Tier", picked[tier_col])
    out = out.rename(columns={score_col: "score"})

    float_cols = [c for c in out.columns if out[c].dtype in ("float64", "float32")]
    out[float_cols] = out[float_cols].round(3)
    return out


def render_3d_structure(
    pdb_text: str,
    top_positions: list = None,
    catalytic_positions: list = None,
    custom_positions: list = None,
    color_mode: str = "Highlight Candidates & Catalytic Sites",
    rep_style: str = "Cartoon",
    bg_hex: str = "#0F172A",
    plddt_dict: dict = None,
    mean_scores_dict: dict = None,
    rsa_dict: dict = None,
    spin: bool = False,
    height: int = 540,
) -> str:
    """Generate 3Dmol.js HTML viewer for the AlphaFold structure."""
    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb_text, "pdb")
    view.setBackgroundColor(bg_hex)

    top_positions = top_positions or []
    catalytic_positions = catalytic_positions or []
    custom_positions = custom_positions or []
    plddt_dict = plddt_dict or {}
    mean_scores_dict = mean_scores_dict or {}
    rsa_dict = rsa_dict or {}

    # Base styling per representation
    if rep_style == "Surface (Semi-transparent)":
        view.setStyle({}, {"cartoon": {"color": "#94A3B8"}})
        view.addSurface(py3Dmol.VDW, {"opacity": 0.65, "color": "#CBD5E1"})
    elif rep_style == "Cartoon + Sidechains":
        view.setStyle({}, {"cartoon": {"color": "#94A3B8"}, "stick": {"colorscheme": "lightGreyCarbon", "radius": 0.15}})
    else:  # Cartoon
        view.setStyle({}, {"cartoon": {"color": "#94A3B8"}})

    # Apply color modes
    if color_mode == "AlphaFold pLDDT Confidence":
        for pos, plddt in plddt_dict.items():
            if plddt >= 90:
                col = "#0053D6"  # Very high (deep blue)
            elif plddt >= 70:
                col = "#65CBF3"  # Confident (cyan)
            elif plddt >= 50:
                col = "#FFDB13"  # Low (yellow)
            else:
                col = "#FF7D45"  # Very low (orange)
            view.setStyle({"resi": pos}, {"cartoon": {"color": col}})

    elif color_mode == "Mean Predicted Tolerance Score" and mean_scores_dict:
        scores = list(mean_scores_dict.values())
        min_s, max_s = min(scores), max(scores)
        rng = max_s - min_s if max_s > min_s else 1.0
        cmap = plt.get_cmap("coolwarm")
        for pos, val in mean_scores_dict.items():
            norm_val = (val - min_s) / rng
            hex_col = mcolors.to_hex(cmap(norm_val))
            view.setStyle({"resi": pos}, {"cartoon": {"color": hex_col}})

    elif color_mode == "Surface Accessibility (RSA)" and rsa_dict:
        for pos, rsa in rsa_dict.items():
            if rsa < 0.20:
                col = "#1E3A8A"  # Buried (dark blue)
            elif rsa < 0.50:
                col = "#10B981"  # Intermediate (emerald)
            else:
                col = "#F59E0B"  # Exposed (amber)
            view.setStyle({"resi": pos}, {"cartoon": {"color": col}})

    # Highlight Top-10 Recommended Candidates (Hot Pink spheres + sticks)
    if top_positions:
        view.addStyle(
            {"resi": top_positions},
            {
                "sphere": {"color": "#EC4899", "radius": 0.95},
                "stick": {"colorscheme": "magentaCarbon", "radius": 0.25},
            },
        )

    # Highlight Catalytic / Active Site residues (Bright Yellow spheres + sticks)
    if catalytic_positions:
        view.addStyle(
            {"resi": catalytic_positions},
            {
                "sphere": {"color": "#FACC15", "radius": 0.85},
                "stick": {"colorscheme": "yellowCarbon", "radius": 0.25},
            },
        )

    # Highlight Custom Positions (Cyan spheres + sticks)
    if custom_positions:
        view.addStyle(
            {"resi": custom_positions},
            {
                "sphere": {"color": "#06B6D4", "radius": 0.90},
                "stick": {"colorscheme": "cyanCarbon", "radius": 0.25},
            },
        )

    if spin:
        view.spin(True)
    else:
        view.spin(False)

    view.setHoverable(
        {},
        True,
        """function(atom,viewer,event,container) {
            if(!atom.label) {
                atom.label = viewer.addLabel(atom.resn + atom.resi + ":" + atom.atom, {
                    position: atom,
                    backgroundColor: 'rgba(15,23,42,0.85)',
                    fontColor: 'white',
                    fontSize: 12
                });
            }
        }""",
        """function(atom,viewer) {
            if(atom.label) {
                viewer.removeLabel(atom.label);
                delete atom.label;
            }
        }""",
    )

    view.zoomTo()
    return view._make_html()


def generate_combinatorial_mutants(
    single_df: pd.DataFrame,
    score_col: str,
    ca_coords: dict = None,
    pool_size: int = 12,
    min_dist_filter: float = 5.0,
    only_tolerant: bool = True,
) -> pd.DataFrame:
    """Generate and rank double mutants using the additive approximation.

    Pairs top single mutants from distinct positions, computes additive scores,
    and calculates 3D C-alpha physical distance (Å) to flag close contacts.
    """
    df_pool = single_df.sort_values(by=score_col, ascending=False)
    if only_tolerant and "Tier" in df_pool.columns:
        df_pool = df_pool[df_pool["Tier"].str.contains("Tier 1|Tier 2", regex=True)]

    df_pool = df_pool.head(pool_size)
    singles = df_pool.to_dict(orient="records")
    pairs = []
    ca_coords = ca_coords or {}

    for m1, m2 in itertools.combinations(singles, 2):
        if m1["pos"] == m2["pos"]:
            continue
        if m1["pos"] > m2["pos"]:
            m1, m2 = m2, m1

        p1, p2 = int(m1["pos"]), int(m2["pos"])
        dist = None
        if p1 in ca_coords and p2 in ca_coords:
            dist = float(np.linalg.norm(ca_coords[p1] - ca_coords[p2]))

        comb_score = float(m1[score_col]) + float(m2[score_col])
        comb_esm = (
            float(m1["esm"]) + float(m2["esm"])
            if ("esm" in m1 and pd.notna(m1.get("esm")) and "esm" in m2 and pd.notna(m2.get("esm")))
            else None
        )
        comb_evo = (
            float(m1["evo"]) + float(m2["evo"])
            if ("evo" in m1 and pd.notna(m1.get("evo")) and "evo" in m2 and pd.notna(m2.get("evo")))
            else None
        )

        is_contact = bool(dist < 5.0) if dist is not None else False
        status_label = "⚠️ Contact (< 5Å)" if is_contact else ("✅ Distant (≥ 5Å)" if dist is not None else "Unknown (No PDB)")

        pairs.append({
            "pair": f"{m1['mutant']} + {m2['mutant']}",
            "pos1": p1,
            "mut1": m1["mutant"],
            "score1": round(float(m1[score_col]), 3),
            "pos2": p2,
            "mut2": m2["mutant"],
            "score2": round(float(m2[score_col]), 3),
            "score_double": round(comb_score, 3),
            "ca_dist": round(dist, 1) if dist is not None else None,
            "contact_risk": is_contact,
            "status": status_label,
            "esm_sum": round(comb_esm, 3) if comb_esm is not None else None,
            "evo_sum": round(comb_evo, 3) if comb_evo is not None else None,
        })

    out_df = pd.DataFrame(pairs)
    if out_df.empty:
        return out_df

    out_df = out_df.sort_values(by="score_double", ascending=False).reset_index(drop=True)

    pct = out_df["score_double"].rank(pct=True)
    tiers = []
    for p in pct:
        if p >= 0.90:
            tiers.append("🟢 Tier 1 (High)")
        elif p >= 0.70:
            tiers.append("🟡 Tier 2 (Moderate)")
        else:
            tiers.append("⚪ Tier 3 (Tolerated)")
    out_df.insert(0, "Tier", tiers)

    return out_df


def render_3d_tab(pdb_text, clean_id, entry_name, is_tem1, top_rec_df, df, seq, plddt_dict, rsa_dict, mean_score_col):
    st.subheader(f"🏛️ Interactive 3D Protein Structure — {entry_name or clean_id}")
    st.caption("AlphaFold 3D structure rendered in real-time with py3Dmol. Visualize beneficial mutations, catalytic residues, and structural confidence.")

    if not pdb_text:
        st.warning("⚠️ No AlphaFold 3D structure available for this protein. Please provide a valid UniProt ID in the sidebar.")
        return

    c_mode, c_rep, c_bg, c_spin = st.columns([3, 2, 2, 1])
    with c_mode:
        color_mode = st.selectbox(
            "Color Scheme",
            [
                "Highlight Candidates & Catalytic Sites",
                "AlphaFold pLDDT Confidence",
                "Mean Predicted Tolerance Score",
                "Surface Accessibility (RSA)",
            ],
            key="viewer_color_mode",
        )
    with c_rep:
        rep_style = st.selectbox(
            "Representation",
            ["Cartoon", "Cartoon + Sidechains", "Surface (Semi-transparent)"],
            key="viewer_rep_style",
        )
    with c_bg:
        bg_choice = st.selectbox(
            "Background",
            ["Dark Slate (#0F172A)", "Deep Charcoal (#1E1E2F)", "Clean White (#FFFFFF)"],
            key="viewer_bg",
        )
        bg_hex = bg_choice.split("(")[-1].replace(")", "").strip()
    with c_spin:
        st.write("")
        st.write("")
        spin_toggle = st.checkbox("Spin", value=False, key="viewer_spin")

    col_cust, col_cat = st.columns([2, 2])
    with col_cust:
        custom_input = st.text_input(
            "Custom residue positions to highlight (cyan)",
            value="",
            placeholder="e.g. 104, 238, 240",
            help="Enter comma-separated residue numbers (1-based sequential) to highlight with cyan spheres/sticks.",
            key="viewer_custom_input",
        )
    with col_cat:
        default_cat = "68, 71, 164, 232" if is_tem1 else ""
        cat_input = st.text_input(
            "Catalytic / Active Site positions (yellow)",
            value=default_cat,
            placeholder="e.g. 68, 71, 164",
            help="Active site positions to highlight in bright yellow sticks/spheres.",
            key="viewer_cat_input",
        )

    def parse_pos_str(s):
        out = []
        for token in s.replace(";", ",").split(","):
            token = token.strip()
            if token.isdigit():
                out.append(int(token))
        return out

    custom_pos_list = parse_pos_str(custom_input)
    catalytic_pos_list = parse_pos_str(cat_input)

    top_candidate_pos = list(dict.fromkeys(top_rec_df["pos"].tolist()))[:10]
    mean_scores_dict = df.groupby("pos")[mean_score_col].mean().to_dict()

    html_3d = render_3d_structure(
        pdb_text=pdb_text,
        top_positions=top_candidate_pos,
        catalytic_positions=catalytic_pos_list,
        custom_positions=custom_pos_list,
        color_mode=color_mode,
        rep_style=rep_style,
        bg_hex=bg_hex,
        plddt_dict=plddt_dict,
        mean_scores_dict=mean_scores_dict,
        rsa_dict=rsa_dict,
        spin=spin_toggle,
        height=540,
    )
    components.html(html_3d, height=560)

    st.markdown("##### 🎨 Structural Color Legend")
    if color_mode == "Highlight Candidates & Catalytic Sites":
        l1, l2, l3, l4 = st.columns(4)
        l1.markdown("🌸 **Hot Pink**: Top-10 Recommended Sites")
        l2.markdown("⭐ **Bright Yellow**: Catalytic / Active Sites")
        l3.markdown("🔷 **Cyan**: Custom Highlighted Residues")
        l4.markdown("🔘 **Slate Grey**: Neutral WT Backbone")
    elif color_mode == "AlphaFold pLDDT Confidence":
        l1, l2, l3, l4 = st.columns(4)
        l1.markdown("🟦 **Deep Blue**: Very High Confidence (pLDDT > 90)")
        l2.markdown("🩵 **Cyan**: Confident (90 ≥ pLDDT > 70)")
        l3.markdown("🟨 **Yellow**: Low Confidence (70 ≥ pLDDT > 50)")
        l4.markdown("🟧 **Orange**: Very Low / Floppy (pLDDT ≤ 50)")
    elif color_mode == "Mean Predicted Tolerance Score":
        st.caption("Mapped along coolwarm gradient: **Deep Blue** = Strictly Conserved / Low Tolerance; **White** = Neutral; **Deep Red** = Highly Plastic / High Tolerance.")
    elif color_mode == "Surface Accessibility (RSA)":
        l1, l2, l3 = st.columns(3)
        l1.markdown("🟦 **Navy**: Buried Core (RSA < 0.20)")
        l2.markdown("🟩 **Emerald**: Partially Exposed (0.20 ≤ RSA < 0.50)")
        l3.markdown("🟧 **Amber**: Highly Surface Exposed (RSA ≥ 0.50)")

    with st.expander("🔍 Inspect Structural Properties of Top Highlighted Residues"):
        summary_rows = []
        for p in top_candidate_pos:
            wt_aa = seq[p - 1] if p <= len(seq) else "?"
            matching_muts = df[df["pos"] == p].sort_values(by=mean_score_col, ascending=False)
            best_mut_name = matching_muts.iloc[0]["mutant"] if not matching_muts.empty else "N/A"
            best_mut_score = matching_muts.iloc[0][mean_score_col] if not matching_muts.empty else np.nan
            summary_rows.append({
                "Position": p,
                "Wild-Type": wt_aa,
                "Top Candidate": best_mut_name,
                "Score": round(float(best_mut_score), 3) if pd.notna(best_mut_score) else np.nan,
                "RSA": round(float(rsa_dict.get(p, np.nan)), 3),
                "pLDDT": round(float(plddt_dict.get(p, np.nan)), 1),
                "AlphaFold Status": "Confident" if plddt_dict.get(p, 0) >= 70 else "Disordered / Floppy",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


def render_combinatorial_tab(candidate_df, target_score_col, ca_coords, clean_id, entry_name):
    st.subheader(f"🧪 Combinatorial (Double) Mutants — {entry_name or clean_id}")
    st.caption(
        "Combines top-scoring single mutations using the additive approximation "
        "(sum of model scores + independent site conservation and ESM log-odds). "
        "Filters or flags adjacent residues within 5Å to avoid steric clashes and negative epistasis."
    )

    st.info(
        "💡 **Additive Approximation Principle**: Beneficial mutations at spatially separated sites "
        "(Cα distance ≥ 5.0 Å) tend to act independently, making their combined fitness effects largely additive. "
        "Mutations in direct physical contact (< 5.0 Å) carry high risk of packing conflicts, backbone clashes, "
        "or deleterious non-linear epistatic interactions."
    )

    col_pool, col_dist, col_tier = st.columns([2, 2, 2])
    with col_pool:
        pool_size = st.slider(
            "Candidate Pool (Top Singles)",
            min_value=6,
            max_value=25,
            value=12,
            help="Number of top single-mutant candidates to combine pairwise.",
            key="comb_pool_size",
        )
    with col_dist:
        min_dist_filter = st.slider(
            "Min Cα Distance Filter (Å)",
            min_value=0.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            help="Exclude double mutants whose Cα atoms are closer than this threshold to protect against steric clashes. Set to 0 to view all pairs.",
            key="comb_dist_filter",
        )
    with col_tier:
        only_tolerant = st.checkbox(
            "Only include Tier 1 & 2 singles",
            value=True,
            help="Restricts combination candidates strictly to High and Moderate confidence single variants.",
            key="comb_only_tolerant",
        )

    df_doubles = generate_combinatorial_mutants(
        single_df=candidate_df,
        score_col=target_score_col,
        ca_coords=ca_coords,
        pool_size=pool_size,
        min_dist_filter=min_dist_filter,
        only_tolerant=only_tolerant,
    )

    if df_doubles.empty:
        st.warning("No combinatorial double mutants found with the selected criteria.")
        return

    # Filter by distance cutoff if specified
    if min_dist_filter > 0.0 and "ca_dist" in df_doubles.columns:
        filtered_doubles = df_doubles[
            df_doubles["ca_dist"].isna() | (df_doubles["ca_dist"] >= min_dist_filter)
        ].copy()
    else:
        filtered_doubles = df_doubles.copy()

    m1, m2, m3, m4 = st.columns(4)
    total_gen = len(df_doubles)
    passed_filter = len(filtered_doubles)
    clashes_flagged = total_gen - passed_filter
    avg_dist = filtered_doubles["ca_dist"].dropna().mean() if not filtered_doubles.empty else 0

    m1.metric("Pairs Generated", f"{total_gen}")
    m2.metric(f"Eligible Distant Pairs (≥ {min_dist_filter}Å)", f"{passed_filter}", f"-{clashes_flagged} close contact" if clashes_flagged else "0 filtered")
    m3.metric("Mean Cα Distance", f"{avg_dist:.1f} Å" if avg_dist else "N/A")
    if not filtered_doubles.empty:
        m4.metric("Top Double Candidate", f"{filtered_doubles.iloc[0]['pair']}")

    st.divider()

    # Altair Scatter: Distance vs Additive Fitness Score
    if "ca_dist" in df_doubles.columns and df_doubles["ca_dist"].notna().any():
        st.markdown("#### 🗺️ Spatial Distance vs Additive Fitness Landscape")
        st.caption("Dotted red vertical line indicates the distance filter threshold. Green points represent safe, non-contact additive pairings.")

        scatter_chart = (
            alt.Chart(df_doubles)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=alt.X("ca_dist:Q", title="Cα–Cα Distance (Å)", scale=alt.Scale(zero=False)),
                y=alt.Y("score_double:Q", title="Additive Predicted Score", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "status:N",
                    title="Spatial Status",
                    scale=alt.Scale(domain=["✅ Distant (≥ 5Å)", "⚠️ Contact (< 5Å)"], range=["#10B981", "#EF4444"]),
                ),
                tooltip=[
                    alt.Tooltip("pair:N", title="Double Mutant"),
                    alt.Tooltip("Tier:N", title="Confidence Tier"),
                    alt.Tooltip("score_double:Q", title="Additive Score", format=".3f"),
                    alt.Tooltip("ca_dist:Q", title="Cα Distance (Å)", format=".1f"),
                    alt.Tooltip("mut1:N", title="Mutation 1"),
                    alt.Tooltip("mut2:N", title="Mutation 2"),
                ],
            )
        )

        rule = (
            alt.Chart(pd.DataFrame({"x": [min_dist_filter]}))
            .mark_rule(color="#EF4444", strokeDash=[4, 4], strokeWidth=2)
            .encode(x="x:Q")
        )

        st.altair_chart((scatter_chart + rule).properties(width="container", height=320), use_container_width=True)

    st.markdown(f"#### 🏆 Top Recommended Double Mutants (Filter: Cα Distance ≥ {min_dist_filter} Å)")
    display_doubles = filtered_doubles.head(30)
    st.dataframe(display_doubles, use_container_width=True)

    csv_doubles = filtered_doubles.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Combinatorial Double Mutants CSV",
        csv_doubles,
        f"{clean_id}_combinatorial_double_mutants.csv",
        "text/csv",
        use_container_width=True,
    )



# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Mutation Picker", layout="wide")
    st.title("🧬 Mutation Picker")
    st.caption("Ranks every single amino-acid substitution by predicted fitness impact.")

    # sidebar inputs
    with st.sidebar:
        st.header("Input")

        # Initialize session state defaults
        if "fasta_input_val" not in st.session_state:
            st.session_state["fasta_input_val"] = TEM1_DEMO_SEQUENCE
        if "uniprot_id_val" not in st.session_state:
            st.session_state["uniprot_id_val"] = "P62593"
        if "use_demo_msa_val" not in st.session_state:
            st.session_state["use_demo_msa_val"] = True

        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            if st.button("🧪 Load TEM-1 Demo", use_container_width=True, help="Load canonical 286-aa TEM-1 sequence & UniProt P62593"):
                st.session_state["fasta_input_val"] = TEM1_DEMO_SEQUENCE
                st.session_state["uniprot_id_val"] = "P62593"
                st.session_state["use_demo_msa_val"] = True
                st.rerun()
        with col_demo2:
            if st.button("🔄 Clear", use_container_width=True, help="Clear input fields"):
                st.session_state["fasta_input_val"] = ""
                st.session_state["uniprot_id_val"] = ""
                st.session_state["use_demo_msa_val"] = False
                st.rerun()

        fasta_text = st.text_area(
            "Protein sequence (FASTA or raw)",
            value=st.session_state["fasta_input_val"],
            height=150,
            placeholder=">MyProtein\nMKTAYIAKQRQ...",
            help="Standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY). Sequential numbering (1-indexed).",
        )
        st.session_state["fasta_input_val"] = fasta_text

        uniprot_id = st.text_input(
            "UniProt ID (for AlphaFold structure)",
            value=st.session_state["uniprot_id_val"],
            placeholder="e.g. P62593 or BLAT_ECOLX",
            help="UniProt Accession or Entry Name used to query AlphaFold 3D coordinates.",
        )
        st.session_state["uniprot_id_val"] = uniprot_id

        msa_file = st.file_uploader("MSA file (.a2m, optional)", type=["a2m", "fasta", "fa"])

        # Check for bundled demo MSA
        tem1_msa_path = ROOT / "data" / "raw" / "msa" / "DMS_msa_files" / "BLAT_ECOLX_full_11-26-2021_b02.a2m"
        clean_upper = uniprot_id.strip().upper()
        is_tem1_demo = clean_upper in ["P62593", "BLAT_ECOLX", "TEM-1", "TEM1", "TEM_1"]
        use_demo_msa = False
        if is_tem1_demo and tem1_msa_path.exists() and msa_file is None:
            use_demo_msa = st.checkbox(
                "Use bundled TEM-1 reference MSA (286 cols, Neff/L = 100.8)",
                value=st.session_state.get("use_demo_msa_val", True),
                help="Automatically loads the high-depth ProteinGym reference MSA for TEM-1 beta-lactamase.",
            )
            st.session_state["use_demo_msa_val"] = use_demo_msa

        st.header("Filters & Options")
        exclude_disordered = st.checkbox(
            "Exclude disordered positions (pLDDT < 70)",
            value=True,
            help="Excludes floppy/disordered positions with low AlphaFold confidence (pLDDT < 70) from recommendations, such as signal peptides, which can get artificially high RSA.",
        )
        max_per_pos = st.slider(
            "Max mutations per position",
            min_value=1,
            max_value=5,
            value=2,
            help="Caps mutations per residue position in recommendations table so lab users get library diversity across distinct positions.",
        )

        with st.expander("⚙️ Advanced Settings"):
            esm_model_choice = st.selectbox(
                "ESM-2 Model",
                [
                    "esm2_t12_35M_UR50D (Fast, ~35M)",
                    "esm2_t33_650M_UR50D (High Accuracy, ~650M)",
                ],
                index=0,
                help=(
                    "Choose ESM-2 model scale. "
                    "35M is fast and lightweight (recommended for quick iterations). "
                    "650M provides higher zero-shot accuracy, recommended for shallow MSAs or GPU systems."
                ),
            )

        run_btn = st.button("Score mutations", type="primary")

    if not run_btn:
        st.info("Fill in the sequence and UniProt ID in the sidebar, then click **Score mutations**.")
        st.stop()

    # parse and validate inputs
    if not fasta_text.strip():
        st.error("Please provide a protein sequence.")
        st.stop()

    seq = parse_fasta(fasta_text)
    is_valid, err_msg = validate_sequence(seq)
    if not is_valid:
        st.error(f"❌ {err_msg}")
        st.stop()

    if not uniprot_id.strip():
        st.error("UniProt ID is required to fetch the AlphaFold structure.")
        st.stop()

    clean_id = uniprot_id.strip()
    primary_acc, entry_name = resolve_uniprot_id(clean_id)
    selected_esm_model = (
        AVAILABLE_MODELS["650M"] if "650M" in esm_model_choice else AVAILABLE_MODELS["35M"]
    )
    model_tag = "650M" if "650M" in selected_esm_model else "35M"

    with st.spinner("Loading models (one-time, ~10 s) …"):
        ridge_full, ridge_fallback = load_baseline_models()

    # ---------------------------------------------------------------------------
    # multi-step scoring pipeline with progress bar
    # ---------------------------------------------------------------------------

    progress_bar = st.progress(0, text="Initializing mutation scoring pipeline …")

    FEAT_CACHE_DIR = ROOT / "data" / "processed" / "features"
    FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_key = entry_name if entry_name else clean_id
    parquet_file = FEAT_CACHE_DIR / f"{parquet_key}_{model_tag}.parquet"

    # Step 1/4: AlphaFold structure
    progress_bar.progress(25, text="Step 1/4: Fetching AlphaFold structure & computing RSA/pLDDT …")
    pdb_text = fetch_alphafold_pdb(clean_id)

    if pdb_text is None:
        st.warning(f"Could not fetch AlphaFold structure for '{uniprot_id}'. RSA and pLDDT will be missing.")
        rsa_dict, plddt_dict = {}, {}
        ca_coords = {}
    else:
        rsa_dict, plddt_dict = rsa_plddt_from_text(pdb_text)
        ca_coords = extract_ca_coords(pdb_text)

    low_plddt_positions = {p for p, v in plddt_dict.items() if v < 70}

    # Step 2/4: MSA & conservation
    progress_bar.progress(50, text="Step 2/4: Processing MSA & computing conservation frequencies …")
    freq = None
    neff_L = None
    is_verified_depth = False
    msa_mat = None
    if msa_file is not None:
        msa_bytes = msa_file.read().decode()
        tmp_path = ROOT / "data" / "processed" / "user_msa_tmp.a2m"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(msa_bytes)
        _, msa_mat = load_msa(tmp_path)
    elif use_demo_msa and tem1_msa_path.exists():
        _, msa_mat = load_msa(tem1_msa_path)

    if msa_mat is not None:
        freq = frequencies(msa_mat)
        ref_map = get_reference_neff_dict()
        ref_match = None
        for cand in [clean_id.upper(), entry_name.upper(), primary_acc.upper()]:
            if cand in ref_map:
                ref_match = ref_map[cand]
                break
        if ref_match is not None:
            neff_L = ref_match
            is_verified_depth = True
        else:
            neff = compute_neff(msa_mat, theta=0.8)
            neff_L = neff / len(seq)
            is_verified_depth = False

    # Check if pre-computed feature parquet exists and matches current sequence
    valid_cache = is_valid_feature_cache(parquet_file, seq)

    if valid_cache:
        progress_bar.progress(85, text=f"Loading features from cache ({parquet_file.name}) …")
        df = pd.read_parquet(parquet_file)
        if freq is not None and "evo" not in df.columns:
            cols = df["pos"].to_numpy() - 1
            df["evo"] = evo_scores(freq, cols, df["wt"], df["mt"])
            df["entropy"] = site_entropy(freq)[cols]
            df.to_parquet(parquet_file, index=False)
    else:
        # Step 3/4: ESM-2
        progress_bar.progress(75, text=f"Step 3/4: Generating ESM-2 ({model_tag}) zero-shot log-likelihoods …")
        tok, _ = _load_model(selected_esm_model)
        esm_cache_key = entry_name if entry_name else clean_id
        logprobs = score_sequence(seq, esm_cache_key, model_name=selected_esm_model)

        # Step 4/4: Scoring & Ranking
        progress_bar.progress(90, text="Step 4/4: Evaluating ensemble weights & ranking variants …")
        df = score_all_mutations(seq, freq, rsa_dict, plddt_dict, logprobs, tok)
        # Cache computed features to parquet for instant future runs
        df.to_parquet(parquet_file, index=False)

    if "plddt" in df.columns:
        df["low_plddt"] = df["plddt"].apply(lambda v: "⚠️ < 70" if pd.notna(v) and v < 70 else "✅ ≥ 70")
    else:
        df["low_plddt"] = "N/A"


    # Fallback model scores (always computed using BLOSUM + RSA + ESM)
    feat_fallback = df[FEATURES_FALLBACK].copy()
    for c in FEATURES_FALLBACK:
        feat_fallback[c] = feat_fallback[c].rank(pct=True)
    df["score_fallback"] = ridge_fallback.predict(feat_fallback.values)
    df["tier_fallback"], df["pct_fallback"] = assign_confidence_tiers(df, "score_fallback")
    df["rank_fallback"] = df["score_fallback"].rank(ascending=False, method="min").astype(int)

    # Full model scores (computed when MSA is uploaded)
    has_full = freq is not None
    if has_full:
        feat_full = df[FEATURES_FULL].copy()
        for c in FEATURES_FULL:
            feat_full[c] = feat_full[c].rank(pct=True)
        df["score_full"] = ridge_full.predict(feat_full.values)
        df["tier_full"], df["pct_full"] = assign_confidence_tiers(df, "score_full")
        df["rank_full"] = df["score_full"].rank(ascending=False, method="min").astype(int)
        df["rank_shift"] = df["rank_fallback"] - df["rank_full"]
        df["score"] = df["score_full"]
    else:
        df["score"] = df["score_fallback"]

    progress_bar.progress(100, text="✅ Scoring complete!")
    time.sleep(0.3)
    progress_bar.empty()

    # ---------------------------------------------------------------------------
    # notices & warnings
    # ---------------------------------------------------------------------------

    # TEM-1 Ambler numbering warning
    is_tem1 = any(c in ["BLAT_ECOLX", "P62593", "TEM-1", "TEM1", "TEM_1"] for c in [clean_id.upper(), entry_name.upper(), primary_acc.upper()])
    if is_tem1:
        st.warning(
            "⚠️ **Numbering Notice**: Predictions use sequential numbering (1–286). "
            "TEM-1 literature commonly uses **Ambler numbering** where **Ambler Pos = App Pos + 2** "
            "(e.g. App M180T is Ambler M182T, App S68A is Ambler S70A)."
        )

    # In-sample training alert
    training_uniprots = get_training_set_uniprots()
    if any(c in training_uniprots for c in [clean_id.upper(), entry_name.upper(), primary_acc.upper()]):
        st.info(
            "ℹ️ **In-sample demonstration**: This protein was part of the training set. "
            "For strict zero-shot evaluation, test on a held-out protein (e.g. UBC9_HUMAN)."
        )

    # Biological Interpretation Warning
    st.info(
        "ℹ️ **Biological Interpretation**: High scores indicate substitutions that are evolutionary tolerated "
        "and structurally benign (the protein is likely to fold and retain function). "
        "High scores **do not guarantee improved catalytic activity or binding affinity** over wild-type."
    )

    # ---------------------------------------------------------------------------
    # display confidence flags
    # ---------------------------------------------------------------------------

    st.subheader("Confidence flags")
    col1, col2, col3 = st.columns(3)
    with col1:
        if neff_L is not None:
            depth_badge = "(ProteinGym reference)" if is_verified_depth else "[Estimated / Unverified depth]"
            if neff_L < 10:
                st.warning(f"⚠️ Shallow MSA — Neff/L = {neff_L:.1f} (< 10) {depth_badge}. Evolutionary signal is unreliable.")
            else:
                st.success(f"✅ MSA depth OK — Neff/L = {neff_L:.1f} {depth_badge}")
        else:
            st.info("No MSA uploaded — using fallback model (ESM-2 + BLOSUM + RSA).")
    with col2:
        if low_plddt_positions:
            st.warning(f"⚠️ {len(low_plddt_positions)} positions have pLDDT < 70 (unreliable structure).")
        elif pdb_text:
            st.success("✅ All positions have pLDDT ≥ 70.")
    with col3:
        model_label = "Full (Evo + Structure + ESM-2)" if has_full \
                      else "Fallback (Structure + ESM-2)"
        st.info(f"Model: {model_label} • ESM-2: {model_tag}")

    # ---------------------------------------------------------------------------
    # candidate recommendation filtering & presentation
    # ---------------------------------------------------------------------------

    candidate_df = df.copy()
    filter_note = ""
    if exclude_disordered and low_plddt_positions:
        candidate_df = candidate_df[~candidate_df["pos"].isin(low_plddt_positions)]
        filter_note = f" *(excluded {len(low_plddt_positions)} disordered positions with pLDDT < 70, capped at max {max_per_pos}/position)*"
    else:
        filter_note = f" *(capped at max {max_per_pos}/position)*"

    if has_full:
        tab_full, tab_fallback, tab_shift, tab_3d, tab_double = st.tabs([
            "🧬 Full Model (Evo + Structure + ESM)",
            "⚡ Fallback Model (Structure + ESM)",
            "🔄 Evolutionary Shift Analysis",
            "🏛️ 3D Structure Viewer",
            "🧪 Combinatorial Mutants",
        ])

        with tab_full:
            st.subheader(f"Top-20 predicted mutations — Full Model{filter_note}")
            top_full = format_candidates_table(
                candidate_df,
                score_col="score_full",
                tier_col="tier_full",
                feat_cols=FEATURES_FULL,
                max_per_pos=max_per_pos,
                top_n=20,
            )
            st.dataframe(top_full, use_container_width=True)

            # Complete Landscape CSV for Full Model
            full_landscape_cols = ["pos", "wt", "mt", "mutant", "score_full", "tier_full", "rank_full", "evo", "entropy", "blosum", "rsa", "plddt", "esm"]
            full_export_df = df[full_landscape_cols].copy().rename(columns={
                "score_full": "final_score",
                "tier_full": "confidence_tier",
                "rank_full": "model_rank",
            }).sort_values(by="final_score", ascending=False)
            float_cols = [c for c in full_export_df.columns if full_export_df[c].dtype in ("float64", "float32")]
            full_export_df[float_cols] = full_export_df[float_cols].round(4)
            csv_bytes_all_full = full_export_df.to_csv(index=False).encode()
            csv_bytes_top_full = top_full.to_csv(index=False).encode()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("📥 Download Top Candidates CSV", csv_bytes_top_full, f"{clean_id}_top20_full_model.csv", "text/csv", use_container_width=True)
            with col_dl2:
                st.download_button("📊 Download Complete Mutational Landscape CSV (~5k)", csv_bytes_all_full, f"{clean_id}_complete_landscape_full_model.csv", "text/csv", use_container_width=True)

            st.divider()
            st.subheader("Signal Attribution (Feature Contributions)")
            col_sel, _ = st.columns([2, 2])
            with col_sel:
                selected_mut_full = st.selectbox(
                    "Select variant to inspect feature contributions:",
                    top_full["mutant"].tolist(),
                    key="sel_full_mut",
                )
            row_idx = df[df["mutant"] == selected_mut_full].index[0]
            f_row = feat_full.loc[row_idx].to_dict()
            chart_full, chart_df_full = make_feature_attribution_chart(
                ridge_full, FEATURES_FULL, f_row, selected_mut_full
            )
            st.altair_chart(chart_full, use_container_width=True)

            st.divider()
            st.subheader("Interactive Position × Amino-Acid Fitness Landscape (Full Model)")
            st.caption("Hover over cells to view detailed signal scores. Dark cells with white dots (•) indicate Wild-Type residues.")
            heatmap_chart_full = make_interactive_heatmap(df, seq, "score_full", f"Interactive Mutational Landscape — {entry_name or clean_id}", has_evo=True)
            st.altair_chart(heatmap_chart_full, use_container_width=True)

        with tab_fallback:
            st.subheader(f"Top-20 predicted mutations — Fallback Model{filter_note}")
            top_fb = format_candidates_table(
                candidate_df,
                score_col="score_fallback",
                tier_col="tier_fallback",
                feat_cols=FEATURES_FALLBACK,
                max_per_pos=max_per_pos,
                top_n=20,
            )
            st.dataframe(top_fb, use_container_width=True)

            # Complete Landscape CSV for Fallback Model
            fb_landscape_cols = ["pos", "wt", "mt", "mutant", "score_fallback", "tier_fallback", "rank_fallback", "blosum", "rsa", "plddt", "esm"]
            fb_export_df = df[fb_landscape_cols].copy().rename(columns={
                "score_fallback": "final_score",
                "tier_fallback": "confidence_tier",
                "rank_fallback": "model_rank",
            }).sort_values(by="final_score", ascending=False)
            float_cols = [c for c in fb_export_df.columns if fb_export_df[c].dtype in ("float64", "float32")]
            fb_export_df[float_cols] = fb_export_df[float_cols].round(4)
            csv_bytes_all_fb = fb_export_df.to_csv(index=False).encode()
            csv_bytes_top_fb = top_fb.to_csv(index=False).encode()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("📥 Download Top Candidates CSV", csv_bytes_top_fb, f"{clean_id}_top20_fallback_model.csv", "text/csv", use_container_width=True)
            with col_dl2:
                st.download_button("📊 Download Complete Mutational Landscape CSV (~5k)", csv_bytes_all_fb, f"{clean_id}_complete_landscape_fallback_model.csv", "text/csv", use_container_width=True)

            st.divider()
            st.subheader("Signal Attribution (Feature Contributions)")
            col_sel_fb, _ = st.columns([2, 2])
            with col_sel_fb:
                selected_mut_fb = st.selectbox(
                    "Select variant to inspect feature contributions:",
                    top_fb["mutant"].tolist(),
                    key="sel_fb_mut",
                )
            row_idx_fb = df[df["mutant"] == selected_mut_fb].index[0]
            f_row_fb = feat_fallback.loc[row_idx_fb].to_dict()
            chart_fb, chart_df_fb = make_feature_attribution_chart(
                ridge_fallback, FEATURES_FALLBACK, f_row_fb, selected_mut_fb
            )
            st.altair_chart(chart_fb, use_container_width=True)

            st.divider()
            st.subheader("Interactive Position × Amino-Acid Fitness Landscape (Fallback Model)")
            st.caption("Hover over cells to view detailed signal scores. Dark cells with white dots (•) indicate Wild-Type residues.")
            heatmap_chart_fb = make_interactive_heatmap(df, seq, "score_fallback", f"Interactive Mutational Landscape — {entry_name or clean_id}", has_evo=False)
            st.altair_chart(heatmap_chart_fb, use_container_width=True)

        with tab_shift:
            st.subheader("🔄 Evolutionary Shift Analysis")
            st.caption(
                "Compares ranking differences between the Full Model (Evo + Structure + ESM) "
                "and the Fallback Model (Structure + ESM). "
                "Positive rank shift indicates variants promoted by evolutionary evidence across homologs."
            )
            rho, _ = spearmanr(df["score_full"], df["score_fallback"])
            max_promo = df.nlargest(1, "rank_shift").iloc[0]
            max_pen = df.nsmallest(1, "rank_shift").iloc[0]

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Model Rank Correlation (ρ)", f"{rho:.3f}", help="Spearman correlation between Full and Fallback scores.")
            mcol2.metric("Top Promoted Variant", f"{max_promo.mutant}", f"+{int(max_promo.rank_shift)} ranks")
            mcol3.metric("Top Penalized Variant", f"{max_pen.mutant}", f"{int(max_pen.rank_shift)} ranks")

            st.divider()

            col_promo, col_pen = st.columns(2)
            with col_promo:
                st.markdown("#### 🚀 Top 10 Variants Promoted by Evolution")
                st.caption("Mutations where homolog conservation reveals tolerance or stability that structure alone missed.")
                promo_df = df.nlargest(10, "rank_shift")[
                    ["mutant", "pos", "wt", "mt", "rank_full", "rank_fallback", "rank_shift", "evo", "rsa", "esm"]
                ].copy()
                promo_df = promo_df.rename(columns={
                    "rank_full": "Full Rank",
                    "rank_fallback": "Fallback Rank",
                    "rank_shift": "Rank Gain (+)",
                    "evo": "Evo Score",
                    "rsa": "RSA",
                    "esm": "ESM",
                })
                float_cols = [c for c in promo_df.columns if promo_df[c].dtype in ("float64", "float32")]
                promo_df[float_cols] = promo_df[float_cols].round(3)
                st.dataframe(promo_df, use_container_width=True)

            with col_pen:
                st.markdown("#### 🛑 Top 10 Variants Penalized by Evolution")
                st.caption("Mutations that looked benign structurally, but are strictly conserved/purged by natural selection.")
                pen_df = df.nsmallest(10, "rank_shift")[
                    ["mutant", "pos", "wt", "mt", "rank_full", "rank_fallback", "rank_shift", "evo", "rsa", "esm"]
                ].copy()
                pen_df = pen_df.rename(columns={
                    "rank_full": "Full Rank",
                    "rank_fallback": "Fallback Rank",
                    "rank_shift": "Rank Drop (−)",
                    "evo": "Evo Score",
                    "rsa": "RSA",
                    "esm": "ESM",
                })
                float_cols = [c for c in pen_df.columns if pen_df[c].dtype in ("float64", "float32")]
                pen_df[float_cols] = pen_df[float_cols].round(3)
                st.dataframe(pen_df, use_container_width=True)

        with tab_3d:
            render_3d_tab(
                pdb_text=pdb_text,
                clean_id=clean_id,
                entry_name=entry_name,
                is_tem1=is_tem1,
                top_rec_df=top_full,
                df=df,
                seq=seq,
                plddt_dict=plddt_dict,
                rsa_dict=rsa_dict,
                mean_score_col="score_full",
            )

        with tab_double:
            render_combinatorial_tab(
                candidate_df=candidate_df,
                target_score_col="score_full",
                ca_coords=ca_coords,
                clean_id=clean_id,
                entry_name=entry_name,
            )

    else:
        st.info("💡 **Tip**: Upload an MSA file (.a2m) in the sidebar to unlock the Full Evolutionary Model and side-by-side shift analysis.")
        tab_fallback, tab_3d, tab_double = st.tabs([
            "⚡ Fallback Model (Structure + ESM)",
            "🏛️ 3D Structure Viewer",
            "🧪 Combinatorial Mutants",
        ])

        with tab_fallback:
            st.subheader(f"Top-20 predicted mutations — Fallback Model{filter_note}")
            top_fb = format_candidates_table(
                candidate_df,
                score_col="score_fallback",
                tier_col="tier_fallback",
                feat_cols=FEATURES_FALLBACK,
                max_per_pos=max_per_pos,
                top_n=20,
            )
            st.dataframe(top_fb, use_container_width=True)

            fb_landscape_cols = ["pos", "wt", "mt", "mutant", "score_fallback", "tier_fallback", "rank_fallback", "blosum", "rsa", "plddt", "esm"]
            fb_export_df = df[fb_landscape_cols].copy().rename(columns={
                "score_fallback": "final_score",
                "tier_fallback": "confidence_tier",
                "rank_fallback": "model_rank",
            }).sort_values(by="final_score", ascending=False)
            float_cols = [c for c in fb_export_df.columns if fb_export_df[c].dtype in ("float64", "float32")]
            fb_export_df[float_cols] = fb_export_df[float_cols].round(4)
            csv_bytes_all_fb = fb_export_df.to_csv(index=False).encode()
            csv_bytes_top_fb = top_fb.to_csv(index=False).encode()

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("📥 Download Top Candidates CSV", csv_bytes_top_fb, f"{clean_id}_top20_fallback_model.csv", "text/csv", use_container_width=True)
            with col_dl2:
                st.download_button("📊 Download Complete Mutational Landscape CSV (~5k)", csv_bytes_all_fb, f"{clean_id}_complete_landscape_fallback_model.csv", "text/csv", use_container_width=True)

            st.divider()
            st.subheader("Signal Attribution (Feature Contributions)")
            col_sel_fb, _ = st.columns([2, 2])
            with col_sel_fb:
                selected_mut_fb = st.selectbox(
                    "Select variant to inspect feature contributions:",
                    top_fb["mutant"].tolist(),
                    key="sel_fb_mut",
                )
            row_idx_fb = df[df["mutant"] == selected_mut_fb].index[0]
            f_row_fb = feat_fallback.loc[row_idx_fb].to_dict()
            chart_fb, chart_df_fb = make_feature_attribution_chart(
                ridge_fallback, FEATURES_FALLBACK, f_row_fb, selected_mut_fb
            )
            st.altair_chart(chart_fb, use_container_width=True)

            st.divider()
            st.subheader("Interactive Position × Amino-Acid Fitness Landscape (Fallback Model)")
            st.caption("Hover over cells to view detailed signal scores. Dark cells with white dots (•) indicate Wild-Type residues.")
            heatmap_chart_fb = make_interactive_heatmap(df, seq, "score_fallback", f"Interactive Mutational Landscape — {entry_name or clean_id}", has_evo=False)
            st.altair_chart(heatmap_chart_fb, use_container_width=True)

        with tab_3d:
            render_3d_tab(
                pdb_text=pdb_text,
                clean_id=clean_id,
                entry_name=entry_name,
                is_tem1=is_tem1,
                top_rec_df=top_fb,
                df=df,
                seq=seq,
                plddt_dict=plddt_dict,
                rsa_dict=rsa_dict,
                mean_score_col="score_fallback",
            )

        with tab_double:
            render_combinatorial_tab(
                candidate_df=candidate_df,
                target_score_col="score_fallback",
                ca_coords=ca_coords,
                clean_id=clean_id,
                entry_name=entry_name,
            )





if __name__ == "__main__":
    main()
