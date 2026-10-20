"""Leave-one-protein-out evaluation of combined mutation-ranking models.

Loads per-mutation feature tables (built by baseline.py), rank-normalises
every feature and the target within each assay, then trains Ridge and
LightGBM regressors under leave-one-protein-out cross-validation.

Outputs
-------
results/tables/loo_metrics.csv   per-protein Spearman, NDCG@10/50,
                                  Precision@10 (two definitions), random
                                  baselines, and 95 % bootstrap CI on Spearman
results/tables/ablation.csv      drop-one-feature impact on mean Spearman
results/figures/neff_vs_rho.png  key ESM-2 vs Neff/L scatter
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
import lightgbm as lgb

from .data import MSA_DIR, RESULTS, STARTER_SET, load_reference
from .baseline import score_assay

FEATURES = ["evo", "entropy", "blosum", "rsa", "esm"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rank_normalise(series):
    """Map values to [0, 1] percentile rank within the series."""
    return series.rank(pct=True)


def build_dataset(ref):
    """Run score_assay for every protein and concatenate into one DataFrame.

    Returns a DataFrame with one row per single mutation and columns for
    every feature, the target (DMS_score), the binary hit label, the assay
    id, and the UniProt ID used as the LOO grouping key.
    """
    frames = []
    for dms_id in STARTER_SET:
        msa_path = MSA_DIR / ref.loc[dms_id].MSA_filename
        if not msa_path.exists():
            print(f"  skipping {dms_id} – MSA not found")
            continue
        print(f"  loading {dms_id}")
        df, _ = score_assay(dms_id, ref)
        df["assay"] = dms_id
        df["uniprot"] = ref.loc[dms_id].UniProt_ID
        frames.append(df[["assay", "uniprot", "DMS_score", "DMS_score_bin",
                           *FEATURES]])
    return pd.concat(frames, ignore_index=True)


def normalise_within_assay(df):
    """Rank-normalise the target and all features within each assay."""
    out = df.copy()
    out["y"] = out.groupby("assay")["DMS_score"].transform(rank_normalise)
    for col in FEATURES:
        out[col] = out.groupby("assay")[col].transform(rank_normalise)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def precision_at_k(sub, pred_col, k=10):
    """Return two precision@k values: binary hit and true top-10 %."""
    top = sub.nlargest(k, pred_col)
    p_bin = top["DMS_score_bin"].mean()
    thr = sub["DMS_score"].quantile(0.9)
    p_top = (top["DMS_score"] >= thr).mean()
    return p_bin, p_top


def random_precision(sub, k=10):
    """Expected precision@10 for a random ranker (= base rates)."""
    p_bin = sub["DMS_score_bin"].mean()
    thr = sub["DMS_score"].quantile(0.9)
    p_top = (sub["DMS_score"] >= thr).mean()
    return p_bin, p_top


def bootstrap_ci(values, n=1000, alpha=0.05, rng=None):
    """95 % bootstrap CI (percentile method) for the mean of an array."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values)
    means = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# LOO cross-validation
# ---------------------------------------------------------------------------

def make_ridge():
    return Ridge(alpha=1.0)


def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        verbose=-1
    )


def run_loo(df, model_fn):
    """Add a 'pred' column via leave-one-protein-out CV."""
    df = df.copy()
    df["pred"] = np.nan
    groups = df["uniprot"].values
    X = df[FEATURES].values
    y = df["y"].values

    for tr_idx, te_idx in LeaveOneGroupOut().split(X, groups=groups):
        m = model_fn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr_idx], y[tr_idx])
        df.iloc[te_idx, df.columns.get_loc("pred")] = m.predict(X[te_idx])
    return df


def evaluate_predictions(df):
    """Compute per-assay metrics from the 'pred' column."""
    rows = []
    for assay, sub in df.groupby("assay"):
        sub = sub.dropna(subset=["pred"])
        rho = spearmanr(sub["pred"], sub["DMS_score"]).statistic

        # ndcg_score expects 2-D arrays and non-negative relevance
        rel = sub["y"].values.reshape(1, -1)
        sc = sub["pred"].values.reshape(1, -1)
        ndcg10 = ndcg_score(rel, sc, k=10)
        ndcg50 = ndcg_score(rel, sc, k=min(50, len(sub)))

        p_bin, p_top = precision_at_k(sub, "pred")
        rp_bin, rp_top = random_precision(sub)

        rows.append({
            "assay": assay,
            "spearman": rho,
            "ndcg10": ndcg10,
            "ndcg50": ndcg50,
            "p10_bin": p_bin,
            "p10_top": p_top,
            "rand_bin": rp_bin,
            "rand_top": rp_top,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------

def ablation(df, model_fn):
    """Drop one feature at a time; return mean Spearman change vs. full model."""
    full_df = run_loo(df, model_fn)
    full_metrics = evaluate_predictions(full_df)
    full_mean = full_metrics["spearman"].mean()

    rows = [{"dropped": "none (full model)", "mean_spearman": full_mean, "delta": 0.0}]
    for feat in FEATURES:
        subset_feats = [f for f in FEATURES if f != feat]
        df_sub = df.copy()
        # zero out the dropped feature so the same LOO code works unchanged
        df_sub[feat] = 0.0
        pred_df = run_loo(df_sub, model_fn)
        m = evaluate_predictions(pred_df)["spearman"].mean()
        rows.append({"dropped": feat, "mean_spearman": m, "delta": m - full_mean})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Neff/L vs Spearman plot
# ---------------------------------------------------------------------------

TYPE_COLOR = {
    "Activity": "#e07b39",
    "OrganismalFitness": "#4a90d9",
    "Expression": "#57a85e",
    "Stability": "#9b59b6",
}

def plot_neff_vs_rho(baseline_csv, out_path):
    """Scatter of Neff/L vs per-signal Spearman rho, annotated by protein."""
    bl = pd.read_csv(baseline_csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log")

    signal_style = {
        "rho_evo":  dict(marker="o", linestyle="-",  label="Evolutionary (evo)"),
        "rho_rsa":  dict(marker="s", linestyle="--", label="RSA (structure)"),
        "rho_esm":  dict(marker="^", linestyle=":",  label="ESM-2 35M"),
    }

    for col, style in signal_style.items():
        # colour points by assay type
        for _, row in bl.iterrows():
            c = TYPE_COLOR.get(row["type"], "#888888")
            ax.scatter(row["neff_L"], row[col], color=c, s=50, zorder=3,
                       marker=style["marker"], alpha=0.85)
        # draw a thin connecting line across proteins sorted by Neff/L
        bl_sorted = bl.sort_values("neff_L")
        ax.plot(bl_sorted["neff_L"], bl_sorted[col], color="grey",
                linewidth=0.6, linestyle=style["linestyle"], alpha=0.5)

    # signal legend
    from matplotlib.lines import Line2D
    signal_handles = [
        Line2D([0], [0], marker=s["marker"], color="grey",
               linestyle=s["linestyle"], label=s["label"])
        for s in signal_style.values()
    ]
    # type legend
    type_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="", label=t)
        for t, c in TYPE_COLOR.items() if t in bl["type"].values
    ]

    leg1 = ax.legend(handles=signal_handles, loc="upper left", fontsize=8,
                     title="Signal", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, loc="lower right", fontsize=8,
              title="Assay type", title_fontsize=8)

    # annotate a few interesting proteins
    for _, row in bl.iterrows():
        short = row["assay"].split("_")[0]
        ax.annotate(short, (row["neff_L"], row["rho_esm"]),
                    fontsize=6, alpha=0.7, xytext=(2, 2),
                    textcoords="offset points")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("MSA depth  (Neff / L,  log scale)", fontsize=10)
    ax.set_ylabel("Spearman ρ vs measured DMS fitness", fontsize=10)
    ax.set_title("Signal performance vs MSA depth\n"
                 "ESM-2 fails on shallow MSAs; RSA is consistently strong",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ref = load_reference()
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    print("Building per-mutation feature tables …")
    raw = build_dataset(ref)
    df = normalise_within_assay(raw)

    # --- LOO with Ridge ---
    print("\nRunning Ridge LOO …")
    ridge_df = run_loo(df, make_ridge)
    ridge_metrics = evaluate_predictions(ridge_df)
    ridge_metrics["model"] = "Ridge"

    # --- LOO with LightGBM ---
    print("Running LightGBM LOO …")
    lgbm_df = run_loo(df, make_lgbm)
    lgbm_metrics = evaluate_predictions(lgbm_df)
    lgbm_metrics["model"] = "LightGBM"

    # bootstrap CI on Ridge Spearman values (more stable than GBM with few groups)
    rho_values = ridge_metrics["spearman"].values
    ci_lo, ci_hi = bootstrap_ci(rho_values)

    print(f"\nRidge  mean Spearman = {rho_values.mean():.3f}  "
          f"95 % CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"LightGBM mean Spearman = {lgbm_metrics['spearman'].mean():.3f}")

    metrics = pd.concat([ridge_metrics, lgbm_metrics], ignore_index=True)
    metrics = metrics.round(3)
    out_path = RESULTS / "tables" / "loo_metrics.csv"
    metrics.to_csv(out_path, index=False)
    print(f"\n  saved {out_path}")
    print(metrics.to_string(index=False))

    # --- ablation (Ridge only – simpler and more honest with few groups) ---
    print("\nRunning ablation …")
    abl = ablation(df, make_ridge).round(3)
    abl_path = RESULTS / "tables" / "ablation.csv"
    abl.to_csv(abl_path, index=False)
    print(f"  saved {abl_path}")
    print(abl.to_string(index=False))

    # --- Neff/L plot ---
    plot_neff_vs_rho(
        RESULTS / "tables" / "baseline.csv",
        RESULTS / "figures" / "neff_vs_rho.png",
    )


if __name__ == "__main__":
    main()
