from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr

from .data import MSA_DIR, RESULTS, ROOT, STARTER_SET, check_numbering, load_assay, load_reference
from .esm import _load_model, esm_mutation_scores, score_sequence
from .msa import evo_scores, frequencies, load_msa, site_entropy
from .plots import heatmap
from .structure import compute_rsa_plddt

B62 = substitution_matrices.load("BLOSUM62")
FEAT_CACHE_DIR: Path = ROOT / "data" / "processed" / "features"


def score_assay(dms_id: str, ref: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Score all mutations in an assay, caching computed features to Parquet."""
    row = ref.loc[dms_id]
    df = load_assay(dms_id, ref)
    FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = FEAT_CACHE_DIR / f"{row.UniProt_ID}.parquet"

    # Fast path: load pre-cached feature table if available
    if parquet_path.exists():
        cached = pd.read_parquet(parquet_path)
        feat_cols = [c for c in ["evo", "entropy", "blosum", "rsa", "plddt", "esm"] if c in cached.columns]
        df = df.merge(cached[["mutant", *feat_cols]], on="mutant", how="left")
    else:
        query, msa = load_msa(MSA_DIR / row.MSA_filename)
        check_numbering(row, df, query)

        freq = frequencies(msa)
        cols = df.pos.to_numpy() - int(row.MSA_start)
        df["evo"] = evo_scores(freq, cols, df.wt, df.mt)
        df["entropy"] = site_entropy(freq)[cols]
        df["blosum"] = [B62[w, m] for w, m in zip(df.wt, df.mt)]

        rsa, plddt = compute_rsa_plddt(row.pdb_file)
        df["rsa"] = df.pos.map(rsa)
        df["plddt"] = df.pos.map(plddt)

        tok, _ = _load_model()
        logprobs = score_sequence(row.target_seq, row.UniProt_ID)
        df["esm"] = esm_mutation_scores(logprobs, tok, df.pos.to_numpy(), df.wt, df.mt)

        # Save to parquet cache for subsequent runs
        df.to_parquet(parquet_path, index=False)

    top10 = df.nlargest(10, "evo")
    summary: Dict[str, Any] = {
        "assay": dms_id,
        "type": row.coarse_selection_type,
        "mutants": len(df),
        "neff_L": row.MSA_Neff_L,
        "rho_evo": float(spearmanr(df.evo, df.DMS_score)[0]),
        "rho_entropy": float(spearmanr(df.entropy, df.DMS_score)[0]),
        "rho_blosum": float(spearmanr(df.blosum, df.DMS_score)[0]),
        "rho_rsa": float(spearmanr(df.rsa, df.DMS_score)[0]),
        "rho_esm": float(spearmanr(df.esm, df.DMS_score)[0]),
        "top10_fn": float(top10.DMS_score_bin.mean()),
        "random_fn": float(df.DMS_score_bin.mean()),
    }
    return df, summary


def main() -> None:
    ref = load_reference()
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    rows = []

    for dms_id in STARTER_SET:
        if not (MSA_DIR / ref.loc[dms_id].MSA_filename).exists():
            print("skipping", dms_id, "- MSA file not downloaded")
            continue
        df, summary = score_assay(dms_id, ref)
        rows.append(summary)
        heatmap(df, dms_id, RESULTS / "figures" / f"{dms_id}_heatmap.png")
        print("done", dms_id)

    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "tables" / "baseline.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

