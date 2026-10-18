import pandas as pd
from Bio.Align import substitution_matrices
from scipy.stats import spearmanr

from .data import MSA_DIR, RESULTS, STARTER_SET, check_numbering, load_assay, load_reference
from .msa import evo_scores, frequencies, load_msa, site_entropy
from .plots import heatmap
from .structure import compute_rsa_plddt

B62 = substitution_matrices.load("BLOSUM62")


def score_assay(dms_id, ref):
    row = ref.loc[dms_id]
    df = load_assay(dms_id, ref)
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

    top10 = df.nlargest(10, "evo")
    summary = {
        "assay": dms_id,
        "type": row.coarse_selection_type,
        "mutants": len(df),
        "neff_L": row.MSA_Neff_L,
        "rho_evo": spearmanr(df.evo, df.DMS_score)[0],
        "rho_entropy": spearmanr(df.entropy, df.DMS_score)[0],
        "rho_blosum": spearmanr(df.blosum, df.DMS_score)[0],
        "rho_rsa": spearmanr(df.rsa, df.DMS_score)[0],
        "top10_fn": top10.DMS_score_bin.mean(),
        "random_fn": df.DMS_score_bin.mean(),
    }
    return df, summary


def main():
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
