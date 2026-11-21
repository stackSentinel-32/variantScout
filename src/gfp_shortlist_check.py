"""Retrospective measured-single GFP check; emits aggregate metrics only."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.app import FEATURES_FULL, apply_diversity_cap
from .baseline import B62
from .cache_io import atomic_output
from .data import DMS_DIR, MSA_DIR, RAW, ROOT, RESULTS, check_numbering, load_assay, load_reference
from .msa import evo_scores, frequencies, load_msa, site_entropy
from .structure import STRUCT_DIR, compute_rsa_plddt, validate_structure_mapping

TRAINING = ("CALM1_HUMAN_Weile_2017", "TPMT_HUMAN_Matreyek_2018")
REPORTER = "GFP_AEQVI_Sarkisyan_2016"
REFERENCE_COLUMN = "ESM2_35M"


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_manifest(ref):
    files = [RAW / "DMS_substitutions.csv"]
    for assay in (*TRAINING, REPORTER):
        row = ref.loc[assay]
        files.extend([DMS_DIR / row.DMS_filename, MSA_DIR / row.MSA_filename,
                      STRUCT_DIR / row.pdb_file, RAW / "zero_shot" / f"{assay}.csv"])
    files.extend([Path(__file__), ROOT / "app" / "app.py"])
    files.extend(ROOT / "src" / name for name in ("baseline.py", "data.py", "msa.py", "structure.py"))
    return {"schema": 1, "training": list(TRAINING), "held_out": REPORTER,
            "reference_column": REFERENCE_COLUMN, "features": FEATURES_FULL,
            "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "biopython")},
            "inputs": {path.relative_to(ROOT).as_posix(): fingerprint(path) for path in files}}


def measured_table(assay, ref):
    row = ref.loc[assay]
    df = load_assay(assay, ref)
    query, alignment = load_msa(MSA_DIR / row.MSA_filename)
    check_numbering(row, df, query)
    freq = frequencies(alignment)
    index = df.pos.to_numpy() - int(row.MSA_start)
    df["evo"] = evo_scores(freq, index, df.wt, df.mt)
    df["entropy"] = site_entropy(freq)[index]
    df["blosum"] = [B62[w, m] for w, m in zip(df.wt, df.mt)]
    structure = (STRUCT_DIR / row.pdb_file).read_text()
    validate_structure_mapping(structure, row.target_seq)
    rsa, confidence = compute_rsa_plddt(structure)
    df["rsa"] = df.pos.map(rsa)
    df["plddt"] = df.pos.map(confidence)
    reference = pd.read_csv(RAW / "zero_shot" / f"{assay}.csv", usecols=["mutant", REFERENCE_COLUMN])
    df = df.merge(reference, on="mutant", how="left", validate="one_to_one").rename(columns={REFERENCE_COLUMN: "esm"})
    if not np.isfinite(df[[*FEATURES_FULL, "plddt", "DMS_score", "DMS_score_bin"]]).all().all():
        raise ValueError(f"{assay}: missing measured values")
    return df


def canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def read_verified(path, manifest):
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf8"))
        result = record["result"]
        if record["manifest"] != manifest or record["result_sha256"] != canonical_digest(result):
            return None
        return result
    except (KeyError, TypeError, ValueError, OSError):
        return None


def evaluate():
    ref = load_reference()
    if ref.loc[REPORTER, "UniProt_ID"] in set(ref.loc[list(TRAINING), "UniProt_ID"]):
        raise ValueError("Reporter overlaps training proteins")
    manifest = input_manifest(ref)
    output = RESULTS / "runs" / "gfp-shortlist" / f"{canonical_digest(manifest)}.json"
    cached = read_verified(output, manifest)
    if cached is not None:
        print(f"Verified cached result: {output}")
        print(json.dumps(cached, indent=2))
        return
    features, targets = [], []
    for assay in TRAINING:
        data = measured_table(assay, ref)
        features.append(data[FEATURES_FULL].rank(pct=True))
        targets.append(data.DMS_score.rank(pct=True))
    model = Ridge(alpha=1.).fit(pd.concat(features), pd.concat(targets))
    data = measured_table(REPORTER, ref)
    data["prediction"] = model.predict(data[FEATURES_FULL].rank(pct=True))
    data["top_decile"] = data.DMS_score.ge(data.DMS_score.quantile(0.9))
    eligible = data.loc[data.plddt.ge(70)].reset_index(drop=True)
    picks = apply_diversity_cap(eligible.sort_values("prediction", ascending=False), 2, 20)
    if picks.empty:
        raise ValueError("No eligible measured GFP substitutions")
    labels = ["DMS_score_bin", "top_decile"]
    rng = np.random.default_rng(42)
    # The first two occurrences of each position in a permutation are exactly
    # the rows admitted by the app's diversity cap.
    random_rates = []
    for _ in range(2000):
        shuffled = eligible.iloc[rng.permutation(len(eligible))]
        first_two = shuffled.groupby("pos", sort=False).cumcount().lt(2)
        sample = shuffled.loc[first_two].head(20)
        random_rates.append(sample[labels].mean().to_numpy())
    result = {"measured": len(data), "eligible": len(eligible), "picks": len(picks),
              "functional_hits": int(picks.DMS_score_bin.sum()), "top_decile_hits": int(picks.top_decile.sum()),
              "precision": dict(picks[labels].mean()), "random_draws": len(random_rates),
              "random_precision": dict(zip(labels, np.mean(random_rates, axis=0))),
              "random_monte_carlo_se": dict(zip(labels, np.std(random_rates, axis=0, ddof=1) / np.sqrt(len(random_rates))))}
    if manifest != input_manifest(ref):
        raise RuntimeError("Inputs changed during evaluation")
    record = {"manifest": manifest, "result": result, "result_sha256": canonical_digest(result)}
    with atomic_output(output) as pending:
        pending.write_text(json.dumps(record, indent=2), encoding="utf8")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    evaluate()
