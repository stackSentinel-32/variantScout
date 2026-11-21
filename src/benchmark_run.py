"""Isolate benchmark artifacts and record exactly which cohort was requested."""
from contextlib import contextmanager
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

from .data import DMS_DIR, MSA_DIR, RAW, RESULTS
from .structure import STRUCT_DIR


@contextmanager
def benchmark_run(name, ref, cohort, output_root=None):
    root = Path(output_root or RESULTS / "runs") / f"{name}-{uuid4().hex}"
    root.mkdir(parents=True)
    (root / "tables").mkdir()
    (root / "figures").mkdir()
    manifest = {"pipeline": name, "status": "incomplete", "cohort": list(cohort),
                "missing": [], "input_sha256": {}}
    path = root / "manifest.json"

    def record():
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp.replace(path)

    manifest["packages"] = {name: version(name) for name in
                            ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "torch", "transformers")}
    manifest["source_sha256"] = {}
    for module in sorted(Path(__file__).parent.glob("*.py")):
        with module.open("rb") as stream:
            manifest["source_sha256"][module.name] = file_digest(stream, "sha256").hexdigest()
    record()
    try:
        files = {RAW / "DMS_substitutions.csv"}
        for assay in cohort:
            if assay not in ref.index:
                manifest["missing"].append(assay)
                continue
            row = ref.loc[assay]
            files.update([MSA_DIR / row.MSA_filename, DMS_DIR / row.DMS_filename,
                          STRUCT_DIR / row.pdb_file])
        for source in sorted(files):
            if source.is_file():
                with source.open("rb") as stream:
                    manifest["input_sha256"][str(source.relative_to(RAW))] = file_digest(stream, "sha256").hexdigest()
            else:
                manifest["missing"].append(str(source))
        if manifest["missing"]:
            raise ValueError("Incomplete benchmark cohort: " + ", ".join(manifest["missing"]))
        record()
        print(f"Benchmark output: {root}")
        yield root
    except BaseException as exc:
        manifest["error"] = str(exc)
        record()
        raise
    manifest["status"] = "complete"
    record()
