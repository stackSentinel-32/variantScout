import json

from src.gfp_shortlist_check import canonical_digest, read_verified


def test_report_requires_matching_inputs_and_complete_result(tmp_path):
    path = tmp_path / "report.json"
    manifest = {"schema": 1, "inputs": {"training.csv": "a", "structure.pdb": "b"}}
    result = {"measured": 30, "picks": 20}
    record = {"manifest": manifest, "result": result, "result_sha256": canonical_digest(result)}
    path.write_text(json.dumps(record))
    assert read_verified(path, manifest) == result
    assert read_verified(path, {**manifest, "schema": 2}) is None
    assert read_verified(path, {**manifest, "inputs": {"training.csv": "c", "structure.pdb": "b"}}) is None
    path.write_text(json.dumps({**record, "result": {"measured": 29}}))
    assert read_verified(path, manifest) is None
    path.write_text("{")
    assert read_verified(path, manifest) is None
