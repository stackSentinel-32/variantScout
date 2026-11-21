from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import app
from src import baseline, benchmark_run, evaluate
from src.esm import compare_to_proteingym
from src.msa import compute_neff, parse_msa
from src.structure import validate_structure_mapping


def test_inputs_and_upload_isolation():
    assert app.parse_fasta(">one\nAC\nDE") == "ACDE"
    with pytest.raises(ValueError, match="exactly one"):
        app.parse_fasta(">one\nAC\n>two\nDE")
    with pytest.raises(ValueError, match="same nonzero width"):
        parse_msa(">one\nAAA\n>two\nA")
    uploads = [">one\nACDE", ">two\nEDCA"]
    with ThreadPoolExecutor(2) as pool:
        assert [q for q, _ in pool.map(parse_msa, uploads)] == ["ACDE", "EDCA"]
    _, rows = parse_msa(">one\nACDE\n" * 3000)
    assert compute_neff(rows) == 1


PDB = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 90.00           C\n"


def test_structure_mapping():
    validate_structure_mapping(PDB, "A")
    with pytest.raises(ValueError, match="entire submitted"):
        validate_structure_mapping(PDB, "AC")
    with pytest.raises(ValueError, match="mismatched"):
        validate_structure_mapping(PDB, "C")
    with pytest.raises(ValueError, match="one chain"):
        validate_structure_mapping(PDB + PDB.replace("ALA A", "ALA B"), "A")


def test_reference_matching(tmp_path):
    ref = tmp_path / "reference.csv"
    local = pd.DataFrame(dict(mutant=["A1C", "A1D", "A1E"], esm=[3, 1, 0], DMS_score=[1, 2, 3]))
    rows = pd.DataFrame(dict(mutant=["A1C", "A1D"], ESM2_35M=[1, 2]))
    rows.to_csv(ref, index=False)
    metrics = compare_to_proteingym(local, ref, "ESM2_35M")
    assert metrics["matched"] == 2 and metrics["measured"] == 3
    assert metrics["local_rho"] == pytest.approx(-1)
    assert metrics["reference_rho"] == pytest.approx(1)
    pd.concat([rows, rows]).to_csv(ref, index=False)
    with pytest.raises(pd.errors.MergeError):
        compare_to_proteingym(local, ref, "ESM2_35M")


def test_group_holdout_and_intervals():
    rows = pd.DataFrame({"assay": ["a", "b", "c"], "uniprot": ["p", "p", "q"], "y": [0, 0, 1]})
    for col in evaluate.FEATURES:
        rows[col] = rows.y

    class SplitCheck:
        def fit(self, features, labels):
            self.seen = set(labels)
        def predict(self, features):
            assert self.seen.isdisjoint(features[:, 0])
            return features[:, 0]

    assert evaluate.run_loo(rows, SplitCheck).pred.tolist() == [0, 0, 1]
    stats = rows.assign(model="test", spearman=rows.y, p10_bin=rows.y, rand_bin=0,
                        p10_top=rows.y, rand_top=0)
    intervals = evaluate.protein_intervals(stats)
    assert intervals.proteins.eq(2).all()
    assert intervals["mean"].eq(0.5).all()


def test_tied_cutoff_and_fewer_than_ten():
    data = pd.DataFrame(dict(pred=[0, 1, 2], DMS_score=[0, 2, 2], DMS_score_bin=[0, 1, 0]))
    assert evaluate.precision_at_k(data, "pred") == pytest.approx((1/3, 2/3))
    assert evaluate.random_precision(data) == pytest.approx((1/3, 2/3))


def test_incomplete_runs_are_isolated(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for key in ("RAW", "DMS_DIR", "MSA_DIR", "STRUCT_DIR"):
        monkeypatch.setattr(benchmark_run, key, raw)
    for name in ["DMS_substitutions.csv", "alignment", "assay", "structure"]:
        (raw / name).write_text("fixture")
    ref = pd.DataFrame(dict(MSA_filename=["alignment"], DMS_filename=["assay"], pdb_file=["structure"]), index=["toy"])
    published = tmp_path / "tables" / "baseline.csv"
    published.parent.mkdir()
    published.write_bytes(b"snapshot")
    with pytest.raises(RuntimeError):
        with benchmark_run.benchmark_run("toy", ref, ["toy"], tmp_path / "runs") as run:
            (run / "tables" / "baseline.csv").write_text("partial")
            raise RuntimeError("interrupted")
    (raw / "alignment").unlink()
    with pytest.raises(ValueError, match="Incomplete benchmark"):
        with benchmark_run.benchmark_run("toy", ref, ["toy"], tmp_path / "runs"):
            pytest.fail("Incomplete cohort entered run")
    manifests = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(manifests) == 2
    assert all(json.loads(path.read_text())["status"] == "incomplete" for path in manifests)
    assert published.read_bytes() == b"snapshot"


def test_missing_ranking_and_downstream_error(monkeypatch, tmp_path):
    ref = pd.DataFrame(dict(UniProt_ID=["toy"], MSA_filename=["toy.a2m"], MSA_Neff_L=[1],
                            coarse_selection_type=["fixture"]), index=["assay"])
    rows = pd.DataFrame(dict(mutant=["A1D", "A1C"], DMS_score=[1, 2], DMS_score_bin=[1, 1]))
    monkeypatch.setattr(baseline, "load_assay", lambda *args: rows.copy())
    monkeypatch.setattr(baseline, "FEAT_CACHE_DIR", tmp_path)
    rows.to_parquet(tmp_path / "toy.parquet")
    _, summary = baseline.score_assay("assay", ref)
    assert np.isnan(summary["top10_fn"]) and summary["top10_reason"]
    rows.iloc[:1].to_parquet(tmp_path / "toy.parquet")
    with pytest.raises(ValueError, match="coverage differs"):
        baseline.score_assay("assay", ref)
    rows.to_parquet(tmp_path / "toy.parquet")
    monkeypatch.setattr(evaluate, "MSA_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "STARTER_SET", ["assay"])
    (tmp_path / "toy.a2m").touch()
    with pytest.raises(ValueError, match="missing evaluation features"):
        evaluate.build_dataset(ref)


@pytest.mark.parametrize("structure,message", [(None, "Structure unavailable"), (PDB, "entire submitted")])
def test_structure_errors_reach_form(monkeypatch, structure, message):
    monkeypatch.setattr(app, "resolve_uniprot_id", lambda name: (name, name))
    monkeypatch.setattr(app, "fetch_alphafold_pdb", lambda name: structure)
    ui = AppTest.from_string("from app.app import main\nmain()").run()
    ui.text_area[0].set_value("ACDEFGHIKL")
    ui.text_input[0].set_value("TOY")
    next(b for b in ui.button if b.label == "Score mutations").click().run()
    assert not ui.exception
    assert any(message in e.value for e in ui.error)


@pytest.mark.parametrize("full", [False, True])
def test_completed_submission_survives_reruns(monkeypatch, full):
    from types import SimpleNamespace

    def fixture(seq, name, *args):
        rows = pd.DataFrame(dict(mutant=["A1C", "A1D"], pos=[1, 1], wt=["A", "A"], mt=["C", "D"],
                                 plddt=[50., 50.], low_plddt=["<70"] * 2, rank_shift=[0, 0]))
        for col in app.FEATURES_FULL:
            rows[col] = [0.2, 0.8]
        for mode in ["fallback", "full"]:
            rows["score_" + mode] = [0.1, 0.2]
            rows["tier_" + mode] = ["low", "low"]
            rows["rank_" + mode] = [2, 1]
        return dict(seq=seq, clean_id=name, entry_name=name, primary_acc=name, model_tag="fixture",
                    pdb_text=PDB, rsa_dict={1: 0.2}, plddt_dict={1: 50}, ca_coords={},
                    low_plddt_positions={1}, neff_L=None, has_full=full, df=rows,
                    ridge_full=SimpleNamespace(coef_=[0]*5), ridge_fallback=SimpleNamespace(coef_=[0]*3),
                    feat_full=rows[app.FEATURES_FULL], feat_fallback=rows[app.FEATURES_FALLBACK])

    scorer = Mock(side_effect=fixture)
    monkeypatch.setattr(app, "score_submission", scorer)
    monkeypatch.setattr(app, "get_training_set_uniprots", lambda: set())
    ui = AppTest.from_string("from app.app import main\nmain()", default_timeout=30).run()
    ui.text_area[0].set_value("ACDEFGHIKL")
    ui.text_input[0].set_value("FIRST")
    def submit():
        next(b for b in ui.button if b.label == "Score mutations").click().run()
        assert not ui.exception
    submit()
    assert any("No eligible" in m.value for m in ui.info)
    assert len(ui.get("download_button")) == 1
    def coverage_filter():
        return next(c for c in ui.checkbox if "Exclude disordered" in c.label)
    coverage_filter().uncheck().run()
    assert not ui.exception
    ui.selectbox(key="viewer_color_mode").select("AlphaFold pLDDT Confidence").run()
    assert not ui.exception and len(ui.get("download_button")) >= 2
    coverage_filter().check().run()
    assert any("No eligible" in m.value for m in ui.info)
    ui.text_input[0].set_value("SECOND").run()
    assert any("last completed" in m.value for m in ui.info)
    assert any("FIRST" in c.value for c in ui.caption)
    assert scorer.call_count == 1
    submit()
    assert any("SECOND" in c.value for c in ui.caption) and scorer.call_count == 2


def test_cache_publish_is_atomic(tmp_path):
    from src.cache_io import atomic_output

    cache = tmp_path / "entry.parquet"
    cache.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with atomic_output(cache) as pending:
            pending.write_bytes(b"incomplete")
            raise RuntimeError("interruption")
    assert list(tmp_path.iterdir()) == [cache]
    assert cache.read_bytes() == b"original"
    with atomic_output(cache) as pending:
        pending.write_bytes(b"complete")
    assert cache.read_bytes() == b"complete"


@pytest.mark.parametrize("button,expected_id", [("Clear", ""), ("Load TEM-1 Demo", "P62593")])
def test_sidebar_reset_discards_previous_upload(button, expected_id):
    ui = AppTest.from_string("from app.app import main\nmain()", default_timeout=30).run()
    ui.text_area[0].set_value("ACDEFGHIKL")
    ui.text_input[0].set_value("FIXTURE")
    ui.file_uploader[0].upload("example.fa", b">fixture\nACDEFGHIKL\n").run()
    assert ui.file_uploader[0].value.name == "example.fa"
    next(b for b in ui.button if button in b.label).click().run()
    assert not ui.exception
    assert ui.file_uploader[0].value is None
    assert ui.text_input[0].value == expected_id
    assert ui.text_area[0].value == (app.TEM1_DEMO_SEQUENCE if expected_id else "")


def test_app_entrypoint_preserves_shared_module_state_on_rerun():
    import sys
    from src import esm

    model_cache = esm._model_cache
    ui = AppTest.from_file(str(app.ROOT / "app" / "app.py"), default_timeout=30).run()
    path_count = sys.path.count(str(app.ROOT))
    ui.slider[0].set_value(3).run()
    assert not ui.exception
    assert esm._model_cache is model_cache
    assert sys.path.count(str(app.ROOT)) == path_count
