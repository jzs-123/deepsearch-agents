import json
from dataclasses import replace

import pytest

from app.rag.calibration import CalibrationRequiredError, save_calibration
from app.rag.config import RagConfig
from app.rag.eval import EvalItem, evaluate
from app.rag.pipeline import RagPipeline


def make_pipeline(tmp_path, embedder):
    pipeline = RagPipeline(kb="test", config=RagConfig(
        data_dir=tmp_path, embedding_provider="hash", query_rewrite="off",
        retry_on_low_score=False, cache_enabled=True, audit_enabled=False,
    ), embedder=embedder)
    pipeline.add_texts(["授信额度上限为 500 元。"], source="policy.md")
    return pipeline


def persist(pipeline, tmp_path):
    items = [EvalItem(question="授信额度", expected_sources=["policy.md"]),
             EvalItem(question="月球餐厅菜单", expect_no_answer=True)]
    eval_set = tmp_path / "eval.json"
    eval_set.write_text("[]", encoding="utf-8")
    return save_calibration(pipeline, evaluate(pipeline, items), eval_set)


def test_saved_threshold_is_used_and_generation_cache_changes(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder)
    persist(pipeline, tmp_path)
    path = pipeline.kb_dir / "calibration.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["recommended_threshold"] = -1
    path.write_text(json.dumps(record), encoding="utf-8")
    first = pipeline.query("授信额度")
    assert pipeline.config.min_score == -1
    assert first.debug["calibration"]["status"] == "calibrated"
    assert pipeline.query("授信额度").debug["cache_hit"]
    record["recommended_threshold"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    answer = pipeline.query("授信额度")
    assert not answer.debug["cache_hit"]
    assert not answer.grounded


def test_changed_data_blocks_stale_calibration_and_can_be_recalibrated(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder)
    persist(pipeline, tmp_path)
    pipeline.add_texts(["新规定：复核时限为 3 天。"], source="new.md")
    with pytest.raises(CalibrationRequiredError):
        pipeline.search("授信额度")
    pipeline._base_config = replace(pipeline._base_config, use_calibration=False)
    persist(pipeline, tmp_path)
    pipeline._base_config = replace(pipeline._base_config, use_calibration=True)
    pipeline.search("授信额度")
    assert pipeline.last_search_debug["calibration"]["status"] == "calibrated"


@pytest.mark.parametrize("field,value", [("embedding_fingerprint", "hash:256"),
                                           ("recommended_threshold", float("nan"))])
def test_invalid_calibration_fails_closed(tmp_path, hash_embedder, field, value):
    pipeline = make_pipeline(tmp_path, hash_embedder)
    record = persist(pipeline, tmp_path)
    record[field] = value
    (pipeline.kb_dir / "calibration.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CalibrationRequiredError):
        pipeline.search("授信额度")


def test_cli_saves_and_refreshes_stale_calibration(tmp_path, hash_embedder, monkeypatch):
    from app.rag import cli
    pipeline = make_pipeline(tmp_path, hash_embedder)
    monkeypatch.setattr(cli, "_build_config", lambda args: pipeline._base_config)
    eval_set = tmp_path / "examples.json"
    eval_set.write_text(json.dumps([
        {"question": "授信额度", "expected_sources": ["policy.md"]},
        {"question": "月球餐厅菜单", "expect_no_answer": True},
    ]), encoding="utf-8")
    argv = ["eval", "-k", "test", "-e", str(eval_set), "-n", "4", "--save-calibration", "--json"]
    assert cli.main(argv) == 0
    pipeline.add_texts(["复核时限为三天。"], source="new.md")
    assert cli.main(argv) == 0
    pipeline.search("授信额度")
    assert pipeline.last_search_debug["calibration"]["status"] == "calibrated"
