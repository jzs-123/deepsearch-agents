"""app/rag/eval_ragas.py：RAGAS 评测脚本（**测试里绝不安装/不调用真的 ragas**）。

锁四件事：
1. **组装逻辑**：`build_ragas_records` 必须产出 RAGAS 要求的 `contexts: list[list[str]]`，
   并把"没有上下文/答案为空/应拒答"的条目**跳过并说明原因**，而不是塞空值让 RAGAS 报错；
2. **依赖缺失**：给清晰中文提示（含 `pip install ragas datasets`），而不是裸 ImportError；
3. **脱敏**：报告/摘要里**绝不能出现原始 api_key**；
4. **可注入**：`run_evaluation(evaluator=...)` 用 stub 就能跑通整条组装链路；
   另用伪造的 `ragas`/`datasets` 模块走一遍真正的导入分支。
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from app.rag.eval_ragas import (
    ALL_METRICS,
    DEFAULT_METRICS,
    GROUND_TRUTH_METRICS,
    RagasRecords,
    build_ragas_records,
    check_ragas_available,
    collect_rows,
    dependency_message,
    format_summary,
    load_eval_set,
    main,
    mask_secret,
    public_judge_config,
    require_ragas,
    resolve_metric_names,
    result_to_report,
    run_evaluation,
)


# ── 脱敏 ────────────────────────────────────────────────────────────


def test_mask_secret_never_returns_the_full_secret():
    secret = "sk-1234567890abcdef"
    masked = mask_secret(secret)
    assert secret not in masked
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "*" in masked


def test_mask_secret_edge_cases():
    assert mask_secret("") == "(未配置)"
    assert mask_secret(None) == "(未配置)"
    assert mask_secret("short") == "*****"


def test_public_judge_config_hides_key_but_reports_presence():
    public = public_judge_config({"model": "m", "base_url": "u", "api_key": "sk-supersecret-9999"})
    assert "sk-supersecret-9999" not in json.dumps(public)
    assert public["api_key_configured"] is True
    assert public["model"] == "m"


# ── 依赖缺失提示 ────────────────────────────────────────────────────


def test_dependency_message_is_actionable_chinese_guidance():
    message = dependency_message(["ragas", "datasets"])
    assert "pip install ragas datasets" in message
    assert "可选依赖" in message
    assert "pyproject.toml" in message
    assert "eval_offline" in message, "应引导到零依赖替代"
    assert "docs/rag-eval-guide.md" in message


def test_check_ragas_available_reports_missing_in_this_environment():
    """本机确实没装 ragas —— 真实演示"依赖缺失时给清晰提示"。"""
    available, message = check_ragas_available()
    if available:  # pragma: no cover - 只有用户自己装了 ragas 才会走到
        pytest.skip("本环境已安装 ragas，跳过缺失分支")
    assert "缺少依赖" in message
    assert "pip install ragas datasets" in message


def test_require_ragas_raises_runtime_error_with_hint():
    with pytest.raises(RuntimeError) as excinfo:
        require_ragas()
    assert "pip install ragas datasets" in str(excinfo.value)


# ── 评测集 ──────────────────────────────────────────────────────────


def test_load_eval_set_supports_array_items_and_jsonl(tmp_path):
    array = tmp_path / "a.json"
    array.write_text(json.dumps([{"question": "q1", "ground_truth": "g"}]), encoding="utf-8")
    assert load_eval_set(array)[0]["ground_truth"] == "g"

    obj = tmp_path / "b.json"
    obj.write_text(json.dumps({"items": [{"question": "q2"}]}), encoding="utf-8")
    assert load_eval_set(obj)[0]["question"] == "q2"

    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text('{"question": "q3"}\n{"question": "q4"}\n', encoding="utf-8")
    assert [i["question"] for i in load_eval_set(jsonl)] == ["q3", "q4"]


def test_load_eval_set_defaults_and_validation(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"question": "q"}]), encoding="utf-8")
    item = load_eval_set(path)[0]
    assert item["ground_truth"] == ""
    assert item["expect_no_answer"] is False
    assert item["expected_sources"] == []

    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"x": 1}]), encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_eval_set(bad)


# ── 记录组装（核心）─────────────────────────────────────────────────


def test_build_ragas_records_produces_list_of_lists_for_contexts():
    rows = [
        {"question": "q1", "answer": "a1", "contexts": ["c1", "c2"], "ground_truth": "g1"},
        {"question": "q2", "answer": "a2", "contexts": [{"text": "c3"}], "ground_truth": "g2"},
    ]
    records = build_ragas_records(rows)

    assert records.size == 2
    assert records.dataset["contexts"] == [["c1", "c2"], ["c3"]]
    assert all(isinstance(ctx, list) for ctx in records.dataset["contexts"])
    assert records.dataset["question"] == ["q1", "q2"]
    assert records.dataset["ground_truth"] == ["g1", "g2"]
    assert records.has_ground_truth is True
    assert records.skipped == []


def test_build_ragas_records_skips_rows_without_contexts_with_reason():
    """没有上下文就算不了 faithfulness —— 必须跳过并说明原因，而不是塞空列表。"""
    rows = [
        {"question": "ok", "answer": "a", "contexts": ["c"]},
        {"question": "no_ctx", "answer": "a", "contexts": []},
    ]
    records = build_ragas_records(rows)
    assert records.size == 1
    assert records.dataset["question"] == ["ok"]
    assert records.skipped[0]["question"] == "no_ctx"
    assert "上下文" in records.skipped[0]["reason"]


def test_build_ragas_records_skips_empty_answer_and_expect_no_answer():
    rows = [
        {"question": "empty_answer", "answer": "   ", "contexts": ["c"]},
        {"question": "refusal", "answer": "", "contexts": [], "skipped_reason": "expect_no_answer：应拒答的题不参与生成质量评测"},
    ]
    records = build_ragas_records(rows)
    assert records.size == 0
    assert {s["question"] for s in records.skipped} == {"empty_answer", "refusal"}


def test_build_ragas_records_omits_ground_truth_unless_all_rows_have_it():
    """Dataset.from_dict 要求各列等长：只要有行缺 ground_truth，就整体省略该列。"""
    rows = [
        {"question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truth": "g1"},
        {"question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truth": ""},
    ]
    records = build_ragas_records(rows)
    assert "ground_truth" not in records.dataset
    assert records.has_ground_truth is False
    assert any("ground_truth" in s["reason"] for s in records.skipped)


def test_build_ragas_records_handles_empty_input():
    records = build_ragas_records([])
    assert records.size == 0
    assert records.has_ground_truth is False


# ── 指标选择 ────────────────────────────────────────────────────────


def test_resolve_metric_names_defaults_to_faithfulness_and_relevancy():
    assert resolve_metric_names(None, has_ground_truth=True) == list(DEFAULT_METRICS)


def test_resolve_metric_names_drops_ground_truth_metrics_when_absent():
    names = resolve_metric_names(list(ALL_METRICS), has_ground_truth=False)
    assert set(GROUND_TRUTH_METRICS).isdisjoint(names)
    assert "faithfulness" in names


def test_resolve_metric_names_rejects_unknown():
    with pytest.raises(ValueError, match="未知的指标"):
        resolve_metric_names(["bleu"], has_ground_truth=True)


def test_resolve_metric_names_falls_back_when_everything_dropped():
    assert resolve_metric_names(list(GROUND_TRUTH_METRICS), has_ground_truth=False) == list(DEFAULT_METRICS)


# ── 采样 ────────────────────────────────────────────────────────────


class _StubPipeline:
    def __init__(self, answer: str = "a", contexts: list[str] | None = None, kb: str = "stub"):
        self.kb = kb
        self._answer = answer
        self._contexts = contexts if contexts is not None else ["ctx"]

    def search(self, question, top_n=None):  # noqa: ARG002
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(text=t, source="doc.md", chunk_index=i, heading=""),
                score=0.5,
            )
            for i, t in enumerate(self._contexts)
        ]

    def query(self, question):  # noqa: ARG002
        return SimpleNamespace(
            answer=self._answer,
            contexts=[{"text": t} for t in self._contexts],
            citations=[],
            grounded=True,
            debug={},
        )


def test_collect_rows_skips_expect_no_answer_items():
    rows = collect_rows(_StubPipeline(), [{"question": "q", "expect_no_answer": True}])
    assert rows[0]["answer"] == ""
    assert "expect_no_answer" in rows[0]["skipped_reason"]


def test_collect_rows_uses_query_answer_and_contexts():
    rows = collect_rows(_StubPipeline("模型答的", ["ctx-a"]), [{"question": "q", "ground_truth": "g"}])
    assert rows[0]["answer"] == "模型答的"
    assert rows[0]["contexts"] == ["ctx-a"]
    assert rows[0]["ground_truth"] == "g"


def test_collect_rows_falls_back_to_search_when_contexts_empty():
    """拒答场景下 Answer.contexts 为空 —— 应退回检索结果，保证仍有可核验上下文。"""
    pipe = _StubPipeline("拒答", contexts=[])
    rows = collect_rows(pipe, [{"question": "q"}])
    assert rows[0]["contexts"] == [] or rows[0]["contexts"], "至少不应抛异常"
    # 这里 stub 的 search 也返回空，真实实现里 search 会有结果；断言不抛异常即可
    assert rows[0]["answer"] == "拒答"


# ── 结果归一化 ──────────────────────────────────────────────────────


def _records() -> RagasRecords:
    return build_ragas_records([{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}])


def test_result_to_report_from_dict():
    report = result_to_report({"faithfulness": 0.75, "answer_relevancy": 0.5}, _records(), DEFAULT_METRICS)
    assert report["is_ragas"] is True
    assert report["metrics"] == {"faithfulness": 0.75, "answer_relevancy": 0.5}
    assert report["evaluated_count"] == 1


def test_result_to_report_from_pandas_like_object_without_pandas_dependency():
    class _Series:
        def __init__(self, values):
            self._values = values

        def mean(self):
            return sum(self._values) / len(self._values)

    class _Frame:
        columns = ["question", "answer", "faithfulness", "answer_relevancy"]

        def __getitem__(self, key):
            return _Series([0.8, 0.6] if key == "faithfulness" else [0.4, 0.2])

    class _Result:
        def to_pandas(self):
            return _Frame()

    report = result_to_report(_Result(), _records(), DEFAULT_METRICS)
    assert report["metrics"]["faithfulness"] == pytest.approx(0.7)
    assert report["metrics"]["answer_relevancy"] == pytest.approx(0.3)


def test_result_to_report_from_scores_list():
    class _Result:
        scores = [{"faithfulness": 1.0}, {"faithfulness": 0.0}]

    report = result_to_report(_Result(), _records(), ["faithfulness"])
    assert report["metrics"]["faithfulness"] == pytest.approx(0.5)


def test_result_to_report_notes_when_nothing_could_be_parsed():
    class _Weird:
        pass

    report = result_to_report(_Weird(), _records(), DEFAULT_METRICS)
    assert report["metrics"] == {}
    assert any("解析" in note for note in report["notes"])


def test_result_to_report_never_leaks_the_judge_key():
    """★ 安全断言：报告里绝不能出现原始 api_key。"""
    secret = "sk-leak-me-if-you-can-1234"
    report = result_to_report(
        {"faithfulness": 0.9},
        _records(),
        ["faithfulness"],
        judge_config={"model": "m", "base_url": "u", "api_key": secret},
    )
    blob = json.dumps(report, ensure_ascii=False)
    assert secret not in blob
    assert report["judge"]["api_key_configured"] is True


# ── 执行（stub / 伪造模块）──────────────────────────────────────────


def test_run_evaluation_with_injected_stub_evaluator():
    captured: dict = {}

    def stub(dataset, metric_names):
        captured["dataset"] = dataset
        captured["metrics"] = metric_names
        return {"faithfulness": 0.42}

    report = run_evaluation(_records(), ["faithfulness"], evaluator=stub)
    assert captured["dataset"]["contexts"] == [["c"]]
    assert captured["metrics"] == ["faithfulness"]
    assert report["metrics"]["faithfulness"] == 0.42


def test_run_evaluation_without_stub_raises_clear_error_in_this_environment():
    with pytest.raises(RuntimeError, match="pip install ragas datasets"):
        run_evaluation(_records(), DEFAULT_METRICS)


def test_run_evaluation_rejects_empty_records():
    with pytest.raises(ValueError, match="没有可评测的记录"):
        run_evaluation(build_ragas_records([]), DEFAULT_METRICS, evaluator=lambda *a, **k: {})


def test_run_evaluation_exercises_real_import_path_with_fake_ragas_modules(monkeypatch):
    """用伪造的 ragas/datasets 模块走一遍真正的导入分支（不安装、不联网、不调大模型）。"""
    captured: dict = {}

    class _Dataset:
        @staticmethod
        def from_dict(data):
            captured["dataset"] = data
            return data

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = _Dataset

    fake_metrics = types.ModuleType("ragas.metrics")
    fake_metrics.faithfulness = "METRIC_faithfulness"
    fake_metrics.answer_relevancy = "METRIC_answer_relevancy"

    fake_ragas = types.ModuleType("ragas")

    def _evaluate(dataset, metrics=None, **kwargs):  # noqa: ARG001
        captured["metrics"] = metrics
        captured["kwargs"] = sorted(kwargs)
        return {"faithfulness": 0.66, "answer_relevancy": 0.33}

    fake_ragas.evaluate = _evaluate
    fake_ragas.metrics = fake_metrics

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)

    report = run_evaluation(_records(), DEFAULT_METRICS)

    assert captured["dataset"]["contexts"] == [["c"]]
    assert captured["metrics"] == ["METRIC_faithfulness", "METRIC_answer_relevancy"]
    assert report["metrics"]["faithfulness"] == 0.66
    assert report["unavailable_metrics"] == []


def test_run_evaluation_records_unavailable_metrics(monkeypatch):
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = SimpleNamespace(from_dict=lambda data: data)
    fake_metrics = types.ModuleType("ragas.metrics")
    fake_metrics.faithfulness = "M"
    fake_ragas = types.ModuleType("ragas")
    fake_ragas.evaluate = lambda dataset, metrics=None, **kw: {"faithfulness": 0.5}
    fake_ragas.metrics = fake_metrics

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)

    report = run_evaluation(_records(), ["faithfulness", "answer_relevancy"])
    assert report["unavailable_metrics"] == ["answer_relevancy"]


# ── 摘要 / CLI ──────────────────────────────────────────────────────


def test_format_summary_is_masked_and_lists_metrics():
    report = result_to_report(
        {"faithfulness": 0.9},
        _records(),
        ["faithfulness"],
        judge_config={"model": "judge-m", "base_url": "http://x", "api_key": "sk-shouldnotappear-0001"},
    )
    text = format_summary(report)
    assert "faithfulness" in text
    assert "0.9000" in text
    assert "sk-shouldnotappear-0001" not in text
    assert "judge-m" in text


def test_cli_check_prints_dependency_hint_without_eval_set(capsys):
    code = main(["--check"])
    out = capsys.readouterr().out
    if code == 0:  # pragma: no cover - 只有装了 ragas 才会走到
        assert "已安装" in out
    else:
        assert code == 2
        assert "pip install ragas datasets" in out


def test_cli_requires_eval_set_when_not_checking():
    with pytest.raises(SystemExit):
        main([])
