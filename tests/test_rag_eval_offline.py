"""app/rag/eval_offline.py：零依赖离线「生成质量」近似评测。

重点锁三件事：
1. **幻觉数字识别**（本任务的核心要求）：答案里出现上下文中没有的数字/金额/型号时，
   必须被标记为"疑似幻觉"，而不是静默算作合格；
2. **诚实性**：报告必须自证"非 RAGAS"，没有可核验内容时给 `None` 而不是给 1.0；
3. 评测集解析与报告结构稳定。

全程不联网、不调用大模型（用 stub pipeline）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.rag.eval_offline import (
    OFFLINE_DISCLAIMER,
    build_extractive_answer,
    context_coverage_approx,
    context_faithfulness_approx,
    evaluate_offline,
    extract_facts,
    extract_key_terms,
    format_offline_report,
    load_eval_set,
    main,
    normalize_contexts,
)
from app.rag.types import Chunk, RetrievedChunk

CONTEXTS = [
    "风控模型的准确率为 91.5%。",
    "授信规则要求 A13 型号必须复核，编号 3.2.1。",
]


# ── 事实抽取 ────────────────────────────────────────────────────────


def test_extract_facts_picks_numbers_percent_money_and_ids():
    facts = extract_facts("准确率为 91.5%，金额 1.5 亿元，型号 A13，编号 3.2.1，同比增长 12%")
    for expected in ("91.5%", "1.5 亿元", "A13", "3.2.1", "12%"):
        assert expected in facts


def test_extract_facts_deduplicates():
    assert extract_facts("91.5% 与 91.5% 是一样的") == ["91.5%"]


def test_extract_facts_handles_full_width_percent():
    assert len(extract_facts("占比 3.5％")) == 1


def test_extract_facts_on_empty_or_textless_input():
    assert extract_facts("") == []
    assert extract_facts("这段文字没有任何数字") == []


def test_extract_facts_keeps_thousands_separator():
    assert "1,200 元" in extract_facts("收入 1,200 元")


def test_extract_key_terms_is_wider_than_facts():
    facts = set(extract_facts("风控模型的准确率为 91.5%"))
    terms = set(extract_key_terms("风控模型的准确率为 91.5%"))
    assert facts <= terms
    assert any(term.startswith("风控") for term in terms), "中文长串应按功能字切出词"
    assert any(term.lower() == "gdp" for term in extract_key_terms("GDP 增速"))


def test_extract_key_terms_drops_single_char_noise():
    terms = extract_key_terms("的了在")
    assert all(len(t) >= 2 for t in terms)


# ── 上下文归一 ──────────────────────────────────────────────────────


def test_normalize_contexts_accepts_several_shapes():
    assert normalize_contexts(["甲", "乙"]) == ["甲", "乙"]
    assert normalize_contexts([{"text": "甲"}, {"content": "乙"}]) == ["甲", "乙"]
    assert normalize_contexts("甲") == ["甲"]


def test_normalize_contexts_drops_blanks_and_none():
    assert normalize_contexts(["甲", "", "  ", None]) == ["甲"]
    assert normalize_contexts(None) == []
    assert normalize_contexts([]) == []


# ── 核心：幻觉识别 ──────────────────────────────────────────────────


def test_faithfulness_flags_numbers_absent_from_context():
    """★ 核心要求：答案里的数字若在上下文中找不到，必须被标记为疑似幻觉。"""
    answer = "风控模型准确率为 91.5%，A13 型号返修率为 2.1%，编号 3.2.1，金额 1.5 亿元。"
    result = context_faithfulness_approx(answer, CONTEXTS)

    assert result["checkable"] is True
    assert result["fact_count"] == 5
    # 91.5% / A13 / 3.2.1 在上下文里 → 支持
    assert set(result["supported"]) == {"91.5%", "A13", "3.2.1"}
    # 2.1% / 1.5 亿元 是答案自己编的 → 疑似幻觉
    assert set(result["suspected_hallucinations"]) == {"2.1%", "1.5 亿元"}
    assert result["score"] == pytest.approx(3 / 5)


def test_faithfulness_full_score_when_all_facts_are_grounded():
    result = context_faithfulness_approx("准确率为 91.5%，A13 型号，编号 3.2.1。", CONTEXTS)
    assert result["score"] == 1.0
    assert result["suspected_hallucinations"] == []


def test_faithfulness_is_none_when_answer_has_nothing_checkable():
    """没有可核验内容时必须给 None，不能给 1.0 —— 否则等于把"空话"判成满分。"""
    result = context_faithfulness_approx("知识库中没有找到与该问题相关的资料。", CONTEXTS)
    assert result["score"] is None
    assert result["checkable"] is False
    assert result["fact_count"] == 0


def test_faithfulness_is_whitespace_and_case_insensitive():
    """上下文写 "1.5 亿元"、答案写 "1.5亿元" 时应判为一致，而不是误报幻觉。"""
    result = context_faithfulness_approx("金额 1.5亿元", ["报告显示金额为 1.5 亿元。"])
    assert result["suspected_hallucinations"] == []
    assert result["score"] == 1.0


def test_faithfulness_with_no_contexts_flags_everything():
    result = context_faithfulness_approx("准确率为 91.5%", [])
    assert result["score"] == 0.0
    assert result["suspected_hallucinations"] == ["91.5%"]


def test_coverage_reports_missing_terms():
    result = context_coverage_approx("风控模型准确率为 91.5%，另外提到量子纠错码。", CONTEXTS)
    assert 0.0 <= result["score"] < 1.0
    assert result["term_count"] > 0
    assert result["missing"], "未覆盖的术语应列出来"


def test_coverage_is_none_without_terms():
    assert context_coverage_approx("", CONTEXTS)["score"] is None


# ── 评测集 ──────────────────────────────────────────────────────────


def test_load_eval_set_from_json_array(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"question": "q1", "ground_truth": "g1"}]), encoding="utf-8")
    items = load_eval_set(path)
    assert items[0]["question"] == "q1"
    assert items[0]["ground_truth"] == "g1"
    assert items[0]["expect_no_answer"] is False


def test_load_eval_set_from_items_object_and_jsonl(tmp_path):
    obj = tmp_path / "a.json"
    obj.write_text(json.dumps({"items": [{"question": "q"}]}), encoding="utf-8")
    assert load_eval_set(obj)[0]["question"] == "q"

    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text('{"question": "q1"}\n\n{"question": "q2"}\n', encoding="utf-8")
    assert [i["question"] for i in load_eval_set(jsonl)] == ["q1", "q2"]


def test_load_eval_set_accepts_sources_alias(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps([{"question": "q", "sources": "a.md"}]), encoding="utf-8")
    assert load_eval_set(path)[0]["expected_sources"] == ["a.md"]


def test_load_eval_set_rejects_missing_question_and_missing_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"ground_truth": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_eval_set(bad)
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "nope.json")


def test_example_eval_set_shipped_with_repo_is_loadable():
    """交付的示例评测集必须自身可加载（防止示例文件写错）。"""
    example = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "ragas_eval_set.example.json"
    items = load_eval_set(example)
    assert len(items) >= 3
    assert any(i["expect_no_answer"] for i in items)
    assert any(i["ground_truth"] for i in items)


# ── 抽取式基线答案 ──────────────────────────────────────────────────


def test_build_extractive_answer_concatenates_and_truncates():
    chunks = [Chunk(chunk_id="a", text="甲" * 10, source="s", source_path="s", chunk_index=0),
              Chunk(chunk_id="b", text="乙" * 10, source="s", source_path="s", chunk_index=1)]
    answer = build_extractive_answer(chunks, max_chars=15)
    assert answer.startswith("甲")
    assert len(answer) <= 15
    assert "乙" not in answer, "超出 max_chars 后不应再拼接下一段"


def test_build_extractive_answer_empty():
    assert build_extractive_answer([]) == ""


# ── 端到端（stub pipeline，不联网）──────────────────────────────────


class _StubPipeline:
    """只实现 evaluate_offline 用到的那几个接口。"""

    def __init__(self, answer: str, contexts: list[str], grounded: bool = True, kb: str = "stub"):
        self.kb = kb
        self.embedder = SimpleNamespace(fingerprint="hash:512")
        self.config = SimpleNamespace(min_score=0.2)
        self._answer = answer
        self._contexts = contexts
        self._grounded = grounded

    def _chunks(self) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=Chunk(chunk_id=f"c{i}", text=text, source="doc.md", source_path="doc.md", chunk_index=i),
                score=0.9 - i * 0.1,
            )
            for i, text in enumerate(self._contexts)
        ]

    def search(self, question, top_n=None):  # noqa: ARG002
        return self._chunks()

    def query(self, question):  # noqa: ARG002
        return SimpleNamespace(
            question=question,
            answer=self._answer,
            contexts=[{"index": i + 1, "text": t, "source": "doc.md"} for i, t in enumerate(self._contexts)],
            citations=[{"index": 1, "source": "doc.md", "chunk_index": 0, "heading": "", "score": 0.9}],
            grounded=self._grounded,
            debug={"gate_score": 0.8} if self._grounded else {"reason": "score_below_threshold"},
        )


def test_evaluate_offline_report_marks_itself_as_not_ragas():
    """报告必须自证不是 RAGAS，禁止被当成 RAGAS 结果消费。"""
    pipe = _StubPipeline("准确率为 91.5%。", CONTEXTS)
    report = evaluate_offline([{"question": "准确率是多少", "expect_no_answer": False}], pipe)

    assert report["report_type"] == "offline_approximation"
    assert report["is_ragas"] is False
    assert report["not_ragas"] is True
    assert report["disclaimer"] == OFFLINE_DISCLAIMER
    assert "RAGAS" in report["disclaimer"]
    assert "faithfulness" in report["ragas_metrics_not_included"]
    assert report["metrics"]["context_faithfulness_approx"] == 1.0


def test_evaluate_offline_flags_hallucination_in_report():
    pipe = _StubPipeline("准确率为 99.9%，A13 型号。", CONTEXTS)
    report = evaluate_offline([{"question": "准确率是多少"}], pipe)
    assert report["metrics"]["suspected_hallucination_rate"] == 1.0
    assert "99.9%" in report["per_query"][0]["suspected_hallucinations"]


def test_evaluate_offline_extractive_mode_marks_refusal_as_na_and_explains():
    """抽取式基线不会拒答 —— 拒答率必须记 N/A 并给出解释，不能记 0% 让人误读成失败。"""
    pipe = _StubPipeline("甲", CONTEXTS)
    report = evaluate_offline(
        [{"question": "食堂菜单", "expect_no_answer": True}], pipe, answer_mode="extractive"
    )
    assert report["metrics"]["refusal_accuracy"] is None
    assert any("extractive" in note for note in report["notes"])
    assert any("100%" in note for note in report["notes"])


def test_evaluate_offline_pipeline_mode_counts_correct_refusal():
    pipe = _StubPipeline("知识库中没有找到与该问题相关的资料。", CONTEXTS, grounded=False)
    report = evaluate_offline([{"question": "食堂菜单", "expect_no_answer": True}], pipe, answer_mode="pipeline")
    assert report["metrics"]["refusal_accuracy"] == 1.0
    assert report["per_query"][0]["refused"] is True
    assert report["counts"]["no_answer"] == 1


def test_evaluate_offline_rejects_unknown_answer_mode():
    with pytest.raises(ValueError, match="answer_mode"):
        evaluate_offline([], _StubPipeline("x", CONTEXTS), answer_mode="magic")


def test_format_offline_report_shows_warning_and_metrics():
    pipe = _StubPipeline("准确率为 91.5%。", CONTEXTS)
    text = format_offline_report(evaluate_offline([{"question": "q"}], pipe))
    assert "非 RAGAS" in text
    assert "context_faithfulness_approx" in text
    assert "100.00%" in text


# ── CLI ─────────────────────────────────────────────────────────────


def test_cli_writes_report_and_returns_zero(tmp_path, monkeypatch, capsys):
    import app.rag.pipeline as pipeline_module

    pipe = _StubPipeline("准确率为 91.5%。", CONTEXTS)
    monkeypatch.setattr(pipeline_module, "get_pipeline", lambda kb=None: pipe)

    eval_set = tmp_path / "s.json"
    eval_set.write_text(json.dumps([{"question": "准确率是多少"}]), encoding="utf-8")
    out = tmp_path / "report.json"

    code = main(["-e", str(eval_set), "-k", "stub", "--answer-mode", "extractive", "-o", str(out)])
    assert code == 0
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["is_ragas"] is False
    assert written["kb"] == "stub"
    assert "非 RAGAS" in capsys.readouterr().out
