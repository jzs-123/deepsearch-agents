"""app/rag/eval.py：检索质量评测。

本文件锁死两条历史 bug（docs/rag-self-built.md 5.11「评测脚本自身的两个 bug」）——
**评测脚本是尺子，尺子本身必须先校准**：

* **假失败**：入库时 `source_path` 是相对路径（`sub_dir/报告.md`），评测集里只写文件名
  （`报告.md`），用 `==` 比对永远不相等 → Hit@5 报成 0.00%，而检索其实完全正确。
  解法：来源按**文件名**归一化比对（忽略目录与大小写）。见 `test_matches_*`。

* **假通过**：拒答判定用了排序分，与线上真实判定（`grounded_score()` 的向量余弦）不一致，
  于是「评测通过了但线上行为不一样」。解法：与线上共用 `grounded_score()`。
  见 `test_refusal_uses_vector_cosine_not_rank_score`。
"""

from __future__ import annotations

import json

import pytest

from app.rag.eval import EvalItem, evaluate, format_report, load_eval_set

KB_DOC = (
    "# 风控\n\n"
    "风控模型的准确率为 91.5%。\n\n"
    "## 授信\n\n"
    "授信规则要求 A13 型号必须复核，编号 3.2.1。"
)
RELEVANT_Q = "风控模型的准确率是多少"
IRRELEVANT_Q = "公司食堂本周的菜单"


# ── 来源归一化与比对（历史 bug：假失败）────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sub_dir/报告.md", "报告.md"),
        ("sub_dir\\报告.md", "报告.md"),
        ("报告.MD", "报告.md"),
        ("  报告.md  ", "报告.md"),
        ("a/b/c/报告.md", "报告.md"),
    ],
)
def test_norm_source_strips_directory_case_and_whitespace(raw, expected):
    from app.rag.eval import _norm_source

    assert _norm_source(raw) == expected


def test_matches_ignores_directory_prefix():
    """★ 回归锁：入库是相对路径、评测集只写文件名时，必须判定为命中。"""
    from app.rag.eval import _matches

    assert _matches("sub_dir/报告.md", ["报告.md"]) is True
    assert _matches("报告.md", ["sub_dir/报告.md"]) is True


def test_matches_ignores_case_and_separators():
    from app.rag.eval import _matches

    assert _matches("sub_dir\\报告.MD", ["报告.md"]) is True


def test_matches_rejects_unrelated_source():
    from app.rag.eval import _matches

    assert _matches("报告.md", ["白皮书.pdf"]) is False


# ── EvalItem / load_eval_set ────────────────────────────────────────


def test_eval_item_accepts_sources_alias_and_string():
    assert EvalItem.from_dict({"question": "q", "sources": "a.md"}).expected_sources == ["a.md"]
    assert EvalItem.from_dict({"question": "q", "expected_sources": ["a.md"]}).expected_sources == ["a.md"]


def test_eval_item_defaults():
    item = EvalItem.from_dict({"question": "q"})
    assert item.expected_sources == []
    assert item.expect_no_answer is False
    assert EvalItem.from_dict({"question": "q", "expect_no_answer": 1}).expect_no_answer is True


def test_load_eval_set_from_json_array(tmp_path):
    path = tmp_path / "set.json"
    path.write_text(json.dumps([{"question": "q1", "expected_sources": ["a.md"]}]), encoding="utf-8")
    items = load_eval_set(path)
    assert len(items) == 1
    assert items[0].question == "q1"


def test_load_eval_set_from_json_object_with_items(tmp_path):
    path = tmp_path / "set.json"
    path.write_text(json.dumps({"items": [{"question": "q1"}]}), encoding="utf-8")
    assert load_eval_set(path)[0].question == "q1"


def test_load_eval_set_from_jsonl(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text('{"question": "q1"}\n\n{"question": "q2"}\n', encoding="utf-8")
    assert [item.question for item in load_eval_set(path)] == ["q1", "q2"]


def test_load_eval_set_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "nope.json")


# ── evaluate ────────────────────────────────────────────────────────


def test_evaluate_hit_when_expected_source_only_written_as_filename(pipeline):
    """★ 假失败回归锁：检索正确 + 评测集只写文件名 → Hit@k 必须是 100%，不是 0%。"""
    pipeline.add_texts([KB_DOC], source="doc.md")
    result = evaluate(
        pipeline, [EvalItem(question=RELEVANT_Q, expected_sources=["sub_dir/DOC.MD"])], k=5
    )
    assert result.hit_at_k == 1.0
    assert result.mrr == 1.0
    assert result.recall_at_k == 1.0
    assert result.per_query[0]["hit"] is True
    assert result.per_query[0]["first_hit_rank"] == 1


def test_evaluate_miss_when_source_unrelated(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    result = evaluate(pipeline, [EvalItem(question=RELEVANT_Q, expected_sources=["其他.pdf"])], k=5)
    assert result.hit_at_k == 0.0
    assert result.per_query[0]["first_hit_rank"] == 0


def test_evaluate_recall_uses_all_expected_sources(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    result = evaluate(
        pipeline,
        [EvalItem(question=RELEVANT_Q, expected_sources=["doc.md", "缺失的.pdf"])],
        k=5,
    )
    assert result.recall_at_k == pytest.approx(0.5)


def test_refusal_uses_vector_cosine_not_rank_score(pipeline, make_chunk, monkeypatch):
    """★ 假通过回归锁：拒答判定必须与线上 `grounded_score()` 一致（向量余弦）。

    刻意造一个「排序分 0.95 越过阈值、向量余弦 0.01 远低于阈值」的候选。
    若评测退回去用排序分，这条 `expect_no_answer` 会被判成失败，
    refusal_accuracy 就从 1.0 掉下来 —— 正是当年"评测通过了但线上行为不一样"的成因。
    """
    from app.rag.types import RetrievedChunk

    crafted = [
        RetrievedChunk(chunk=make_chunk(text="无关内容"), score=0.95, dense_score=0.01, rank=1)
    ]
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: crafted)

    result = evaluate(pipeline, [EvalItem(question=IRRELEVANT_Q, expect_no_answer=True)], k=5)
    assert result.refusal_accuracy == 1.0
    assert result.per_query[0]["passed"] is True
    assert result.per_query[0]["top_score"] == pytest.approx(0.01)


def test_refusal_miss_when_cosine_above_threshold(pipeline, make_chunk, monkeypatch):
    """反向对照：余弦越过阈值时必须算作"没拒答"（防止上一条恒真而假通过）。"""
    from app.rag.types import RetrievedChunk

    crafted = [
        RetrievedChunk(chunk=make_chunk(text="相关内容"), score=0.95, dense_score=0.9, rank=1)
    ]
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: crafted)

    result = evaluate(pipeline, [EvalItem(question=IRRELEVANT_Q, expect_no_answer=True)], k=5)
    assert result.refusal_accuracy == 0.0


def test_refusal_counts_miss_when_cosine_above_threshold(pipeline):
    """反向：真正有依据的问题被标成"应拒答"时必须算失败（避免恒真）。"""
    pipeline.add_texts([KB_DOC], source="doc.md")
    result = evaluate(pipeline, [EvalItem(question=RELEVANT_Q, expect_no_answer=True)], k=5)
    assert result.refusal_accuracy == 0.0


def test_evaluate_empty_items_gives_zero_metrics(pipeline):
    result = evaluate(pipeline, [], k=5)
    assert result.total == 0
    assert result.hit_at_k == 0.0
    assert result.refusal_accuracy == 0.0


def test_evaluate_records_config_for_reproducibility(pipeline):
    result = evaluate(pipeline, [], k=7)
    assert result.config["k"] == 7
    assert result.config["rrf_k"] == pipeline.config.rrf_k
    assert result.config["embedding"].startswith("hash:")


def test_evaluate_does_not_generate_answers(pipeline, monkeypatch, stub_client):
    """评测的是检索层，绝不能顺带调用生成模型（生成质量波动会污染指标）。

    注意：`search()` 会把 `self._llm_client()` 作为实参**求值**后传给 `build_reranker`，
    所以断言对象是「没有发起生成调用」，而不是「没有构造客户端」。
    """
    pipeline.add_texts([KB_DOC], source="doc.md")
    stub = stub_client()
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)
    evaluate(pipeline, [EvalItem(question=RELEVANT_Q, expected_sources=["doc.md"])], k=5)
    assert stub.calls == []


# ── format_report ───────────────────────────────────────────────────


def test_format_report_contains_metrics_and_rows(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    result = evaluate(
        pipeline,
        [
            EvalItem(question=RELEVANT_Q, expected_sources=["doc.md"]),
            EvalItem(question=IRRELEVANT_Q, expect_no_answer=True),
        ],
        k=5,
    )
    text = format_report(result)
    assert "RAG 检索评测报告" in text
    assert "Hit@5" in text
    assert "MRR" in text
    assert "拒答准确率" in text
    assert "应拒答" in text
    assert "[HIT" in text


def test_format_report_marks_miss(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    text = format_report(evaluate(pipeline, [EvalItem(question=RELEVANT_Q, expected_sources=["无.pdf"])], k=5))
    assert "[MISS" in text
