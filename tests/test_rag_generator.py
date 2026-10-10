"""app/rag/generator.py：上下文拼装、拒答、降级。

三层防幻觉约束（docs/rag-self-built.md 5.12）在本文件逐层锁死：
① 入口拒答：`gate_score` 低于阈值直接返回拒答文案，**不进入大模型**（省钱且最可靠）；
② 不可用降级：没有大模型或调用失败时返回「依据 + 明确提示」，而不是抛异常；
③ 出口识别：模型自己承认没依据时标记 `grounded=False`。
"""

from __future__ import annotations

import pytest

from app.rag.generator import (
    LLM_UNAVAILABLE_ANSWER,
    NO_EVIDENCE_ANSWER,
    AnswerGenerator,
    build_context,
)
from app.rag.types import RetrievedChunk


def _hit(make_chunk, text: str = "风控准确率为 91.5%。", index: int = 0, score: float = 0.8,
         dense: float | None = 0.75, heading: str = "") -> RetrievedChunk:
    item = RetrievedChunk(
        chunk=make_chunk(text=text, index=index, heading=heading), score=score, dense_score=dense
    )
    item.channels = ["dense", "sparse"]
    return item


# ── build_context ───────────────────────────────────────────────────


def test_build_context_numbers_sources(make_chunk):
    context, used = build_context([_hit(make_chunk, "甲", 0), _hit(make_chunk, "乙", 1)])
    assert "[1] 来源:doc.md#0" in context
    assert "[2] 来源:doc.md#1" in context
    assert len(used) == 2


def test_build_context_includes_heading(make_chunk):
    context, _ = build_context([_hit(make_chunk, "甲", heading="第三章 风控")])
    assert "第三章 风控" in context


def test_build_context_truncates_at_whole_chunk_granularity(make_chunk):
    """max_chars 是二次防线：截断以「整条片段」为粒度，不切出半个句子。"""
    big = _hit(make_chunk, "甲" * 100, 0)
    second = _hit(make_chunk, "乙" * 100, 1)
    context, used = build_context([big, second], max_chars=150)
    assert len(used) == 1
    assert "乙" not in context


def test_build_context_keeps_first_chunk_even_if_oversized(make_chunk):
    """第一条即使超长也要保留，否则会退化成"没有任何依据"。"""
    context, used = build_context([_hit(make_chunk, "甲" * 500)], max_chars=10)
    assert len(used) == 1
    assert context


def test_build_context_empty(make_chunk):
    context, used = build_context([])
    assert context == ""
    assert used == []


# ── 拒答（入口门槛）──────────────────────────────────────────────────


def test_no_chunks_yields_refusal():
    answer = AnswerGenerator().generate("问题", [])
    assert answer.answer == NO_EVIDENCE_ANSWER
    assert answer.grounded is False
    assert answer.citations == []


def test_gate_below_threshold_refuses_without_calling_llm(make_chunk, stub_client):
    """★ 核心：低于阈值必须直接拒答，且一次大模型都不调。"""
    client = stub_client()
    answer = AnswerGenerator(client=client, min_score=0.2).generate(
        "公司食堂本周的菜单", [_hit(make_chunk, dense=0.05)], gate_score=0.05
    )
    assert answer.answer == NO_EVIDENCE_ANSWER
    assert answer.grounded is False
    assert answer.debug["reason"] == "score_below_threshold"
    assert answer.debug["gate_score"] == pytest.approx(0.05)
    assert client.calls == [], "命中拒答门槛时绝不能调用大模型"


def test_gate_uses_gate_score_not_rank_score(make_chunk, stub_client):
    """排序分与判定分是两套尺度：判定只看 gate_score（向量余弦）。"""
    client = stub_client()
    # rank(排序)分很高，但 gate(向量余弦)很低 → 必须拒答
    answer = AnswerGenerator(client=client, min_score=0.2).generate(
        "无关问题", [_hit(make_chunk, score=0.95, dense=0.01)], gate_score=0.01
    )
    assert answer.grounded is False
    assert answer.debug["reason"] == "score_below_threshold"
    assert answer.debug["rank_score"] == pytest.approx(0.95)
    assert answer.debug["gate_score"] == pytest.approx(0.01)


def test_missing_gate_score_falls_back_to_rank_score(make_chunk):
    """不传 gate_score 时退回排序分（文档明说：开启重排后阈值需重新标定）。"""
    answer = AnswerGenerator(min_score=0.2).generate("q", [_hit(make_chunk, score=0.1, dense=0.9)])
    assert answer.grounded is False
    assert answer.debug["reason"] == "score_below_threshold"


# ── 正常生成 ────────────────────────────────────────────────────────


def test_successful_generation_populates_citations(make_chunk, stub_client):
    client = stub_client(reply="准确率为 91.5%[1]。")
    answer = AnswerGenerator(client=client, min_score=0.2).generate(
        "风控准确率", [_hit(make_chunk, "风控准确率为 91.5%。", heading="第三章")], gate_score=0.8
    )
    assert answer.answer == "准确率为 91.5%[1]。"
    assert answer.grounded is True
    assert len(client.calls) == 1
    citation = answer.citations[0]
    assert citation["index"] == 1
    assert citation["source"] == "doc.md"
    assert citation["chunk_index"] == 0
    assert citation["heading"] == "第三章"
    assert citation["channels"] == ["dense", "sparse"]
    assert answer.contexts[0]["text"] == "风控准确率为 91.5%。"


def test_system_prompt_carries_hard_rules(make_chunk, stub_client):
    client = stub_client()
    AnswerGenerator(client=client, min_score=0.2).generate("q", [_hit(make_chunk)], gate_score=0.9)
    system_prompt, user_prompt = client.calls[0]
    assert "只使用" in system_prompt and "不要使用你自己的先验知识" in system_prompt
    assert "【用户问题】" in user_prompt
    assert "【参考资料】" in user_prompt


# ── 降级 ────────────────────────────────────────────────────────────


def test_unavailable_client_degrades_to_evidence_only(make_chunk, stub_client):
    """没有大模型时检索价值仍要保留，不能抛异常。"""
    answer = AnswerGenerator(client=stub_client(available=False), min_score=0.2).generate(
        "q", [_hit(make_chunk)], gate_score=0.9
    )
    assert answer.answer == LLM_UNAVAILABLE_ANSWER
    assert answer.grounded is False
    assert answer.debug["reason"] == "llm_unavailable"
    assert answer.citations, "降级时仍应给出引用"


def test_no_client_at_all_degrades(make_chunk):
    answer = AnswerGenerator(client=None, min_score=0.2).generate("q", [_hit(make_chunk)], gate_score=0.9)
    assert answer.debug["reason"] == "llm_unavailable"


def test_client_exception_is_contained(make_chunk, stub_client):
    answer = AnswerGenerator(client=stub_client(error=RuntimeError("模型超时")), min_score=0.2).generate(
        "q", [_hit(make_chunk)], gate_score=0.9
    )
    assert answer.grounded is False
    assert answer.debug["reason"] == "llm_error"
    assert "模型超时" in answer.answer
    assert answer.citations


def test_empty_model_output_treated_as_refusal(make_chunk, stub_client):
    answer = AnswerGenerator(client=stub_client(reply=""), min_score=0.2).generate(
        "q", [_hit(make_chunk)], gate_score=0.9
    )
    assert answer.answer == NO_EVIDENCE_ANSWER


# ── 出口识别（refusal marker）───────────────────────────────────────


@pytest.mark.parametrize(
    "reply",
    ["知识库中没有找到相关信息", "资料中没有相关说明", "无法回答该问题", "文档中没有包含相关内容"],
)
def test_refusal_markers_mark_ungrounded(make_chunk, stub_client, reply):
    answer = AnswerGenerator(client=stub_client(reply=reply), min_score=0.2).generate(
        "q", [_hit(make_chunk)], gate_score=0.9
    )
    assert answer.grounded is False
    assert answer.debug["reason"] == "model_refused"


def test_long_answer_with_marker_stays_grounded(make_chunk, stub_client):
    """长回答里出现「无法回答」多半是在引用资料，不能误判为拒答。"""
    reply = "根据资料，风控准确率为 91.5%[1]。" + "补充说明。" * 60
    answer = AnswerGenerator(client=stub_client(reply=reply), min_score=0.2).generate(
        "q", [_hit(make_chunk)], gate_score=0.9
    )
    assert answer.grounded is True
    assert "reason" not in answer.debug


def test_normal_answer_without_marker_is_grounded(make_chunk, stub_client):
    answer = AnswerGenerator(client=stub_client(reply="准确率为 91.5%[1]。"), min_score=0.2).generate(
        "q", [_hit(make_chunk)], gate_score=0.9
    )
    assert answer.grounded is True
