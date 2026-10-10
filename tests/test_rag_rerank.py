"""app/rag/rerank.py：重排（词面 / 大模型 / 关闭）。

本文件锁死历史 bug（docs/rag-self-built.md 5.4）：
查询「风控模型的准确率是多少」与文档「准确率为 91.5%」在 jieba 下切分不一致
（准确/率 vs 准确率为），只按分词 token 做子串匹配会**一个都命中不了**，
于是所有候选重排分都是 0、名次完全不变 —— 等于没重排，且不报任何错。

实现用两级兜底修掉了它：① 特征词额外加中文相邻二元组；② 再加字符级覆盖率。
两条兜底分别单独锁死，避免有人"简化"掉其中一条时回归。
"""

from __future__ import annotations

import pytest

from app.rag.rerank import (
    LexicalReranker,
    LLMReranker,
    NoopReranker,
    Reranker,
    build_reranker,
)
from app.rag.types import RetrievedChunk


def _cand(make_chunk, text: str, rank: int, score: float = 0.0, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(text=text, index=index), score=score, rank=rank)


# ── 历史 bug 回归：中文分词差异不得让重排整体失效 ───────────────────


def test_feature_extraction_adds_adjacent_bigrams():
    """兜底①：即使 jieba 把「准确率」切成别的样子，二元组也能绕开分词分歧。"""
    features = set(LexicalReranker()._features("准确率"))
    assert {"准确", "确率"} <= features


def test_char_coverage_ignores_low_signal_chars():
    """兜底②：字符级覆盖率（剔除无区分度单字）。"""
    hit, total = LexicalReranker()._char_coverage("准确率", "准确率为")
    assert (hit, total) == (3, 3)


def test_score_for_split_words_is_not_zero():
    """★ 核心回归锁：查询「准确率」对文档「准确率为」的重排分必须 > 0。

    失效时这里恒为 0.0，且名次不变、不抛异常 —— 正是当年没被发现的原因。
    """
    score = LexicalReranker().score("风控模型的准确率是多少", "风控模型的准确率为 91.5%。")
    assert score > 0.0, "重排分归零即代表中文分词差异已让重排整体失效"


def test_relevantly_matching_candidate_outranks_irrelevant_one():
    assert LexicalReranker().score("风控模型的准确率是多少", "风控模型的准确率为 91.5%。") > \
        LexicalReranker().score("风控模型的准确率是多少", "公司食堂本周菜单：红烧肉、青菜。")


def test_rerank_actually_reorders_candidates(make_chunk):
    """端到端：把相关候选放在最后，重排后必须把它提到第一。"""
    irrelevant = _cand(make_chunk, "公司食堂本周菜单：红烧肉、青菜。", rank=1, score=0.5, index=1)
    relevant = _cand(make_chunk, "风控模型的准确率为 91.5%。", rank=2, score=0.3, index=2)

    result = LexicalReranker().rerank("风控模型的准确率是多少", [irrelevant, relevant], top_n=2)
    assert result[0].chunk is relevant.chunk
    assert result[0].rank == 1
    assert result[1].rank == 2


def test_empty_query_scores_zero(make_chunk):
    assert LexicalReranker().score("", "任意内容") == 0.0


# ── NoopReranker / 通用契约 ─────────────────────────────────────────


def test_noop_reranker_keeps_fusion_order_and_truncates(make_chunk):
    candidates = [_cand(make_chunk, f"c{i}", rank=i + 1, index=i) for i in range(5)]
    result = NoopReranker().rerank("q", candidates, top_n=3)
    assert [item.chunk.text for item in result] == ["c0", "c1", "c2"]
    assert [item.rank for item in result] == [1, 2, 3]


def test_rerank_empty_candidates_returns_empty():
    assert LexicalReranker().rerank("q", [], top_n=3) == []
    assert NoopReranker().rerank("q", [], top_n=3) == []


def test_rerank_truncates_to_top_n(make_chunk):
    candidates = [_cand(make_chunk, f"风控内容{i}", rank=i + 1, index=i) for i in range(6)]
    assert len(LexicalReranker().rerank("风控", candidates, top_n=2)) == 2


def test_ties_keep_original_fusion_rank(make_chunk):
    """同分时必须保留融合名次（稳定排序），不能把双路共识的候选无意义打乱。"""
    candidates = [_cand(make_chunk, "内容", rank=1, index=0), _cand(make_chunk, "内容", rank=2, index=1)]
    result = LexicalReranker().rerank("完全无关的查询词", candidates, top_n=2)
    assert [item.rank for item in result] == [1, 2]


# ── LLMReranker ─────────────────────────────────────────────────────


def test_parse_order_drops_invalid_and_duplicate_numbers():
    assert LLMReranker._parse_order("3,1,99,1,0", size=3) == [3, 1]
    assert LLMReranker._parse_order("没有数字", size=3) == []
    assert LLMReranker._parse_order("2 2 2", size=3) == [2]


def test_llm_rerank_reorders_by_model_output(make_chunk, stub_openai):
    client = stub_openai(content="3,1")
    candidates = [_cand(make_chunk, f"c{i}", rank=i + 1, index=i) for i in range(3)]
    result = LLMReranker(client=client, model="stub-model").rerank("q", candidates, top_n=3)
    assert [item.chunk.text for item in result] == ["c2", "c0", "c1"]


def test_llm_rerank_keeps_candidates_not_mentioned_by_model(make_chunk, stub_openai):
    """模型没排到的候选必须补在后面，绝不能丢候选。"""
    client = stub_openai(content="3")
    candidates = [_cand(make_chunk, f"c{i}", rank=i + 1, index=i) for i in range(3)]
    result = LLMReranker(client=client, model="m").rerank("q", candidates, top_n=3)
    assert [item.chunk.text for item in result] == ["c2", "c0", "c1"]


def test_llm_rerank_fails_open_on_exception(make_chunk, stub_openai):
    """重排是优化项：模型输出异常不能让整个问答失败，保持原顺序。"""
    client = stub_openai(error=RuntimeError("接口炸了"))
    candidates = [_cand(make_chunk, f"c{i}", rank=i + 1, index=i) for i in range(3)]
    result = LLMReranker(client=client, model="m").rerank("q", candidates, top_n=3)
    assert [item.chunk.text for item in result] == ["c0", "c1", "c2"]


def test_llm_rerank_marks_channel(make_chunk, stub_openai):
    candidates = [_cand(make_chunk, f"c{i}", rank=i + 1, index=i) for i in range(2)]
    result = LLMReranker(client=stub_openai(content="2,1"), model="m").rerank("q", candidates, top_n=2)
    assert all("llm_rerank" in item.channels for item in result)


def test_llm_rerank_single_candidate_short_circuits(make_chunk, stub_openai):
    client = stub_openai()
    result = LLMReranker(client=client, model="m").rerank("q", [_cand(make_chunk, "c0", rank=1)], top_n=1)
    assert len(result) == 1
    assert client.calls == [], "只有一个候选时不该浪费一次大模型调用"


# ── 工厂分派 ────────────────────────────────────────────────────────


def test_build_reranker_dispatch():
    assert isinstance(build_reranker("off"), NoopReranker)
    assert isinstance(build_reranker("lexical"), LexicalReranker)
    assert isinstance(build_reranker("LEXICAL"), LexicalReranker)  # 大小写不敏感
    assert isinstance(build_reranker(""), NoopReranker)  # 空值默认 off


def test_build_reranker_llm_without_client_falls_back_to_lexical():
    """没有可用大模型时必须降级而不是报错（llm 是优化项，不是必需品）。"""
    assert isinstance(build_reranker("llm", llm_client=None, llm_model="m"), LexicalReranker)
    assert isinstance(build_reranker("llm", llm_client=object(), llm_model=""), LexicalReranker)


def test_build_reranker_llm_with_client(stub_openai):
    reranker = build_reranker("llm", llm_client=stub_openai(), llm_model="m")
    assert isinstance(reranker, LLMReranker)


def test_build_reranker_unknown_mode_raises():
    with pytest.raises(ValueError, match="未知的重排模式"):
        build_reranker("cross-encoder")


def test_all_rerankers_implement_interface():
    assert issubclass(NoopReranker, Reranker)
    assert issubclass(LexicalReranker, Reranker)
    assert issubclass(LLMReranker, Reranker)
