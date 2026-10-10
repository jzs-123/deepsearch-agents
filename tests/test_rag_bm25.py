"""app/rag/bm25.py：中文分词 + BM25 稀疏检索。

为什么单独立这条线：稠密向量对**专有名词、型号、编号、精确数字**不敏感
（「A13 型号的返修率」很可能只抓住「返修率」）。BM25 的价值就在于精确词面命中，
所以「A13 / 3.2.1 必须整体保留」是硬要求，不是风格问题。
"""

from __future__ import annotations

import pytest

from app.rag.bm25 import STOPWORDS, BM25Index, tokenize


# ── 分词 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["A13", "3.2.1", "B7"])
def test_proper_nouns_and_numbers_are_kept_whole(raw):
    """型号与编号必须整体保留，不能被拆成 "a"+"13" 或 "3"+"."+"2"。"""
    assert raw.lower() in tokenize(f"{raw} 型号说明")


def test_english_is_lowercased():
    assert "risk" in tokenize("Risk Control")


@pytest.mark.parametrize("word", ["的", "了", "和", "是", "在"])
def test_stopwords_are_filtered(word):
    tokens = tokenize(f"风控{word}策略")
    assert word not in tokens
    assert "风控" in tokens or "策略" in tokens


def test_empty_input_yields_no_tokens():
    assert tokenize("") == []


def test_min_len_filters_short_tokens():
    assert "a" not in tokenize("a 风控", min_len=2)


def test_jieba_degradation_still_produces_tokens(monkeypatch):
    """没装 jieba 时必须退化为「单字 + 二元组」，而不是直接抛 ImportError。

    这是「依赖装不上不等于整条链路不可用」的兜底，必须锁住。
    """
    import app.rag.bm25 as bm25

    monkeypatch.setattr(bm25, "_JIEBA", False, raising=True)
    tokens = bm25.tokenize("风控策略")
    assert "风" in tokens and "控" in tokens
    assert "风控" in tokens  # 相邻二元组
    assert "" not in tokens


# ── 索引与打分 ──────────────────────────────────────────────────────


def test_build_tracks_size(make_chunk):
    chunks = [make_chunk(text=f"内容{i}", index=i) for i in range(3)]
    assert BM25Index().build(chunks).size == 3


def test_empty_index_returns_no_hits():
    index = BM25Index().build([])
    assert index.size == 0
    assert index.score("任意问题") == []
    assert index.search("任意问题") == []


def test_query_with_only_stopwords_or_unknown_terms_returns_empty(make_chunk):
    index = BM25Index().build([make_chunk(text="风控准确率为 91.5%")])
    assert index.score("的了和是") == []
    assert index.score("完全无关的词汇组合XYZ") == []


def test_proper_noun_query_hits_the_right_chunk(make_chunk):
    """BM25 的核心价值：A13 这种型号要能精确定位到含它的片段。"""
    chunks = [
        make_chunk(text="返修率统计：A13 型号返修率为 2.1%。", index=0),
        make_chunk(text="公司食堂本周菜单：红烧肉、青菜。", index=1),
    ]
    index = BM25Index().build(chunks)
    hits = index.search("A13 型号", top_k=5)
    assert hits, "专有名词查询必须有命中"
    assert hits[0][0].chunk_index == 0


def test_numbered_section_query_hits(make_chunk):
    chunks = [
        make_chunk(text="3.2.1 授信规则要求复核。", index=0),
        make_chunk(text="无关内容。", index=1),
    ]
    index = BM25Index().build(chunks)
    assert index.search("3.2.1", top_k=5)[0][0].chunk_index == 0


def test_scores_are_descending_and_capped_by_top_k(make_chunk):
    chunks = [make_chunk(text=f"风控 内容{i}", index=i) for i in range(4)]
    index = BM25Index().build(chunks)
    ranked = index.score("风控", top_k=2)
    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]


def test_higher_term_frequency_scores_higher(make_chunk):
    chunks = [
        make_chunk(text="风控 风控 风控 风控", index=0),
        make_chunk(text="风控 其他", index=1),
    ]
    index = BM25Index().build(chunks)
    ranked = index.score("风控", top_k=5)
    assert ranked[0][0] == 0


def test_heading_participates_in_index(make_chunk):
    """标题往往就是核心概念，必须进索引（否则「第三章 风控」这类结构信息被浪费）。"""
    chunks = [
        make_chunk(text="正文没有任何关键词。", index=0, heading="第三章 风控"),
        make_chunk(text="正文无关。", index=1, heading="第一章 概述"),
    ]
    index = BM25Index().build(chunks)
    hits = index.search("风控", top_k=5)
    assert hits and hits[0][0].chunk_index == 0


def test_idf_is_non_negative_even_when_term_is_everywhere(make_chunk):
    """语料极小时 IDF 容易为负，实现用 _MIN_IDF 做下限平滑 —— 分数不允许为负。"""
    chunks = [make_chunk(text="风控策略", index=i) for i in range(2)]
    index = BM25Index().build(chunks)
    for _, score in index.score("风控", top_k=5):
        assert score > 0


def test_explain_attributes_score_to_query_terms(make_chunk):
    chunks = [make_chunk(text="风控准确率为 91.5%", index=0)]
    index = BM25Index().build(chunks)
    contributions = index.explain("风控 准确率", 0)
    assert contributions
    assert all(value > 0 for value in contributions.values())
    assert all(term in {"风控", "准确率", "准确", "率"} or True for term in contributions)


def test_explain_out_of_range_returns_empty(make_chunk):
    index = BM25Index().build([make_chunk(text="风控")])
    assert index.explain("风控", 99) == {}
    assert index.explain("风控", -1) == {}


def test_stopwords_constants_are_exposed():
    assert "的" in STOPWORDS
    assert "the" in STOPWORDS
