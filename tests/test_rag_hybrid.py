"""app/rag/hybrid.py：稠密 + 稀疏的 RRF 融合。

RRF 只用「名次」不用「分数」，正是为了免除量纲问题（余弦在 [-1,1]，BM25 无上界）。
因此这里断言的是**名次融合的算术**与 `channels`/`dense_score`/`sparse_score` 的归属，
而不是某个绝对分数阈值 —— 后者会随语料漂移。
"""

from __future__ import annotations

import pytest

from app.rag.hybrid import (
    DEFAULT_CHANNEL_WEIGHTS,
    hybrid_search,
    reciprocal_rank_fusion,
)


def _pair(chunk, score: float):
    return (chunk, score)


def _entry(fused, chunk_id: str):
    return next(item for item in fused if item.chunk.chunk_id == chunk_id)


# ── 三态：只有稠密 / 只有稀疏 / 两路都命中 ──────────────────────────


def test_dense_only_hit_records_dense_channel(make_chunk):
    chunk = make_chunk(text="稠密命中", index=0)
    fused = reciprocal_rank_fusion({"dense": [_pair(chunk, 0.9)]}, k=60)
    item = _entry(fused, chunk.chunk_id)
    assert item.channels == ["dense"]
    assert item.dense_score == pytest.approx(0.9)
    assert item.sparse_score is None
    assert item.rrf_score == pytest.approx(1.0 / 61)


def test_sparse_only_hit_records_sparse_channel(make_chunk):
    chunk = make_chunk(text="稀疏命中", index=0)
    fused = reciprocal_rank_fusion({"sparse": [_pair(chunk, 7.5)]}, k=60)
    item = _entry(fused, chunk.chunk_id)
    assert item.channels == ["sparse"]
    assert item.sparse_score == pytest.approx(7.5)
    assert item.dense_score is None
    assert item.rrf_score == pytest.approx(0.8 / 61)  # 稀疏默认权重 0.8


def test_two_channel_hit_accumulates_both_scores(make_chunk):
    """两路都命中 = 「双路共识」，分数应为两路贡献之和。"""
    chunk = make_chunk(text="两路都命中", index=0)
    fused = reciprocal_rank_fusion(
        {"dense": [_pair(chunk, 0.8)], "sparse": [_pair(chunk, 4.2)]}, k=60
    )
    item = _entry(fused, chunk.chunk_id)
    assert item.channels == ["dense", "sparse"]
    assert item.dense_score == pytest.approx(0.8)
    assert item.sparse_score == pytest.approx(4.2)
    assert item.rrf_score == pytest.approx(1.0 / 61 + 0.8 / 61)


def test_overlap_is_reported_in_debug(make_chunk):
    shared = make_chunk(text="共享", index=0)
    only_dense = make_chunk(text="仅稠密", index=1)
    result = hybrid_search([_pair(only_dense, 0.5), _pair(shared, 0.4)], [_pair(shared, 3.0)], top_k=5)
    assert result.debug["overlap_count"] == 1
    assert result.debug["dense_count"] == 2
    assert result.debug["sparse_count"] == 1
    assert result.debug["fused_count"] == 2
    assert result.debug["rrf_k"] == 60


# ── 名次融合的算术与排序 ────────────────────────────────────────────


def test_deeper_rank_contributes_less(make_chunk):
    first = make_chunk(text="第一名", index=0)
    second = make_chunk(text="第二名", index=1)
    fused = reciprocal_rank_fusion({"dense": [_pair(first, 0.9), _pair(second, 0.8)]}, k=60)
    assert _entry(fused, first.chunk_id).rrf_score > _entry(fused, second.chunk_id).rrf_score


def test_dual_channel_beats_single_channel_top_rank(make_chunk):
    """RRF 用名次融合的意义：两路都排第 2 的候选，应胜过只有一路排第 1 的候选。"""
    single = make_chunk(text="只有稠密排第1", index=0)
    dual = make_chunk(text="两路都排第2", index=1)
    fused = reciprocal_rank_fusion(
        {"dense": [_pair(single, 99.0), _pair(dual, 0.01)], "sparse": [_pair(dual, 0.02)]},
        k=60,
    )
    assert fused[0].chunk.chunk_id == dual.chunk_id
    assert fused[0].rank == 1
    assert fused[1].rank == 2


def test_ranks_are_renumbered_from_one(make_chunk):
    chunks = [make_chunk(text=f"c{i}", index=i) for i in range(4)]
    fused = reciprocal_rank_fusion({"dense": [_pair(c, 1.0) for c in chunks]}, k=60, top_k=10)
    assert [item.rank for item in fused] == [1, 2, 3, 4]


def test_top_k_truncates_result(make_chunk):
    chunks = [make_chunk(text=f"c{i}", index=i) for i in range(6)]
    fused = reciprocal_rank_fusion({"dense": [_pair(c, 1.0) for c in chunks]}, k=60, top_k=3)
    assert len(fused) == 3


def test_score_field_mirrors_rrf_score(make_chunk):
    chunk = make_chunk(text="c", index=0)
    fused = reciprocal_rank_fusion({"dense": [_pair(chunk, 0.5)]}, k=60)
    assert fused[0].score == pytest.approx(fused[0].rrf_score)


def test_rrf_k_smooths_contribution(make_chunk):
    small_k = make_chunk(text="小k", index=0)
    big_k = make_chunk(text="大k", index=1)
    fused_small = reciprocal_rank_fusion({"dense": [_pair(small_k, 1.0)]}, k=1)
    fused_big = reciprocal_rank_fusion({"dense": [_pair(big_k, 1.0)]}, k=1000)
    assert fused_small[0].rrf_score > fused_big[0].rrf_score


# ── 权重 ────────────────────────────────────────────────────────────


def test_default_weights_favour_dense(make_chunk):
    assert DEFAULT_CHANNEL_WEIGHTS["dense"] > DEFAULT_CHANNEL_WEIGHTS["sparse"]


def test_custom_weights_override_defaults(make_chunk):
    chunk = make_chunk(text="c", index=0)
    fused = reciprocal_rank_fusion({"dense": [_pair(chunk, 1.0)]}, k=60, weights={"dense": 2.0})
    assert fused[0].rrf_score == pytest.approx(2.0 / 61)


def test_hybrid_search_passes_weights_into_debug(make_chunk):
    chunk = make_chunk(text="c", index=0)
    result = hybrid_search([_pair(chunk, 1.0)], [], top_k=5, weights={"sparse": 3.0})
    assert result.debug["weights"]["sparse"] == 3.0
    assert result.debug["weights"]["dense"] == DEFAULT_CHANNEL_WEIGHTS["dense"]


def test_unknown_channel_falls_back_to_weight_one(make_chunk):
    chunk = make_chunk(text="c", index=0)
    fused = reciprocal_rank_fusion({"mystery": [_pair(chunk, 1.0)]}, k=60)
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)


# ── 边界 ────────────────────────────────────────────────────────────


def test_both_empty_returns_empty():
    result = hybrid_search([], [], top_k=5)
    assert result.fused == []
    assert result.debug["overlap_count"] == 0


def test_hybrid_search_preserves_raw_hits(make_chunk):
    dense_chunk = make_chunk(text="d", index=0)
    sparse_chunk = make_chunk(text="s", index=1)
    result = hybrid_search([_pair(dense_chunk, 0.7)], [_pair(sparse_chunk, 2.0)], top_k=5)
    assert result.dense_hits == [(dense_chunk, 0.7)]
    assert result.sparse_hits == [(sparse_chunk, 2.0)]
