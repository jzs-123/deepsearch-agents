"""
混合检索:稠密(向量)+ 稀疏(BM25)用 RRF 融合

为什么用 RRF(Reciprocal Rank Fusion)而不是把两路分数加权求和:
1. 两路分数的量纲完全不同 —— 余弦相似度在 [-1, 1],BM25 是无上界的正数,直接相加等于
   让 BM25 的分值尺度单方面决定结果(某个含高频词的 chunk 能轻松碾压语义最相关的 chunk);
2. 归一化后再加权看似可行,但归一化依赖当次候选集的最大/最小值,同一问题在不同知识库上
   权重含义会漂移,调参结果无法迁移;
3. RRF 只用「名次」不用「分数」,天然免量纲,且对单路的异常高分不敏感,是工业界默认做法。

公式:score(d) = Σ_channel w_c / (k + rank_c(d)),k 默认 60(经验值,越大越平滑)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.rag.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

# 默认通道权重:稠密略高于稀疏。稀疏路的分数噪声更大(长文档、重复词),不宜等权。
DEFAULT_CHANNEL_WEIGHTS = {"dense": 1.0, "sparse": 0.8}


@dataclass
class FusionResult:
    """融合结果,同时保留融合前的分路原始结果,便于排障与对比。"""

    fused: list[RetrievedChunk]
    dense_hits: list[tuple[Chunk, float]] = field(default_factory=list)
    sparse_hits: list[tuple[Chunk, float]] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[Chunk, float]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """把多路已排序结果融合成一个排序。

    :param ranked_lists: 通道名 -> [(chunk, 原始分数)] 降序列表
    :param k: RRF 平滑常数
    :param weights: 通道权重,缺省用 DEFAULT_CHANNEL_WEIGHTS
    :param top_k: 返回条数
    """
    weights = {**DEFAULT_CHANNEL_WEIGHTS, **(weights or {})}
    # 用 chunk_id 而不是下标做 key:同一段内容可能来自不同通道、不同取数方式
    table: dict[str, RetrievedChunk] = {}

    for channel, items in ranked_lists.items():
        weight = weights.get(channel, 1.0)
        for rank, (chunk, raw_score) in enumerate(items, start=1):
            entry = table.get(chunk.chunk_id)
            if entry is None:
                entry = RetrievedChunk(chunk=chunk, score=0.0)
                table[chunk.chunk_id] = entry
            entry.rrf_score += weight / (k + rank)
            entry.channels.append(channel)
            if channel == "dense":
                entry.dense_score = float(raw_score)
            elif channel == "sparse":
                entry.sparse_score = float(raw_score)

    fused = sorted(table.values(), key=lambda item: -item.rrf_score)[: max(1, top_k)]
    for rank, item in enumerate(fused, start=1):
        item.rank = rank
        item.score = item.rrf_score
    return fused


def hybrid_search(
    dense_hits: list[tuple[Chunk, float]],
    sparse_hits: list[tuple[Chunk, float]],
    top_k: int = 10,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> FusionResult:
    """稠密 + 稀疏两路混合检索。

    :param dense_hits: 向量检索结果 [(chunk, 余弦相似度)]
    :param sparse_hits: BM25 检索结果 [(chunk, BM25 分数)]
    :param top_k: 融合后返回条数
    """
    ranked_lists = {"dense": dense_hits, "sparse": sparse_hits}
    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, weights=weights, top_k=top_k)
    debug = {
        "dense_count": len(dense_hits),
        "sparse_count": len(sparse_hits),
        "fused_count": len(fused),
        "rrf_k": rrf_k,
        "weights": {**DEFAULT_CHANNEL_WEIGHTS, **(weights or {})},
        # 命中两个通道的候选被认为是「双路共识」,通常最可靠,单独统计便于评估融合收益
        "overlap_count": sum(1 for item in fused if len(item.channels) > 1),
    }
    logger.debug("混合检索:%s", debug)
    return FusionResult(fused=fused, dense_hits=dense_hits, sparse_hits=sparse_hits, debug=debug)


__all__ = ["hybrid_search", "reciprocal_rank_fusion", "FusionResult", "DEFAULT_CHANNEL_WEIGHTS"]
