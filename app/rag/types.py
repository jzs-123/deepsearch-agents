"""
RAG 数据模型

用 dataclass 而不是裸 dict,是因为检索链路里 chunk 会被多路复用(向量库、BM25、
重排、生成),字段一旦拼错很难发现;显式字段 + to_dict/from_dict 保证落盘和
跨模块传递时结构一致。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """入库的最小检索单元。

    :param chunk_id: 稳定 ID(来源+序号+内容指纹),重复入库时用于去重
    :param text: chunk 正文
    :param source: 来源文件名(引用时展示给用户)
    :param source_path: 来源文件相对路径
    :param chunk_index: 在该文档内的序号
    :param heading: 所属标题路径,如 "第三章 > 3.2 风控",用于给模型补充结构上下文
    :param doc_id: 文档指纹,同一文件重复入库时用于整篇替换
    """

    chunk_id: str
    text: str
    source: str
    source_path: str
    chunk_index: int
    heading: str = ""
    doc_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """生成给用户看的引用标记,如 [电商白皮书.pdf#3]。"""
        return f"[{self.source}#{self.chunk_index}]"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        # 只取已知字段,兼容后续新增字段的旧数据
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class RetrievedChunk:
    """带分数的检索结果,保留每一路分数便于排障与调参。

    :param chunk: 命中的 chunk
    :param score: 最终用于排序的分数(重排后为重排分)
    :param dense_score: 稠密(向量)相似度,未命中该路时为 None
    :param sparse_score: 稀疏(BM25)分数,未命中该路时为 None
    :param rrf_score: RRF 融合分
    :param rank: 在当前结果列表中的位次(从 1 开始,重排后重算)
    :param channels: 命中该 chunk 的检索通道,如 ["dense", "sparse"]
    :param cached: 本条结果是否直接来自查询结果缓存(命中缓存时必须显式标注)
    """

    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rank: int = 0
    channels: list[str] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["chunk"] = self.chunk.to_dict()
        return data


@dataclass
class Answer:
    """RAG 问答结果。

    :param question: 用户问题
    :param answer: 模型回答;检索不到依据时为拒答文案
    :param citations: 引用列表,每项含 source/chunk_index/score
    :param contexts: 送入模型的上下文片段(便于前端展示「依据」)
    :param grounded: 是否基于检索结果作答(False 表示拒答/无依据)
    :param debug: 各阶段耗时与候选数量,排障用
    """

    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[dict[str, Any]] = field(default_factory=list)
    grounded: bool = True
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def content_fingerprint(text: str) -> str:
    """计算内容指纹,用于去重与文档级替换。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_chunk_id(source_path: str, index: int, text: str) -> str:
    """构造稳定 chunk_id:同一份文件同一位置同一内容 -> 同一 ID。"""
    raw = f"{source_path}|{index}|{text[:120]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
