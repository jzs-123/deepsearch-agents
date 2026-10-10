"""
稀疏检索:中文分词 + BM25

为什么在向量检索之外还要做一路 BM25:
- 稠密向量擅长「语义相近」,但对专有名词、编号、型号、精确数字不敏感
  (例如「A13 型号的返修率」,向量模型很可能只抓住「返修率」);
- BM25 擅长精确词面匹配,两路互补正是混合检索的收益来源;
- 而且 BM25 可以解释:命中了哪些 query 词、各自贡献多少分,排障时非常好用。

实现要点:
1. jieba 分词 + 中英文/数字统一处理(编号如 "A13"、"3.2.1" 必须整体保留);
2. 倒排索引(chunk_id -> 词频),查询时只遍历含查询词的文档,不必全量打分;
3. 带 IDF 的词权重缓存,并支持自定义停用词与最小词长过滤。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.rag.types import Chunk

logger = logging.getLogger(__name__)

# 语料极小时 IDF 容易出现负值(词在所有文档都出现),用该下限做平滑
_MIN_IDF = 1e-6
# 常见中文停用词(精简版,只保留高频且无区分度的)
STOPWORDS = {
    "的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个", "上", "也",
    "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那",
    "与", "及", "或", "对", "为", "以", "并", "而", "等", "中", "按", "从", "把", "被", "让",
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "on", "with", "as", "by",
}

# 英文单词、数字、型号(字母+数字)、小数编号;中文按单字交给 jieba 组合
_TOKEN_PATTERN = re.compile(r"[A-Za-z]+\d+[A-Za-z0-9]*|\d+(?:\.\d+)+|[A-Za-z]+|\d+|[\u4e00-\u9fff]+")


def tokenize(text: str, stopwords: set[str] | None = None, min_len: int = 1) -> list[str]:
    """中文分词入口。

    jieba 的 cut_for_search 会额外产出细粒度子词,对关键词召回有帮助,但会让索引膨胀,
    这里用默认精确模式 + 型号/编号整体保留,兼顾召回与体积。
    """
    if not text:
        return []
    stopwords = STOPWORDS if stopwords is None else stopwords
    tokens: list[str] = []
    for raw in _TOKEN_PATTERN.findall(text):
        if "\u4e00" <= raw[0] <= "\u9fff":
            # 中文片段交给 jieba;纯符号/单字停用词过滤掉
            for word in _jieba_cut(raw):
                word = word.strip()
                if not word or len(word) < min_len or word in stopwords:
                    continue
                # 过滤纯标点与纯空白
                if not re.search(r"[\w\u4e00-\u9fff]", word):
                    continue
                tokens.append(word.lower())
        else:
            # 英文/数字/型号整体保留,大小写归一
            if len(raw) >= min_len and raw.lower() not in stopwords:
                tokens.append(raw.lower())
    return tokens


def _jieba_cut(text: str) -> list[str]:
    """延迟加载 jieba:不装也能跑(退化为单字 + 二元组),装了则用真实分词。"""
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba  # type: ignore

            jieba.setLogLevel(logging.WARNING)  # 关掉 jieba 的构建缓存日志
            _JIEBA = jieba.lcut
        except ImportError:
            logger.warning("未安装 jieba,中文将退化为单字切分。建议: pip install jieba")
            _JIEBA = False

    if _JIEBA is False:
        chars = list(text)
        # 退化模式:单字 + 相邻二元组,尽量保住一点词组信息
        return chars + [a + b for a, b in zip(chars, chars[1:])]
    return _JIEBA(text)


_JIEBA = None


@dataclass
class BM25Index:
    """内存版 BM25 倒排索引(Okapi BM25)。

    :param k1: 词频饱和系数,越大则高频词增益越持久
    :param b: 文档长度归一化强度,1 表示完全归一化
    """

    k1: float = 1.5
    b: float = 0.75
    stopwords: set[str] | None = None

    _postings: dict[str, dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    _doc_len: list[int] = field(default_factory=list)
    _chunks: list[Chunk] = field(default_factory=list)
    _chunk_tokens: list[list[str]] = field(default_factory=list)
    _avgdl: float = 0.0

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        """基于全量 chunk 重建索引。

        为什么每次都重建而不是增量维护:知识库入库是批量操作,重建一次 O(N) 分词远比
        维护倒排索引的一致性简单可靠;查询路径上不会重建。
        """
        self._postings = defaultdict(dict)
        self._doc_len = []
        self._chunks = []
        self._chunk_tokens = []

        for idx, chunk in enumerate(chunks):
            # 标题也参与索引:标题里往往是核心概念,能显著提升召回
            tokens = tokenize(f"{chunk.heading} {chunk.text}", self.stopwords)
            self._chunks.append(chunk)
            self._chunk_tokens.append(tokens)
            self._doc_len.append(len(tokens) or 1)
            for token, tf in Counter(tokens).items():
                self._postings[token][idx] = tf

        total = sum(self._doc_len)
        self._avgdl = (total / len(self._doc_len)) if self._doc_len else 0.0
        logger.debug("BM25 索引完成:docs=%d terms=%d", len(self._chunks), len(self._postings))
        return self

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _idf(self, term: str) -> float:
        """BM25 的 IDF(带 +0.5 平滑),返回非负值。"""
        n = len(self._chunks)
        df = len(self._postings.get(term, {}))
        if n == 0 or df == 0:
            return 0.0
        return max(_MIN_IDF, math.log(1 + (n - df + 0.5) / (df + 0.5)))

    def score(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """对查询打分,返回 (chunk 下标, BM25 分数) 降序列表。"""
        if not self._chunks:
            return []
        query_tokens = tokenize(query, self.stopwords)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_idx, tf in postings.items():
                dl = self._doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / (denom or 1)

        ranked = sorted(scores.items(), key=lambda item: -item[1])
        return ranked[: max(1, min(top_k, len(ranked)))]

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """返回 (chunk, 分数) 列表。"""
        return [(self._chunks[idx], score) for idx, score in self.score(query, top_k)]

    def explain(self, query: str, chunk_index: int) -> dict[str, float]:
        """解释某个 chunk 的 BM25 分数来自哪些查询词(调参与排障用)。"""
        if chunk_index < 0 or chunk_index >= len(self._chunks):
            return {}
        contributions: dict[str, float] = {}
        dl = self._doc_len[chunk_index]
        for term in set(tokenize(query, self.stopwords)):
            tf = self._postings.get(term, {}).get(chunk_index)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            contributions[term] = self._idf(term) * (tf * (self.k1 + 1)) / (denom or 1)
        return contributions


__all__ = ["BM25Index", "tokenize", "STOPWORDS"]
