"""
缓存(Cache):embedding / 查询结果 / 生成结果

三块缓存的定位完全不同,所以失效策略也不同——**不要用一套规则套三种东西**:

| 缓存 | 键 | 失效 | 默认 |
| --- | --- | --- | --- |
| Embedding(`embeddings.jsonl`) | `(文本 SHA1, 向量后端指纹)` | 键里含指纹 → 换后端自然不命中;`reset()` 清空缓存目录 | 由 `RAG_CACHE_ENABLED` 控制(默认关) |
| 查询结果(`query_cache.jsonl`) | `(原始 query, 归一化 filter, 指纹, 数据版本号)` | 库内任何 chunk 增删改 → 数据版本号 +1 → 整体失效;`reset()` 清空 | 由 `RAG_CACHE_ENABLED` 控制(默认关) |
| 生成结果(`generation_cache.jsonl`) | `(query, filter, 命中 chunk_id 集合, 模型名, 指纹, 版本号)` | 同上 | 同上 |

为什么整体关掉时也不影响正确性:**默认关闭**是刻意的(见 `docs/rag-enterprise-features.md`):
查询/生成缓存一旦陈旧就是"返回过期答案",这种风险不该由默认配置承担。Embedding 缓存虽然
是内容寻址(同一段文本 + 同一个后端 = 同一个向量,语义上不可能陈旧),但也一起受总开关控制,
保持"一个开关解释全部行为"。

落盘格式统一用 **JSONL 追加写**(便于 `grep` 与人工查看),进程内维护一份索引避免重复解析:

    rag_data/<kb>/.cache/embeddings.jsonl     {"k": "<sha1>|<fingerprint>", "dim": 512, "v": [...]}
    rag_data/<kb>/.cache/query_cache.jsonl    {"k": "<sha1>", "v": {"results": [...]}}
    rag_data/<kb>/.cache/generation_cache.jsonl

已知未实现(文档里也写了):没有 TTL、没有容量上限与 LRU 淘汰、没有跨进程文件锁、没有压缩。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".cache"
EMBEDDINGS_FILE = "embeddings.jsonl"
QUERY_CACHE_FILE = "query_cache.jsonl"
GENERATION_CACHE_FILE = "generation_cache.jsonl"


def content_key(text: str) -> str:
    """文本内容键:SHA1(与 chunk_id 用的是同一套哈希思路,稳定且足够短)。"""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def compound_key(*parts: Any) -> str:
    """把多个维度拼成一个稳定键(维度之间用 `|` 分隔,便于人工排查命中原因)。"""
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    """一次会话内的缓存命中统计(给审计日志与 CLI 用)。"""

    embedding_hits: int = 0
    embedding_misses: int = 0
    query_hits: int = 0
    query_misses: int = 0
    generation_hits: int = 0
    generation_misses: int = 0
    writes: int = 0
    errors: int = 0

    def to_dict(self) -> dict:
        return {
            "embedding_hits": self.embedding_hits,
            "embedding_misses": self.embedding_misses,
            "query_hits": self.query_hits,
            "query_misses": self.query_misses,
            "generation_hits": self.generation_hits,
            "generation_misses": self.generation_misses,
            "writes": self.writes,
            "errors": self.errors,
        }


class JsonlCache:
    """JSONL 追加写缓存的最小实现(键 -> JSON 可序列化值)。

    读写失败一律降级为 warning:缓存是优化项,不能因为它把检索搞挂。
    """

    def __init__(self, path: Path, label: str):
        self.path = Path(path)
        self.label = label
        self._index: dict[str, Any] = {}
        self._loaded = False
        self.stats = CacheStats()

    # ── 内部 ──
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 半截行(上次写一半崩了)直接跳过
                    key = record.get("k")
                    if key:
                        self._index[str(key)] = record.get("v")
        except OSError as exc:
            self.stats.errors += 1
            logger.warning("%s 缓存读取失败(降级为不命中):%s", self.label, exc)

    def get(self, key: str) -> Any | None:
        self._load()
        return self._index.get(key)

    def put(self, key: str, value: Any) -> bool:
        self._load()
        self._index[str(key)] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")
            self.stats.writes += 1
            return True
        except OSError as exc:
            self.stats.errors += 1
            logger.warning("%s 缓存写入失败(不影响本次结果):%s", self.label, exc)
            return False

    def __len__(self) -> int:
        self._load()
        return len(self._index)


class PipelineCache:
    """一个知识库的全部缓存入口(未启用时所有方法都是空操作)。"""

    def __init__(
        self,
        kb_dir: Path,
        fingerprint: str,
        data_version: int = 0,
        enabled: bool = False,
    ):
        self.kb_dir = Path(kb_dir)
        self.cache_dir = self.kb_dir / CACHE_DIR_NAME
        self.fingerprint = fingerprint or ""
        self.data_version = int(data_version or 0)
        self.enabled = bool(enabled)
        self.result_policy = ""
        self.stats = CacheStats()
        self._embeddings: JsonlCache | None = None
        self._queries: JsonlCache | None = None
        self._generations: JsonlCache | None = None

    # ── 路径 ──
    @property
    def embeddings(self) -> JsonlCache:
        if self._embeddings is None:
            self._embeddings = JsonlCache(self.cache_dir / EMBEDDINGS_FILE, "embedding")
        return self._embeddings

    @property
    def queries(self) -> JsonlCache:
        if self._queries is None:
            self._queries = JsonlCache(self.cache_dir / QUERY_CACHE_FILE, "query")
        return self._queries

    @property
    def generations(self) -> JsonlCache:
        if self._generations is None:
            self._generations = JsonlCache(self.cache_dir / GENERATION_CACHE_FILE, "generation")
        return self._generations

    def set_version(self, version: int) -> None:
        """数据版本变化(库内 chunk 增删改)后调用:查询/生成缓存整体失效。"""
        self.data_version = int(version or 0)

    # ── Embedding ──
    def embedding_key(self, text: str) -> str:
        return f"{content_key(text)}|{self.fingerprint}"

    def get_embeddings(self, texts: list[str]) -> dict[int, list[float]]:
        """返回「下标 -> 向量」的命中集合(未启用时恒为空 → 调用方全部重新编码)。"""
        if not self.enabled:
            return {}
        hits: dict[int, list[float]] = {}
        for index, text in enumerate(texts):
            vector = self.embeddings.get(self.embedding_key(text))
            if isinstance(vector, list) and vector:
                hits[index] = vector
        self.stats.embedding_hits += len(hits)
        self.stats.embedding_misses += len(texts) - len(hits)
        return hits

    def put_embeddings(self, texts: list[str], vectors: Iterable[Iterable[float]]) -> None:
        if not self.enabled:
            return
        for text, vector in zip(texts, vectors):
            values = [float(v) for v in vector]
            self.embeddings.put(self.embedding_key(text), values)

    # ── 查询结果 ──
    def query_cache_key(self, query: str, canonical_filter: str, top_n: int | None = None) -> str:
        parts = [query, canonical_filter, str(top_n or ""), self.fingerprint, self.data_version]
        return compound_key(*parts, self.result_policy) if self.result_policy else compound_key(*parts)

    def get_query(self, query: str, canonical_filter: str, top_n: int | None = None) -> tuple[str, dict | None]:
        """返回 (缓存键, 载荷)。载荷必须是含 `results` 列表的字典,否则按未命中处理。"""
        key = self.query_cache_key(query, canonical_filter, top_n)
        if not self.enabled:
            return key, None
        payload = self.queries.get(key)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            self.stats.query_misses += 1
            return key, None
        self.stats.query_hits += 1
        return key, payload

    def put_query(self, query: str, canonical_filter: str, payload: dict, top_n: int | None = None) -> None:
        """按原样存载荷(调用方负责给出 `{"results": [...]}` 结构)。

        注意不要把 payload 再包一层——早先版本在这里多包了一层 `{"results": payload}`,
        结果读回来时 `payload["results"]` 是 dict 而不是 list,缓存永远"看起来没命中"。
        """
        if not self.enabled:
            return
        self.queries.put(self.query_cache_key(query, canonical_filter, top_n), payload)

    # ── 生成结果 ──
    def generation_cache_key(
        self, query: str, canonical_filter: str, chunk_ids: list[str], model: str
    ) -> str:
        parts = (
            query,
            canonical_filter,
            ",".join(chunk_ids),
            model,
            self.fingerprint,
            self.data_version,
        )
        return compound_key(*parts, self.result_policy) if self.result_policy else compound_key(*parts)

    def get_generation(
        self, query: str, canonical_filter: str, chunk_ids: list[str], model: str
    ) -> tuple[str, dict | None]:
        key = self.generation_cache_key(query, canonical_filter, chunk_ids, model)
        if not self.enabled:
            return key, None
        payload = self.generations.get(key)
        if not isinstance(payload, dict) or "answer" not in payload:
            self.stats.generation_misses += 1
            return key, None
        self.stats.generation_hits += 1
        return key, payload

    def put_generation(
        self,
        query: str,
        canonical_filter: str,
        chunk_ids: list[str],
        model: str,
        payload: dict,
    ) -> None:
        if not self.enabled:
            return
        self.generations.put(
            self.generation_cache_key(query, canonical_filter, chunk_ids, model), payload
        )

    # ── 运维 ──
    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "dir": str(self.cache_dir),
            "fingerprint": self.fingerprint,
            "data_version": self.data_version,
            "embedding_entries": len(self._embeddings) if self._embeddings else 0,
            "query_entries": len(self._queries) if self._queries else 0,
            "generation_entries": len(self._generations) if self._generations else 0,
        }

    def clear(self) -> bool:
        """清空该库的全部缓存(reset / 换向量后端时调用)。"""
        removed = 0
        for target in (self.cache_dir / EMBEDDINGS_FILE,
                       self.cache_dir / QUERY_CACHE_FILE,
                       self.cache_dir / GENERATION_CACHE_FILE):
            try:
                if target.exists():
                    target.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning("清空缓存文件失败 %s:%s", target, exc)
        # 重新打开索引,避免内存里还留着已删除的旧键
        self._embeddings = None
        self._queries = None
        self._generations = None
        return bool(removed)


__all__ = [
    "CACHE_DIR_NAME",
    "EMBEDDINGS_FILE",
    "QUERY_CACHE_FILE",
    "GENERATION_CACHE_FILE",
    "CacheStats",
    "JsonlCache",
    "PipelineCache",
    "content_key",
    "compound_key",
]
