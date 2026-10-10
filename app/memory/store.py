"""
持久化长期记忆存储层

实现 LangGraph 的 BaseStore 接口,让本项目能直接把语义记忆注入
create_deep_agent(store=...),并与 DeepAgents 内置的 StoreBackend / memory
特性无缝衔接。底层用 SQLAlchemy(SQLite 起步),连接串换成 Postgres 即可升级。

检索策略(词面召回,预留 embed_fn 钩子):
    - query 为空  → 按更新时间倒序返回
    - query 非空  → 对 value/key 做中英文切分后的重合度打分
    - filter 提供 → 元数据过滤
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.store.base import BaseStore, Item, SearchItem
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("memory.store")

# 命名空间分隔符:选一个几乎不会出现在 key 中的控制字符,便于前缀 LIKE 检索
_SEP = "\x1f"

# 中文按二元组切分,英文/数字按单词切分,用于词面相似度打分
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[一-鿿]")


class PersistentMemoryStore(BaseStore):
    """SQLAlchemy 持久化的 BaseStore 实现(同步 + 异步)。"""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = Path(db_path or settings.memory_db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # 用文件型 SQLite;替换为 postgresql://... 连接串即可升级
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False},
        )
        # 写操作串行化,读操作并发安全
        self._lock = threading.Lock()
        self._ensure_table()

    # ─── 建表 ──────────────────────────────────────────────

    def _ensure_table(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace     TEXT NOT NULL,
                        key           TEXT NOT NULL,
                        value_json    TEXT NOT NULL,
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL,
                        access_count  INTEGER NOT NULL DEFAULT 0,
                        last_access_at TEXT,
                        UNIQUE(namespace, key)
                    )
                    """
                )
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_memory_ns ON memory_items(namespace)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_items(updated_at)")
            )

    # ─── 序列化辅助 ────────────────────────────────────────

    @staticmethod
    def _join(namespace: tuple[str, ...]) -> str:
        return _SEP.join(namespace)

    @staticmethod
    def _split(namespace: str) -> tuple[str, ...]:
        return tuple(namespace.split(_SEP)) if namespace else ()

    @staticmethod
    def _parse_dt(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """粗粒度切分,得到用于重合度打分的词元集合。"""
        raw = str(text).lower()
        tokens = set(_TOKEN_RE.findall(raw))
        # 补充中文二元组,提升中文短句的召回质量
        cjk = re.findall(r"[一-鿿]", raw)
        for a, b in zip(cjk, cjk[1:]):
            tokens.add(a + b)
        return tokens

    def _score(self, query: str, value: dict[str, Any], key: str) -> float:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return 0.0
        doc = json.dumps(value, ensure_ascii=False) + " " + key
        d_tokens = self._tokenize(doc)
        overlap = len(q_tokens & d_tokens)
        return overlap / len(q_tokens) if overlap else 0.0

    # ─── 同步接口 ──────────────────────────────────────────

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any], index: Any = None, *, ttl: Any = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        ns = self._join(namespace)
        with self._lock:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO memory_items
                            (namespace, key, value_json, created_at, updated_at, access_count)
                        VALUES (:ns, :key, :val, :now, :now, 0)
                        ON CONFLICT(namespace, key) DO UPDATE SET
                            value_json = excluded.value_json,
                            updated_at = excluded.updated_at
                        """
                    ),
                    {"ns": ns, "key": key, "val": json.dumps(value, ensure_ascii=False), "now": now},
                )

    def get(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None) -> Item | None:
        ns = self._join(namespace)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._engine.begin() as conn:
                row = conn.execute(
                    text("SELECT * FROM memory_items WHERE namespace = :ns AND key = :key"),
                    {"ns": ns, "key": key},
                ).fetchone()
                if row is None:
                    return None
                # 读即更新访问统计
                conn.execute(
                    text(
                        "UPDATE memory_items SET access_count = access_count + 1, last_access_at = :now "
                        "WHERE namespace = :ns AND key = :key"
                    ),
                    {"now": now, "ns": ns, "key": key},
                )
        return self._row_to_item(row, score=None)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        ns = self._join(namespace)
        with self._lock:
            with self._engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM memory_items WHERE namespace = :ns AND key = :key"),
                    {"ns": ns, "key": key},
                )

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        ns = self._join(namespace_prefix)
        with self._engine.connect() as conn:
            if ns:
                rows = conn.execute(
                    text(
                        "SELECT * FROM memory_items "
                        "WHERE namespace = :ns OR namespace LIKE :child "
                        "ORDER BY updated_at DESC"
                    ),
                    {"ns": ns, "child": ns + _SEP + "%"},
                ).fetchall()
            else:
                rows = conn.execute(
                    text("SELECT * FROM memory_items ORDER BY updated_at DESC")
                ).fetchall()

        items = [self._row_to_item(r, score=None) for r in rows]

        # 元数据过滤
        if filter:
            items = [it for it in items if self._match_filter(it, filter)]

        # 词面检索打分:命中才保留,按相关度倒序
        if query:
            scored: list[SearchItem] = []
            for it in items:
                s = self._score(query, it.value, it.key)
                if s > 0:
                    scored.append(
                        SearchItem(
                            namespace=it.namespace,
                            key=it.key,
                            value=it.value,
                            created_at=it.created_at,
                            updated_at=it.updated_at,
                            score=s,
                        )
                    )
            scored.sort(key=lambda x: x.score or 0.0, reverse=True)
            results = scored
        else:
            results = [
                SearchItem(
                    namespace=it.namespace,
                    key=it.key,
                    value=it.value,
                    created_at=it.created_at,
                    updated_at=it.updated_at,
                    score=None,
                )
                for it in items
            ]

        return results[offset : offset + limit]

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT namespace FROM memory_items")).fetchall()
        namespaces = [self._split(r[0]) for r in rows]
        if prefix is not None:
            namespaces = [ns for ns in namespaces if ns[: len(prefix)] == prefix]
        if suffix is not None:
            namespaces = [ns for ns in namespaces if ns[-len(suffix) :] == suffix]
        if max_depth is not None:
            namespaces = [ns for ns in namespaces if len(ns) <= max_depth]
        return namespaces[offset : offset + limit]

    def batch(self, ops: list[dict[str, Any]]) -> list[Any]:
        """按操作类型批量执行(同步)。"""
        results: list[Any] = []
        for op in ops:
            if "namespace_prefix" in op:
                results.append(
                    self.search(
                        op["namespace_prefix"],
                        query=op.get("query"),
                        filter=op.get("filter"),
                        limit=op.get("limit", 10),
                        offset=op.get("offset", 0),
                    )
                )
            elif "namespace" in op and "value" in op:
                self.put(op["namespace"], op["key"], op["value"], index=op.get("index"))
                results.append(None)
            elif "namespace" in op and "key" in op:
                results.append(self.get(op["namespace"], op["key"]))
            else:
                # ListNamespacesOp:match_conditions 为 [(match_type, path), ...]
                prefix = suffix = None
                for cond in op.get("match_conditions") or []:
                    if cond.get("match_type") == "prefix":
                        prefix = tuple(cond.get("path", ()))
                    elif cond.get("match_type") == "suffix":
                        suffix = tuple(cond.get("path", ()))
                results.append(
                    self.list_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.get("max_depth"),
                        limit=op.get("limit", 100),
                        offset=op.get("offset", 0),
                    )
                )
        return results

    # ─── 异步接口(桥接到同步实现) ─────────────────────────

    async def aput(self, namespace: tuple[str, ...], key: str, value: dict[str, Any], index: Any = None, *, ttl: Any = None) -> None:
        await asyncio.to_thread(self.put, namespace, key, value, index)

    async def aget(self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None) -> Item | None:
        return await asyncio.to_thread(self.get, namespace, key)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        await asyncio.to_thread(self.delete, namespace, key)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return await asyncio.to_thread(self.search, namespace_prefix, query=query, filter=filter, limit=limit, offset=offset)

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return await asyncio.to_thread(self.list_namespaces, prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset)

    async def abatch(self, ops: list[dict[str, Any]]) -> list[Any]:
        """按操作类型批量执行(异步,桥接到同步实现)。"""
        return await asyncio.to_thread(self.batch, ops)

    # ─── 内部工具 ──────────────────────────────────────────

    @staticmethod
    def _match_filter(item: Item, filter: dict[str, Any]) -> bool:
        """按 value.metadata 中的字段做等值过滤。"""
        meta = item.value.get("metadata", {}) if isinstance(item.value, dict) else {}
        for k, v in filter.items():
            if meta.get(k) != v:
                return False
        return True

    def _row_to_item(self, row: Any, score: float | None) -> Item | SearchItem:
        value = json.loads(row.value_json or "{}")
        created = self._parse_dt(row.created_at)
        updated = self._parse_dt(row.updated_at)
        if score is None:
            return Item(
                value=value,
                key=row.key,
                namespace=self._split(row.namespace),
                created_at=created,
                updated_at=updated,
            )
        return SearchItem(
            namespace=self._split(row.namespace),
            key=row.key,
            value=value,
            created_at=created,
            updated_at=updated,
            score=score,
        )


# 进程内共享的单例 store,保证 create_deep_agent 与 MemoryManager 指向同一份数据
_memory_store: PersistentMemoryStore | None = None
_store_lock = threading.Lock()


def get_memory_store() -> PersistentMemoryStore:
    """返回进程内共享的持久化记忆存储单例。"""
    global _memory_store
    if _memory_store is None:
        with _store_lock:
            if _memory_store is None:
                _memory_store = PersistentMemoryStore()
                logger.info("记忆存储已初始化: %s", _memory_store._db_path)
    return _memory_store
