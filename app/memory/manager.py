"""
记忆管理器

在 PersistentMemoryStore 之上提供面向业务语义的 API:按用户/租户命名空间
写入、召回、遗忘与衰减。召回结果按「相关度 × 重要度 × 新鲜度」综合排序。

记忆类别约定见 MemoryKind:fact / preference / project / procedure。
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Iterable, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.memory.schemas import MemoryKind, MemoryRecallResult, MemoryRecord
from app.memory.store import PersistentMemoryStore, get_memory_store
from app.security.auth import Identity, tenant_namespace, user_namespace

logger = get_logger("memory.manager")

# 默认召回的全部类别
ALL_KINDS: tuple[MemoryKind, ...] = ("fact", "preference", "project", "procedure")


class MemoryManager:
    """语义记忆的业务门面。"""

    def __init__(self, store: Optional[PersistentMemoryStore] = None) -> None:
        self.store = store or get_memory_store()

    # ─── 写入 ──────────────────────────────────────────────

    def remember(
        self,
        identity: Identity,
        kind: MemoryKind,
        content: str,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        key: Optional[str] = None,
    ) -> str:
        """
        保存一条长期记忆到当前用户命名空间。

        :return: 记忆的 key,供 forget 使用
        """
        namespace = user_namespace(identity) + (kind,)
        key = key or f"{kind}_{uuid.uuid4().hex[:12]}"
        value = {
            "content": content,
            "metadata": {"importance": importance, "tags": tags or [], "kind": kind},
        }
        self.store.put(namespace, key, value)
        logger.debug("已写入记忆 %s/%s", "/".join(namespace), key)
        return key

    # ─── 召回 ──────────────────────────────────────────────

    def recall(
        self,
        identity: Identity,
        query: str,
        kinds: Optional[Iterable[MemoryKind]] = None,
        limit: Optional[int] = None,
    ) -> list[MemoryRecallResult]:
        """
        按当前用户 + 租户共享命名空间召回相关记忆。

        :param query: 用于相关度打分的查询文本
        :param kinds: 限定类别;None 时召回全部类别
        :param limit: 返回条数上限
        """
        kinds = tuple(kinds) if kinds else ALL_KINDS
        limit = limit or settings.memory_recall_limit
        namespaces = [user_namespace(identity) + (k,) for k in kinds]
        namespaces.append(tenant_namespace(identity))  # 租户共享记忆兜底

        now = datetime.now(timezone.utc)
        results: list[MemoryRecallResult] = []
        seen: set[tuple] = set()
        for ns in namespaces:
            for item in self.store.search(ns, query=query, limit=limit * 2):
                dedup_key = (item.namespace, item.key)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                results.append(
                    MemoryRecallResult(
                        record=MemoryRecord(
                            namespace=item.namespace,
                            key=item.key,
                            value=item.value,
                            created_at=item.created_at,
                            updated_at=item.updated_at,
                        ),
                        score=self._composite_score(item, now),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    @staticmethod
    def _composite_score(item, now: datetime) -> float:
        """相关度 × 重要度 × 新鲜度 的综合打分。"""
        lex = item.score or 0.0
        meta = item.value.get("metadata", {}) if isinstance(item.value, dict) else {}
        try:
            importance = float(meta.get("importance", 0.5) or 0.5)
        except (TypeError, ValueError):
            importance = 0.5
        age_days = max((now - item.updated_at).total_seconds() / 86400.0, 0.0)
        recency = max(0.0, 1.0 - age_days / 30.0)  # 30 天线性衰减
        return 0.5 * lex + 0.3 * importance + 0.2 * recency

    @staticmethod
    def render(results: list[MemoryRecallResult]) -> str:
        """把召回结果格式化为可注入上下文的文本块。"""
        if not results:
            return ""
        lines = []
        for r in results:
            rec = r.record
            meta = rec.value.get("metadata", {}) if isinstance(rec.value, dict) else {}
            kind = meta.get("kind", "fact")
            tags = rec.tags
            tag_str = f" #{' #'.join(tags)}" if tags else ""
            lines.append(f"- [{kind}]{tag_str} {rec.content}")
        return "\n".join(lines)

    # ─── 遗忘与衰减 ────────────────────────────────────────

    def forget(self, identity: Identity, key: str) -> bool:
        """按 key 删除当前用户命名空间下的记忆。"""
        deleted = False
        for kind in ALL_KINDS:
            ns = user_namespace(identity) + (kind,)
            self.store.delete(ns, key)
            deleted = True
        return deleted

    def decay(
        self,
        identity: Identity,
        threshold: float = 0.1,
        stale_days: int = 30,
    ) -> int:
        """
        衰减:长时间未更新的记忆降低重要度,低于阈值则删除。
        返回受影响(删除或降级)的记录数。
        """
        now = datetime.now(timezone.utc)
        affected = 0
        for ns in [user_namespace(identity), tenant_namespace(identity)]:
            for item in self.store.search(ns, limit=1000):
                value = dict(item.value) if isinstance(item.value, dict) else {}
                meta = value.get("metadata", {}) if isinstance(value, dict) else {}
                try:
                    importance = float(meta.get("importance", 0.5) or 0.5)
                except (TypeError, ValueError):
                    importance = 0.5
                age_days = (now - item.updated_at).total_seconds() / 86400.0
                if age_days <= stale_days:
                    continue
                new_importance = importance * 0.5
                if new_importance < threshold:
                    self.store.delete(item.namespace, item.key)
                else:
                    meta["importance"] = new_importance
                    value["metadata"] = meta
                    self.store.put(item.namespace, item.key, value)
                affected += 1
        return affected


_manager: Optional[MemoryManager] = None
_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """返回进程内共享的 MemoryManager 单例。"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = MemoryManager()
    return _manager
