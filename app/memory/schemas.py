"""
记忆数据结构定义

统一记忆的字段与类别约定,供存储层、管理层与工具层复用。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# 记忆类别约定:fact=事实,preference=偏好,project=项目知识,procedure=流程
MemoryKind = Literal["fact", "preference", "project", "procedure"]


class MemoryRecord(BaseModel):
    """一条语义记忆的完整描述。"""

    namespace: tuple[str, ...] = Field(description="命名空间,例如 ('user','alice','fact')")
    key: str = Field(description="记录唯一键")
    value: dict[str, Any] = Field(default_factory=dict, description="记忆内容(JSON)")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="最近更新时间")
    access_count: int = 0
    last_access_at: datetime | None = None

    @property
    def content(self) -> str:
        """取记忆正文文本。"""
        return str(self.value.get("content", ""))

    @property
    def importance(self) -> float:
        """重要度,默认 0.5。"""
        meta = self.value.get("metadata", {}) if isinstance(self.value, dict) else {}
        try:
            return float(meta.get("importance", 0.5))
        except (TypeError, ValueError):
            return 0.5

    @property
    def tags(self) -> list[str]:
        meta = self.value.get("metadata", {}) if isinstance(self.value, dict) else {}
        tags = meta.get("tags", [])
        return list(tags) if isinstance(tags, list) else []


class MemoryRecallResult(BaseModel):
    """召回结果条目,附带相关度分数。"""

    record: MemoryRecord
    score: float = 0.0
