"""知识库 HTTP 接口使用的轻量校验与展示辅助函数。"""

from __future__ import annotations

import re
from pathlib import Path

from app.rag.loaders import SUPPORTED_SUFFIXES


_KB_NAME_RE = re.compile(r"^[\w\-.\u4e00-\u9fff]{1,64}$", re.UNICODE)


def normalize_kb_name(value: str) -> str:
    """返回安全的知识库名称，拒绝路径片段和空白名称。"""
    name = str(value or "").strip()
    if not _KB_NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise ValueError("知识库名称仅支持 1-64 位中文、字母、数字、下划线、点和短横线")
    return name


def safe_upload_name(value: str) -> str:
    """校验上传文件名与格式，防止路径穿越。"""
    raw = str(value or "").strip()
    name = Path(raw).name
    if not name or name != raw or raw in {".", ".."}:
        raise ValueError("文件名不合法")
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = "、".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"暂不支持 {Path(name).suffix or '无后缀'} 文件，可用格式：{supported}")
    return name


def public_kb_summary(item: dict, backend: str) -> dict:
    """去掉磁盘路径等内部字段，并附上规模建议。"""
    chunk_count = int(item.get("chunk_count", 0) or 0)
    recommendation = "当前规模适合本地索引"
    if chunk_count >= 100_000 and backend != "qdrant":
        recommendation = "片段已达十万级，建议切换 Qdrant；远程模式可关闭 BM25 避免首查全量加载"
    return {
        "name": item.get("name", ""),
        "chunk_count": chunk_count,
        "embedding": item.get("embedding", ""),
        "updated_at": item.get("updated_at", ""),
        "data_version": int(item.get("data_version", 0) or 0),
        "filter_fields": list(item.get("filter_fields") or []),
        "backend": backend,
        "recommendation": recommendation,
    }


__all__ = ["normalize_kb_name", "safe_upload_name", "public_kb_summary"]
