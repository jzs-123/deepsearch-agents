"""API 资源归属校验。"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.api.knowledge_bases import normalize_thread_id
from app.security.auth import Identity
from app.utils.history_manager import SessionAccessError, get_session, upsert_session


def claim_session(thread_id: str, identity: Identity) -> str:
    """创建当前用户的空会话，或确认已有会话归当前用户所有。"""
    try:
        safe_thread_id = normalize_thread_id(thread_id)
        upsert_session(safe_thread_id, "", identity.user_id, identity.tenant_id)
        return safe_thread_id
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SessionAccessError as exc:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问") from exc


def require_session(thread_id: str, identity: Identity) -> dict:
    """返回当前用户拥有的会话；隐藏其他用户会话是否存在。"""
    try:
        safe_thread_id = normalize_thread_id(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = get_session(safe_thread_id, identity.user_id, identity.tenant_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或无权访问")
    return session


def resolve_owned_output_path(path: str, output_root: Path, identity: Identity) -> Path:
    """校验输出路径位于当前用户拥有的 session 目录。"""
    try:
        candidate = Path(path).resolve()
        root = output_root.resolve()
        relative = candidate.relative_to(root)
        session_part = relative.parts[0]
        if not session_part.startswith("session_"):
            raise ValueError
        thread_id = session_part.removeprefix("session_")
        require_session(thread_id, identity)
        return candidate
    except HTTPException:
        raise
    except (IndexError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无效的输出路径") from exc


def task_key(thread_id: str, identity: Identity) -> tuple[str, str, str]:
    return identity.tenant_id, identity.user_id, thread_id


__all__ = ["claim_session", "require_session", "resolve_owned_output_path", "task_key"]
