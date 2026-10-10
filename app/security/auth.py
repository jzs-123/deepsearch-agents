"""
身份与多租户隔离模块

从请求头解析用户/租户身份,为记忆、历史与线程 ID 提供命名空间隔离。
AUTH_ENABLED=false 时退化为匿名身份,不破坏现有前端;开启后强制校验 API Key。

用法示例:
    identity = resolve_identity(request.headers)
    set_identity_context(identity)   # 供深层工具读取当前用户身份
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Mapping, Optional

from app.core.config import settings


class AuthError(Exception):
    """鉴权失败异常,由 API 层统一转换为 401。"""


@dataclass
class Identity:
    """一次请求解析出的身份信息。"""

    user_id: str
    tenant_id: str = "default"
    roles: tuple[str, ...] = ()
    authenticated: bool = False


# 身份上下文:run_deep_agent 写入,记忆工具等深层代码读取
_identity_ctx: ContextVar[Optional[Identity]] = ContextVar("identity", default=None)


def set_identity_context(identity: Identity) -> Token:
    return _identity_ctx.set(identity)


def get_identity_context() -> Optional[Identity]:
    return _identity_ctx.get()


def reset_identity_context(token: Token) -> None:
    _identity_ctx.reset(token)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """从 Authorization 头提取 Bearer token。"""
    if not authorization:
        return None
    parts = authorization.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return parts[0] if len(parts) == 1 else None


def resolve_identity(headers: Optional[Mapping[str, str]] = None) -> Identity:
    """
    从请求头解析身份。开启鉴权时校验 API Key,失败抛 AuthError。

    :param headers: 请求头字典,默认从 X-User-Id / X-Tenant-Id / Authorization 解析
    :return: 解析后的 Identity
    """
    h = dict(headers or {})
    user_id = h.get("x-user-id") or "anonymous"
    tenant_id = h.get("x-tenant-id") or "default"
    roles = tuple(r for r in (h.get("x-roles") or "").split(",") if r)

    if settings.auth_enabled:
        token = _extract_bearer(h.get("authorization"))
        if not token or token != settings.auth_api_key:
            raise AuthError("未授权: API Key 缺失或无效")
        return Identity(user_id=user_id, tenant_id=tenant_id, roles=roles, authenticated=True)

    # 未开启鉴权:显式携带身份头视为已认证,否则匿名
    return Identity(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=roles,
        authenticated=bool(h.get("x-user-id")),
    )


def user_namespace(identity: Identity) -> tuple[str, ...]:
    """当前用户的记忆命名空间前缀。"""
    return ("user", identity.user_id)


def tenant_namespace(identity: Identity) -> tuple[str, ...]:
    """当前租户的共享记忆命名空间前缀。"""
    return ("tenant", identity.tenant_id)


def scope_thread_id(identity: Identity, raw_thread_id: str) -> str:
    """
    给线程 ID 加租户/用户前缀,防止不同租户复用同一 thread_id 导致串扰。
    未开启鉴权时保持原样,兼容现有前端。
    """
    if not settings.auth_enabled:
        return raw_thread_id
    return f"{identity.tenant_id}:{identity.user_id}:{raw_thread_id}"
