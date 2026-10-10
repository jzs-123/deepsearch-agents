"""
结构化日志模块

统一使用 app.* 命名空间的 logger,通过 contextvars 注入 request_id 与
thread_id,让跨模块日志能按「请求/会话」关联起来。支持 text 与 json 两种
输出格式,由 Settings.log_format 控制。
"""

import json
import logging
import sys
import threading
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings

# 请求级上下文:由中间件/执行入口写入,日志 Filter 读取后注入到每条记录
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_thread_id_ctx: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)


class _ContextFilter(logging.Filter):
    """把 contextvars 中的请求上下文挂到 LogRecord 上。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get() or "-"
        record.thread_id = _thread_id_ctx.get() or "-"
        return True


class _TextFormatter(logging.Formatter):
    """人类可读的 key=value 格式。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        base = (
            f"{ts} [{record.levelname}] {record.name} "
            f"rid={getattr(record, 'request_id', '-')} "
            f"tid={getattr(record, 'thread_id', '-')} "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class _JsonFormatter(logging.Formatter):
    """机器可解析的 JSON 格式,便于接入日志采集系统。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "thread_id": getattr(record, "thread_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter())
    return handler


_configured = False
_configure_lock = threading.Lock()


def configure_logging() -> None:
    """初始化应用日志(幂等,进程内只配置一次)。"""
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        root = logging.getLogger("app")
        root.setLevel(settings.log_level.upper())
        root.addHandler(_build_handler())
        root.propagate = False
        _configured = True


def get_logger(name: str = "") -> logging.Logger:
    """获取 app.* 命名空间下的 logger。"""
    configure_logging()
    return logging.getLogger(f"app.{name}".strip("."))


# ─── 请求上下文绑定 ──────────────────────────────────────────


def set_request_context(
    request_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> tuple[Token, Token]:
    """绑定当前请求的 request_id / thread_id,返回用于恢复的 token。"""
    return _request_id_ctx.set(request_id), _thread_id_ctx.set(thread_id)


def reset_request_context(req_token: Token, thread_token: Token) -> None:
    """恢复请求上下文,避免残留到后续请求。"""
    _request_id_ctx.reset(req_token)
    _thread_id_ctx.reset(thread_token)


def bind_thread_id(thread_id: Optional[str]) -> Token:
    """单独绑定会话 ID(供 Agent 执行入口在后台任务中使用)。"""
    return _thread_id_ctx.set(thread_id)


def reset_thread_id(token: Token) -> None:
    _thread_id_ctx.reset(token)
