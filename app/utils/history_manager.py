"""
会话历史持久化管理模块

使用 SQLite 存储对话历史，支持会话列表、历史加载和删除。
解决 InMemorySaver 进程重启即丢失的问题。
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 数据库文件放在 app 目录下，与 output、updated 同级
_db_dir = Path(__file__).resolve().parents[1]
_db_path = _db_dir / "sessions.db"

# 线程锁，保证多协程并发写入安全
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接，自动创建表。"""
    conn = sqlite3.connect(str(_db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    """创建历史表（如不存在）。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id   TEXT PRIMARY KEY,
            title       TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id   TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL DEFAULT '',
            events_json TEXT NOT NULL DEFAULT '[]',
            created_at  TEXT NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES sessions(thread_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


# 模块加载时初始化
_conn = _get_conn()
_init_db(_conn)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_session(thread_id: str, title: str = "") -> None:
    """创建或更新会话记录。"""
    with _lock:
        now = _now_iso()
        _conn.execute(
            """
            INSERT INTO sessions (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = CASE WHEN sessions.title = '' AND excluded.title != ''
                             THEN excluded.title ELSE sessions.title END,
                updated_at = excluded.updated_at
            """,
            (thread_id, title or "", now, now),
        )
        _conn.commit()


def save_message(
    thread_id: str,
    role: str,
    content: str,
    events_json: str = "[]",
) -> None:
    """保存一条对话消息。events_json 存储与该消息关联的 monitor 事件。"""
    with _lock:
        _conn.execute(
            """
            INSERT INTO messages (thread_id, role, content, events_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, role, content, events_json, _now_iso()),
        )
        # 同步更新会话修改时间
        _conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE thread_id = ?",
            (_now_iso(), thread_id),
        )
        _conn.commit()


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """列出最近会话，按更新时间倒序。"""
    rows = _conn.execute(
        "SELECT thread_id, title, created_at, updated_at "
        "FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(thread_id: str) -> list[dict[str, Any]]:
    """加载一个会话的所有历史消息。"""
    rows = _conn.execute(
        "SELECT role, content, events_json, created_at "
        "FROM messages WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session(thread_id: str) -> Optional[dict[str, Any]]:
    """获取单个会话摘要。"""
    row = _conn.execute(
        "SELECT thread_id, title, created_at, updated_at "
        "FROM sessions WHERE thread_id = ?",
        (thread_id,),
    ).fetchone()
    return dict(row) if row else None


def delete_session(thread_id: str) -> bool:
    """删除一个会话及其所有消息。"""
    with _lock:
        cur = _conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        _conn.commit()
        return cur.rowcount > 0
