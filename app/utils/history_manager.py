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
from app.core.config import settings

# 数据库文件放在 app 目录下，与 output、updated 同级
_db_path = Path(settings.sessions_db_path)
_db_path.parent.mkdir(parents=True, exist_ok=True)

# 线程锁，保证多协程并发写入安全
_lock = threading.Lock()


class SessionAccessError(PermissionError):
    """thread_id 已被其他用户或租户占用。"""


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
            user_id     TEXT NOT NULL DEFAULT 'anonymous',
            tenant_id   TEXT NOT NULL DEFAULT 'default',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "user_id" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'")
    if "tenant_id" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
    if "deleted" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_owner_updated "
        "ON sessions(tenant_id, user_id, updated_at DESC)"
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
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            status TEXT NOT NULL,
            query TEXT NOT NULL,
            result TEXT NOT NULL DEFAULT '',
            error_code TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            FOREIGN KEY(thread_id) REFERENCES sessions(thread_id)
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_thread ON tasks(thread_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS task_events (
            task_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(task_id, seq),
            FOREIGN KEY(task_id) REFERENCES tasks(task_id)
        );
    """)
    task_columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "output_json" not in task_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN output_json TEXT NOT NULL DEFAULT '{}'")
    conn.commit()


# 模块加载时初始化
_conn = _get_conn()
_init_db(_conn)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_session(
    thread_id: str,
    title: str = "",
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> None:
    """创建或更新当前 owner 的会话；同 ID 属于别人时拒绝接管。"""
    with _lock:
        now = _now_iso()
        cursor = _conn.execute(
            """
            INSERT INTO sessions (thread_id, title, user_id, tenant_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = CASE WHEN sessions.title = '' AND excluded.title != ''
                             THEN excluded.title ELSE sessions.title END,
                updated_at = excluded.updated_at
            WHERE sessions.user_id = excluded.user_id
              AND sessions.tenant_id = excluded.tenant_id
              AND sessions.deleted = 0
            """,
            (thread_id, title or "", user_id, tenant_id, now, now),
        )
        if cursor.rowcount == 0:
            _conn.rollback()
            raise SessionAccessError("会话不存在或无权访问")
        _conn.commit()


def save_message(
    thread_id: str,
    role: str,
    content: str,
    events_json: str = "[]",
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> None:
    """保存一条对话消息。events_json 存储与该消息关联的 monitor 事件。"""
    with _lock:
        owned = _conn.execute(
            "SELECT 1 FROM sessions WHERE thread_id = ? AND user_id = ? AND tenant_id = ? AND deleted = 0",
            (thread_id, user_id, tenant_id),
        ).fetchone()
        if not owned:
            raise SessionAccessError("会话不存在或无权访问")
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


def list_sessions(
    limit: int = 50,
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> list[dict[str, Any]]:
    """列出最近会话，按更新时间倒序。"""
    rows = _conn.execute(
        "SELECT thread_id, title, created_at, updated_at "
        "FROM sessions WHERE user_id = ? AND tenant_id = ? AND deleted = 0 "
        "ORDER BY updated_at DESC LIMIT ?",
        (user_id, tenant_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(
    thread_id: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> list[dict[str, Any]]:
    """加载一个会话的所有历史消息。"""
    rows = _conn.execute(
        "SELECT m.role, m.content, m.events_json, m.created_at "
        "FROM messages m JOIN sessions s ON s.thread_id = m.thread_id "
        "WHERE m.thread_id = ? AND s.user_id = ? AND s.tenant_id = ? AND s.deleted = 0 ORDER BY m.id ASC",
        (thread_id, user_id, tenant_id),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session(
    thread_id: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> Optional[dict[str, Any]]:
    """获取单个会话摘要。"""
    row = _conn.execute(
        "SELECT thread_id, title, created_at, updated_at "
        "FROM sessions WHERE thread_id = ? AND user_id = ? AND tenant_id = ? AND deleted = 0",
        (thread_id, user_id, tenant_id),
    ).fetchone()
    return dict(row) if row else None


def delete_session(
    thread_id: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> bool:
    """删除一个会话及其所有消息。"""
    with _lock:
        cur = _conn.execute(
            "UPDATE sessions SET deleted = 1, title = '' WHERE thread_id = ? AND user_id = ? AND tenant_id = ? AND deleted = 0",
            (thread_id, user_id, tenant_id),
        )
        if cur.rowcount:
            _conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        _conn.commit()
        return cur.rowcount > 0
