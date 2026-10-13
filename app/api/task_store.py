"""单进程任务快照与可补发事件，复用会话 SQLite 数据库。"""

import json
import uuid
from contextvars import ContextVar
from app.utils import history_manager as history

current_task_id: ContextVar[str | None] = ContextVar("current_task_id", default=None)
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
TRANSITIONS = {
    "queued": {"running", "cancelling", "cancelled", "interrupted"},
    "running": {"succeeded", "failed", "cancelling", "cancelled", "interrupted"},
    "cancelling": {"cancelled", "failed", "interrupted"},
}


def create_task(thread_id, identity, query):
    task_id = uuid.uuid4().hex
    with history._lock:
        history._conn.execute(
            "INSERT INTO tasks(task_id,thread_id,user_id,tenant_id,status,query,created_at) "
            "VALUES(?,?,?,?, 'queued', ?,?)",
            (task_id, thread_id, identity.user_id, identity.tenant_id, query, history._now_iso()),
        )
        history._conn.commit()
    return task_id


def get_task(task_id, identity):
    with history._lock:
        row = history._conn.execute(
            "SELECT t.* FROM tasks t JOIN sessions s ON s.thread_id=t.thread_id "
            "WHERE t.task_id=? AND t.user_id=? AND t.tenant_id=? AND s.deleted=0",
            (task_id, identity.user_id, identity.tenant_id),
        ).fetchone()
    return _snapshot(row)


def latest_task(thread_id, identity):
    with history._lock:
        row = history._conn.execute(
            "SELECT t.* FROM tasks t JOIN sessions s ON s.thread_id=t.thread_id "
            "WHERE t.thread_id=? AND t.user_id=? AND t.tenant_id=? AND s.deleted=0 "
            "ORDER BY t.created_at DESC LIMIT 1",
            (thread_id, identity.user_id, identity.tenant_id),
        ).fetchone()
    return _snapshot(row)


def _snapshot(row):
    if not row:
        return None
    snapshot = dict(row)
    snapshot["output"] = json.loads(snapshot.pop("output_json")) or None
    return snapshot


def transition(task_id, status, *, result="", error_code="", error_message="", output=None):
    """终态不可覆盖；非法流转直接失败，避免取消与成功互相覆盖。"""
    with history._lock:
        row = history._conn.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            raise ValueError("任务不存在")
        if status not in TRANSITIONS.get(row["status"], set()):
            raise ValueError(f"非法任务状态转换: {row['status']} -> {status}")
        now = history._now_iso()
        history._conn.execute(
            "UPDATE tasks SET status=?, result=?, error_code=?, error_message=?, output_json=?, "
            "started_at=CASE WHEN ?='running' THEN ? ELSE started_at END, "
            "finished_at=CASE WHEN ? THEN ? ELSE finished_at END WHERE task_id=?",
            (status, result, error_code, error_message, json.dumps(output or {}, ensure_ascii=False),
             status, now, status in TERMINAL, now, task_id),
        )
        history._conn.commit()


def append_event(task_id, payload):
    with history._lock:
        seq = history._conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        event = {**payload, "task_id": task_id, "seq": seq}
        history._conn.execute(
            "INSERT INTO task_events VALUES(?,?,?)",
            (task_id, seq, json.dumps(event, ensure_ascii=False, default=str)),
        )
        history._conn.commit()
    return event


def get_events(task_id, after=0, limit=500):
    with history._lock:
        rows = history._conn.execute(
            "SELECT payload_json FROM task_events WHERE task_id=? AND seq>? ORDER BY seq LIMIT ?",
            (task_id, after, limit),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def interrupt_unfinished():
    """仅单进程启动时调用；保留失败原因，不尝试重新执行有副作用的工具。"""
    with history._lock:
        history._conn.execute(
            "UPDATE tasks SET status='interrupted', finished_at=?, error_code='SERVICE_RESTART', "
            "error_message='服务已重启，任务中断，请重新提交' "
            "WHERE status IN ('queued','running','cancelling')",
            (history._now_iso(),),
        )
        history._conn.commit()
