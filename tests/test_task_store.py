import sqlite3

import pytest

from app.api import task_store
from app.security.auth import Identity
from app.utils import history_manager as history


@pytest.fixture
def task_db(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "tasks.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    history._init_db(conn)
    monkeypatch.setattr(history, "_conn", conn)
    owner = Identity("alice", "a", authenticated=True)
    history.upsert_session("thread", user_id=owner.user_id, tenant_id=owner.tenant_id)
    yield conn, owner
    conn.close()


def test_task_transitions_and_immutable_terminal_state(task_db):
    _, owner = task_db
    task_id = task_store.create_task("thread", owner, "query")
    task_store.transition(task_id, "running")
    task_store.transition(task_id, "succeeded", result="answer")
    assert task_store.get_task(task_id, owner)["result"] == "answer"
    with pytest.raises(ValueError):
        task_store.transition(task_id, "cancelled")
    assert task_store.get_task(task_id, Identity("bob", "a")) is None
    assert task_store.get_task(task_id, Identity("alice", "b")) is None


def test_cancel_and_restart_leave_no_active_states(task_db):
    _, owner = task_db
    task_id = task_store.create_task("thread", owner, "query")
    task_store.transition(task_id, "running")
    task_store.transition(task_id, "cancelling")
    task_store.interrupt_unfinished()
    snapshot = task_store.get_task(task_id, owner)
    assert snapshot["status"] == "interrupted"
    assert snapshot["error_code"] == "SERVICE_RESTART"
    assert snapshot["finished_at"]


def test_events_survive_reopen_and_can_be_paged_without_duplicates(task_db):
    conn, owner = task_db
    task_id = task_store.create_task("thread", owner, "query")
    for number in range(3):
        event = task_store.append_event(task_id, {"event": "tool_start", "number": number})
        assert event["seq"] == number + 1
    reopened = sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2])
    assert reopened.execute("SELECT count(*) FROM task_events").fetchone()[0] == 3
    reopened.close()
    assert [event["seq"] for event in task_store.get_events(task_id, 1, 1)] == [2]
    assert [event["seq"] for event in task_store.get_events(task_id, 2)] == [3]


def test_deleted_session_hides_task_and_events_owner(task_db):
    _, owner = task_db
    task_id = task_store.create_task("thread", owner, "query")
    history.delete_session("thread", "alice", "a")
    assert task_store.get_task(task_id, owner) is None
    assert task_store.latest_task("thread", owner) is None
