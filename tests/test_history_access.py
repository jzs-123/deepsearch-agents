import sqlite3

import pytest
from fastapi import HTTPException

from app.api.access import claim_session, require_session, resolve_owned_output_path
from app.security.auth import Identity
from app.utils import history_manager as history


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "sessions-test.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    history._init_db(conn)
    monkeypatch.setattr(history, "_conn", conn)
    yield conn
    conn.close()


def test_history_is_isolated_by_user_and_tenant(isolated_history):
    history.upsert_session("thread_a", "A", "alice", "tenant-1")
    history.upsert_session("thread_b", "B", "bob", "tenant-1")
    history.upsert_session("thread_c", "C", "alice", "tenant-2")
    history.save_message("thread_a", "user", "secret", user_id="alice", tenant_id="tenant-1")

    assert [item["thread_id"] for item in history.list_sessions(50, "alice", "tenant-1")] == ["thread_a"]
    assert history.get_session("thread_a", "bob", "tenant-1") is None
    assert history.get_session_messages("thread_a", "bob", "tenant-1") == []
    assert history.get_session_messages("thread_a", "alice", "tenant-1")[0]["content"] == "secret"
    assert not history.delete_session("thread_a", "bob", "tenant-1")
    assert history.delete_session("thread_a", "alice", "tenant-1")


def test_existing_thread_id_cannot_be_claimed_by_another_owner(isolated_history):
    history.upsert_session("shared_id", "", "alice", "tenant-1")

    with pytest.raises(history.SessionAccessError):
        history.upsert_session("shared_id", "", "bob", "tenant-1")


def test_deleted_session_cannot_be_reused_to_take_over_files(isolated_history):
    history.upsert_session("shared_id", "", "alice", "tenant-1")
    assert history.delete_session("shared_id", "alice", "tenant-1")
    with pytest.raises(history.SessionAccessError):
        history.upsert_session("shared_id", "", "bob", "tenant-1")
    assert history.get_session("shared_id", "alice", "tenant-1") is None


def test_old_sessions_are_migrated_to_anonymous_owner(tmp_path):
    conn = sqlite3.connect(tmp_path / "legacy.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE sessions (thread_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('legacy', 'old', '2026-01-01', '2026-01-01')"
    )

    history._init_db(conn)

    row = conn.execute("SELECT user_id, tenant_id FROM sessions WHERE thread_id='legacy'").fetchone()
    assert tuple(row) == ("anonymous", "default")
    conn.close()


def test_access_helpers_hide_other_users_sessions(isolated_history, tmp_path):
    alice = Identity(user_id="alice", tenant_id="tenant-1", authenticated=True)
    bob = Identity(user_id="bob", tenant_id="tenant-1", authenticated=True)
    claim_session("thread_a", alice)

    assert require_session("thread_a", alice)["thread_id"] == "thread_a"
    with pytest.raises(HTTPException) as error:
        require_session("thread_a", bob)
    assert error.value.status_code == 404

    output_root = tmp_path / "output"
    session_dir = output_root / "session_thread_a"
    session_dir.mkdir(parents=True)
    report = session_dir / "report.md"
    report.write_text("ok", encoding="utf-8")
    assert resolve_owned_output_path(str(report), output_root, alice) == report.resolve()
    with pytest.raises(HTTPException) as error:
        resolve_owned_output_path(str(report), output_root, bob)
    assert error.value.status_code == 404


def test_access_helpers_reject_paths_outside_output(isolated_history, tmp_path):
    identity = Identity(user_id="alice", tenant_id="tenant-1", authenticated=True)
    claim_session("thread_a", identity)

    with pytest.raises(HTTPException) as error:
        resolve_owned_output_path(str(tmp_path / "outside.txt"), tmp_path / "output", identity)
    assert error.value.status_code == 400
