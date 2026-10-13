import asyncio
import importlib.util
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.security import auth, knowledge_bases
from app.utils import history_manager as history


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    async def fake_agent(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setitem(sys.modules, "app.agent.main_agent", SimpleNamespace(run_deep_agent=fake_agent))
    config = settings.model_copy(update={
        "auth_enabled": True, "auth_api_key": "", "rate_limit_per_minute": 10000,
        "auth_users": {
            "alice-key": {"user_id": "alice", "tenant_id": "a"},
            "bob-key": {"user_id": "bob", "tenant_id": "b"},
        },
    })
    monkeypatch.setattr(auth, "settings", config)
    monkeypatch.setattr(knowledge_bases, "settings", config)
    path = Path(__file__).resolve().parents[1] / "app/api/server.py"
    spec = importlib.util.spec_from_file_location("resource_access_test_server", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    monkeypatch.setattr(server, "settings", config)
    server.output_dir = tmp_path / "output"
    server.updated_dir = tmp_path / "uploads"
    server.kb_upload_dir = tmp_path / "kb_uploads"
    server.output_dir.mkdir()
    server.updated_dir.mkdir()
    server.kb_upload_dir.mkdir()
    connection = sqlite3.connect(tmp_path / "history.db", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    history._init_db(connection)
    monkeypatch.setattr(history, "_conn", connection)
    with TestClient(server.app) as client:
        yield client, server
        for task in list(server.active_tasks.values()):
            client.portal.call(task.cancel)
    server.manager.active_connections.clear()
    connection.close()


def _login(client, key):
    response = client.post("/api/auth/login", json={"api_key": key})
    assert response.status_code == 200
    return response


def test_login_cookie_history_download_and_cross_user_denials(api_client):
    client, server = api_client
    assert client.get("/api/sessions").status_code == 401
    login = _login(client, "alice-key")
    assert "HttpOnly" in login.headers["set-cookie"]
    assert client.post("/api/upload", data={"thread_id": "alice_thread"}, files={"files": ("notes.txt", b"hello")}).status_code == 200
    directory = server.output_dir / "session_alice_thread"
    directory.mkdir()
    report = directory / "report.md"
    report.write_text("alice secret", encoding="utf-8")
    assert client.get("/api/download", params={"path": str(report)}).text == "alice secret"
    assert client.get("/api/sessions/alice_thread").status_code == 200
    _login(client, "bob-key")
    assert client.get("/api/sessions").json()["sessions"] == []
    assert client.get("/api/sessions/alice_thread").status_code == 404
    assert client.delete("/api/sessions/alice_thread").status_code == 404
    assert client.post("/api/task/alice_thread/cancel").status_code == 404
    assert client.post("/api/task", json={"thread_id": "alice_thread", "query": "steal"}).status_code == 404
    assert client.get("/api/files", params={"path": str(directory)}).status_code == 404
    assert client.get("/api/download", params={"path": str(report)}).status_code == 404
    assert client.post("/api/upload", data={"thread_id": "alice_thread"}, files={"files": ("notes.txt", b"overwrite")}).status_code == 404


def test_websocket_requires_cookie_and_owner(api_client):
    client, _ = api_client
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/alice_thread"):
            pass
    _login(client, "alice-key")
    with client.websocket_connect("/ws/alice_thread") as socket:
        socket.send_text("ping")
        assert socket.receive_json()["type"] == "pong"
    _login(client, "bob-key")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/alice_thread"):
            pass


def test_cancel_only_owner_can_stop_task(api_client):
    client, server = api_client
    _login(client, "alice-key")
    assert client.post("/api/task", json={"thread_id": "alice_thread", "query": "hello"}).status_code == 200
    _login(client, "bob-key")
    assert client.post("/api/task/alice_thread/cancel").status_code == 404
    assert ("a", "alice", "alice_thread") in server.active_tasks
    _login(client, "alice-key")
    response = client.post("/api/task/alice_thread/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert _wait_terminal(client, response.json()["task_id"])["status"] == "cancelled"


def test_anonymous_mode_keeps_existing_frontend_flow(api_client, monkeypatch):
    client, server = api_client
    anonymous = settings.model_copy(update={"auth_enabled": False})
    monkeypatch.setattr(server, "settings", anonymous)
    monkeypatch.setattr(auth, "settings", anonymous)
    monkeypatch.setattr(knowledge_bases, "settings", anonymous)
    assert client.get("/api/auth/status").json()["authenticated"] is True
    response = client.post("/api/upload", data={"thread_id": "local_thread"}, files={"files": ("notes.txt", b"local")})
    assert response.status_code == 200
    assert client.get("/api/sessions/local_thread").status_code == 200


def _wait_terminal(client, task_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/tasks/{task_id}").json()
        if snapshot["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return snapshot
        time.sleep(0.01)
    pytest.fail("任务未进入终态")


@pytest.mark.parametrize("error,error_code", [
    (RuntimeError("secret backend credential"), "AGENT_EXECUTION_FAILED"),
    (TimeoutError("secret provider address"), "EXECUTION_TIMEOUT"),
])
def test_task_snapshot_failure_and_event_replay_are_owned_and_durable(api_client, monkeypatch, error, error_code):
    client, server = api_client

    async def failed_agent(*args, **kwargs):
        server.monitor.report_tool("test", {"query": "question"})
        raise error

    monkeypatch.setattr(server, "run_deep_agent", failed_agent)
    _login(client, "alice-key")
    task_id = client.post("/api/task", json={"thread_id": "alice_thread", "query": "hello"}).json()["task_id"]
    snapshot = _wait_terminal(client, task_id)
    assert snapshot["status"] == "failed"
    assert snapshot["error_code"] == error_code
    assert "secret" not in snapshot["error_message"]
    events = client.get(f"/api/tasks/{task_id}/events").json()["events"]
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert any(event["event"] == "tool_start" for event in events)
    assert events[-1]["data"]["status"] == "failed"
    assert client.get(f"/api/tasks/{task_id}/events", params={"after": events[-1]["seq"]}).json()["events"] == []
    _login(client, "bob-key")
    assert client.get(f"/api/tasks/{task_id}").status_code == 404
    assert client.get(f"/api/tasks/{task_id}/events").status_code == 404
    assert client.get("/api/sessions/alice_thread/task").status_code == 404


def test_task_success_replacement_waits_for_old_agent_and_releases_capacity(api_client, monkeypatch):
    client, server = api_client
    stopped = []

    async def agent(query, *args, **kwargs):
        if query == "slow":
            try:
                await asyncio.sleep(3600)
            finally:
                stopped.append("slow")
        assert stopped == ["slow"]
        return "completed"

    monkeypatch.setattr(server, "run_deep_agent", agent)
    _login(client, "alice-key")
    old_id = client.post("/api/task", json={"thread_id": "thread", "query": "slow"}).json()["task_id"]
    new_id = client.post("/api/task", json={"thread_id": "thread", "query": "new"}).json()["task_id"]
    assert _wait_terminal(client, old_id)["status"] == "cancelled"
    snapshot = _wait_terminal(client, new_id)
    assert snapshot["status"] == "succeeded"
    assert snapshot["result"] == "completed"
    assert client.get("/api/sessions/thread/task").json()["task"]["task_id"] == new_id


def test_multiple_websockets_receive_same_task_events(api_client, monkeypatch):
    client, server = api_client

    async def agent(*args, **kwargs):
        server.monitor.report_tool("test")
        return "answer"

    monkeypatch.setattr(server, "run_deep_agent", agent)
    _login(client, "alice-key")
    with client.websocket_connect("/ws/thread") as first:
        with client.websocket_connect("/ws/thread") as second:
            task_id = client.post("/api/task", json={"thread_id": "thread", "query": "test"}).json()["task_id"]
            for socket in (first, second):
                event = socket.receive_json()
                assert event["task_id"] == task_id
                assert event["seq"] == 1
            assert len(server.manager.active_connections["thread"]) == 2


def test_service_shutdown_cancellation_is_interruption(api_client, monkeypatch):
    client, server = api_client
    _login(client, "alice-key")
    task_id = client.post("/api/task", json={"thread_id": "thread", "query": "slow"}).json()["task_id"]
    monkeypatch.setattr(server, "_shutting_down", True)
    task = server.active_tasks[("a", "alice", "thread")]
    client.portal.call(task.cancel)
    snapshot = _wait_terminal(client, task_id)
    assert snapshot["status"] == "interrupted"
    assert snapshot["error_code"] == "SERVICE_SHUTDOWN"


def test_cancel_before_coroutine_starts_releases_slot_and_records_terminal(api_client):
    client, server = api_client
    _login(client, "alice-key")
    from app.security.auth import Identity
    from app.api import task_store
    identity = Identity("alice", "a", authenticated=True)
    history.upsert_session("thread", user_id="alice", tenant_id="a")

    async def cancel_immediately():
        await server._task_semaphore.acquire()
        task_id = task_store.create_task("thread", identity, "query")
        key = ("a", "alice", "thread")
        task = asyncio.create_task(server._run_task_safely("query", "thread", identity, task_id))
        server.active_tasks[key] = task
        task.add_done_callback(lambda done: server._task_done(key, done, task_id, identity))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return task_id

    task_id = client.portal.call(cancel_immediately)
    assert _wait_terminal(client, task_id)["status"] == "cancelled"
    assert server._task_semaphore._value == server.settings.max_concurrent_tasks
