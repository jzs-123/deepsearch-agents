import asyncio
import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from app.api.context import set_session_context, reset_session_context, get_session_context
from app.security.auth import Identity, get_identity_context
from app.utils import history_manager as history
from app.agent.result import AgentResult, StructuredResultError


class ToolCompatibleFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("failure", [None, "empty", "after_answer"])
def test_agent_runs_with_private_workspace_and_persists_answer(tmp_path, monkeypatch, failure):
    model = ToolCompatibleFakeModel(responses=[AIMessage(content="", tool_calls=[{
        "name": "AgentResult", "args": {"status": "success", "answer": "测试回答"}, "id": "result-call",
    }])])
    monkeypatch.setitem(sys.modules, "app.agent.llm", SimpleNamespace(model=model))
    path = Path(__file__).resolve().parents[1] / "app/agent/main_agent.py"
    spec = importlib.util.spec_from_file_location("test_private_workspace_agent", path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    skills = tmp_path / "skills" / "demo"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: demo\ndescription: test\n---\nTest skill", encoding="utf-8")
    monkeypatch.setattr(agent, "project_root_path", tmp_path)
    conn = sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    history._init_db(conn)
    monkeypatch.setattr(history, "_conn", conn)
    published_results = []
    original_report = agent.monitor.report_task_result
    def record_result(result):
        published_results.append(result)
        original_report(result)
    monkeypatch.setattr(agent.monitor, "report_task_result", record_result)
    if failure:
        async def failing_stream(*args, **kwargs):
            if failure == "after_answer":
                yield {"model": {"messages": [AIMessage(content="unfinished answer")]}}
                raise RuntimeError("secret provider failure")
            yield {}
        monkeypatch.setattr(agent, "main_agent", SimpleNamespace(astream=failing_stream))
    try:
        run = agent.run_deep_agent("hello", "private_run", Identity("alice", "team-a", authenticated=True))
        if failure:
            with pytest.raises((RuntimeError, StructuredResultError)):
                asyncio.run(run)
        else:
            assert asyncio.run(run).answer == "测试回答"
        assert get_session_context() is None
        assert get_identity_context() is None
        messages = history.get_session_messages("private_run", "alice", "team-a")
        assert [message["role"] for message in messages] == ["user", "system" if failure else "assistant"]
        assert messages[-1]["content"] == ("任务执行失败，请稍后重试" if failure else "测试回答")
        assert [result.answer for result in published_results] == ([] if failure else ["测试回答"])
        workspace = tmp_path / "output" / "session_private_run"
        assert (workspace / "skills/demo/SKILL.md").exists()
        token = set_session_context(str(workspace))
        try:
            backend = agent.file_backend(None)
            assert backend.cwd == workspace.resolve()
            assert backend._resolve_path("/report.md").is_relative_to(workspace.resolve())
        finally:
            reset_session_context(token)
    finally:
        conn.close()
