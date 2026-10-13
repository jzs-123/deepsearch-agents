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


def test_deepseek_structured_request_disables_thinking(tmp_path, monkeypatch):
    """走真实 LangChain/OpenAI 请求链路，模拟 DeepSeek 的思考模式限制。"""
    import httpx
    import json
    from langchain_openai import ChatOpenAI
    import app.core.config as config

    requests = []
    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("thinking") != {"type": "disabled"}:
            return httpx.Response(400, json={"error": {
                "message": "Thinking mode does not support this tool_choice", "type": "invalid_request_error",
            }})
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "final", "type": "function", "function": {"name": "AgentResult",
                        "arguments": json.dumps({"status": "success", "answer": "4"})},
                }]}}],
        })
    transport = httpx.MockTransport(provider)
    model = ChatOpenAI(model="deepseek-v4-pro", api_key="test-key",
        timeout=2, max_retries=0,
        base_url="https://api.deepseek.com", extra_body={"other_option": "preserved"},
        http_client=httpx.Client(transport=transport), http_async_client=httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(config, "settings", config.settings.model_copy(update={
        "openai_base_url": "https://api.deepseek.com/v1",
    }))
    monkeypatch.setitem(sys.modules, "app.agent.llm", SimpleNamespace(model=model))
    path = Path(__file__).resolve().parents[1] / "app/agent/main_agent.py"
    spec = importlib.util.spec_from_file_location("test_deepseek_agent", path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    (tmp_path / "skills").mkdir()
    monkeypatch.setattr(agent, "project_root_path", tmp_path)
    conn = sqlite3.connect(tmp_path / "sessions.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    history._init_db(conn)
    monkeypatch.setattr(history, "_conn", conn)
    try:
        assert asyncio.run(agent.run_deep_agent("2+2", "deepseek_test")).answer == "4"
        assert requests[0]["tool_choice"] == "required"
        assert requests[0]["other_option"] == "preserved"
        assert model.extra_body == {"other_option": "preserved"}
    finally:
        conn.close()
