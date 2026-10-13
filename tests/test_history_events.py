from __future__ import annotations

from app.api.monitor import ToolMonitor


def test_monitor_collector_keeps_complete_events(capsys):
    monitor = ToolMonitor()
    events: list[dict] = []
    token = monitor.bind_event_collector(events)
    try:
        monitor.report_tool("测试工具", {"query": "问题"})
    finally:
        monitor.reset_event_collector(token)

    assert len(events) == 1
    assert events[0]["type"] == "monitor_event"
    assert events[0]["event"] == "tool_start"
    assert events[0]["data"] == {"tool_name": "测试工具", "args": {"query": "问题"}}
    assert events[0]["timestamp"]

    monitor.report_tool("后续工具")


def test_task_result_has_stable_json_envelope(capsys):
    monitor = ToolMonitor()
    events = []
    token = monitor.bind_event_collector(events)
    try:
        monitor.report_task_result("最终答案")
    finally:
        monitor.reset_event_collector(token)

    output = events[0]["data"]["output"]
    assert output == {
        "schema_version": "2.0",
        "status": "success",
        "answer": "最终答案",
        "sources": [], "artifacts": [], "warnings": [],
    }
    assert len(events) == 1
    capsys.readouterr()
