"""
Agent 执行过程监控模块

负责把工具调用、子智能体调用、任务结果和会话目录等事件统一包装后推送给前端
在 Web 服务中优先通过 WebSocket 定向推送；在脚本调试场景中保留控制台输出
"""

import asyncio
import builtins
import datetime
from contextvars import ContextVar, Token
from typing import Any, Optional

from fastapi import WebSocket

from app.api.context import get_thread_context
from app.api.task_store import append_event, current_task_id
from app.agent.result import AgentResult


_event_collector: ContextVar[Optional[list[dict[str, Any]]]] = ContextVar(
    "monitor_event_collector",
    default=None,
)


class ToolMonitor:
    """
    工具和助手调用的统一监控入口

    业务工具只需要导入全局 monitor，并调用 report_tool/report_assistant 等方法
    具体是通过 WebSocket 推送，还是输出到脚本运行时，由本类内部统一处理
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ToolMonitor, cls).__new__(cls)
            cls._instance.websocket_manager = None
        return cls._instance

    def set_websocket_manager(self, manager: "ConnectionManager") -> None:
        """绑定 FastAPI WebSocket 连接管理器"""
        self.websocket_manager = manager

    def bind_event_collector(self, collector: list[dict[str, Any]]) -> Token:
        """收集本次 Agent 运行产生的完整事件，供会话历史持久化。"""
        return _event_collector.set(collector)

    def reset_event_collector(self, token: Token) -> None:
        _event_collector.reset(token)

    def _emit(
        self,
        event_type: str,
        message: str,
        data: Optional[dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """
        构造统一监控事件，并尝试推送到当前 thread_id 对应的前端连接

        :param event_type: 事件类型，例如 tool_start、assistant_call
        :param message: 面向前端展示的事件说明
        :param data: 附加结构化数据
        """
        payload = {
            "type": "monitor_event",
            "event": event_type,
            "message": message,
            "data": data or {},
            "timestamp": datetime.datetime.now().isoformat(),
        }

        task_id = current_task_id.get()
        if task_id:
            payload = append_event(task_id, payload)

        collector = _event_collector.get()
        if collector is not None:
            collector.append(payload)

        if self.websocket_manager:
            try:
                thread_id = thread_id or get_thread_context()
                manager_loop = self.websocket_manager.loop

                if manager_loop and not manager_loop.is_closed() and thread_id:
                    self._send_to_websocket(payload, thread_id, manager_loop)
            except Exception as e:
                print(f"[Monitor] WebSocket send failed: {e}")

        # DeepAgents 脚本调试时，如果运行时暴露了 stream_writer，也同步写入流式输出
        if hasattr(builtins, "runtime") and hasattr(builtins.runtime, "stream_writer"):
            try:
                builtins.runtime.stream_writer(payload)
            except Exception:
                pass

        # 控制台保底输出，便于无前端场景下观察执行过程
        print(f"\n[Monitor:{event_type}] {message}")

    def _send_to_websocket(
        self,
        payload: dict[str, Any],
        thread_id: str,
        manager_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        将监控事件投递到 WebSocket 所在事件循环

        FastAPI 的 WebSocket 必须在创建它的事件循环中发送消息
        如果当前代码已经在同一个循环里，直接 create_task；否则使用线程安全投递
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        coroutine = self.websocket_manager.send_to_thread(payload, thread_id)
        if current_loop and current_loop == manager_loop:
            current_loop.create_task(coroutine)
        else:
            asyncio.run_coroutine_threadsafe(coroutine, manager_loop)

    def report_tool(
        self,
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """报告开始执行某个工具"""
        self._emit(
            "tool_start",
            f"开始执行工具: {tool_name}",
            {"tool_name": tool_name, "args": args},
        )

    def report_assistant(
        self,
        assistant_name: str,
        args: Optional[dict[str, Any]] = None,
    ) -> None:
        """报告正在调用某个子智能体"""
        self._emit(
            "assistant_call",
            f"正在调用助手: {assistant_name}",
            {"assistant_name": assistant_name, "args": args},
        )

    def report_task_result(self, result: AgentResult | str) -> None:
        """报告任务最终结果"""
        output = result if isinstance(result, AgentResult) else AgentResult(status="success", answer=result)
        self._emit(
            "task_result",
            "任务执行完成",
            {
                "result": output.answer,
                "output": output.model_dump(),
            },
        )

    def report_rag_sources(self, payload: dict[str, Any]) -> None:
        """报告一次 RAG 检索使用的结构化原文依据。"""
        count = len(payload.get("sources") or [])
        self._emit("rag_sources", f"检索到 {count} 条知识库依据", payload)

    def report_task_cancelled(self, thread_id: Optional[str] = None) -> None:
        """报告任务已被用户取消"""
        self._emit("task_cancelled", "任务已取消", thread_id=thread_id)

    def report_session_dir(self, path: str) -> None:
        """报告当前任务工作目录"""
        self._emit("session_created", f"工作目录已创建: {path}", {"path": path})


monitor = ToolMonitor()


class ConnectionManager:
    """
    WebSocket 连接管理器

    active_connections 使用 thread_id 作为 key，保证监控事件只推送给对应任务的前端连接
    """

    def __init__(self) -> None:
        self.active_connections: dict[str, set[WebSocket]] = {}
        # WebSocket 发送必须回到创建连接的事件循环，因此启动时需要显式绑定 loop
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """绑定 FastAPI 主事件循环，并同步注册到 monitor"""
        self.loop = loop
        monitor.set_websocket_manager(self)
        print(f"[Monitor] ConnectionManager manually bound to loop: {id(self.loop)}")

    async def connect(self, websocket: WebSocket, thread_id: str) -> None:
        """接受 WebSocket 连接，并按 thread_id 保存"""
        await websocket.accept()
        self.active_connections.setdefault(thread_id, set()).add(websocket)
        print(f"Client connected: {thread_id}")

    def disconnect(self, websocket: WebSocket, thread_id: str) -> None:
        """移除已经断开的 WebSocket 连接"""
        connections = self.active_connections.get(thread_id, set())
        if websocket in connections:
            connections.remove(websocket)
            if not connections:
                self.active_connections.pop(thread_id, None)
            print(f"Client disconnected: {thread_id}")
        else:
            print(f"Stale websocket disconnected, current connection kept: {thread_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """向指定 WebSocket 发送纯文本消息"""
        await websocket.send_text(message)

    async def send_to_thread(self, message: dict[str, Any], thread_id: str) -> None:
        """向指定 thread_id 对应的前端连接发送 JSON 消息"""
        for websocket in tuple(self.active_connections.get(thread_id, set())):
            try:
                await websocket.send_json(message)
            except Exception:
                self.disconnect(websocket, thread_id)


manager = ConnectionManager()
