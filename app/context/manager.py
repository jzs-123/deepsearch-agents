"""
上下文装配管理器

每次任务启动前,把「长期记忆 + 会话历史重放 + 当前问题」按序装配成送入模型
的消息列表,并在超预算时从最旧的历史开始裁剪。历史重放解决了 InMemorySaver
进程重启即丢状态的问题;预算裁剪是 DeepAgents 内置摘要压缩之前的第一道防线。

装配结果附带 context_report,便于通过 monitor 推送给前端观测。
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from app.context.token_counter import TokenCounter
from app.core.config import settings
from app.core.logging import get_logger
from app.memory.manager import get_memory_manager
from app.security.auth import Identity
from app.utils.history_manager import get_session_messages

logger = get_logger("context.manager")

# 长期记忆块的前缀标记,用于裁剪时识别并保护
MEMORY_BLOCK_PREFIX = "[系统注入·长期记忆]"


class ContextManager:
    """上下文装配与预算管理的统一入口。"""

    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        memory_manager=None,
    ) -> None:
        self.counter = token_counter or TokenCounter()
        self.memory = memory_manager or get_memory_manager()

    def assemble(
        self,
        session_id: str,
        user_query: str,
        identity: Identity,
        suffix: str = "",
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        装配本次任务送入模型的消息列表。

        :param session_id: 会话 ID,用于读取历史
        :param user_query: 用户本次问题
        :param identity: 当前用户身份,用于记忆召回
        :param suffix: 追加到当前问题末尾的运行时指令(如工作目录约束)
        :return: (messages, context_report)
        """
        report: dict[str, Any] = {
            "memories": 0,
            "history_turns": 0,
            "tokens_before": 0,
            "tokens_after": 0,
            "trimmed": 0,
        }

        # 1) 长期语义记忆召回
        memories = self.memory.recall(identity, user_query, limit=settings.memory_recall_limit)
        memory_block = self.memory.render(memories)
        report["memories"] = len(memories)

        # 2) 情节记忆(会话历史)重放
        history_msgs: list[dict[str, str]] = []
        for m in get_session_messages(session_id, identity.user_id, identity.tenant_id):
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                history_msgs.append({"role": role, "content": content})
        history_msgs = history_msgs[-settings.context_history_messages :]
        report["history_turns"] = len(history_msgs)

        # 3) 按序装配:记忆块 -> 历史 -> 当前问题
        messages: list[dict[str, str]] = []
        if memory_block:
            messages.append({"role": "user", "content": f"{MEMORY_BLOCK_PREFIX}\n{memory_block}"})
        messages.extend(history_msgs)
        content = (user_query + suffix) if suffix else user_query
        messages.append({"role": "user", "content": content})

        report["tokens_before"] = self.counter.count_messages(messages)

        # 4) 预算裁剪
        messages, trimmed = self._trim(messages, settings.context_max_tokens)
        report["trimmed"] = trimmed
        report["tokens_after"] = self.counter.count_messages(messages)

        logger.info(
            "上下文装配完成 session=%s 历史=%d 记忆=%d 裁剪=%d tokens=%d/%d",
            session_id,
            report["history_turns"],
            report["memories"],
            trimmed,
            report["tokens_after"],
            report["tokens_before"],
        )
        return messages, report

    def _trim(self, messages: list[dict[str, str]], budget: int) -> tuple[list[dict[str, str]], int]:
        """
        超预算时从最旧的历史开始裁剪,始终保留记忆块(若存在)与当前问题。

        :return: (裁剪后的 messages, 被裁剪条数)
        """
        total = self.counter.count_messages(messages)
        if total <= budget or len(messages) < 3:
            return messages, 0

        # 保护范围:记忆块(若存在)在首位,当前问题在末位,其余为可裁剪的历史
        has_memory = messages[0]["content"].startswith(MEMORY_BLOCK_PREFIX)
        start = 1 if has_memory else 0
        end = len(messages) - 1  # 末位当前问题不可裁剪
        trimmed = 0

        while total > budget and start < end:
            total -= self.counter.count(messages[start]["content"]) + 4
            messages.pop(start)
            end -= 1
            trimmed += 1

        return messages, trimmed


_manager: Optional[ContextManager] = None
_manager_lock = threading.Lock()


def get_context_manager() -> ContextManager:
    """返回进程内共享的 ContextManager 单例。"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ContextManager()
    return _manager
