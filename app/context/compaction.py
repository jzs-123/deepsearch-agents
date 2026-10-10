"""
上下文摘要压缩封装

DeepAgents 在 create_deep_agent 内部已默认装配 SummarizationMiddleware:
当会话 Token 超过阈值时,自动把旧消息摘要化并卸载到
conversation_history/{thread_id}.md,再触发 ContextOverflowError 时兜底压缩。

本模块不重复注册该中间件(避免双重压缩),只提供两类能力:
1. build_summarization_middleware:按需显式构建/配置压缩中间件
2. summarize_messages:一次性摘要工具,供入口前滚动摘要等场景复用
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, get_buffer_string

from app.core.logging import get_logger

logger = get_logger("context.compaction")

# 摘要提示词:要求保留关键事实、结论与待办
_SUMMARY_PROMPT = (
    "请把以下对话内容压缩成简洁的要点摘要,保留关键事实、结论、用户偏好与未完成事项,"
    "去掉过程性、重复的细节。用中文输出。"
)


def build_summarization_middleware(model, backend):
    """
    构建 DeepAgents 自动摘要压缩中间件(默认阈值由模型 profile 推导)。

    :param model: 已解析的 BaseChatModel 实例
    :param backend: 用于卸载历史的后端(如 FilesystemBackend)
    :return: SummarizationMiddleware 实例
    """
    from deepagents.middleware.summarization import create_summarization_middleware

    return create_summarization_middleware(model, backend)


def summarize_messages(
    model,
    messages: list[AnyMessage],
    summary_prompt: Optional[str] = None,
) -> str:
    """
    用模型把一段消息列表压缩为摘要文本(同步,阻塞调用)。

    :param model: BaseChatModel 实例
    :param messages: 待压缩的消息列表
    :param summary_prompt: 自定义摘要提示词
    :return: 摘要文本
    """
    prompt = summary_prompt or _SUMMARY_PROMPT
    text = get_buffer_string(messages)
    try:
        response = model.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=f"以下是需要压缩的对话内容:\n\n{text}"),
            ]
        )
        return str(response.content) if response.content else ""
    except Exception as e:
        logger.warning("摘要生成失败,返回空摘要: %s", e)
        return ""


async def asummarize_messages(
    model,
    messages: list[AnyMessage],
    summary_prompt: Optional[str] = None,
) -> str:
    """summarize_messages 的异步版本。"""
    prompt = summary_prompt or _SUMMARY_PROMPT
    text = get_buffer_string(messages)
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=f"以下是需要压缩的对话内容:\n\n{text}"),
            ]
        )
        return str(response.content) if response.content else ""
    except Exception as e:
        logger.warning("异步摘要生成失败,返回空摘要: %s", e)
        return ""
