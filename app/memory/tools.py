"""
记忆工具

把 MemoryManager 的能力封装成 LangChain 工具,交给主智能体在对话中主动
记忆、回忆与遗忘长期信息。工具通过 ContextVar 读取当前用户身份,从而把
记忆自动隔离到当前用户命名空间。
"""

from langchain_core.tools import tool

from app.core.logging import get_logger
from app.memory.manager import get_memory_manager
from app.security.auth import get_identity_context

logger = get_logger("memory.tools")


def _require_identity() -> str:
    """取当前身份;无身份时返回错误提示。"""
    identity = get_identity_context()
    if identity is None:
        return "错误: 当前没有可用的用户身份上下文,无法操作记忆"
    return ""


@tool
def save_memory(
    content: str,
    kind: str = "fact",
    importance: float = 0.5,
) -> str:
    """
    保存一条长期记忆,供跨会话复用。

    适用场景:用户明确要求记住某信息、描述了偏好、提供了后续任务会用到的
    稳定事实或项目背景。不要保存临时、一次性的信息。
    :param content: 要记住的内容(一句话)
    :param kind: 类别,可选 fact(事实)/preference(偏好)/project(项目)/procedure(流程)
    :param importance: 重要度 0~1,默认 0.5
    :return: 保存结果
    """
    err = _require_identity()
    if err:
        return err
    if kind not in ("fact", "preference", "project", "procedure"):
        return f"错误: 未知的记忆类别 {kind!r}"
    identity = get_identity_context()
    key = get_memory_manager().remember(identity, kind, content, importance=importance)
    logger.info("save_memory kind=%s key=%s", kind, key)
    return f"已保存长期记忆 [{kind}]: {content}"


@tool
def recall_memory(query: str, kind: str = "") -> str:
    """
    回忆与当前任务相关的长期记忆。

    :param query: 检索关键词或问题
    :param kind: 限定类别,留空表示召回全部类别
    :return: 命中的记忆内容,或"未找到"
    """
    err = _require_identity()
    if err:
        return err
    identity = get_identity_context()
    kinds = [kind] if kind in ("fact", "preference", "project", "procedure") else None
    results = get_memory_manager().recall(identity, query, kinds=kinds, limit=5)
    if not results:
        return "未找到与当前任务相关的长期记忆"
    rendered = get_memory_manager().render(results)
    logger.info("recall_memory query=%s 命中 %d 条", query, len(results))
    return f"相关长期记忆如下:\n{rendered}"


@tool
def forget_memory(key: str) -> str:
    """
    删除一条长期记忆。

    :param key: 记忆的唯一键(保存时返回)
    :return: 删除结果
    """
    err = _require_identity()
    if err:
        return err
    identity = get_identity_context()
    get_memory_manager().forget(identity, key)
    logger.info("forget_memory key=%s", key)
    return f"已删除记忆: {key}"
