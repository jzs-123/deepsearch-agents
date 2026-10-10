"""
Token 计数模块

优先使用 tiktoken(cl100k_base)精确计数,不可用时回退到字符长度启发式。
统一暴露 count / count_messages,供上下文预算裁剪与日志复用。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖可选
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

# 每条消息的固定开销(角色标记、分隔符等)
_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=1)
def _get_encoder():
    """获取共享的 tiktoken 编码器。"""
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, model: str | None = None) -> int:
    """
    估算文本的 token 数。

    :param text: 待计数的文本
    :param model: 模型名(当前未使用,保留以便未来按模型选编码器)
    :return: token 数
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # 回退:中文约 1 token/字,英文约 4 字符/token,折中按 4 字符估算
    return max(1, len(text) // 4)


class TokenCounter:
    """可复用的 Token 计数对象。"""

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def count(self, text: str) -> int:
        return count_tokens(text, self.model)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """计算一组消息的总 token 数(含每条消息的固定开销)。"""
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += count_tokens(str(content), self.model) + _MESSAGE_OVERHEAD
        return total
