"""主 Agent 最终结果契约与有限的 Schema 纠错预算。"""

from contextvars import ContextVar
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_id: str
    type: Literal["web", "database", "upload", "rag"]
    title: str
    uri: str = ""
    locator: str = ""
    excerpt: str = ""


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    title: str = ""


class AgentResult(BaseModel):
    """通过此工具提交最终答案；sources/artifacts 只列实际使用或生成的对象。"""
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["2.0"] = "2.0"
    status: Literal["success", "partial", "refused", "failed"]
    answer: str = Field(min_length=1)
    sources: list[SourceRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def nonempty_answer(cls, value):
        if not value.strip():
            raise ValueError("answer 不能为空")
        return value


class StructuredResultError(ValueError):
    pass


structure_retry_budget: ContextVar[list[int] | None] = ContextVar("structure_retry_budget", default=None)


def retry_structure(error: Exception) -> str:
    budget = structure_retry_budget.get()
    if not budget or budget[0] <= 0:
        raise StructuredResultError("Agent 最终结果未通过 Schema 校验") from error
    budget[0] -= 1
    return f"最终结果格式无效：{error}。请调用 AgentResult 工具提交符合 Schema 的完整结果，不要用 Markdown 代码块或自由文本替代。"
