"""
RAG 运行时配置

app/core/config.py 里的 Settings 面向整个项目,这里把其中 rag_* 字段整理成
RAG 模块专用的不可变配置对象,好处是:
1. 模块内不依赖全局单例,单元测试可以直接构造配置;
2. 集中校验非法取值(如 chunk_overlap >= chunk_size),避免脏配置跑到检索阶段才炸。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app.core.config import PROJECT_ROOT, settings

# 合法取值枚举,便于 CLI 直接校验用户输入
EMBEDDING_PROVIDERS = ("local", "openai", "hash")
RERANK_MODES = ("lexical", "llm", "off")


@dataclass(frozen=True)
class RagConfig:
    """RAG 模块全部可调参数。"""

    data_dir: Path
    default_kb: str = "default"

    # 向量化
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_device: str = "auto"

    # 切块
    chunk_size: int = 500
    chunk_overlap: int = 80

    # 检索
    top_k: int = 10
    top_n: int = 4
    rrf_k: int = 60
    rerank_mode: str = "lexical"
    min_score: float = 0.2

    # 生成所用大模型(与主项目共用一套 OpenAI 兼容配置)
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.1
    llm_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"embedding_provider 必须是 {EMBEDDING_PROVIDERS} 之一,当前为 {self.embedding_provider!r}"
            )
        if self.rerank_mode not in RERANK_MODES:
            raise ValueError(f"rerank_mode 必须是 {RERANK_MODES} 之一,当前为 {self.rerank_mode!r}")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        # 重叠必须小于 chunk 本身,否则切块窗口无法前进(会退化成死循环)
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                f"chunk_overlap 必须满足 0 <= overlap < chunk_size,当前 {self.chunk_overlap} / {self.chunk_size}"
            )
        if self.top_n > self.top_k:
            raise ValueError(f"top_n({self.top_n}) 不能大于 top_k({self.top_k})")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    def with_kb(self, kb: str | None) -> "RagConfig":
        """返回指定知识库的配置副本(知识库名不影响其它参数)。"""
        return replace(self, default_kb=kb or self.default_kb)

    def kb_dir(self, kb: str | None = None) -> Path:
        """知识库目录:<data_dir>/<kb>。"""
        return self.data_path / (kb or self.default_kb)


def load_rag_config(**overrides) -> RagConfig:
    """从项目 Settings 装配 RAG 配置,overrides 用于 CLI/测试临时覆盖。"""
    base = RagConfig(
        data_dir=Path(settings.rag_data_dir).resolve(),
        default_kb=settings.rag_default_kb,
        embedding_provider=settings.rag_embedding_provider,
        embedding_model=settings.rag_embedding_model,
        embedding_dim=settings.rag_embedding_dim,
        embedding_device=settings.rag_embedding_device,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        top_n=settings.rag_top_n,
        rrf_k=settings.rag_rrf_k,
        rerank_mode=settings.rag_rerank_mode,
        min_score=settings.rag_min_score,
        llm_model=settings.llm_model,
        llm_base_url=settings.openai_base_url,
        llm_api_key=settings.openai_api_key,
    )
    return replace(base, **overrides) if overrides else base


def ensure_within(base: Path, target: Path) -> Path:
    """校验 target 位于 base 之内,防止知识库名里带 ../ 越界读写。"""
    base_resolved = base.resolve()
    target_resolved = target.resolve()
    if base_resolved != target_resolved and base_resolved not in target_resolved.parents:
        raise ValueError(f"非法路径 {target},必须位于 {base_resolved} 之内")
    return target_resolved


__all__ = ["RagConfig", "load_rag_config", "ensure_within", "PROJECT_ROOT"]
