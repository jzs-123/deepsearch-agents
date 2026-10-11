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
from app.rag.ocr import OCR_BACKENDS, OcrOptions
from app.rag.rewrite import REWRITE_MODES

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

    # 查询改写(Query Rewrite)
    # 注意:**数据类默认值是 off**,保证直接构造 RagConfig 的调用方(含既有单元测试)行为与改写功能
    # 引入前逐字节一致;应用级默认由 Settings.rag_query_rewrite 提供(= local),CLI 与 Agent 工具
    # 都走 load_rag_config(),所以生产路径默认开启改写。见 docs/rag-query-rewrite.md 第 6 节。
    query_rewrite: str = "off"
    # 变体总数上限(含原始查询):原始查询永远保留,所以 >= 1
    query_variants: int = 3
    # 首轮判定分(向量余弦)低于 min_score 时,是否用改写变体自动重试一次
    retry_on_low_score: bool = True
    # 补充同义词表,格式 "key=alt1|alt2;key2=alt"
    rewrite_synonyms: str = ""

    # OCR(图片/扫描件兜底)
    # 默认开启:但只对**文本层过少的页**生效 —— 有文本层的页面不会被 OCR 覆盖
    # (否则就是用更差的识别结果替换干净原文,是净损失,见 docs/rag-ocr.md)
    ocr_enabled: bool = True
    # windows(系统内置,仅 Windows) / rapidocr(需装 rapidocr-onnxruntime) / none
    ocr_backend: str = "windows"
    # PDF 单页文本层字符数低于该值时对该页做 OCR(0 = 关闭页级 OCR)
    ocr_min_text_chars: int = 50
    ocr_language: str = "zh-Hans-CN"

    # 缓存(embedding / 查询结果 / 生成结果)
    # **默认关闭**:查询与生成缓存一旦陈旧就是"返回过期答案",这种风险不该由默认配置承担;
    # embedding 缓存是内容寻址(不陈旧)但共用这个总开关,保证"一个开关解释全部行为"。
    # 见 docs/rag-enterprise-features.md。
    cache_enabled: bool = False

    # 审计日志(JSONL;每次 search/query 一条,不含原文)
    # 默认开启:审计是这次明确要的能力,且写入失败只 warning、不影响查询。
    audit_enabled: bool = True
    # 审计目录;留空 = 跟随知识库数据根目录下的 .audit/(默认即 rag_data/.audit/)
    audit_dir: str = ""

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
        if self.query_rewrite not in REWRITE_MODES:
            raise ValueError(
                f"query_rewrite 必须是 {REWRITE_MODES} 之一,当前为 {self.query_rewrite!r}"
            )
        if self.query_variants < 1:
            raise ValueError("query_variants 至少为 1(原始查询本身也算一个变体)")
        if self.ocr_backend not in OCR_BACKENDS:
            raise ValueError(
                f"ocr_backend 必须是 {OCR_BACKENDS} 之一,当前为 {self.ocr_backend!r}"
            )
        if self.ocr_min_text_chars < 0:
            raise ValueError("ocr_min_text_chars 不能为负数(0 表示关闭页级 OCR)")

    def ocr_options(self) -> OcrOptions:
        """转成 OCR 层使用的选项对象(loaders 只认 OcrOptions,不反向依赖 RagConfig)。"""
        return OcrOptions(
            enabled=self.ocr_enabled,
            backend=self.ocr_backend,
            min_text_chars=self.ocr_min_text_chars,
            language=self.ocr_language,
        )

    def cache_dir(self, kb: str | None = None) -> Path:
        """缓存目录:`<kb 目录>/.cache/`(缓存与数据同库同生共死,reset 时一起清)。"""
        return self.kb_dir(kb) / ".cache"

    def audit_path(self, kb: str | None = None) -> Path:
        """审计目录:默认 `<数据根>/.audit/`,可用 RAG_AUDIT_DIR 覆盖成独立目录。

        为什么默认跟着数据根走而不是写死 `rag_data/.audit/`:测试与 selftest 用的是临时数据目录,
        跟着走就不会往真实的 rag_data/ 里写审计文件;而默认数据根就是 rag_data,所以默认行为
        仍然是任务要求的 `rag_data/.audit/`。
        """
        if self.audit_dir:
            return Path(self.audit_dir)
        return self.data_path / ".audit"

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
        query_rewrite=settings.rag_query_rewrite,
        query_variants=settings.rag_query_variants,
        retry_on_low_score=settings.rag_retry_on_low_score,
        rewrite_synonyms=settings.rag_rewrite_synonyms,
        ocr_enabled=settings.rag_ocr_enabled,
        ocr_backend=settings.rag_ocr_backend,
        ocr_min_text_chars=settings.rag_ocr_min_text_chars,
        ocr_language=settings.rag_ocr_language,
        cache_enabled=settings.rag_cache_enabled,
        audit_enabled=settings.rag_audit_enabled,
        audit_dir=settings.rag_audit_dir,
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
