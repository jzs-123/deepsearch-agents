"""
向量化(Embedding)层

三种后端,统一的 `Embedder` 接口,目的是让「换模型」这件事只改一个配置项:

1. local  —— sentence-transformers 加载 BGE 系列中文模型,语义效果最好,首次需下载模型;
2. openai —— 走 OpenAI 兼容的 /embeddings 接口(如 DashScope text-embedding-v3),无需本地 GPU;
3. hash   —— 零依赖离线兜底(字符 n-gram 哈希 + L2 归一化)。

为什么一定要有 hash 兜底:本地模型下载需要联网、需要几百 MB 磁盘、还要装 torch。
一旦这些前置条件不满足,RAG 链路会整体不可用。hash 向量没有真正的语义能力,但能让
「解析→切块→入库→检索→生成」全链路跑通,便于在没有模型的机器上开发调试与写测试。
注意:三种后端的向量空间互不兼容,同一知识库里严禁混用(见 vectorstore 的维度/模型校验)。
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# BGE 系列官方建议:查询侧加指令前缀能明显提升短查询对长文档的检索效果
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章:"


class Embedder(ABC):
    """向量化后端统一接口。"""

    #: 后端标识,落盘到知识库元数据里用于校验一致性
    name: str = "base"
    #: 向量维度
    dim: int = 0

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """把一批文本编码成 (n, dim) 的 float32 矩阵,行已 L2 归一化。"""

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    @property
    def fingerprint(self) -> str:
        """后端指纹,写入知识库元数据,加载时比对防止「换了模型还读旧向量」。"""
        return f"{self.name}:{self.dim}"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化,归一化后内积等价于余弦相似度。"""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 零向量(如空文本)保持为零,避免除零产生 NaN 污染整个矩阵
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# ────────────────────────────── hash 兜底后端 ──────────────────────────────


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _hash_tokens(text: str) -> list[str]:
    """把文本拆成「英文单词 + 中文单字 + 中文二元组」的 token 列表。

    中文单字区分度低,所以额外生成相邻二元组(bigram),相当于最粗糙的语义单元。
    """
    lowered = text.lower()
    raw = _TOKEN_RE.findall(lowered)
    tokens = list(raw)
    cjk = [t for t in raw if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


class HashEmbedder(Embedder):
    """零依赖离线向量:token 哈希到固定维度 + 词频加权(sublinear TF) + L2 归一化。

    本质是「哈希技巧版的 TF 向量」,只能捕捉词面重叠,不具备真正的语义泛化能力。
    它的价值是让整个 RAG 链路在没有任何模型和网络的环境下依然可运行、可测试。
    """

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            for token in _hash_tokens(text or ""):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                # 用哈希的另一段决定符号,降低哈希冲突带来的系统性偏置
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                counts[idx] = counts.get(idx, 0.0) + sign
            for idx, value in counts.items():
                # sublinear scaling:出现多次的词不应线性放大权重
                matrix[row, idx] = np.sign(value) * (1.0 + np.log1p(abs(value)))
        return _l2_normalize(matrix)


# ────────────────────────────── OpenAI 兼容接口后端 ──────────────────────────────


class OpenAIEmbedder(Embedder):
    """调用 OpenAI 兼容 /embeddings 接口的向量后端(如 DashScope、硅基流动等)。"""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: str = "",
        base_url: str = "",
        dim: int = 1024,
        batch_size: int = 16,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("使用 openai 向量后端必须配置 OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("缺少 openai 依赖,请执行: pip install openai") from exc

        self.model = model
        self.dim = int(dim)
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        vectors: list[list[float]] = []
        # 接口一般限制单次条数,分批提交
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dim,
            )
            vectors.extend(item.embedding for item in response.data)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.size and matrix.shape[1] != self.dim:
            # 服务端可能忽略 dimensions 参数,以实际返回为准并同步维度
            self.dim = int(matrix.shape[1])
        return _l2_normalize(matrix)

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.model}:{self.dim}"


# ────────────────────────────── 本地模型后端 ──────────────────────────────


class LocalEmbedder(Embedder):
    """sentence-transformers 本地模型后端,默认 BGE-small-zh-v1.5。

    懒加载:只有真正 encode 时才加载模型,避免 import 阶段就吃几百 MB 内存和几秒启动时间。
    """

    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "auto"):
        self.model_name = model_name
        self._device_pref = device
        self._model = None
        self.dim = 0

    def _resolve_device(self) -> str:
        if self._device_pref and self._device_pref != "auto":
            return self._device_pref
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 sentence-transformers 依赖,请执行: pip install sentence-transformers,"
                "或把 RAG_EMBEDDING_PROVIDER 改为 hash 使用离线兜底"
            ) from exc

        device = self._resolve_device()
        logger.info("加载本地向量模型 %s (device=%s)", self.model_name, device)
        try:
            # 先只查本机缓存，避免每次启动都向 Hugging Face 发送 HEAD 请求。
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                local_files_only=True,
            )
        except Exception:
            # 首次使用且本机没有模型时才联网下载；下载完成后后续启动走上面的本地路径。
            logger.info("本机未缓存向量模型，首次从 Hugging Face 获取 %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=device)
        # 以模型实际输出维度为准,不硬编码 512/768,避免换模型后维度写死出错
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:  # 兼容旧版 sentence-transformers
            get_dimension = self._model.get_sentence_embedding_dimension
        self.dim = int(get_dimension())
        logger.info("本地向量模型加载完成,dim=%d", self.dim)
        return self._model

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        model = self._load()
        payload = list(texts)
        if is_query and "bge" in self.model_name.lower():
            # BGE 官方推荐的查询前缀,对短查询召回提升明显
            payload = [f"{BGE_QUERY_INSTRUCTION}{t}" for t in payload]
        vectors = model.encode(
            payload,
            batch_size=32,
            normalize_embeddings=True,  # 归一化后内积即余弦相似度
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.model_name}:{self.dim}"


# ────────────────────────────── 工厂 ──────────────────────────────


def build_embedder(
    provider: str = "local",
    model: str = "BAAI/bge-small-zh-v1.5",
    dim: int = 512,
    device: str = "auto",
    api_key: str = "",
    base_url: str = "",
    allow_fallback: bool = True,
) -> Embedder:
    """按配置构造向量后端。

    allow_fallback=True 时,local 后端加载失败(没装依赖 / 模型下载不了)会自动降级到
    hash 后端,保证链路可用;降级会打 warning,方便发现「其实没用到语义模型」。
    """
    provider = (provider or "local").lower()
    if provider == "hash":
        return HashEmbedder(dim=dim)
    if provider == "openai":
        return OpenAIEmbedder(model=model, api_key=api_key, base_url=base_url, dim=dim)
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=model, device=device)
            # 触发一次真实加载,把「下载失败/依赖缺失」的问题在构造阶段暴露出来
            embedder.encode(["预热"], is_query=False)
            return embedder
        except Exception as exc:
            if not allow_fallback:
                raise
            logger.warning("本地向量模型不可用(%s),降级为 hash 离线向量。检索质量会明显下降。", exc)
            return HashEmbedder(dim=dim)
    raise ValueError(f"未知的向量后端 {provider!r},可选:local / openai / hash")


__all__ = [
    "Embedder",
    "HashEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
    "BGE_QUERY_INSTRUCTION",
]
