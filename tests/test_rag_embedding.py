"""app/rag/embedding.py：三种向量后端与工厂降级。

测试只碰 hash 后端与降级逻辑，**不加载任何真实模型、不联网**。

降级是工程可用性的兜底（docs/rag-self-built.md 4.5、5.8）：
本地模型需要联网下载 + torch + 磁盘，任一不满足整条链路就不可用。
hash 后端没有语义能力，但能让全链路跑通 —— 所以 `allow_fallback` 的两条路径都要锁死。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.rag.embedding import (
    BGE_QUERY_INSTRUCTION,
    Embedder,
    HashEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    build_embedder,
)


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


# ── HashEmbedder ────────────────────────────────────────────────────


def test_hash_embedder_shape_and_dim():
    embedder = HashEmbedder(dim=64)
    assert embedder.dim == 64
    assert embedder.encode_one("甲").shape == (64,)


def test_hash_embedder_batch_shape():
    embedder = HashEmbedder(dim=64)
    matrix = embedder.encode(["甲", "乙", "丙"])
    assert matrix.shape == (3, 64)
    assert matrix.dtype == np.float32


def test_hash_embedder_is_l2_normalised():
    embedder = HashEmbedder(dim=128)
    vector = embedder.encode_one("风控模型的准确率")
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_hash_embedder_is_deterministic():
    a = HashEmbedder(dim=32).encode_one("同样的文本")
    b = HashEmbedder(dim=32).encode_one("同样的文本")
    assert np.allclose(a, b)


def test_hash_embedder_similar_text_scores_higher():
    """虽然无语义能力，但词面重合度高的问题应当更接近原文。"""
    embedder = HashEmbedder(dim=512)
    query = embedder.encode_one("风控模型的准确率")
    near = embedder.encode_one("风控模型的准确率为 91.5%")
    far = embedder.encode_one("公司食堂本周的菜单")
    assert float(near @ query) > float(far @ query)


def test_hash_embedder_empty_text_yields_zero_vector_without_nan():
    """零向量必须保持为零，不能让除零产生 NaN 污染整个矩阵。"""
    vector = HashEmbedder(dim=32).encode_one("")
    assert not np.isnan(vector).any()
    assert np.allclose(vector, 0.0)


def test_hash_embedder_fingerprint_includes_dim():
    assert HashEmbedder(dim=512).fingerprint == "hash:512"


def test_hash_embedder_satisfies_interface():
    assert isinstance(HashEmbedder(), Embedder)


def test_encode_one_matches_batch_first_row():
    embedder = HashEmbedder(dim=32)
    assert np.allclose(embedder.encode_one("风控"), embedder.encode(["风控"])[0])


# ── OpenAIEmbedder ──────────────────────────────────────────────────


def test_openai_embedder_requires_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(api_key="")


def test_openai_embedder_fingerprint_format():
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)  # 不触发网络客户端构造
    embedder.model = "text-embedding-v3"
    embedder.dim = 1024
    assert embedder.fingerprint == "openai:text-embedding-v3:1024"


# ── LocalEmbedder ───────────────────────────────────────────────────


def test_local_embedder_fingerprint_includes_model_name():
    embedder = LocalEmbedder(model_name="BAAI/bge-small-zh-v1.5")
    assert embedder.fingerprint.startswith("local:BAAI/bge-small-zh-v1.5")
    assert embedder.dim == 0  # 懒加载：构造时不加载模型


def test_local_embedder_is_lazy():
    assert LocalEmbedder()._model is None


@pytest.mark.skipif(
    _has_sentence_transformers(), reason="本机装了 sentence-transformers，不适用缺失分支"
)
def test_local_embedder_raises_actionable_error_without_dependency():
    """依赖缺失必须给出可执行提示，而不是一个裸 ImportError。"""
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        LocalEmbedder().encode(["预热"])


def test_bge_query_instruction_is_defined():
    assert "检索" in BGE_QUERY_INSTRUCTION


# ── build_embedder 工厂 ─────────────────────────────────────────────


def test_build_embedder_hash_provider():
    embedder = build_embedder(provider="hash", dim=64)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dim == 64


def test_build_embedder_is_case_insensitive():
    assert isinstance(build_embedder(provider="HASH", dim=8), HashEmbedder)


def test_build_embedder_defaults_to_local_when_provider_empty():
    try:
        embedder = build_embedder(provider="", dim=8)
    except RuntimeError:
        pytest.skip("本机可用本地模型且 allow_fallback=False，跳过")
    assert isinstance(embedder, (LocalEmbedder, HashEmbedder))


def test_build_embedder_unknown_provider_raises():
    with pytest.raises(ValueError, match="未知的向量后端"):
        build_embedder(provider="magic")


@pytest.mark.skipif(
    _has_sentence_transformers(), reason="本机装了 sentence-transformers，不适用降级分支"
)
def test_build_embedder_local_falls_back_to_hash_by_default():
    """默认 allow_fallback=True：本地模型不可用时静默降级，保证整条链路仍可用。"""
    embedder = build_embedder(provider="local", model="BAAI/bge-small-zh-v1.5", dim=64)
    assert isinstance(embedder, HashEmbedder)


@pytest.mark.skipif(
    _has_sentence_transformers(), reason="本机装了 sentence-transformers，不适用降级分支"
)
def test_build_embedder_local_without_fallback_raises():
    """allow_fallback=False 时必须把问题暴露出来，而不是悄悄降级掩盖。"""
    with pytest.raises(RuntimeError):
        build_embedder(provider="local", model="BAAI/bge-small-zh-v1.5", allow_fallback=False)


def test_build_embedder_openai_requires_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_embedder(provider="openai", api_key="")
