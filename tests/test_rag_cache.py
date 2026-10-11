"""缓存机制(embedding / 查询结果 / 生成结果):命中、失效与"绝不假装重算"。

三条主线:

1. **默认关闭**:`cache_enabled=False` 时既不读也不写缓存,`search/query` 行为与引入缓存前一致(不产生 .cache 目录);
2. **命中与标注**:开启后第二次同查询必须命中,结果对象 `cached=True`(可审计),分数与首次一致;
3. **失效**:库内 chunk 增删改(数据版本号 +1)→ 查询/生成缓存整体失效;`reset()` → 缓存目录清空;
   embedding 缓存按 `(文本 SHA1, 向量指纹)` 键,换后端不会误命中。

另覆盖降级:缓存文件损坏、缓存目录不可写时,查询必须照常成功。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.rag.cache import (
    EMBEDDINGS_FILE,
    GENERATION_CACHE_FILE,
    QUERY_CACHE_FILE,
    PipelineCache,
    content_key,
)
from app.rag.config import RagConfig
from app.rag.embedding import HashEmbedder
from app.rag.pipeline import RagPipeline

DOC = "# 风控制度\n\n授信额度上限为 500 元；连续履约 6 个月可提额至 5000 元。\n"
DOC2 = "# 市场方案\n\n全年新增会员 50 万。\n"
DOC3 = "# 财务制度\n\n差旅费报销上限为每日 800 元，需附发票。\n"


class CountingEmbedder:
    """带调用计数的 embedder:用它证明"命中缓存时确实没有重新编码"。"""

    def __init__(self, dim: int = 512):
        self._inner = HashEmbedder(dim=dim)
        self.dim = dim
        self.encoded_batches: list[list[str]] = []
        self.single_calls: list[str] = []

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    def encode(self, texts, is_query: bool = False):
        self.encoded_batches.append(list(texts))
        return self._inner.encode(texts, is_query=is_query)

    def encode_one(self, text: str, is_query: bool = False):
        self.single_calls.append(text)
        return self._inner.encode_one(text, is_query=is_query)


def make_pipeline(tmp_path, embedder, kb: str = "cachekb", **overrides):
    params = dict(
        data_dir=tmp_path / "_rag_data",
        default_kb=kb,
        embedding_provider="hash",
        chunk_size=500,
        chunk_overlap=80,
        query_rewrite="off",
        cache_enabled=False,
    )
    params.update(overrides)
    return RagPipeline(kb=kb, config=RagConfig(**params), embedder=embedder)


# ── 1. 默认关闭 ─────────────────────────────────────────────────────


def test_cache_disabled_by_default_creates_no_cache_dir(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder)
    assert pipeline.config.cache_enabled is False
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度", top_n=2)
    assert not (pipeline.kb_dir / ".cache").exists(), "缓存关闭时不得产生缓存文件"
    assert pipeline.last_search_debug["cache"]["used"] is False


def test_search_with_cache_disabled_matches_uncached_content(tmp_path, hash_embedder):
    """关闭缓存时的检索结果必须与非缓存路径完全一致(顺序/分数/通道)。"""
    pipeline = make_pipeline(tmp_path, hash_embedder)
    pipeline.add_texts([DOC, DOC2], source="doc.md")
    first = pipeline.search("授信额度", top_n=2)
    second = pipeline.search("授信额度", top_n=2)
    signature = lambda items: [  # noqa: E731
        (i.chunk.chunk_id, i.rank, round(i.score, 6), i.cached) for i in items
    ]
    assert signature(first) == signature(second)
    assert all(item.cached is False for item in first)


# ── 2. 查询结果缓存:命中与标注 ──────────────────────────────────────


def test_query_cache_miss_then_hit(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    pipeline.add_texts([DOC, DOC2], source="doc.md")

    first = pipeline.search("授信额度上限", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is False
    assert all(item.cached is False for item in first)

    second = pipeline.search("授信额度上限", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is True, "第二次同查询必须命中缓存"
    assert pipeline.last_search_debug["cached"] is True
    assert second and all(item.cached is True for item in second), "命中缓存的结果必须标注 cached=True"

    signature = lambda items: [  # noqa: E731
        (i.chunk.chunk_id, i.rank, round(i.score, 6), tuple(i.channels)) for i in items
    ]
    assert signature(second) == signature(first), "缓存必须返回与首次一致的结果"

    cache_file = pipeline.kb_dir / ".cache" / QUERY_CACHE_FILE
    assert cache_file.exists()
    lines = [json.loads(line) for line in cache_file.read_text(encoding="utf-8").splitlines() if line]
    assert lines and lines[0]["v"]["results"], "缓存文件里要有结果载荷"
    assert "text" not in json.dumps(lines[0]), "缓存里不存原文"


def test_query_cache_is_isolated_by_filter(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    pipeline.add_texts([DOC, DOC2], source="doc.md")
    pipeline.search("制度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is False


def test_query_cache_is_isolated_by_top_n(tmp_path, hash_embedder):
    """top_n 必须进缓存键:top_n=2 写入的缓存不得被 top_n=3 命中,否则会少拿结果。"""
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    pipeline.add_texts([DOC, DOC2, DOC3], source="doc.md")

    first = pipeline.search("制度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is False
    assert len(first) == 2

    # 同 top_n 命中
    second = pipeline.search("制度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is True

    # 不同 top_n 不得命中旧缓存
    third = pipeline.search("制度", top_n=3)
    assert pipeline.last_search_debug["cache"]["hit"] is False, "top_n 不同不能命中同一缓存"
    assert len(third) == 3


def test_query_cache_is_isolated_by_filter(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    pipeline.add_texts([DOC, DOC2], source="doc.md")
    pipeline.search("制度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is False
    pipeline.search("制度", top_n=2, filter={"部门": "风控"})
    assert pipeline.last_search_debug["cache"]["hit"] is False, "不同 filter 不能互相命中"
    pipeline.search("制度", top_n=2, filter={"部门": "风控"})
    assert pipeline.last_search_debug["cache"]["hit"] is True


def test_query_cache_invalidated_when_store_changes(tmp_path, hash_embedder):
    """★ 库内任何 chunk 增删改 → 数据版本号 +1 → 该库查询缓存整体失效。"""
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    version_before = pipeline.store.data_version

    pipeline.search("授信额度", top_n=2)
    pipeline.search("授信额度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is True

    pipeline.add_texts([DOC2], source="doc2.md")
    assert pipeline.store.data_version > version_before, "数据变更必须让版本号递增"
    pipeline.cache.set_version(pipeline.store.data_version)

    pipeline.search("授信额度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is False, "库变了以后旧缓存必须失效"
    pipeline.search("授信额度", top_n=2)
    assert pipeline.last_search_debug["cache"]["hit"] is True, "失效后重新写入,再查又能命中"


def test_delete_document_also_bumps_version(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    added = pipeline.add_texts([DOC], source="doc.md")
    assert added > 0
    version = pipeline.store.data_version
    doc_id = pipeline.store.all_chunks()[0].doc_id
    removed = pipeline.store.delete_document(doc_id)
    assert removed > 0
    assert pipeline.store.data_version > version


def test_reset_clears_cache_files(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度", top_n=2)
    cache_dir = pipeline.kb_dir / ".cache"
    assert cache_dir.exists() and any(cache_dir.iterdir())

    pipeline.reset()
    assert pipeline.store.data_version == 0
    remaining = list(cache_dir.iterdir()) if cache_dir.exists() else []
    assert remaining == [], f"reset 后缓存必须清空,仍有:{remaining}"


# ── 3. embedding 缓存 ───────────────────────────────────────────────


def test_embedding_cache_hits_on_reingest_without_reencoding(tmp_path):
    embedder = CountingEmbedder()
    pipeline = make_pipeline(tmp_path, embedder, kb="embedkb", cache_enabled=True)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "a.md").write_text(DOC, encoding="utf-8")

    pipeline.ingest(docs)
    first_batches = len(embedder.encoded_batches)
    first_report = pipeline.ingest(docs)  # 同一份文档再入库一次
    assert first_batches >= 1
    assert len(embedder.encoded_batches) == first_batches, "第二次入库不应再调用向量后端"
    assert first_report.embedding_cache["hits"] == first_report.embedding_cache["texts"]
    assert first_report.embedding_cache["misses"] == 0


def test_embedding_cache_does_not_hit_when_fingerprint_changes(tmp_path):
    """键里含向量后端指纹:换后端后旧缓存不会被误用(同一库目录也不会串味)。

    说明:这里直接测缓存层——用不同指纹去**打开**已有的非空知识库会被 VectorStore 的
    指纹校验拦下(EmbeddingMismatchError),那是另一条独立保护,不在这里重复测。
    """
    kb_dir = tmp_path / "kb"
    PipelineCache(kb_dir=kb_dir, fingerprint="hash:512", enabled=True).put_embeddings(
        ["甲"], [[1.0] * 512]
    )
    assert (kb_dir / ".cache" / EMBEDDINGS_FILE).exists()

    other = PipelineCache(kb_dir=kb_dir, fingerprint="hash:128", enabled=True)
    assert other.get_embeddings(["甲"]) == {}, "指纹不同不得命中"

    same = PipelineCache(kb_dir=kb_dir, fingerprint="hash:512", enabled=True)
    assert set(same.get_embeddings(["甲"])) == {0}, "指纹相同(新进程实例)必须命中"


def test_pipeline_cache_embedding_roundtrip(tmp_path):
    cache = PipelineCache(kb_dir=tmp_path / "kb", fingerprint="hash:4", enabled=True)
    texts = ["甲", "乙"]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    cache.put_embeddings(texts, vectors)
    hits = PipelineCache(kb_dir=tmp_path / "kb", fingerprint="hash:4", enabled=True).get_embeddings(texts)
    assert set(hits) == {0, 1}
    assert hits[0] == pytest.approx(vectors[0])
    assert content_key("甲") in cache.embedding_key("甲")


def test_embedding_cache_dim_mismatch_falls_back_to_full_encode(tmp_path):
    """缓存里是 4 维、当前后端是 8 维 → 视为不可用,全量重算而不是拼出畸形矩阵。"""
    kb_dir = tmp_path / "kb"
    seed = PipelineCache(kb_dir=kb_dir, fingerprint="hash:8", enabled=True)
    seed.put_embeddings(["甲"], [[1.0, 0.0, 0.0, 0.0]])

    embedder = HashEmbedder(dim=8)
    pipeline = make_pipeline(tmp_path, embedder, kb="dimmismatch", cache_enabled=True)
    pipeline.kb_dir.mkdir(parents=True, exist_ok=True)
    pipeline.cache = PipelineCache(kb_dir=pipeline.kb_dir, fingerprint=embedder.fingerprint, enabled=True)
    pipeline.cache.put_embeddings(["甲"], [[1.0, 0.0, 0.0, 0.0]])
    matrix = pipeline._encode_documents(["甲"])
    assert matrix.shape == (1, 8), "维度不一致时必须重新编码成当前后端维度"


# ── 4. 生成结果缓存 ─────────────────────────────────────────────────


def test_generation_cache_skips_second_llm_call(tmp_path, hash_embedder, stub_client, monkeypatch):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="genkb", cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    stub = stub_client(reply="授信额度上限为 500 元[1]。")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)

    first = pipeline.query("授信额度上限")
    assert len(stub.calls) == 1
    assert first.answer == "授信额度上限为 500 元[1]。"
    assert first.debug["cache_hit"] is False

    second = pipeline.query("授信额度上限")
    assert len(stub.calls) == 1, "第二次问答应直接命中生成缓存,不再调用大模型"
    assert second.answer == first.answer
    assert second.debug["cache_hit"] is True
    assert second.debug["cache_type"] == "generation"
    assert (pipeline.kb_dir / ".cache" / GENERATION_CACHE_FILE).exists()


def test_generation_cache_disabled_by_default_calls_llm_every_time(
    tmp_path, hash_embedder, stub_client, monkeypatch
):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="genoff", cache_enabled=False)
    pipeline.add_texts([DOC], source="doc.md")
    stub = stub_client(reply="回答[1]。")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)
    pipeline.query("授信额度上限")
    pipeline.query("授信额度上限")
    assert len(stub.calls) == 2
    assert not (pipeline.kb_dir / ".cache").exists()


def test_generation_cache_invalidated_after_store_change(
    tmp_path, hash_embedder, stub_client, monkeypatch
):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="geninv", cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    stub = stub_client(reply="回答[1]。")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)
    pipeline.query("授信额度上限")
    pipeline.query("授信额度上限")
    assert len(stub.calls) == 1

    pipeline.add_texts([DOC2], source="doc2.md")  # 库变了
    pipeline.cache.set_version(pipeline.store.data_version)
    pipeline.query("授信额度上限")
    assert len(stub.calls) == 2, "库变更后生成缓存必须失效并重新生成"


# ── 5. 降级:缓存坏了不能拖垮查询 ────────────────────────────────────


def test_corrupted_cache_line_is_ignored(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="brokencache", cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度", top_n=2)

    cache_file = pipeline.kb_dir / ".cache" / QUERY_CACHE_FILE
    with cache_file.open("a", encoding="utf-8") as handle:
        handle.write("{半截 JSON\n")

    fresh = make_pipeline(tmp_path, hash_embedder, kb="brokencache", cache_enabled=True)
    results = fresh.search("授信额度", top_n=2)
    assert results, "坏行不能影响检索"
    assert fresh.last_search_debug["cache"]["hit"] is True, "有效行仍应命中"


def test_unwritable_cache_dir_does_not_break_search(tmp_path, hash_embedder, caplog):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="unwritable", cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    # 用一个"文件"占住缓存目录路径 → mkdir 必然失败
    cache_path = pipeline.kb_dir / ".cache"
    if cache_path.exists():
        for child in cache_path.iterdir():
            child.unlink()
        cache_path.rmdir()
    cache_path.write_text("我不是目录", encoding="utf-8")

    results = pipeline.search("授信额度", top_n=2)
    assert results, "缓存写不了也必须返回检索结果"


def test_cache_describe_reports_paths_and_versions(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="describekb", cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    described = pipeline.cache.describe()
    assert described["enabled"] is True
    assert described["dir"].endswith(".cache")
    assert described["data_version"] == pipeline.store.data_version
    assert described["fingerprint"].startswith("hash:")


def test_stats_exposes_cache_state(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="statskb", cache_enabled=False)
    stats = pipeline.stats()
    assert stats["cache_enabled"] is False
    assert stats["cache"]["enabled"] is False
    assert "data_version" in stats
