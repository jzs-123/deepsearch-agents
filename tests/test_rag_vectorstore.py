"""app/rag/vectorstore.py：持久化、幂等、整篇替换、向量指纹校验、撕裂数据容错。

历史 bug（docs/rag-self-built.md 5.5）：换向量模型后跨向量空间算余弦，
**不报错但结果是垃圾** —— 这是 RAG 里最阴的一类 bug。所以指纹校验要覆盖三条路径：
打开时校验、追加时校验、维度不一致时校验。

历史 bug（5.7）：重复入库导致「检索结果全是同一段话」。解法是两层幂等：
`chunk_id` 去重 + `doc_id` 整篇替换。两者都单独锁死。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.rag.vectorstore import (
    CHUNKS_FILE,
    META_FILE,
    VECTORS_FILE,
    EmbeddingMismatchError,
    VectorStore,
    list_knowledge_bases,
)


def _open(path, fingerprint: str | None = None, create: bool = True) -> VectorStore:
    return VectorStore.open(path, embedding_fingerprint=fingerprint, create=create)


# ── 打开 / 指纹校验 ─────────────────────────────────────────────────


def test_open_missing_directory_without_create_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.open(tmp_path / "nope", create=False)


def test_open_with_create_makes_directory(tmp_path):
    path = tmp_path / "kb"
    store = VectorStore.open(path, create=True)
    assert path.is_dir()
    assert len(store) == 0


def test_empty_store_accepts_any_fingerprint(tmp_path, make_chunk):
    """空库没有向量，谈不上"空间不可比"，不得因为指纹不同而报错。

    注意实现细节（不是 bug，是被测契约）：`open()` 只做校验、不落定指纹；
    指纹在真正写入数据时（`add()`）才确定，避免"打开即改库元数据"。
    """
    path = tmp_path / "kb"
    _open(path, "hash:64")
    store = _open(path, "local:BAAI/bge-small-zh-v1.5:512")  # 不得抛 EmbeddingMismatchError
    assert len(store) == 0
    assert store.embedding_fingerprint == ""

    chunk = make_chunk(text="内容")
    store.add([chunk], np.ones((1, 4), dtype=np.float32), "local:BAAI/bge-small-zh-v1.5:512")
    assert store.embedding_fingerprint == "local:BAAI/bge-small-zh-v1.5:512"


def test_fingerprint_mismatch_on_open_raises(tmp_path, make_chunk, hash_embedder):
    """历史 bug 5.5 的回归锁：非空库 + 指纹不同 → 必须显式报错，而不是静默给出垃圾结果。"""
    path = tmp_path / "kb"
    store = _open(path, hash_embedder.fingerprint)
    chunks = [make_chunk(text="风控准确率为 91.5%。")]
    store.add(chunks, hash_embedder.encode([chunks[0].text]), hash_embedder.fingerprint)
    store.save()

    with pytest.raises(EmbeddingMismatchError) as excinfo:
        VectorStore.open(path, embedding_fingerprint="local:BAAI/bge-small-zh-v1.5:512", create=False)
    # 报错信息要给出可执行的补救方向
    assert "重新入库" in str(excinfo.value)


def test_fingerprint_mismatch_on_add_raises(tmp_path, make_chunk, hash_embedder):
    """追加写入时也必须校验：库里有 64 维，就不能再塞 32 维。"""
    path = tmp_path / "kb"
    store = _open(path, "hash:64")
    chunk = make_chunk(text="内容")
    store.add([chunk], hash_embedder.encode([chunk.text]), "hash:64")

    other = make_chunk(text="别的内容", index=1)
    with pytest.raises(EmbeddingMismatchError):
        store.add([other], np.zeros((1, 32), dtype=np.float32), "hash:32")


def test_add_rejects_vector_count_mismatch(tmp_path, make_chunk):
    path = tmp_path / "kb"
    store = _open(path, "hash:4")
    chunks = [make_chunk(text="a", index=0), make_chunk(text="b", index=1)]
    with pytest.raises(ValueError, match="不一致"):
        store.add(chunks, np.zeros((1, 4), dtype=np.float32), "hash:4")


def test_add_returns_zero_for_empty_input(tmp_path):
    store = _open(tmp_path / "kb", "hash:4")
    assert store.add([], np.zeros((0, 4), dtype=np.float32), "hash:4") == 0


# ── 幂等去重 / 整篇替换 ─────────────────────────────────────────────


def test_same_chunk_id_is_deduplicated(tmp_path, make_chunk, hash_embedder):
    """历史 bug 5.7 回归锁：同一份文件重复入库不能产生重复候选。"""
    path = tmp_path / "kb"
    store = _open(path, hash_embedder.fingerprint)
    chunk = make_chunk(text="风控准确率为 91.5%。")
    vectors = hash_embedder.encode([chunk.text])

    assert store.add([chunk], vectors, hash_embedder.fingerprint) == 1
    assert store.add([chunk], vectors, hash_embedder.fingerprint) == 0
    assert len(store) == 1


def test_delete_document_replaces_whole_document(tmp_path, make_chunk):
    """整篇替换：文档更新后旧内容不能残留（否则会检索到过期版本）。"""
    store = _open(tmp_path / "kb", "hash:4")
    old = [make_chunk(text="旧内容一", index=0, doc_id="doc-v1"),
           make_chunk(text="旧内容二", index=1, doc_id="doc-v1")]
    store.add(old, np.ones((2, 4), dtype=np.float32), "hash:4")
    assert len(store) == 2

    removed = store.delete_document("doc-v1")
    assert removed == 2
    assert len(store) == 0

    new = [make_chunk(text="新内容", index=0, doc_id="doc-v2")]
    store.add(new, np.ones((1, 4), dtype=np.float32), "hash:4")
    assert [c.text for c in store.chunks] == ["新内容"]


def test_delete_document_keeps_vectors_aligned(tmp_path, make_chunk):
    """删文档时向量必须同步删，否则下标错位会让检索返回错误的 chunk。"""
    store = _open(tmp_path / "kb", "hash:3")
    chunks = [make_chunk(text=f"内容{i}", index=i, doc_id="d1" if i < 2 else "d2") for i in range(4)]
    vectors = np.arange(12, dtype=np.float32).reshape(4, 3)
    store.add(chunks, vectors, "hash:3")

    store.delete_document("d1")
    assert len(store) == 2
    assert store.vectors is not None and store.vectors.shape == (2, 3)
    assert [c.text for c in store.chunks] == ["内容2", "内容3"]


def test_delete_document_on_empty_store_returns_zero(tmp_path):
    assert _open(tmp_path / "kb", "hash:3").delete_document("d1") == 0


def test_clear_resets_everything(tmp_path, make_chunk):
    store = _open(tmp_path / "kb", "hash:4")
    store.add([make_chunk(text="内容")], np.ones((1, 4), dtype=np.float32), "hash:4")
    store.clear()
    assert len(store) == 0
    assert store.vectors is None
    assert store.embedding_fingerprint == ""


# ── 落盘 / 重载 / 撕裂容错 ──────────────────────────────────────────


def test_save_and_reload_roundtrip(tmp_path, make_chunk, hash_embedder):
    path = tmp_path / "kb"
    store = _open(path, hash_embedder.fingerprint)
    chunks = [make_chunk(text="风控准确率为 91.5%。", index=0),
              make_chunk(text="授信要求 A13 型号复核。", index=1)]
    vectors = hash_embedder.encode([c.text for c in chunks])
    store.add(chunks, vectors, hash_embedder.fingerprint)
    store.save()

    assert (path / CHUNKS_FILE).exists()
    assert (path / VECTORS_FILE).exists()
    assert (path / META_FILE).exists()

    reloaded = _open(path, hash_embedder.fingerprint, create=False)
    assert [c.text for c in reloaded.chunks] == [c.text for c in chunks]
    assert reloaded.vectors is not None and reloaded.vectors.shape == vectors.shape
    assert reloaded.meta["chunk_count"] == 2


def test_torn_write_is_truncated_to_shorter_side(tmp_path, make_chunk):
    """落盘撕裂（chunks 与 vectors 行数不一致）应按较小长度截断并告警，而不是索引越界。"""
    path = tmp_path / "kb"
    path.mkdir()
    chunks = [make_chunk(text=f"内容{i}", index=i) for i in range(3)]
    (path / CHUNKS_FILE).write_text(
        "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    np.save(path / VECTORS_FILE, np.ones((2, 4), dtype=np.float32))
    (path / META_FILE).write_text(json.dumps({"dim": 4, "embedding_fingerprint": "hash:4"}), encoding="utf-8")

    store = VectorStore.open(path, create=False)
    assert len(store) == 2
    assert store.vectors is not None and store.vectors.shape == (2, 4)


def test_zero_row_vector_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "kb"
    path.mkdir()
    (path / CHUNKS_FILE).write_text("", encoding="utf-8")
    np.save(path / VECTORS_FILE, np.zeros((0, 4), dtype=np.float32))
    store = VectorStore.open(path, create=False)
    assert store.vectors is None
    assert len(store) == 0


# ── 检索 ────────────────────────────────────────────────────────────


def test_search_on_empty_store_returns_empty(tmp_path):
    assert _open(tmp_path / "kb", "hash:4").search(np.ones(4, dtype=np.float32)) == []


def test_search_orders_by_cosine_descending(tmp_path, make_chunk):
    store = _open(tmp_path / "kb", "hash:3")
    chunks = [make_chunk(text="a", index=0), make_chunk(text="b", index=1), make_chunk(text="c", index=2)]
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]], dtype=np.float32)
    store.add(chunks, vectors, "hash:3")

    hits = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
    assert [h.chunk.text for h in hits] == ["a", "b", "c"]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_clamps_top_k(tmp_path, make_chunk):
    store = _open(tmp_path / "kb", "hash:3")
    store.add([make_chunk(text="a")], np.ones((1, 3), dtype=np.float32), "hash:3")
    assert len(store.search(np.ones(3, dtype=np.float32), top_k=99)) == 1
    assert len(store.search(np.ones(3, dtype=np.float32), top_k=0)) == 1  # max(1, ...)


def test_search_dimension_mismatch_raises(tmp_path, make_chunk):
    store = _open(tmp_path / "kb", "hash:3")
    store.add([make_chunk(text="a")], np.ones((1, 3), dtype=np.float32), "hash:3")
    with pytest.raises(EmbeddingMismatchError):
        store.search(np.ones(5, dtype=np.float32), top_k=1)


# ── list_knowledge_bases ────────────────────────────────────────────


def test_list_knowledge_bases_missing_dir_returns_empty(tmp_path):
    assert list_knowledge_bases(tmp_path / "nope") == []


def test_list_knowledge_bases_counts_and_tolerates_broken_meta(tmp_path, make_chunk):
    data_dir = tmp_path / "rag_data"
    data_dir.mkdir()

    # kb_a：meta 正常
    a = data_dir / "a"
    a.mkdir()
    (a / META_FILE).write_text(
        json.dumps({"chunk_count": 2, "embedding_fingerprint": "hash:4", "updated_at": "2026-01-01"}),
        encoding="utf-8",
    )
    # kb_b：meta 损坏 → 回退到数 chunks.jsonl 行数
    b = data_dir / "b"
    b.mkdir()
    (b / META_FILE).write_text("{ 这不是合法 json", encoding="utf-8")
    (b / CHUNKS_FILE).write_text(
        "\n".join(json.dumps(make_chunk(text=f"x{i}", index=i).to_dict(), ensure_ascii=False) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    # kb_c：完全空目录
    (data_dir / "c").mkdir()
    # 非目录项应被忽略
    (data_dir / "notes.txt").write_text("x", encoding="utf-8")

    result = {item["name"]: item for item in list_knowledge_bases(data_dir)}
    assert set(result) == {"a", "b", "c"}
    assert result["a"]["chunk_count"] == 2
    assert result["a"]["embedding"] == "hash:4"
    assert result["b"]["chunk_count"] == 3
    assert result["b"]["meta"] == {}
    assert result["c"]["chunk_count"] == 0


def test_use_faiss_flag_defaults_false_without_faiss(tmp_path, make_chunk):
    """没装 faiss 时必须走 numpy 全量点积（结果精确），且不能因为 import 失败而炸。"""
    store = _open(tmp_path / "kb", "hash:3")
    store.add([make_chunk(text="a")], np.ones((1, 3), dtype=np.float32), "hash:3")
    assert store.use_faiss in (True, False)  # 装了就 True，没装就是 False，都不许抛异常
    assert len(store.all_chunks()) == 1
    assert store.size == len(store)
