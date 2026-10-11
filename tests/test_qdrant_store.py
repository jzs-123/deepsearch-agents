"""QdrantStore 单元测试（mock qdrant-client，沙箱里不依赖真实 qdrant）。

真实 qdrant 端到端验证在用户机器上做；这里用假 client 验证 QdrantStore 的**逻辑**：
幂等去重、指纹校验、delete_document、search 语义、clear、all_chunks、data_version。
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.qdrant_store import QdrantStore, EmbeddingMismatchError
from app.rag.types import Chunk


def _chunk(chunk_id: str, text: str, doc_id: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source="a.md",
        source_path="a.md",
        chunk_index=0,
        doc_id=doc_id,
    )


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class _FakeClient:
    """模拟 qdrant_client 的最小 API。"""

    def __init__(self):
        self.collections: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def collection_exists(self, name) -> bool:
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {}

    def upsert(self, collection_name, points):
        col = self.collections[collection_name]
        for p in points:
            col[p.id] = (list(p.vector), dict(p.payload or {}))

    def query_points(self, collection_name, query, limit, with_payload=True, with_vectors=False):
        col = self.collections.get(collection_name, {})
        scored = []
        for pid, (vec, payload) in col.items():
            scored.append((_cosine(query, vec), pid, payload))
        scored.sort(key=lambda item: -item[0])
        points = [
            SimpleNamespace(id=pid, score=score, payload=payload)
            for score, pid, payload in scored[:limit]
        ]
        return SimpleNamespace(points=points)

    def scroll(self, collection_name, limit=1000, offset=None, with_vectors=False,
               with_payload=True, scroll_filter=None):
        col = self.collections.get(collection_name, {})
        items = []
        for pid, (vec, payload) in col.items():
            if scroll_filter is not None:
                if not self._match_filter(payload, scroll_filter):
                    continue
            items.append(SimpleNamespace(id=pid, payload=payload))
        return items, None  # 假实现不分页，一次返回全部

    @staticmethod
    def _match_filter(payload: dict, scroll_filter) -> bool:
        for cond in scroll_filter.must:
            if payload.get(cond.key) != cond.match.value:
                return False
        return True

    def delete(self, collection_name, points_selector):
        col = self.collections[collection_name]
        ids = set(points_selector)
        for pid in list(col):
            if pid in ids:
                del col[pid]

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def count(self, collection_name):
        return SimpleNamespace(count=len(self.collections.get(collection_name, {})))


def _install_fake_qdrant(monkeypatch) -> _FakeClient:
    """注入假 qdrant_client / qdrant_client.models，让方法内的 import 成功。"""
    models = types.ModuleType("qdrant_client.models")

    class PointStruct:
        def __init__(self, id, vector, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload or {}

    class Distance:
        COSINE = "Cosine"

    class VectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class FieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must=None):
            self.must = must or []

    class MatchValue:
        def __init__(self, value):
            self.value = value

    models.PointStruct = PointStruct
    models.Distance = Distance
    models.VectorParams = VectorParams
    models.FieldCondition = FieldCondition
    models.Filter = Filter
    models.MatchValue = MatchValue

    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.models = models

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)
    return _FakeClient()


def _make_store(tmp_path, fake_client, monkeypatch, fingerprint: str | None = None):
    store = QdrantStore.open(
        tmp_path / "kb",
        embedding_fingerprint=fingerprint,
        create=True,
    )
    monkeypatch.setattr(store, "_get_client", lambda: fake_client)
    return store


# ────────────────────────── 写入与幂等 ──────────────────────────


def test_add_and_all_chunks_roundtrip(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)

    added = store.add(
        [_chunk("a", "授信额度上限 500 元"), _chunk("b", "差旅费报销 800 元")],
        [[1.0, 0.0], [0.0, 1.0]],
        embedding_fingerprint="hash:512",
    )
    assert added == 2
    assert len(store) == 2
    texts = sorted(c.text for c in store.all_chunks())
    assert texts == ["差旅费报销 800 元", "授信额度上限 500 元"]


def test_add_is_idempotent_by_chunk_id(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)

    store.add([_chunk("a", "正文")], [[1.0, 0.0]], "hash:512")
    again = store.add([_chunk("a", "正文（重复）")], [[1.0, 0.0]], "hash:512")
    assert again == 0, "chunk_id 相同必须幂等跳过"
    assert len(store) == 1


def test_add_rejects_fingerprint_mismatch(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    store.add([_chunk("a", "正文")], [[1.0, 0.0]], "hash:512")
    with pytest.raises(EmbeddingMismatchError):
        store.add([_chunk("b", "新正文")], [[0.0, 1.0]], "other:512")


def test_open_rejects_fingerprint_mismatch_when_nonempty(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    store.add([_chunk("a", "正文")], [[1.0, 0.0]], "hash:512")
    store.save()  # chunk_count 落 meta.json

    with pytest.raises(EmbeddingMismatchError):
        QdrantStore.open(tmp_path / "kb", embedding_fingerprint="other:512", create=True)


# ────────────────────────── 检索 ──────────────────────────


def test_search_returns_cosine_similarity(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    store.add(
        [_chunk("a", "风控"), _chunk("b", "市场")],
        [[1.0, 0.0], [0.0, 1.0]],
        "hash:512",
    )
    hits = store.search([1.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "a"
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].chunk.chunk_id == "b"
    assert hits[1].score == pytest.approx(0.0)


def test_search_empty_collection(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    assert store.search([1.0, 0.0]) == []


# ────────────────────────── 删除与清空 ──────────────────────────


def test_delete_document_removes_all_chunks_of_doc(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    store.add(
        [_chunk("a", "一", doc_id="doc1"), _chunk("b", "二", doc_id="doc1"), _chunk("c", "三", doc_id="doc2")],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "hash:512",
    )
    removed = store.delete_document("doc1")
    assert removed == 2
    assert [c.chunk_id for c in store.all_chunks()] == ["c"]


def test_clear_empties_collection(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    store.add([_chunk("a", "正文")], [[1.0, 0.0]], "hash:512")
    store.clear()
    assert len(store) == 0
    assert store.all_chunks() == []


# ────────────────────────── 版本号 ──────────────────────────


def test_data_version_bumps_on_add_and_delete(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    assert store.data_version == 0
    store.add([_chunk("a", "正文")], [[1.0, 0.0]], "hash:512")
    assert store.data_version == 1
    store.delete_document("doc1")
    assert store.data_version == 2


def test_save_persists_meta(tmp_path, monkeypatch):
    fake = _install_fake_qdrant(monkeypatch)
    store = _make_store(tmp_path, fake, monkeypatch)
    store.add([_chunk("a", "正文")], [[1.0, 0.0]], "hash:512")
    store.save()
    meta_path = tmp_path / "kb" / "meta.json"
    assert meta_path.exists()
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["chunk_count"] == 1
    assert meta["vectorstore"] == "qdrant"
    assert meta["embedding_fingerprint"] == "hash:512"
