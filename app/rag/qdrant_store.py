"""
Qdrant 向量存储后端（可插拔，替代 numpy 全量点积）

与 `app/rag/vectorstore.py::VectorStore` 实现**完全相同的接口**，所以 pipeline 不需要知道
底层是 numpy 还是 qdrant。区别只在"向量与 chunk 存哪"：

- numpy：`chunks.jsonl` + `vectors.npy`（全量内存 + 磁盘持久化）
- qdrant：本地嵌入模式（`QdrantClient(path=...)`）或远程 server（`QdrantClient(url=...)`）

关键设计取舍：
1. **存自己算好的向量**，绝不用 qdrant 的 embedding 能力（本地模式要 embedding 得另装
   fastembed，我们不需要，也不该让它擅自重新向量化）。所以 collection 的 vectors_config
   只用 `Distance.COSINE` 声明"输入向量已就绪，按余弦算相似度"。
2. **sidecar `meta.json` 复用**：fingerprint / dim / data_version / chunk_size / filter_fields
   这类"库级元数据"仍写在 `<kb>/meta.json`，与 numpy 版完全一致 —— 这样
   `list_knowledge_bases`、指纹校验、缓存失效（靠 data_version）零改动就能工作。
3. **本地模式每个知识库一个独立存储目录** `<kb>/qdrant/`：reset 删目录即清库，数据仍落在
   `rag_data/` 下，与"数据与代码分离"的原则一致。远程模式则共享一个 server，collection 名
   用知识库名区分。
4. **numpy 仍是默认 + golden baseline**：`RAG_VECTORSTORE=numpy` 是默认；换 qdrant 后，两边
   Top-K 应当一致（向量都 L2 归一化，余弦=内积），可用来做正确性对拍。

注意：本模块不 import numpy，避免与 embedding 层耦合；向量以 list 形式进出 qdrant API。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Iterable

from app.rag.types import Chunk

logger = logging.getLogger(__name__)

# 复用 numpy 版的库级元数据文件名（保持 meta.json 语义一致）
META_FILE = "meta.json"
# collection 名：本地模式每个知识库一个独立 qdrant 实例，名字无所谓；远程模式共享实例，
# 这里统一用知识库名（path.name）作为 collection 名，保证多库互不串扰。
COLLECTION_NAME = "rag_chunks"


class EmbeddingMismatchError(RuntimeError):
    """知识库向量与当前查询向量不兼容（后端/模型/维度不一致）。"""


class SearchHit:
    """向量检索命中项（与 vectorstore.SearchHit 同形）。"""

    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = float(score)


class QdrantStore:
    """Qdrant 向量存储，接口与 VectorStore 对齐。

    :param path: 知识库目录（sidecar meta.json 与本地 qdrant 数据都在这里）
    :param url: 远程 qdrant server 地址；空 = 本地嵌入模式
    """

    def __init__(self, path: Path, url: str = ""):
        self.path = Path(path)
        self.kb = self.path.name
        self.url = url or ""
        self.meta: dict = {}
        self.embedding_fingerprint = ""
        self.dim = 0
        self.data_version = 0
        self._client = None

    # ────────────────────────── 构造与持久化 ──────────────────────────

    @classmethod
    def open(
        cls,
        path: Path,
        embedding_fingerprint: str | None = None,
        create: bool = False,
        url: str = "",
    ) -> "QdrantStore":
        """打开（或创建）知识库。指纹校验与 numpy 版语义一致。"""
        path = Path(path)
        if not path.exists():
            if not create:
                raise FileNotFoundError(f"知识库不存在:{path}")
            path.mkdir(parents=True, exist_ok=True)

        store = cls(path=path, url=url)
        store._load_meta()

        if embedding_fingerprint and store.embedding_fingerprint:
            # 用 meta.json 里记录的 chunk_count 判断库是否非空，避免 open 时就去连 qdrant
            chunk_count = int(store.meta.get("chunk_count", 0) or 0)
            if chunk_count > 0 and store.embedding_fingerprint != embedding_fingerprint:
                raise EmbeddingMismatchError(
                    f"知识库 {path.name} 的向量由 [{store.embedding_fingerprint}] 生成,"
                    f"当前配置为 [{embedding_fingerprint}]。向量空间不可比,"
                    f"请统一向量模型后重新入库。"
                )
        return store

    def _load_meta(self) -> None:
        meta_path = self.path / META_FILE
        if meta_path.exists():
            try:
                self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.meta = {}
            self.embedding_fingerprint = self.meta.get("embedding_fingerprint", "")
            self.dim = int(self.meta.get("dim", 0) or 0)
            self.data_version = int(self.meta.get("data_version", 0) or 0)

    def save(self) -> None:
        """写 sidecar meta.json（qdrant 数据由 qdrant 自己持久化，这里只落库级元数据）。"""
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta.update(
            {
                "embedding_fingerprint": self.embedding_fingerprint,
                "dim": int(self.dim),
                "chunk_count": len(self),
                "data_version": int(self.data_version),
                "vectorstore": "qdrant",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        (self.path / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ────────────────────────── qdrant 客户端 ──────────────────────────

    def _get_client(self):
        """懒加载 qdrant client。做成方法是为了测试里可 monkeypatch 注入假 client。"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:  # pragma: no cover - 依赖缺失提示
                raise RuntimeError(
                    "缺少 qdrant-client 依赖,请执行: pip install 'qdrant-client[local]'（本地模式）"
                    " 或 pip install qdrant-client（远程 server 模式）"
                ) from exc
            if self.url:
                self._client = QdrantClient(url=self.url)
            else:
                self._client = QdrantClient(path=str(self.path / "qdrant"))
        return self._client

    def _ensure_collection(self, dim: int) -> None:
        client = self._get_client()
        if client.collection_exists(COLLECTION_NAME):
            return
        from qdrant_client.models import Distance, VectorParams

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=int(dim), distance=Distance.COSINE),
        )

    def _scroll_all(self, with_payload: bool = True, with_vectors: bool = False):
        """分页读回 collection 里的全部 point，返回 point 列表。"""
        client = self._get_client()
        if not client.collection_exists(COLLECTION_NAME):
            return []
        points: list = []
        offset = None
        while True:
            page, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_vectors=with_vectors,
                with_payload=with_payload,
            )
            points.extend(page)
            if offset is None:
                break
        return points

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """把 chunk_id 稳定映射成合法 UUID。

        Qdrant 本地模式要求 point id 是 UUID，而本项目 chunk_id 是 20 字符 hex（sha1 截断），
        不是合法 UUID。用 uuid5(NAMESPACE_OID, chunk_id) 做确定性映射：同一个 chunk_id 永远
        映射到同一个 UUID，幂等语义不变；原始 chunk_id 仍完整保留在 payload 里（chunk.to_dict()）。
        """
        return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))

    # ────────────────────────── 写入 ──────────────────────────

    def add(self, chunks: list[Chunk], vectors, embedding_fingerprint: str) -> int:
        """追加写入一批 chunk 及其向量，返回新增条数（幂等：chunk_id 已存在则跳过）。"""
        if not chunks:
            return 0

        vectors = [list(map(float, v)) for v in vectors]
        if len(vectors) != len(chunks):
            raise ValueError(f"向量条数({len(vectors)})与 chunk 条数({len(chunks)})不一致")

        if self.embedding_fingerprint and self.embedding_fingerprint != embedding_fingerprint:
            raise EmbeddingMismatchError(
                f"知识库已有向量来自 [{self.embedding_fingerprint}],不能用 [{embedding_fingerprint}] 追加"
            )

        # 幂等去重：先读回已存在的 id（UUID），只 upsert 新的
        existing_ids = {p.id for p in self._scroll_all(with_payload=False)}

        from qdrant_client.models import PointStruct

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            point_id = self._point_id(chunk.chunk_id)
            if point_id in existing_ids:
                continue
            points.append(
                PointStruct(id=point_id, vector=vector, payload=chunk.to_dict())
            )
            existing_ids.add(point_id)

        if not points:
            return 0

        self.embedding_fingerprint = embedding_fingerprint
        self.dim = len(vectors[0])
        self._ensure_collection(self.dim)
        self._get_client().upsert(collection_name=COLLECTION_NAME, points=points)
        self.bump_version()
        logger.debug("知识库 %s 追加 %d 条", self.kb, len(points))
        return len(points)

    def delete_document(self, doc_id: str) -> int:
        """按文档指纹删除整篇（用 payload 过滤 scroll 找匹配 point，再批量删除）。"""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        if not client.collection_exists(COLLECTION_NAME):
            return 0

        ids = []
        offset = None
        while True:
            page, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_vectors=False,
                with_payload=True,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            ids.extend(p.id for p in page)
            if offset is None:
                break

        if not ids:
            return 0
        client.delete(collection_name=COLLECTION_NAME, points_selector=ids)
        self.bump_version()
        return len(ids)

    def bump_version(self) -> int:
        self.data_version = int(self.data_version or 0) + 1
        return self.data_version

    def clear(self) -> None:
        client = self._get_client()
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
        self.embedding_fingerprint = ""
        self.dim = 0
        self.meta = {}
        self.data_version = 0

    # ────────────────────────── 检索 ──────────────────────────

    def search(self, query_vector, top_k: int = 10) -> list[SearchHit]:
        client = self._get_client()
        if not client.collection_exists(COLLECTION_NAME):
            return []
        if top_k <= 0:
            return []

        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(map(float, query_vector)),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[SearchHit] = []
        for point in result.points:
            payload = getattr(point, "payload", None) or {}
            try:
                chunk = Chunk.from_dict(payload)
            except Exception as exc:  # 载荷损坏就当没这条
                logger.warning("还原 chunk 失败(%s):%s", point.id, exc)
                continue
            hits.append(SearchHit(chunk=chunk, score=float(point.score)))
        return hits

    def all_chunks(self) -> Iterable[Chunk]:
        chunks: list[Chunk] = []
        for point in self._scroll_all(with_payload=True, with_vectors=False):
            payload = getattr(point, "payload", None) or {}
            try:
                chunks.append(Chunk.from_dict(payload))
            except Exception as exc:
                logger.warning("还原 chunk 失败(%s):%s", point.id, exc)
        return chunks

    def __len__(self) -> int:
        client = self._get_client()
        if not client.collection_exists(COLLECTION_NAME):
            return 0
        return client.count(collection_name=COLLECTION_NAME).count

    @property
    def size(self) -> int:
        return len(self)


__all__ = ["QdrantStore", "SearchHit", "EmbeddingMismatchError", "META_FILE", "COLLECTION_NAME"]
