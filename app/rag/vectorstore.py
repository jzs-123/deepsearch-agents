"""
向量库(持久化 + 相似度检索)

MVP 刻意不引入 FAISS / Milvus / Qdrant,而是用 numpy 实现一个「全量内存 + 磁盘持久化」
的最小向量库,原因是:
1. 知识库规模在万级 chunk 以内时,全量矩阵乘法(一次 numpy 点积)只要毫秒级,足够快;
2. 少一个二进制依赖(FAISS 在 Windows 上装起来很折腾),项目更容易跨机器复现;
3. 全量检索的结果是精确的,可以当作后续接入 ANN 索引时的「golden baseline」做正确性对照。

落盘结构(每个知识库一个目录):
    chunks.jsonl   每个 chunk 一行 JSON,便于 grep 和增量追加
    vectors.npy    与 chunks 顺序严格一一对应的 (n, dim) float32 矩阵
    meta.json      向量后端指纹 / 维度 / 切块参数 / 更新时间

关键防御:meta.json 里记录 embedding 指纹(后端+模型+维度)。加载时如果指纹不一致
(例如入库用的是 BGE-small,查询换成了别的模型),向量空间不可比,检索结果会是垃圾。
这种情况必须显式报错,而不是静默返回错误结果。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from app.rag.types import Chunk

logger = logging.getLogger(__name__)

CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"


class EmbeddingMismatchError(RuntimeError):
    """知识库向量与当前查询向量不兼容(后端/模型/维度不一致)。"""


@dataclass
class SearchHit:
    """向量检索命中项。"""

    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    """一个知识库的向量存储。"""

    path: Path
    embedding_fingerprint: str = ""
    dim: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    # 是否启用 faiss(若环境里恰好装了 faiss-cpu,用 IndexFlatIP 加速精确检索)
    use_faiss: bool = False
    _index: object | None = None

    # ────────────────────────── 构造与持久化 ──────────────────────────

    @classmethod
    def open(
        cls,
        path: Path,
        embedding_fingerprint: str | None = None,
        create: bool = False,
    ) -> "VectorStore":
        """打开(或创建)知识库目录。

        :param embedding_fingerprint: 当前向量后端指纹;传入且与库内不一致时抛 EmbeddingMismatchError
        :param create: 目录不存在时是否创建
        """
        path = Path(path)
        if not path.exists():
            if not create:
                raise FileNotFoundError(f"知识库不存在:{path}")
            path.mkdir(parents=True, exist_ok=True)

        store = cls(path=path)
        store._load_meta()
        store._load_data()

        if embedding_fingerprint and store.embedding_fingerprint:
            # 只有「库里已有向量」时才需要校验;空库可直接沿用新指纹
            if store.chunks and store.embedding_fingerprint != embedding_fingerprint:
                raise EmbeddingMismatchError(
                    f"知识库 {path.name} 的向量由 [{store.embedding_fingerprint}] 生成,"
                    f"当前配置为 [{embedding_fingerprint}]。向量空间不可比,"
                    f"请统一向量模型后重新入库(python -m app.rag.cli reset -k {path.name})。"
                )
        return store

    def _load_meta(self) -> None:
        meta_path = self.path / META_FILE
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.embedding_fingerprint = self.meta.get("embedding_fingerprint", "")
            self.dim = int(self.meta.get("dim", 0) or 0)

    def _load_data(self) -> None:
        chunks_path = self.path / CHUNKS_FILE
        vectors_path = self.path / VECTORS_FILE
        if not chunks_path.exists():
            self.chunks = []
            self.vectors = None
            return

        chunks: list[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.from_dict(json.loads(line)))

        vectors = None
        if vectors_path.exists():
            vectors = np.load(vectors_path)
            if vectors.size == 0:
                vectors = None
            elif vectors.shape[0] != len(chunks):
                # 落盘撕裂(chunks 与 vectors 不同步):以较小的那份为准并告警,避免索引越界
                logger.warning(
                    "知识库 %s 数据不一致:chunks=%d 但 vectors=%d,已按较小长度截断",
                    self.path.name,
                    len(chunks),
                    vectors.shape[0],
                )
                keep = min(len(chunks), vectors.shape[0])
                chunks = chunks[:keep]
                vectors = vectors[:keep]

        self.chunks = chunks
        self.vectors = vectors
        if vectors is not None and not self.dim:
            self.dim = int(vectors.shape[1])
        if self.dim:
            self._build_index()

    def _build_index(self) -> None:
        """构建 faiss 精确内积索引(可选加速);没有 faiss 就退回 numpy 全量计算。"""
        if self.vectors is None or not len(self.vectors):
            self._index = None
            return
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._index = index
            self.use_faiss = True
        except Exception:
            self._index = None
            self.use_faiss = False

    def save(self) -> None:
        """把 chunk 与向量写回磁盘(先写临时文件再替换,避免中途失败留下半个文件)。"""
        self.path.mkdir(parents=True, exist_ok=True)

        chunks_tmp = self.path / (CHUNKS_FILE + ".tmp")
        with chunks_tmp.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        vectors_tmp = self.path / (VECTORS_FILE + ".tmp.npy")
        if self.vectors is None or not len(self.vectors):
            np.save(vectors_tmp, np.zeros((0, self.dim or 1), dtype=np.float32))
        else:
            np.save(vectors_tmp, self.vectors.astype(np.float32))

        # 顺序很重要:向量先就位,再替换 chunk 清单,最后写 meta
        vectors_tmp.replace(self.path / VECTORS_FILE)
        chunks_tmp.replace(self.path / CHUNKS_FILE)

        self.meta.update(
            {
                "embedding_fingerprint": self.embedding_fingerprint,
                "dim": int(self.dim),
                "chunk_count": len(self.chunks),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        (self.path / META_FILE).write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ────────────────────────── 写入 ──────────────────────────

    def add(self, chunks: list[Chunk], vectors: np.ndarray, embedding_fingerprint: str) -> int:
        """追加写入一批 chunk 及其向量,返回新增条数。

        幂等性:chunk_id 已存在则跳过,因此同一文件重复入库不会产生重复候选
        (这是「重复入库导致检索结果全是同一段话」这个常见坑的解法)。
        """
        if not chunks:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(chunks):
            raise ValueError(f"向量条数({vectors.shape[0]})与 chunk 条数({len(chunks)})不一致")

        if self.embedding_fingerprint and self.embedding_fingerprint != embedding_fingerprint:
            raise EmbeddingMismatchError(
                f"知识库已有向量来自 [{self.embedding_fingerprint}],不能用 [{embedding_fingerprint}] 追加"
            )
        self.embedding_fingerprint = embedding_fingerprint
        self.dim = int(vectors.shape[1])

        # 保留第一次写入的顺序,新 chunk 追加在后面
        existing_total = len(self.chunks)
        known_ids = {c.chunk_id for c in self.chunks}
        new_chunks: list[Chunk] = []
        new_vectors: list[np.ndarray] = []
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in known_ids:
                continue
            known_ids.add(chunk.chunk_id)
            new_chunks.append(chunk)
            new_vectors.append(vector)

        if not new_chunks:
            return 0

        self.chunks.extend(new_chunks)
        added = np.vstack(new_vectors).astype(np.float32)
        if self.vectors is None or not len(self.vectors):
            self.vectors = added
        else:
            if self.vectors.shape[1] != added.shape[1]:
                raise EmbeddingMismatchError(
                    f"向量维度不一致:库内 {self.vectors.shape[1]},新增 {added.shape[1]}"
                )
            self.vectors = np.vstack([self.vectors, added])
        logger.debug("知识库 %s 追加 %d 条(原有 %d 条)", self.path.name, len(new_chunks), existing_total)
        self._build_index()
        return len(new_chunks)

    def delete_document(self, doc_id: str) -> int:
        """按文档指纹删除整篇(重新入库同一文件前先清理旧版本,避免新旧内容并存)。"""
        if not self.chunks:
            return 0
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            if self.vectors is not None and len(self.vectors):
                self.vectors = self.vectors[keep]
            self._build_index()
        return removed

    def clear(self) -> None:
        """清空知识库数据(保留目录)。"""
        self.chunks = []
        self.vectors = None
        self._index = None
        self.embedding_fingerprint = ""
        self.meta = {}

    # ────────────────────────── 检索 ──────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchHit]:
        """稠密向量检索,返回按余弦相似度降序的 top_k。

        因为入库时向量已 L2 归一化,内积即为余弦相似度,越大越相似。
        """
        if not self.chunks or self.vectors is None or not len(self.vectors):
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.vectors.shape[1]:
            raise EmbeddingMismatchError(
                f"查询向量维度 {query.shape[0]} 与库内 {self.vectors.shape[1]} 不一致"
            )
        top_k = max(1, min(top_k, len(self.chunks)))

        if self._index is not None and self.use_faiss:
            scores, indices = self._index.search(query.reshape(1, -1), top_k)  # type: ignore[attr-defined]
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            # 全量内积:万级 chunk 下 numpy 一次矩阵乘法即可,精确且无需额外依赖
            scores = self.vectors @ query
            top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            pairs = ((int(i), float(scores[i])) for i in top_idx)

        hits: list[SearchHit] = []
        for idx, score in pairs:
            if idx < 0 or idx >= len(self.chunks):
                continue
            hits.append(SearchHit(chunk=self.chunks[idx], score=float(score)))
        return hits

    def all_chunks(self) -> Iterable[Chunk]:
        return list(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def size(self) -> int:
        return len(self.chunks)


def list_knowledge_bases(data_dir: Path) -> list[dict]:
    """列出数据根目录下所有知识库及其规模。"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    result: list[dict] = []
    for item in sorted(data_dir.iterdir()):
        if not item.is_dir():
            continue
        meta: dict = {}
        meta_path = item / META_FILE
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        chunks_path = item / CHUNKS_FILE
        count = meta.get("chunk_count")
        if count is None:
            count = 0
            if chunks_path.exists():
                with chunks_path.open("r", encoding="utf-8") as f:
                    count = sum(1 for line in f if line.strip())
        result.append(
            {
                "name": item.name,
                "path": str(item),
                "chunk_count": int(count),
                "embedding": meta.get("embedding_fingerprint", ""),
                "updated_at": meta.get("updated_at", ""),
                "meta": meta,
            }
        )
    return result


__all__ = [
    "VectorStore",
    "SearchHit",
    "EmbeddingMismatchError",
    "list_knowledge_bases",
    "CHUNKS_FILE",
    "VECTORS_FILE",
    "META_FILE",
]
