"""
端到端编排:入库(Ingest)与检索问答(Query)

这一层把前九个模块串起来,对外只暴露两个动作:
    ingest(...)  —— 文件/目录 -> 解析 -> 切块 -> 向量化 -> 写入知识库
    search(...)  —— 问题 -> 稠密 + 稀疏 -> RRF 融合 -> 重排 -> top_n
    query(...)   —— search 之后接答案生成(带引用)

把编排集中在这里的好处:CLI、Agent 工具、单元测试都调用同一套逻辑,行为不会分叉。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.bm25 import BM25Index
from app.rag.chunking import chunk_document
from app.rag.config import RagConfig, load_rag_config
from app.rag.embedding import Embedder, build_embedder
from app.rag.generator import AnswerGenerator, ChatClient
from app.rag.hybrid import hybrid_search
from app.rag.loaders import LoadedDocument, iter_supported_files, load_document
from app.rag.rerank import build_reranker
from app.rag.types import Answer, Chunk, RetrievedChunk, content_fingerprint
from app.rag.vectorstore import EmbeddingMismatchError, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """入库结果报告。"""

    kb: str
    files_total: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_added: int = 0
    chunks_replaced: int = 0
    total_chunks: int = 0
    elapsed: float = 0.0
    details: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kb": self.kb,
            "files_total": self.files_total,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "chunks_added": self.chunks_added,
            "chunks_replaced": self.chunks_replaced,
            "total_chunks": self.total_chunks,
            "elapsed": round(self.elapsed, 3),
            "details": self.details,
            "warnings": self.warnings,
        }


class RagPipeline:
    """一个知识库的完整 RAG 流水线。"""

    def __init__(
        self,
        kb: str | None = None,
        config: RagConfig | None = None,
        embedder: Embedder | None = None,
        allow_embedding_fallback: bool = True,
    ):
        self.config = config or load_rag_config()
        self.kb = kb or self.config.default_kb
        self.kb_dir = self.config.kb_dir(self.kb)
        self._bm25: BM25Index | None = None

        # 向量后端:允许外部注入(测试用 stub,避免每次跑测试都加载模型)
        self.embedder = embedder or build_embedder(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
            dim=self.config.embedding_dim,
            device=self.config.embedding_device,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            allow_fallback=allow_embedding_fallback,
        )

        self.store = VectorStore.open(
            self.kb_dir,
            embedding_fingerprint=self.embedder.fingerprint,
            create=True,
        )
        # 切块参数变化时给出提示:旧 chunk 是按老参数切的,混用会导致检索质量下降
        old_size = self.store.meta.get("chunk_size")
        if old_size and int(old_size) != self.config.chunk_size:
            logger.warning(
                "知识库 %s 的既有 chunk 按 chunk_size=%s 切分,当前配置为 %s;建议重新入库",
                self.kb,
                old_size,
                self.config.chunk_size,
            )

    # ────────────────────────── 入库 ──────────────────────────

    def ingest(self, path: str | Path) -> IngestReport:
        """把一个文件或目录入库。

        同名文件重复入库会先按 doc_id 删除旧版本再写入新版本(整篇替换),
        避免「文档更新后新旧内容同时在库里」导致检索到过期信息。
        """
        started = time.time()
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"路径不存在:{root}")

        base_dir = root if root.is_dir() else root.parent
        files = iter_supported_files(root)
        report = IngestReport(kb=self.kb, files_total=len(files))
        if not files:
            report.warnings.append(f"{root} 下没有找到支持的文件类型")

        pending_chunks: list[Chunk] = []
        pending_texts: list[str] = []

        for file_path in files:
            try:
                doc = load_document(file_path, base_dir=base_dir)
            except Exception as exc:
                report.files_skipped += 1
                report.warnings.append(f"跳过 {file_path.name}:{exc}")
                continue

            if not doc.text.strip():
                report.files_skipped += 1
                report.warnings.append(f"跳过 {file_path.name}:未解析出文本")
                continue

            chunks = chunk_document(
                doc,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if not chunks:
                report.files_skipped += 1
                report.warnings.append(f"跳过 {file_path.name}:切块结果为空")
                continue

            # 整篇替换:同一 doc_id 的旧 chunk 先删掉
            replaced = self.store.delete_document(chunks[0].doc_id)
            report.chunks_replaced += replaced

            pending_chunks.extend(chunks)
            pending_texts.extend(chunk.text for chunk in chunks)
            report.files_indexed += 1
            report.details.append(
                {
                    "file": doc.source_path,
                    "chars": doc.char_count,
                    "chunks": len(chunks),
                    "replaced": replaced,
                    "title": doc.title,
                }
            )
            for warning in doc.warnings:
                report.warnings.append(f"{file_path.name}:{warning}")

        if pending_chunks:
            # 向量化是整批做一次:批量编码远快于逐条 encode
            vectors = self.embedder.encode(pending_texts, is_query=False)
            report.chunks_added = self.store.add(
                pending_chunks, vectors, embedding_fingerprint=self.embedder.fingerprint
            )
            self.store.meta.update(
                {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "embedding_provider": self.config.embedding_provider,
                    "kb": self.kb,
                }
            )
            self.store.save()
            # 数据变了,BM25 索引必须失效重建,否则稀疏路会检索到已删除的旧文档
            self._bm25 = None

        report.total_chunks = len(self.store)
        report.elapsed = time.time() - started
        logger.info(
            "入库完成 kb=%s files=%d/%d chunks_added=%d total=%d 用时=%.2fs",
            self.kb,
            report.files_indexed,
            report.files_total,
            report.chunks_added,
            report.total_chunks,
            report.elapsed,
        )
        return report

    def add_texts(self, texts: list[str], source: str = "manual_input") -> int:
        """直接写入文本(不经过文件解析),便于测试与「用户粘贴一段文字」的场景。

        注意:这里同样走 chunk_document 切块,而不是「一段文本 = 一个 chunk」。
        早期实现直接把整段文本当单个 chunk 写入,结果是长文入库后只有一条 3000 字的
        chunk,检索时要么整体命中要么完全漏掉,是本项目实际踩过的坑。
        """
        chunks: list[Chunk] = []
        combined = "\n\n".join(texts)
        for index, text in enumerate(texts):
            text = text.strip()
            if not text:
                continue
            # 用与文件入库一致的解析 + 切块路径,保证两种入口的行为完全一致
            doc = LoadedDocument(text=text, source_path=source, title=source)
            chunked = chunk_document(
                doc,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            # 多个输入文本时把序号前缀让开,避免 chunk_id 冲突
            for offset, chunk in enumerate(chunked):
                chunk.chunk_index = index * 1000 + offset
                chunk.doc_id = content_fingerprint(combined)
                chunks.append(chunk)
        if not chunks:
            return 0
        # 与文件入库保持一致:先按 doc_id 清掉同一份文本的旧版本,避免重复写入
        self.store.delete_document(content_fingerprint(combined))
        vectors = self.embedder.encode([c.text for c in chunks], is_query=False)
        added = self.store.add(chunks, vectors, embedding_fingerprint=self.embedder.fingerprint)
        self.store.meta.update(
            {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "embedding_provider": self.config.embedding_provider,
                "kb": self.kb,
            }
        )
        self.store.save()
        self._bm25 = None
        return added

    # ────────────────────────── 检索 ──────────────────────────

    def _bm25_index(self) -> BM25Index:
        """惰性构建 BM25 索引,并在数据变更后自动重建。"""
        if self._bm25 is None or self._bm25.size != len(self.store):
            self._bm25 = BM25Index().build(list(self.store.all_chunks()))
        return self._bm25

    def search(
        self,
        question: str,
        top_k: int | None = None,
        top_n: int | None = None,
        rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """混合检索 + 重排,返回最终 top_n 结果。"""
        top_k = top_k or self.config.top_k
        top_n = top_n or self.config.top_n
        if not len(self.store):
            return []

        # ① 稠密路:向量相似度
        query_vector = self.embedder.encode_one(question, is_query=True)
        dense_hits = [(hit.chunk, hit.score) for hit in self.store.search(query_vector, top_k)]

        # ② 稀疏路:BM25 词面匹配
        sparse_hits = self._bm25_index().search(question, top_k)

        # ③ RRF 融合
        fusion = hybrid_search(dense_hits, sparse_hits, top_k=top_k, rrf_k=self.config.rrf_k)
        candidates = fusion.fused

        # ④ 重排
        use_rerank = (self.config.rerank_mode != "off") if rerank is None else rerank
        if use_rerank:
            reranker = build_reranker(
                self.config.rerank_mode,
                llm_client=self._llm_client(),
                llm_model=self.config.llm_model,
            )
            return reranker.rerank(question, candidates, top_n)
        for rank, item in enumerate(candidates[:top_n], start=1):
            item.rank = rank
        return candidates[:top_n]

    @staticmethod
    def grounded_score(chunks: list[RetrievedChunk]) -> float:
        """返回用于「是否有依据」判定的分数。

        为什么不用排序分(item.score):开启重排后 score 是重排分(词面覆盖或大模型打分),
        它的绝对尺度与向量余弦完全不同——实测同一批数据里,词面重排会把无关问题的最高分
        从 0.17 抬到 0.22,于是 0.2 的拒答阈值直接失效(拒答准确率 100% -> 50%)。
        向量余弦是唯一跨模式稳定、且有明确语义含义(相似度)的量,所以:
        排序用重排分,拒答门槛一律用向量余弦。
        若某条候选在稠密路没有命中(没有 dense_score),退回用其排序分判定。
        """
        if not chunks:
            return 0.0
        for item in chunks:
            if item.dense_score is not None:
                return float(item.dense_score)
        return float(chunks[0].score)

    def _llm_client(self) -> ChatClient | None:
        client = ChatClient(
            model=self.config.llm_model,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            temperature=self.config.llm_temperature,
            timeout=self.config.llm_timeout,
        )
        return client if client.available else None

    def query(self, question: str, top_k: int | None = None, top_n: int | None = None) -> Answer:
        """检索 + 生成,返回带引用的完整答案。"""
        started = time.time()
        chunks = self.search(question, top_k=top_k, top_n=top_n)
        answer = AnswerGenerator(
            client=self._llm_client(), min_score=self.config.min_score
        ).generate(question, chunks, gate_score=self.grounded_score(chunks))
        answer.debug["elapsed"] = round(time.time() - started, 3)
        return answer

    # ────────────────────────── 运维 ──────────────────────────

    def reset(self) -> None:
        """清空当前知识库(删除所有 chunk 与向量,保留目录)。"""
        self.store.clear()
        # 写回一个空库:直接删除文件比留空 meta 更干净
        for name in ("chunks.jsonl", "vectors.npy", "meta.json"):
            target = self.kb_dir / name
            if target.exists():
                target.unlink()
        self._bm25 = None

    def stats(self) -> dict:
        return {
            "kb": self.kb,
            "path": str(self.kb_dir),
            "chunk_count": len(self.store),
            "dim": self.store.dim or self.embedder.dim,
            "embedding": self.embedder.fingerprint,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "top_k": self.config.top_k,
            "top_n": self.config.top_n,
            "rerank_mode": self.config.rerank_mode,
        }


# 进程内缓存:Agent 工具会在一次对话里被多次调用,重复构造 embedder 会反复加载模型
_PIPELINES: dict[tuple, RagPipeline] = {}


def get_pipeline(
    kb: str | None = None,
    config: RagConfig | None = None,
    embedder: Embedder | None = None,
) -> RagPipeline:
    """获取(带缓存的)知识库流水线。"""
    config = config or load_rag_config()
    kb_name = kb or config.default_kb
    key = (kb_name, config.embedding_provider, config.embedding_model, config.chunk_size, id(embedder))
    if embedder is not None:
        # 外部注入的 embedder 不缓存,避免测试之间互相污染
        return RagPipeline(kb=kb_name, config=config, embedder=embedder)
    if key not in _PIPELINES:
        _PIPELINES[key] = RagPipeline(kb=kb_name, config=config)
    return _PIPELINES[key]


def clear_pipeline_cache() -> None:
    """清空流水线缓存(配置变更或测试时使用)。"""
    _PIPELINES.clear()


__all__ = [
    "RagPipeline",
    "IngestReport",
    "get_pipeline",
    "clear_pipeline_cache",
    "EmbeddingMismatchError",
]
