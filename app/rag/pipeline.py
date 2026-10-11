"""
端到端编排:入库(Ingest)与检索问答(Query)

这一层把前九个模块串起来,对外只暴露两个动作:
    ingest(...)  —— 文件/目录 -> 解析 -> 切块 -> 向量化 -> 写入知识库
    search(...)  —— 问题 -> (可选)查询改写 -> 稠密 + 稀疏 -> RRF 融合 -> 重排 -> top_n
    query(...)   —— search 之后接答案生成(带引用)

把编排集中在这里的好处:CLI、Agent 工具、单元测试都调用同一套逻辑,行为不会分叉。

查询改写(Query Rewrite)的接入方式(见 docs/rag-query-rewrite.md):
1. 首轮永远先用**原始查询**检索,语义与改写功能引入前逐字节一致,零额外成本;
2. 只有「首轮判定分(向量余弦)低于 min_score」且开启重试时,才用改写变体再检一次;
   显式传 `rewrite=True` 则首轮就多查询(CLI 的 `--rewrite-always` 用它);
3. 多查询时按变体分别走稠密+稀疏,再用 RRF 融合(去重靠 chunk_id),并把
   「哪个变体命中、谁胜出」写进 `last_search_debug` 与 `RetrievedChunk.channels`。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from app.rag.audit import AuditLogger, build_query_entry, build_search_entry
from app.rag.bm25 import BM25Index
from app.rag.cache import PipelineCache
from app.rag.chunking import chunk_document
from app.rag.config import RagConfig, load_rag_config
from app.rag.embedding import Embedder, build_embedder
from app.rag.generator import AnswerGenerator, ChatClient
from app.rag.hybrid import hybrid_search, reciprocal_rank_fusion
from app.rag.loaders import LoadedDocument, iter_supported_files, load_document
from app.rag.metadata import (
    apply_filter,
    canonical_filter,
    extract_metadata,
    filter_fields_of,
    format_filter_hint,
    normalize_filter,
)
from app.rag.rerank import build_reranker
from app.rag.rewrite import RewriteResult, build_rewriter
from app.rag.types import Answer, Chunk, RetrievedChunk, content_fingerprint
from app.rag.vectorstore import EmbeddingMismatchError, VectorStore

logger = logging.getLogger(__name__)

#: 多查询融合时,非原始查询的变体权重折扣(原始查询权重要略高,避免被变体挤掉)
VARIANT_DISCOUNT = 0.85

#: 查询缓存里最多缓存多少条结果(与 top_k 同量级即可,避免缓存文件无谓膨胀)
MAX_CACHED_RESULTS = 50


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
    # embedding 缓存的本次命中/未命中(缓存关闭时恒为 0/全量,便于对比"开没开缓存")
    embedding_cache: dict = field(default_factory=dict)

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
            "embedding_cache": self.embedding_cache,
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
        create: bool = True,
    ):
        """
        :param create: 知识库目录不存在时是否自动创建。
            入库/建库场景应为 True;只读查询场景应为 False —— 否则用户传错库名时会静默
            在 rag_data/ 下建出一个空目录,后续查询都得到「库是空的」这种误导性结果,
            而不是「库不存在」这个真实原因。
        """
        self.config = config or load_rag_config()
        self.kb = kb or self.config.default_kb
        self.kb_dir = self.config.kb_dir(self.kb)
        self._bm25: BM25Index | None = None
        # 最近一次 search 的审计信息(用了哪些变体、各自命中几条、是否重试、谁胜出)。
        # 检索过程对本模块是"有状态"的,但对外仍只返回结果列表 —— 不改变既有返回类型。
        self.last_search_debug: dict = {}

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
            create=create,
        )
        # 缓存与审计都是"旁路":缓存默认关闭,审计默认开启但写入失败只 warning
        self.cache = PipelineCache(
            kb_dir=self.kb_dir,
            fingerprint=self.embedder.fingerprint,
            data_version=self.store.data_version,
            enabled=self.config.cache_enabled,
        )
        self.audit = AuditLogger(
            audit_dir=self.config.audit_path(self.kb),
            kb=self.kb,
            enabled=self.config.audit_enabled,
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
                doc = load_document(file_path, base_dir=base_dir, ocr=self.config.ocr_options())
            except Exception as exc:
                report.files_skipped += 1
                report.warnings.append(f"跳过 {file_path.name}:{exc}")
                continue

            if not doc.text.strip():
                report.files_skipped += 1
                report.warnings.append(f"跳过 {file_path.name}:未解析出文本")
                continue

            # 元数据:文件级(名称/路径/后缀/修改时间/大小)+ 内容级(Markdown YAML frontmatter)。
            # frontmatter 会从正文里剥掉——它是元数据,不是正文,留在正文里既污染检索也会挤占 chunk。
            meta_result = extract_metadata(file_path, doc.text, base_dir=base_dir)
            for warning in meta_result.warnings:
                report.warnings.append(f"{file_path.name}:{warning}")
            if meta_result.stripped:
                doc = replace(doc, text=meta_result.body)

            chunks = chunk_document(
                doc,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if not chunks:
                report.files_skipped += 1
                report.warnings.append(f"跳过 {file_path.name}:切块结果为空")
                continue

            # 元数据挂到每个 chunk 上,随 chunks.jsonl 一起持久化
            for chunk in chunks:
                chunk.metadata = dict(meta_result.fields)

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
                    # 文本来源(text_layer/ocr/mixed/none):排障时用于判断该文件的正文
                    # 是原文还是 OCR 识别结果(OCR 会带错字,见 docs/rag-ocr.md)
                    "extraction": doc.extraction,
                    # 该文件携带的可过滤字段(frontmatter 字段名,便于入库时人工核对)
                    "filter_fields": meta_result.filter_fields,
                }
            )
            for warning in doc.warnings:
                report.warnings.append(f"{file_path.name}:{warning}")

        if pending_chunks:
            # 向量化是整批做一次:批量编码远快于逐条 encode;开了缓存则命中部分直接复用向量
            hits_before = self.cache.stats.embedding_hits
            misses_before = self.cache.stats.embedding_misses
            vectors = self._encode_documents(pending_texts)
            report.embedding_cache = {
                "enabled": bool(self.cache.enabled),
                "hits": self.cache.stats.embedding_hits - hits_before,
                "misses": self.cache.stats.embedding_misses - misses_before,
                "texts": len(pending_texts),
            }
            report.chunks_added = self.store.add(
                pending_chunks, vectors, embedding_fingerprint=self.embedder.fingerprint
            )
            self.store.meta.update(
                {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "embedding_provider": self.config.embedding_provider,
                    "kb": self.kb,
                    # 可过滤字段清单:自动去重,供 list/stats 与 Agent 工具展示
                    "filter_fields": self.filter_fields,
                }
            )
            self.store.save()
            # 数据变了 → 查询/生成缓存整体失效(版本号已随 add/delete 递增)
            self.cache.set_version(self.store.data_version)
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
                # 手工写入的文本没有文件级元数据,只留来源(过滤时仍可按 path/file 用)
                chunk.metadata = {"file": source, "path": source}
                chunks.append(chunk)
        if not chunks:
            return 0
        # 与文件入库保持一致:先按 doc_id 清掉同一份文本的旧版本,避免重复写入
        self.store.delete_document(content_fingerprint(combined))
        vectors = self._encode_documents([c.text for c in chunks])
        added = self.store.add(chunks, vectors, embedding_fingerprint=self.embedder.fingerprint)
        self.store.meta.update(
            {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "embedding_provider": self.config.embedding_provider,
                "kb": self.kb,
                "filter_fields": self.filter_fields,
            }
        )
        self.store.save()
        self.cache.set_version(self.store.data_version)
        self._bm25 = None
        return added

    # ────────────────────────── 元数据 / 缓存 工具 ──────────────────────────

    @property
    def filter_fields(self) -> list[str]:
        """该知识库当前可用的过滤字段(优先用 meta.json 里缓存的清单,缺失时现算)。"""
        cached = self.store.meta.get("filter_fields")
        if isinstance(cached, list) and cached:
            return sorted({str(item) for item in cached})
        return filter_fields_of(list(self.store.all_chunks()))

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        """文档向量化(带 embedding 缓存)。

        缓存关闭时就是 `self.embedder.encode(texts)`,行为与引入缓存前完全一致;
        开启时命中的文本直接复用向量,只对未命中的部分调用向量后端——这是入库最耗时的一块。
        任何维度不一致都视为缓存不可用(降级为全量重算),不让脏缓存污染向量库。
        """
        if not texts:
            return np.zeros((0, self.embedder.dim or 1), dtype=np.float32)
        hits = self.cache.get_embeddings(texts)
        if hits:
            cached_dim = len(next(iter(hits.values())))
            expected_dim = int(self.embedder.dim or cached_dim)
            if expected_dim != cached_dim:
                logger.warning(
                    "embedding 缓存维度(%d)与当前后端(%d)不一致,已忽略缓存重新编码",
                    cached_dim,
                    expected_dim,
                )
                hits = {}
        missing = [index for index in range(len(texts)) if index not in hits]
        fresh: np.ndarray | None = None
        if missing:
            fresh = np.asarray(
                self.embedder.encode([texts[index] for index in missing], is_query=False),
                dtype=np.float32,
            )
            if fresh.ndim == 1:
                fresh = fresh.reshape(1, -1)
            self.cache.put_embeddings([texts[index] for index in missing], fresh)
        if not hits:
            return fresh if fresh is not None else np.zeros((0, self.embedder.dim or 1), dtype=np.float32)

        dim = len(next(iter(hits.values())))
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for index, vector in hits.items():
            matrix[index] = np.asarray(vector, dtype=np.float32)
        if fresh is not None:
            for row, index in enumerate(missing):
                matrix[index] = fresh[row]
        return matrix

    # ────────────────────────── 检索 ──────────────────────────

    def _bm25_index(self) -> BM25Index:
        """惰性构建 BM25 索引,并在数据变更后自动重建。"""
        if self._bm25 is None or self._bm25.size != len(self.store):
            self._bm25 = BM25Index().build(list(self.store.all_chunks()))
        return self._bm25

    def _rewriter(self, mode: str):
        """按当前配置构造改写器(复用项目已有的 ChatClient,不引入新的大模型依赖)。"""
        return build_rewriter(
            mode=mode,
            max_variants=self.config.query_variants,
            synonym_spec=self.config.rewrite_synonyms,
            llm_client=self._llm_client(),
            llm_model=self.config.llm_model,
        )

    def _resolve_rewrite_mode(self, rewrite: bool | None) -> str:
        """把 `search(rewrite=...)` 的三种取值解析成实际模式。

        - `rewrite is False`:强制关闭(与改写功能引入前的路径完全一致)
        - `rewrite is True` :显式要求改写(配置为 off 时也做本地规则改写)
        - `rewrite is None` :按配置(默认 local)
        """
        mode = self.config.query_rewrite
        if rewrite is False:
            return "off"
        if rewrite is True and mode == "off":
            return "local"
        return mode

    def _retrieve_variants(
        self, queries: list[str], top_k: int
    ) -> tuple[list[RetrievedChunk], dict]:
        """按若干查询变体做稠密+稀疏检索并融合。

        **单个查询走的就是历史路径**(`hybrid_search`),所以改写关闭时结果逐字节一致;
        多查询时每个变体各占一条 RRF 通道(`dense@v0` / `sparse@v1` …),既做融合也留下审计痕迹。
        """
        if len(queries) <= 1:
            question = queries[0] if queries else ""
            query_vector = self.embedder.encode_one(question, is_query=True)
            dense_hits = [
                (hit.chunk, hit.score) for hit in self.store.search(query_vector, top_k)
            ]
            sparse_hits = self._bm25_index().search(question, top_k)
            fusion = hybrid_search(dense_hits, sparse_hits, top_k=top_k, rrf_k=self.config.rrf_k)
            info = {
                "variant_hits": [
                    {
                        "variant": question,
                        "dense_hits": len(dense_hits),
                        "sparse_hits": len(sparse_hits),
                    }
                ],
                "fused_count": len(fusion.fused),
                "single_query": True,
            }
            return fusion.fused, info

        ranked_lists: dict[str, list[tuple[Chunk, float]]] = {}
        dense_best: dict[str, float] = {}
        sparse_best: dict[str, float] = {}
        variant_hits: list[dict] = []
        for index, variant in enumerate(queries):
            query_vector = self.embedder.encode_one(variant, is_query=True)
            dense_hits = [
                (hit.chunk, hit.score) for hit in self.store.search(query_vector, top_k)
            ]
            sparse_hits = self._bm25_index().search(variant, top_k)
            ranked_lists[f"dense@v{index}"] = dense_hits
            ranked_lists[f"sparse@v{index}"] = sparse_hits
            for chunk, score in dense_hits:
                # 每个 chunk 保留它在所有变体里的最高向量余弦:判定分(dense_score)必须仍是余弦,
                # 不能被 RRF 分或重排分替代(见 docs/rag-self-built.md 5.10)。
                if score > dense_best.get(chunk.chunk_id, float("-inf")):
                    dense_best[chunk.chunk_id] = float(score)
            for chunk, score in sparse_hits:
                if score > sparse_best.get(chunk.chunk_id, float("-inf")):
                    sparse_best[chunk.chunk_id] = float(score)
            variant_hits.append(
                {
                    "variant": variant,
                    "dense_hits": len(dense_hits),
                    "sparse_hits": len(sparse_hits),
                }
            )

        weights: dict[str, float] = {}
        for index in range(len(queries)):
            discount = 1.0 if index == 0 else VARIANT_DISCOUNT
            weights[f"dense@v{index}"] = 1.0 * discount
            weights[f"sparse@v{index}"] = 0.8 * discount
        fused = reciprocal_rank_fusion(
            ranked_lists, k=self.config.rrf_k, weights=weights, top_k=top_k
        )
        for item in fused:
            # 通道名带了变体后缀,RRF 不会自动填 dense_score/sparse_score,这里补回来,
            # 否则审计里看不到"某片段是靠哪一路、哪个变体进来的"(排查问题时最关键的两个数)
            item.dense_score = dense_best.get(item.chunk.chunk_id)
            item.sparse_score = sparse_best.get(item.chunk.chunk_id)
        info = {
            "variant_hits": variant_hits,
            "fused_count": len(fused),
            "single_query": False,
            "channels": sorted(ranked_lists),
        }
        return fused, info

    def _rerank_with_variants(
        self,
        reranker,
        variants: list[str],
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """重排;多变体时用「在所有变体里的最高词面分」排序。

        为什么不能只拿原始查询重排:原始查询往往很短(这正是需要改写的情形),用它打分会让
        变体召回的片段重新掉下去,等于白改。LLM 重排没有 `score()` 接口,按原问题做列表式重排。
        """
        if len(variants) <= 1 or not hasattr(reranker, "score"):
            return reranker.rerank(variants[0] if variants else "", candidates, top_n)
        for item in candidates:
            item.score = max(reranker.score(variant, item.chunk.text) for variant in variants)
        ordered = sorted(candidates, key=lambda item: (-item.score, item.rank))
        result = ordered[:top_n]
        for rank, item in enumerate(result, start=1):
            item.rank = rank
        return result

    def _apply_rerank(
        self,
        candidates: list[RetrievedChunk],
        variants: list[str],
        top_n: int,
        use_rerank: bool,
    ) -> list[RetrievedChunk]:
        """重排(或直接截断);单变体时走的就是历史路径,行为逐字节一致。"""
        if use_rerank:
            reranker = build_reranker(
                self.config.rerank_mode,
                llm_client=self._llm_client(),
                llm_model=self.config.llm_model,
            )
            return self._rerank_with_variants(reranker, variants, candidates, top_n)
        results = candidates[:top_n]
        for rank, item in enumerate(results, start=1):
            item.rank = rank
        return results

    def _search_once(
        self,
        variants: list[str],
        top_k: int,
        top_n: int,
        use_rerank: bool,
        normalized_filter: dict[str, list[str]] | None = None,
    ) -> tuple[list[RetrievedChunk], dict, float]:
        """跑一轮「检索 → (元数据过滤) → 重排」,返回 (最终结果, 检索审计信息, 最终判定分)。

        判定分取的是**最终结果的** `grounded_score()`(向量余弦)。这一点很关键:
        重排前候选按 RRF 排、重排后按词面分排,若拿重排前的候选算分,审计里就会出现
        「重试判定分 0.1541,而实际用的是 0.2181」这种对不上账的情况(本项目真实踩到,已修)。

        元数据过滤插在「召回之后、重排之前」:先按范围剔除候选,再让重排把名额花在可用的片段上
        (见 app/rag/metadata.py 顶部的说明)。
        """
        candidates, info = self._retrieve_variants(variants, top_k)
        before = len(candidates)
        if normalized_filter:
            candidates = apply_filter(candidates, normalized_filter)
        info = dict(info)
        info["filtered_out"] = before - len(candidates)
        info["pre_filter_count"] = before
        results = self._apply_rerank(candidates, variants, top_n, use_rerank)
        return results, info, self.grounded_score(results)

    # ── 结果序列化(查询缓存用) ──

    @staticmethod
    def _serialize_results(results: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "chunk_id": item.chunk.chunk_id,
                "rank": item.rank,
                "score": float(item.score),
                "dense_score": None if item.dense_score is None else float(item.dense_score),
                "sparse_score": None if item.sparse_score is None else float(item.sparse_score),
                "rrf_score": float(item.rrf_score),
                "channels": list(item.channels),
            }
            for item in results[:MAX_CACHED_RESULTS]
        ]

    def _restore_results(self, payload: list[dict]) -> list[RetrievedChunk] | None:
        """从缓存载荷还原结果;任何一条 chunk 已不存在(库变过)就视为缓存不可用。"""
        index = {chunk.chunk_id: chunk for chunk in self.store.all_chunks()}
        restored: list[RetrievedChunk] = []
        for row in payload:
            chunk = index.get(str(row.get("chunk_id")))
            if chunk is None:
                logger.info("查询缓存命中但 chunk 已不存在,按未命处理:%s", row.get("chunk_id"))
                return None
            restored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(row.get("score", 0.0)),
                    dense_score=row.get("dense_score"),
                    sparse_score=row.get("sparse_score"),
                    rrf_score=float(row.get("rrf_score", 0.0)),
                    rank=int(row.get("rank", 0) or 0),
                    channels=list(row.get("channels") or []),
                    cached=True,
                )
            )
        return restored

    def _results_summary(self, results: list[RetrievedChunk]) -> list[dict]:
        """可审计的结果摘要(不含原文;与 audit.summarize_results 同字段)。"""
        return [
            {
                "rank": item.rank,
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "chunk_index": item.chunk.chunk_index,
                "channels": list(item.channels),
                "score": round(item.score, 4),
                "dense": None if item.dense_score is None else round(item.dense_score, 4),
                "sparse": None if item.sparse_score is None else round(item.sparse_score, 4),
                "rrf": round(item.rrf_score, 4),
                "cached": bool(item.cached),
            }
            for item in results
        ]

    def _record_search_audit(
        self,
        question: str,
        results: list[RetrievedChunk],
        cache_info: dict | None,
        kind: str = "search",
    ) -> None:
        """写一条审计记录;失败只 warning(审计是旁路,不能拖垮查询)。"""
        entry = build_search_entry(
            question=question,
            debug=self.last_search_debug,
            filter_spec=self.last_search_debug.get("filter") or {},
            fingerprint=self.embedder.fingerprint,
            results=results,
            cache_info=cache_info,
            audit_dir=str(self.audit.audit_dir),
        )
        entry["kind"] = kind
        self.audit.record(entry)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        top_n: int | None = None,
        rerank: bool | None = None,
        rewrite: bool | None = None,
        filter: dict | None = None,
        _record_audit: bool = True,
    ) -> list[RetrievedChunk]:
        """混合检索 + 元数据过滤 + 重排,返回最终 top_n 结果。

        :param rewrite: None=按配置(默认 local,低分时自动重试);False=强制关闭(历史行为);
            True=首轮就多查询(不等低分)
        :param filter: 元数据过滤,如 `{"部门": "风控"}` / `{"类型": ["制度", "规范"]}`;
            缺字段或值不匹配的片段会被剔除(白名单语义),过滤发生在召回之后、重排之前
        :param _record_audit: 内部参数。`query()` 会自己写一条包含生成阶段的审计记录,
            因此调用 search 时把它置 False,避免一次问答产生两条日志
        """
        top_k = top_k or self.config.top_k
        top_n = top_n or self.config.top_n
        normalized_filter = normalize_filter(filter)
        filter_spec = {str(key): value for key, value in (filter or {}).items()}
        available_fields = self.filter_fields

        if not len(self.store):
            self.last_search_debug = {
                "question": question,
                "rewrite_mode": self._resolve_rewrite_mode(rewrite),
                "empty_store": True,
                "variants": [],
                "retry": {"enabled": bool(self.config.retry_on_low_score), "triggered": False},
                "filter": filter_spec,
                "filter_fields": available_fields,
                "filtered_out": 0,
                "cache": {"used": bool(self.cache.enabled), "type": "query", "hit": False},
                "cached": False,
                "results": [],
            }
            if _record_audit:
                self._record_search_audit(question, [], self.last_search_debug["cache"])
            return []

        mode = self._resolve_rewrite_mode(rewrite)
        use_rerank = (self.config.rerank_mode != "off") if rerank is None else rerank

        # ── 查询结果缓存:命中则直接返回缓存结果(并显式标注 cached=True) ──
        cache_key, cached_payload = self.cache.get_query(question, canonical_filter(filter), top_n)
        cache_info = {
            "used": bool(self.cache.enabled),
            "type": "query",
            "hit": False,
            "key": cache_key,
            "data_version": self.store.data_version,
        }
        if cached_payload is not None:
            restored = self._restore_results(cached_payload.get("results") or [])
            if restored is not None:
                cache_info["hit"] = True
                self.last_search_debug = {
                    "question": question,
                    "rewrite_mode": cached_payload.get("rewrite_mode", mode),
                    "variants": cached_payload.get("variants") or [question],
                    "variant_count": len(cached_payload.get("variants") or [question]),
                    "variant_hits": cached_payload.get("variant_hits", []),
                    "fused_count": cached_payload.get("fused_count"),
                    "gate_score": cached_payload.get("gate_score"),
                    "retry": cached_payload.get("retry") or {"enabled": False, "triggered": False},
                    "rewrite": cached_payload.get("rewrite"),
                    "rerank": cached_payload.get("rerank", use_rerank),
                    "filter": filter_spec,
                    "filter_fields": available_fields,
                    "filtered_out": cached_payload.get("filtered_out", 0),
                    "cache": cache_info,
                    "cached": True,
                    "results": self._results_summary(restored),
                }
                if _record_audit:
                    self._record_search_audit(question, restored, cache_info)
                return restored

        rewrite_result: RewriteResult | None = None
        retry: dict = {
            "enabled": bool(self.config.retry_on_low_score) and mode != "off",
            "triggered": False,
            "threshold": self.config.min_score,
        }

        if rewrite is True and mode != "off":
            # 显式要求改写:首轮即多查询,不存在"低分重试"这一步
            rewrite_result = self._rewriter(mode).rewrite_detailed(question)
            used_variants = rewrite_result.variants
            results, search_info, gate = self._search_once(
                used_variants, top_k, top_n, use_rerank, normalized_filter
            )
            retry["gate_before"] = round(gate, 4)
            retry["reason"] = "首轮即改写,无需重试"
        else:
            # 首轮:只用原始查询(与改写功能引入前的路径一致)
            used_variants = [question]
            results, search_info, gate_before = self._search_once(
                used_variants, top_k, top_n, use_rerank, normalized_filter
            )
            retry["gate_before"] = round(gate_before, 4)

            # 低分重试:判定分用向量余弦(grounded_score),不用重排分
            if mode != "off" and retry["enabled"] and gate_before < self.config.min_score:
                rewrite_result = self._rewriter(mode).rewrite_detailed(question)
                retry_variants = rewrite_result.variants
                if len(retry_variants) > 1:
                    retry_results, retry_info, gate_after = self._search_once(
                        retry_variants, top_k, top_n, use_rerank, normalized_filter
                    )
                    retry.update(
                        triggered=True,
                        variants=list(retry_variants),
                        gate_after=round(gate_after, 4),
                        improved=bool(gate_after > gate_before),
                    )
                    # 只有真的变好才换结果:改写不能把原本召回的东西弄丢
                    if gate_after >= gate_before:
                        results, search_info = retry_results, retry_info
                        used_variants = retry_variants
                else:
                    retry["reason"] = "改写未产生新变体,跳过重试"

        self.last_search_debug = {
            "question": question,
            "rewrite_mode": mode,
            "variants": list(used_variants),
            "variant_count": len(used_variants),
            "variant_hits": search_info.get("variant_hits", []),
            "fused_count": search_info.get("fused_count", len(results)),
            "gate_score": round(self.grounded_score(results), 4),
            "retry": retry,
            "rewrite": rewrite_result.to_dict() if rewrite_result else None,
            "rerank": bool(use_rerank),
            "filter": filter_spec,
            "filter_fields": available_fields,
            "filtered_out": search_info.get("filtered_out", 0),
            "filtered_empty": bool(normalized_filter) and not results,
            "cache": cache_info,
            "cached": False,
            "results": self._results_summary(results),
        }
        # 写查询缓存(数据版本号已写进键里,库一变就整体失效)
        if self.cache.enabled:
            self.cache.put_query(
                question,
                canonical_filter(filter),
                {
                    "results": self._serialize_results(results),
                    "variants": list(used_variants),
                    "variant_hits": search_info.get("variant_hits", []),
                    "fused_count": search_info.get("fused_count"),
                    "gate_score": self.last_search_debug["gate_score"],
                    "retry": retry,
                    "rewrite": rewrite_result.to_dict() if rewrite_result else None,
                    "rewrite_mode": mode,
                    "rerank": bool(use_rerank),
                    "filtered_out": search_info.get("filtered_out", 0),
                    "data_version": self.store.data_version,
                },
                top_n=top_n,
            )
        if _record_audit:
            self._record_search_audit(question, results, cache_info)
        return results

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

    def query(
        self,
        question: str,
        top_k: int | None = None,
        top_n: int | None = None,
        filter: dict | None = None,
    ) -> Answer:
        """检索 + 生成,返回带引用的完整答案。

        :param filter: 元数据过滤,直接透传给 `search()`(见 metadata.py 的匹配规则)
        """
        started = time.time()
        chunks = self.search(question, top_k=top_k, top_n=top_n, filter=filter, _record_audit=False)
        filter_spec = {str(key): value for key, value in (filter or {}).items()}
        canonical = canonical_filter(filter)

        llm_called = False
        cache_info = {
            "used": bool(self.cache.enabled),
            "type": "generation",
            "hit": False,
            "key": "",
            "data_version": self.store.data_version,
        }

        # ── 生成结果缓存:命中则跳过整次大模型调用 ──
        chunk_ids = [item.chunk.chunk_id for item in chunks]
        gen_key, cached_answer = self.cache.get_generation(
            question, canonical, chunk_ids, self.config.llm_model
        )
        cache_info["key"] = gen_key
        answer: Answer | None = None
        if cached_answer is not None:
            try:
                answer = Answer(
                    question=cached_answer.get("question", question),
                    answer=cached_answer.get("answer", ""),
                    citations=list(cached_answer.get("citations") or []),
                    contexts=list(cached_answer.get("contexts") or []),
                    grounded=bool(cached_answer.get("grounded", False)),
                    debug=dict(cached_answer.get("debug") or {}),
                )
                cache_info["hit"] = True
            except Exception as exc:  # noqa: BLE001 - 缓存内容坏掉就当未命中
                logger.warning("生成缓存载荷无法还原(按未命处理):%s", exc)
                answer = None

        if answer is None:
            llm_called = self._llm_client() is not None
            answer = AnswerGenerator(
                client=self._llm_client(), min_score=self.config.min_score
            ).generate(question, chunks, gate_score=self.grounded_score(chunks))
            if self.cache.enabled and answer.answer:
                self.cache.put_generation(
                    question,
                    canonical,
                    chunk_ids,
                    self.config.llm_model,
                    answer.to_dict(),
                )

        answer.debug["elapsed"] = round(time.time() - started, 3)
        # 把改写/重试的审计信息一并带出:只有能看到"改了什么、有没有重试",才谈得上调参
        debug = self.last_search_debug
        answer.debug["rewrite_mode"] = debug.get("rewrite_mode", "off")
        answer.debug["variants"] = debug.get("variants", [])
        answer.debug["retry_triggered"] = bool(debug.get("retry", {}).get("triggered"))
        if debug.get("retry", {}).get("gate_before") is not None:
            answer.debug["gate_before"] = debug["retry"]["gate_before"]
        # 元数据过滤与缓存同样要可见:过滤掉了多少、有没有命中缓存
        answer.debug["filter"] = filter_spec
        answer.debug["filtered_out"] = debug.get("filtered_out", 0)
        answer.debug["filter_fields"] = debug.get("filter_fields", [])
        answer.debug["cache_hit"] = bool(cache_info.get("hit"))
        answer.debug["cache_type"] = cache_info.get("type")
        if cache_info.get("hit"):
            answer.debug["cached"] = True
            answer.debug["contexts"] = answer.contexts

        # 一次问答只写一条审计(检索 + 生成合并),避免同一次 query 产生两条日志
        entry = build_query_entry(
            search_entry=build_search_entry(
                question=question,
                debug=debug,
                filter_spec=filter_spec,
                fingerprint=self.embedder.fingerprint,
                results=chunks,
                cache_info=cache_info,
                audit_dir=str(self.audit.audit_dir),
            ),
            answer=answer,
            llm_called=llm_called,
            elapsed=answer.debug["elapsed"],
            cache_info=cache_info,
        )
        self.audit.record(entry)
        return answer

    # ────────────────────────── 运维 ──────────────────────────

    def reset(self) -> None:
        """清空当前知识库(删除所有 chunk 与向量,保留目录);缓存一起清掉。"""
        self.store.clear()
        # 写回一个空库:直接删除文件比留空 meta 更干净
        for name in ("chunks.jsonl", "vectors.npy", "meta.json"):
            target = self.kb_dir / name
            if target.exists():
                target.unlink()
        # 缓存必须跟着失效:embedding 缓存虽然内容寻址,但 reset 后库里已经没有对应向量,
        # 留着只会让缓存文件与数据不一致(下次入库会重建)
        self.cache.clear()
        self.cache.set_version(0)
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
            "query_rewrite": self.config.query_rewrite,
            "query_variants": self.config.query_variants,
            "retry_on_low_score": self.config.retry_on_low_score,
            # 企业级三项能力的状态(便于 `stats` / Agent 工具一眼看到是否生效)
            "data_version": self.store.data_version,
            "filter_fields": self.filter_fields,
            "cache_enabled": self.config.cache_enabled,
            "cache": self.cache.describe(),
            "audit_enabled": self.config.audit_enabled,
            "audit": self.audit.describe(),
        }


# 进程内缓存:Agent 工具会在一次对话里被多次调用,重复构造 embedder 会反复加载模型
_PIPELINES: dict[tuple, RagPipeline] = {}


def format_search_audit(debug: dict | None) -> str:
    """把 `last_search_debug` 渲染成人类可读的几行(CLI 与 Agent 工具的默认输出)。

    刻意**只**输出「改写/融合/重试」这类过程信息,不输出排序分——排序分与判定分混用是本项目
    踩过的坑(见 docs/rag-self-built.md 5.10),审计信息里再混进去只会让读数的人更迷糊。

    没有可用信息时返回空串,调用方无需特判。
    """
    if not debug or not debug.get("variants") or debug.get("empty_store"):
        return ""
    lines = [
        f"查询改写:{debug.get('rewrite_mode', 'off')}"
        f"(变体 {debug.get('variant_count', len(debug.get('variants', []))) } 个"
        f" | 融合候选 {debug.get('fused_count', '-')} 条"
        f" | 判定分(向量余弦) {debug.get('gate_score', '-')})"
    ]
    for hit in debug.get("variant_hits", []):
        lines.append(
            f"  · 变体「{hit.get('variant', '')}」→ 向量命中 {hit.get('dense_hits', 0)} 条"
            f" / 关键词命中 {hit.get('sparse_hits', 0)} 条"
        )
    retry = debug.get("retry") or {}
    if retry.get("triggered"):
        lines.append(
            f"  · 低分重试:已触发(判定分 {retry.get('gate_before')} → {retry.get('gate_after')},"
            f"阈值 {retry.get('threshold')},改善={'是' if retry.get('improved') else '否'})"
        )
    elif retry.get("reason"):
        lines.append(f"  · 低分重试:{retry['reason']}")
    elif retry.get("enabled"):
        lines.append(
            f"  · 低分重试:未触发(首轮判定分 {retry.get('gate_before')} ≥ 阈值 {retry.get('threshold')})"
        )
    elif retry:
        lines.append("  · 低分重试:已关闭")
    rewrite = debug.get("rewrite") or {}
    if rewrite.get("llm_used"):
        lines.append(
            f"  · 大模型改写:成功(新增 {len(rewrite.get('llm_variants') or [])} 个变体,"
            f"耗时 {rewrite.get('elapsed')}s)"
        )
    elif rewrite.get("llm_error"):
        lines.append(f"  · 大模型改写:失败→已降级为本地规则({rewrite.get('llm_error')})")
    if debug.get("filter"):
        lines.append(
            f"  · 元数据过滤:{debug['filter']}"
            f"(过滤掉 {debug.get('filtered_out', 0)} 条候选)"
        )
    cache = debug.get("cache") or {}
    if cache.get("used"):
        lines.append(
            f"  · 缓存:{cache.get('type')} {'命中' if cache.get('hit') else '未命中'}"
            f"(数据版本 {cache.get('data_version')})"
        )
    return "\n".join(lines)


def get_pipeline(
    kb: str | None = None,
    config: RagConfig | None = None,
    embedder: Embedder | None = None,
    create: bool = True,
) -> RagPipeline:
    """获取(带缓存的)知识库流水线。

    :param create: 见 RagPipeline.__init__。Agent 工具等只读场景应传 False,
        让「库不存在」直接暴露为 FileNotFoundError,而不是建出一个空库。
    """
    config = config or load_rag_config()
    kb_name = kb or config.default_kb
    key = (
        kb_name,
        config.embedding_provider,
        config.embedding_model,
        config.chunk_size,
        config.cache_enabled,
        config.audit_enabled,
        create,
        id(embedder),
    )
    if embedder is not None:
        # 外部注入的 embedder 不缓存,避免测试之间互相污染
        return RagPipeline(kb=kb_name, config=config, embedder=embedder, create=create)
    if key not in _PIPELINES:
        _PIPELINES[key] = RagPipeline(kb=kb_name, config=config, create=create)
    return _PIPELINES[key]


def clear_pipeline_cache() -> None:
    """清空流水线缓存(配置变更或测试时使用)。"""
    _PIPELINES.clear()


__all__ = [
    "RagPipeline",
    "IngestReport",
    "get_pipeline",
    "clear_pipeline_cache",
    "format_search_audit",
    "EmbeddingMismatchError",
]
