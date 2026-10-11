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
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.bm25 import BM25Index
from app.rag.chunking import chunk_document
from app.rag.config import RagConfig, load_rag_config
from app.rag.embedding import Embedder, build_embedder
from app.rag.generator import AnswerGenerator, ChatClient
from app.rag.hybrid import hybrid_search, reciprocal_rank_fusion
from app.rag.loaders import LoadedDocument, iter_supported_files, load_document
from app.rag.rerank import build_reranker
from app.rag.rewrite import RewriteResult, build_rewriter
from app.rag.types import Answer, Chunk, RetrievedChunk, content_fingerprint
from app.rag.vectorstore import EmbeddingMismatchError, VectorStore

logger = logging.getLogger(__name__)

#: 多查询融合时,非原始查询的变体权重折扣(原始查询权重要略高,避免被变体挤掉)
VARIANT_DISCOUNT = 0.85


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
                    # 文本来源(text_layer/ocr/mixed/none):排障时用于判断该文件的正文
                    # 是原文还是 OCR 识别结果(OCR 会带错字,见 docs/rag-ocr.md)
                    "extraction": doc.extraction,
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
    ) -> tuple[list[RetrievedChunk], dict, float]:
        """跑一轮「检索 → 重排」,返回 (最终结果, 检索审计信息, 最终判定分)。

        判定分取的是**最终结果的** `grounded_score()`(向量余弦)。这一点很关键:
        重排前候选按 RRF 排、重排后按词面分排,若拿重排前的候选算分,审计里就会出现
        「重试判定分 0.1541,而实际用的是 0.2181」这种对不上账的情况(本项目真实踩到,已修)。
        """
        candidates, info = self._retrieve_variants(variants, top_k)
        results = self._apply_rerank(candidates, variants, top_n, use_rerank)
        return results, info, self.grounded_score(results)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        top_n: int | None = None,
        rerank: bool | None = None,
        rewrite: bool | None = None,
    ) -> list[RetrievedChunk]:
        """混合检索 + 重排,返回最终 top_n 结果。

        :param rewrite: None=按配置(默认 local,低分时自动重试);False=强制关闭(历史行为);
            True=首轮就多查询(不等低分)
        """
        top_k = top_k or self.config.top_k
        top_n = top_n or self.config.top_n
        if not len(self.store):
            self.last_search_debug = {
                "question": question,
                "rewrite_mode": self._resolve_rewrite_mode(rewrite),
                "empty_store": True,
                "variants": [],
                "retry": {"enabled": bool(self.config.retry_on_low_score), "triggered": False},
                "results": [],
            }
            return []

        mode = self._resolve_rewrite_mode(rewrite)
        use_rerank = (self.config.rerank_mode != "off") if rerank is None else rerank
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
                used_variants, top_k, top_n, use_rerank
            )
            retry["gate_before"] = round(gate, 4)
            retry["reason"] = "首轮即改写,无需重试"
        else:
            # 首轮:只用原始查询(与改写功能引入前的路径一致)
            used_variants = [question]
            results, search_info, gate_before = self._search_once(
                used_variants, top_k, top_n, use_rerank
            )
            retry["gate_before"] = round(gate_before, 4)

            # 低分重试:判定分用向量余弦(grounded_score),不用重排分
            if mode != "off" and retry["enabled"] and gate_before < self.config.min_score:
                rewrite_result = self._rewriter(mode).rewrite_detailed(question)
                retry_variants = rewrite_result.variants
                if len(retry_variants) > 1:
                    retry_results, retry_info, gate_after = self._search_once(
                        retry_variants, top_k, top_n, use_rerank
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
            "results": [
                {
                    "rank": item.rank,
                    "source": item.chunk.source,
                    "chunk_index": item.chunk.chunk_index,
                    "channels": list(item.channels),
                    "score": round(item.score, 4),
                    "dense": None if item.dense_score is None else round(item.dense_score, 4),
                    "sparse": None if item.sparse_score is None else round(item.sparse_score, 4),
                    "rrf": round(item.rrf_score, 4),
                }
                for item in results
            ],
        }
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

    def query(self, question: str, top_k: int | None = None, top_n: int | None = None) -> Answer:
        """检索 + 生成,返回带引用的完整答案。"""
        started = time.time()
        chunks = self.search(question, top_k=top_k, top_n=top_n)
        answer = AnswerGenerator(
            client=self._llm_client(), min_score=self.config.min_score
        ).generate(question, chunks, gate_score=self.grounded_score(chunks))
        answer.debug["elapsed"] = round(time.time() - started, 3)
        # 把改写/重试的审计信息一并带出:只有能看到"改了什么、有没有重试",才谈得上调参
        debug = self.last_search_debug
        answer.debug["rewrite_mode"] = debug.get("rewrite_mode", "off")
        answer.debug["variants"] = debug.get("variants", [])
        answer.debug["retry_triggered"] = bool(debug.get("retry", {}).get("triggered"))
        if debug.get("retry", {}).get("gate_before") is not None:
            answer.debug["gate_before"] = debug["retry"]["gate_before"]
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
            "query_rewrite": self.config.query_rewrite,
            "query_variants": self.config.query_variants,
            "retry_on_low_score": self.config.retry_on_low_score,
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
