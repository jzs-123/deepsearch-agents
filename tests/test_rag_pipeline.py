"""app/rag/pipeline.py：入库 / 检索 / 生成 的编排层。

本文件锁死历史 bug（docs/rag-self-built.md 5.10）：
`min_score` 一直是拿**向量余弦**标定的，但开启词面重排后 `item.score` 变成了**重排分**，
两者尺度完全不同。混用后同一批数据、同一阈值，拒答准确率从 100% 掉到 50%
（无关问题的最高分被重排从 0.17 抬到 0.22，越过阈值 0.2，于是开始拿无关片段生成答案）。
解法是把「排序」和「判定」两个职责分开：排序用重排分，**判定一律用 `grounded_score()`（向量余弦）**。
见 `test_grounded_score_prefers_dense_cosine_over_rank_score`。
"""

from __future__ import annotations

import logging

import pytest

from app.rag.config import RagConfig
from app.rag.embedding import HashEmbedder
from app.rag.pipeline import RagPipeline, clear_pipeline_cache, get_pipeline
from app.rag.types import RetrievedChunk


KB_DOC = (
    "# 风控\n\n"
    "风控模型的准确率为 91.5%。\n\n"
    "## 授信\n\n"
    "授信规则要求 A13 型号必须复核，编号 3.2.1。"
)
RELEVANT_Q = "风控模型的准确率是多少"
IRRELEVANT_Q = "公司食堂本周的菜单"


# ── grounded_score：排序分 vs 判定分 ────────────────────────────────


def test_grounded_score_prefers_dense_cosine_over_rank_score(make_chunk):
    """★ 核心回归锁：判定必须取向量余弦（dense_score），不能取排序分。

    这里刻意构造「排序分高、余弦低」的候选：若实现退回用 item.score，
    就会把本该拒答的问题判成有依据。
    """
    chunk = make_chunk(text="无关内容")
    items = [
        RetrievedChunk(chunk=chunk, score=0.9, dense_score=0.05),
        RetrievedChunk(chunk=chunk, score=0.8, dense_score=0.01),
    ]
    assert RagPipeline.grounded_score(items) == pytest.approx(0.05)


def test_grounded_score_picks_first_available_dense_score(make_chunk):
    """取候选里**第一条有** dense_score 的（第一条可能只在稀疏路命中）。"""
    chunk = make_chunk(text="x")
    items = [
        RetrievedChunk(chunk=chunk, score=0.95, dense_score=None),
        RetrievedChunk(chunk=chunk, score=0.30, dense_score=0.44),
    ]
    assert RagPipeline.grounded_score(items) == pytest.approx(0.44)


def test_grounded_score_falls_back_to_rank_score_when_no_dense(make_chunk):
    chunk = make_chunk(text="x")
    items = [RetrievedChunk(chunk=chunk, score=0.77, dense_score=None)]
    assert RagPipeline.grounded_score(items) == pytest.approx(0.77)


def test_grounded_score_empty_is_zero():
    assert RagPipeline.grounded_score([]) == 0.0


def test_grounded_score_ignores_rerank_score_scale(make_chunk):
    """重排后 item.score 是重排分；grounded_score 必须对它完全不敏感。"""
    chunk = make_chunk(text="x")
    before = RetrievedChunk(chunk=chunk, score=0.1, dense_score=0.3)
    after = RetrievedChunk(chunk=chunk, score=0.99, dense_score=0.3)
    assert RagPipeline.grounded_score([before]) == RagPipeline.grounded_score([after])


# ── add_texts / 入库 ────────────────────────────────────────────────


def test_add_texts_goes_through_chunk_document(pipeline):
    """历史 bug（5.7 踩坑）：早期实现「一段文本 = 一个 chunk」，
    导致长文入库后只有一条 3000 字 chunk，检索要么全中要么全漏。
    """
    added = pipeline.add_texts([KB_DOC], source="doc.md")
    assert added >= 2, "长文本必须被切块，而不是整段当成一个 chunk"
    assert len(pipeline.store) == added


def test_add_texts_is_idempotent(pipeline):
    first = pipeline.add_texts([KB_DOC], source="doc.md")
    second = pipeline.add_texts([KB_DOC], source="doc.md")
    assert first == second
    assert len(pipeline.store) == first, "重复写入同一文本不应产生重复候选"


def test_add_texts_same_source_replaces_changed_content(pipeline):
    pipeline.add_texts(["旧规定：学生十五点放学。"], source="放学时间.md")
    pipeline.add_texts(["新规定：学生十六点放学。"], source="放学时间.md")

    chunks = list(pipeline.store.all_chunks())
    assert chunks
    assert all("十五点" not in chunk.text for chunk in chunks)
    assert any("十六点" in chunk.text for chunk in chunks)
    assert {chunk.metadata["document_key"] for chunk in chunks} == {"放学时间.md"}


def test_add_texts_skips_blank_input(pipeline):
    assert pipeline.add_texts(["   ", "\n"]) == 0
    assert len(pipeline.store) == 0


def test_add_texts_offsets_indices_across_inputs(pipeline):
    pipeline.add_texts(["甲内容。", "乙内容。"], source="multi")
    indices = [chunk.chunk_index for chunk in pipeline.store.all_chunks()]
    assert len(indices) == len(set(indices)), "多个输入的 chunk_index 不能撞车"


def test_ingest_embeds_heading_prefixed_text(rag_config):
    """向量编码输入 = heading 前置 + 正文(与 BM25「标题参与索引」对齐)。

    防止有人改回只编码 chunk.text:那样 heading 又退化为纯展示字段,
    「XX的项目经历」这类命中标题词的 query 在向量路上拿不到任何增益。
    """
    from app.rag.pipeline import RagPipeline

    class RecordingEmbedder(HashEmbedder):
        def __init__(self):
            super().__init__(dim=512)
            self.doc_inputs: list[str] = []

        def encode(self, texts, is_query=False, **kwargs):
            if not is_query:
                self.doc_inputs.extend(texts)
            return super().encode(texts, is_query=is_query, **kwargs)

    embedder = RecordingEmbedder()
    pipeline = RagPipeline(kb="headingtest", config=rag_config, embedder=embedder)
    pipeline.add_texts(["# 风控总章\n\n准确率为 91.5%。"], source="doc.md")
    assert embedder.doc_inputs, "入库必须调用向量编码"
    assert all(text.startswith("风控总章") for text in embedder.doc_inputs)
    # 持久化的 chunk.text 保持原文,不带 heading 前缀(展示/引用仍是干净正文)
    assert all(
        not chunk.text.startswith("风控总章\n") for chunk in pipeline.store.all_chunks()
    )


# ── search ──────────────────────────────────────────────────────────


def test_search_on_empty_store_returns_empty(pipeline):
    assert pipeline.search("任意问题") == []


def test_search_returns_ranked_top_n(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    results = pipeline.search(RELEVANT_Q, top_n=2)
    assert 0 < len(results) <= 2
    assert [item.rank for item in results] == list(range(1, len(results) + 1))


def test_search_records_channels(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    top = pipeline.search(RELEVANT_Q, top_n=1)[0]
    assert top.channels, "融合结果必须标明命中通道"
    assert top.dense_score is not None


def test_search_can_disable_rerank(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    assert pipeline.search(RELEVANT_Q, top_n=2, rerank=False)
    assert pipeline.search(RELEVANT_Q, top_n=2, rerank=True)


# ── query：门槛与不调用大模型 ───────────────────────────────────────


def test_query_refuses_irrelevant_question_without_calling_llm(pipeline, stub_client, monkeypatch):
    """★ 低于阈值时直接拒答，且**一次大模型都不调**（省钱且最可靠）。"""
    pipeline.add_texts([KB_DOC], source="doc.md")
    stub = stub_client(reply="不该被调用")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)

    answer = pipeline.query(IRRELEVANT_Q)
    assert answer.grounded is False
    assert answer.debug["reason"] == "score_below_threshold"
    assert stub.calls == [], "命中拒答门槛时绝不能进入生成层"
    assert pipeline.grounded_score(pipeline.search(IRRELEVANT_Q)) < pipeline.config.min_score


def test_query_answers_relevant_question(pipeline, stub_client, monkeypatch):
    pipeline.add_texts([KB_DOC], source="doc.md")
    stub = stub_client(reply="风控准确率为 91.5%[1]。")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)

    answer = pipeline.query(RELEVANT_Q)
    assert answer.answer == "风控准确率为 91.5%[1]。"
    assert answer.grounded is True
    assert answer.citations
    assert len(stub.calls) == 1
    assert answer.debug["gate_score"] >= pipeline.config.min_score


def test_query_without_llm_config_degrades_gracefully(pipeline):
    """没有配置大模型时检索能力仍要可用（返回依据 + 明确提示）。"""
    pipeline.add_texts([KB_DOC], source="doc.md")
    answer = pipeline.query(RELEVANT_Q)
    assert answer.grounded is False
    assert answer.debug["reason"] == "llm_unavailable"


def test_query_debug_reports_both_score_families(pipeline, stub_client, monkeypatch):
    """把「排序分」和「判定分」都写进 debug —— 这正是 5.10 修复后要能一眼看清的东西。"""
    pipeline.add_texts([KB_DOC], source="doc.md")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub_client())
    answer = pipeline.query(RELEVANT_Q)
    assert "gate_score" in answer.debug and "rank_score" in answer.debug
    assert "elapsed" in answer.debug


# ── ingest ──────────────────────────────────────────────────────────


def test_ingest_missing_path_raises(pipeline, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.ingest(tmp_path / "nope")


def test_ingest_directory_and_report(pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(KB_DOC, encoding="utf-8")
    (docs / "b.txt").write_text("另一份文档：渠道分成比例为 30%。", encoding="utf-8")

    report = pipeline.ingest(docs)
    assert report.files_total == 2
    assert report.files_indexed == 2
    assert report.chunks_added >= 2
    assert report.total_chunks == len(pipeline.store)
    assert report.warnings == []


def test_ingest_same_file_twice_replaces_instead_of_appending(pipeline, tmp_path):
    """历史 bug（5.7）：文档更新后旧内容不能残留，否则会检索到过期版本。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(KB_DOC, encoding="utf-8")

    first = pipeline.ingest(docs)
    second = pipeline.ingest(docs)
    assert second.total_chunks == first.total_chunks
    assert second.chunks_replaced > 0
    assert len(pipeline.store) == first.total_chunks


def test_ingest_changed_file_same_path_replaces_old_version(pipeline, tmp_path):
    doc = tmp_path / "放学时间.md"
    doc.write_text("旧规定：学生十五点放学。", encoding="utf-8")
    pipeline.ingest(doc)

    doc.write_text("新规定：学生十六点放学。", encoding="utf-8")
    report = pipeline.ingest(doc)

    chunks = list(pipeline.store.all_chunks())
    assert report.chunks_replaced > 0
    assert all("十五点" not in chunk.text for chunk in chunks)
    assert any("十六点" in chunk.text for chunk in chunks)
    assert {chunk.metadata["document_key"] for chunk in chunks} == {"放学时间.md"}


def test_ingest_document_key_replaces_renamed_file(pipeline, tmp_path):
    old = tmp_path / "放学时间通知.txt"
    old.write_text("旧规定：学生十五点放学。", encoding="utf-8")
    pipeline.ingest(old, document_key="school-dismissal-time", revision_id="rev-1")

    new = tmp_path / "冬季作息调整.txt"
    new.write_text("新规定：学生十六点放学。", encoding="utf-8")
    report = pipeline.ingest(
        new,
        document_key="school-dismissal-time",
        revision_id="rev-2",
        effective_from="2026-11-01",
    )

    chunks = list(pipeline.store.all_chunks())
    assert report.chunks_replaced > 0
    assert {chunk.source for chunk in chunks} == {"冬季作息调整.txt"}
    assert {chunk.metadata["document_key"] for chunk in chunks} == {
        "school-dismissal-time"
    }
    assert {chunk.metadata["revision_id"] for chunk in chunks} == {"rev-2"}
    assert {chunk.metadata["effective_from"] for chunk in chunks} == {"2026-11-01"}


def test_ingest_document_key_rejects_directory(pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("文档 A", encoding="utf-8")

    with pytest.raises(ValueError, match="只适用于单文件"):
        pipeline.ingest(docs, document_key="shared-key")


def test_ingest_empty_directory_warns(pipeline, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = pipeline.ingest(empty)
    assert report.files_total == 0
    assert report.warnings


def test_ingest_unsupported_file_is_skipped_with_reason(pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.xyz").write_text("不支持的类型", encoding="utf-8")
    (docs / "ok.md").write_text(KB_DOC, encoding="utf-8")
    report = pipeline.ingest(docs)
    assert report.files_indexed == 1


# ── reset / stats ───────────────────────────────────────────────────


def test_reset_clears_knowledge_base(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    assert len(pipeline.store) > 0
    pipeline.reset()
    assert len(pipeline.store) == 0
    # 重新读盘也应为空
    reopened = RagPipeline(kb="test", config=pipeline.config, embedder=pipeline.embedder)
    assert len(reopened.store) == 0


def test_stats_reports_configuration(pipeline):
    stats = pipeline.stats()
    assert stats["kb"] == "test"
    assert stats["embedding"].startswith("hash:")
    assert stats["chunk_size"] == pipeline.config.chunk_size
    assert stats["rerank_mode"] == pipeline.config.rerank_mode


# ── get_pipeline 缓存 ───────────────────────────────────────────────


def test_get_pipeline_is_cached_by_key():
    clear_pipeline_cache()
    first = get_pipeline(kb="cachetest")
    second = get_pipeline(kb="cachetest")
    assert first is second
    clear_pipeline_cache()
    third = get_pipeline(kb="cachetest")
    assert third is not first


def test_get_pipeline_with_injected_embedder_is_not_cached():
    """注入 embedder 的路径不缓存，避免测试之间互相污染。"""
    clear_pipeline_cache()
    embedder = HashEmbedder(dim=128)
    first = get_pipeline(kb="inject", embedder=embedder)
    second = get_pipeline(kb="inject", embedder=embedder)
    assert first is not second


def test_chunk_size_change_on_existing_kb_warns(pipeline, tmp_path, caplog):
    """旧 chunk 是按老参数切的，混用会降低检索质量 —— 必须给出提示。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(KB_DOC, encoding="utf-8")
    pipeline.ingest(docs)

    other_cfg = RagConfig(
        data_dir=pipeline.config.data_dir,
        default_kb="test",
        embedding_provider="hash",
        chunk_size=pipeline.config.chunk_size + 111,
        chunk_overlap=20,
    )
    with caplog.at_level(logging.WARNING, logger="app.rag.pipeline"):
        RagPipeline(kb="test", config=other_cfg, embedder=HashEmbedder())
    assert "chunk_size" in caplog.text
