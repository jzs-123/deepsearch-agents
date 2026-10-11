"""审计日志(app/rag/audit.py + pipeline 接入):写入、读回、降级。

审计是这次明确要的企业级能力,验收点有三个:

1. **一条完整记录**:时间戳/查询/过滤/改写变体/是否命中缓存/是否低分重试/结果摘要(不含原文)/
   是否调用大模型与耗时/拒答原因/向量指纹;
2. **能读回**:`read_audit()` 与 `python -m app.rag.cli audit -k <kb> --tail N` 都能看到;
3. **不拖垮查询**:目录不可写、文件损坏时只 warning,检索/问答照常返回。
"""

from __future__ import annotations

import json

import pytest

from app.rag.audit import AuditLogger, audit_path, format_audit_records, read_audit
from app.rag.config import RagConfig
from app.rag.pipeline import RagPipeline, clear_pipeline_cache

DOC = "# 风控制度\n\n授信额度上限为 500 元；连续履约 6 个月可提额至 5000 元。\n"
LONG_BODY = "这是一段很长的正文内容" * 40


def make_pipeline(tmp_path, embedder, kb: str = "auditkb", **overrides):
    params = dict(
        data_dir=tmp_path / "_rag_data",
        default_kb=kb,
        embedding_provider="hash",
        chunk_size=500,
        chunk_overlap=80,
        query_rewrite="off",
        audit_enabled=True,
    )
    params.update(overrides)
    return RagPipeline(kb=kb, config=RagConfig(**params), embedder=embedder)


def read_records(pipeline) -> list[dict]:
    return read_audit(pipeline.audit.audit_dir, pipeline.kb, tail=100)


# ── 1. 写入一条完整记录 ─────────────────────────────────────────────


def test_search_writes_one_complete_audit_record(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder)
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度上限", top_n=2, filter={"path": "doc.md"})

    records = read_records(pipeline)
    assert len(records) == 1, "一次 search 只写一条记录"
    record = records[0]
    assert record["kind"] == "search"
    assert record["kb"] == "auditkb"
    assert record["query"] == "授信额度上限"
    assert record["filter"] == {"path": "doc.md"}
    assert record["rewrite_mode"] == "off"
    assert record["variants"] == ["授信额度上限"]
    assert "triggered" in record["retry"]
    assert record["cache"]["used"] is False
    assert record["llm"]["called"] is False
    assert record["embedding_fingerprint"].startswith("hash:")
    assert record["ts"]
    # 结果摘要:定位够用,但不含原文
    assert record["results"] and record["results"][0]["source"] == "doc.md"
    assert set(record["results"][0]) >= {"rank", "chunk_id", "source", "score", "channels"}
    assert "text" not in json.dumps(record, ensure_ascii=False)


def test_audit_records_that_filter_emptied_the_candidates(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditfilter")
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度上限", top_n=2, filter={"部门": "法务"})
    record = read_records(pipeline)[-1]
    assert record["filter"] == {"部门": "法务"}
    assert record["results"] == []
    assert record["filtered_out"] > 0, "被过滤掉的候选数必须入档,便于解释'为什么没结果'"


def test_audit_does_not_store_full_chunk_text(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder)
    pipeline.add_texts([LONG_BODY], source="long.md")
    pipeline.search("正文内容", top_n=1)
    raw = pipeline.audit.path.read_text(encoding="utf-8")
    assert LONG_BODY not in raw, "审计日志不得包含完整原文(避免日志爆炸与二次泄露)"
    assert len(raw) < 4000


def test_query_writes_exactly_one_record_with_generation_info(tmp_path, hash_embedder, stub_client, monkeypatch):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditquery")
    pipeline.add_texts([DOC], source="doc.md")
    stub = stub_client(reply="授信额度上限为 500 元[1]。")
    monkeypatch.setattr(pipeline, "_llm_client", lambda: stub)

    answer = pipeline.query("授信额度上限")
    assert answer.grounded is True

    records = read_records(pipeline)
    assert len(records) == 1, "一次 query 只写一条记录(search 不重复写)"
    record = records[0]
    assert record["kind"] == "query"
    assert record["llm"]["called"] is True
    assert record["llm"]["elapsed"] >= 0
    assert record["llm"]["refusal_reason"] == ""
    assert record["grounded"] is True
    assert record["answer_len"] == len(answer.answer)
    assert record["citations"] and record["citations"][0]["source"] == "doc.md"


def test_query_records_refusal_reason(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditrefuse")
    pipeline.add_texts([DOC], source="doc.md")
    answer = pipeline.query("公司食堂本周的菜单")
    assert answer.grounded is False
    record = read_records(pipeline)[-1]
    assert record["kind"] == "query"
    assert record["llm"]["refusal_reason"] == "score_below_threshold"
    assert record["llm"]["called"] is False, "命中拒答门槛时不应进入大模型"


def test_audit_records_cache_hit(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditcache", cache_enabled=True)
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度上限", top_n=2)
    pipeline.search("授信额度上限", top_n=2)
    records = read_records(pipeline)
    assert records[-1]["cache"]["hit"] is True
    assert records[-1]["cache"]["type"] == "query"
    assert records[-1]["kind"] == "search"


def test_audit_records_retry_and_variants(tmp_path, hash_embedder):
    retry_doc = "# 交易指标\n\n高客单价订单的支付转化率为 18.6%；平台整体的转化率为 3.2%。\n"
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditretry", query_rewrite="local")
    pipeline.add_texts([retry_doc], source="doc.md")
    pipeline.search("成交比例", top_n=2)
    record = read_records(pipeline)[-1]
    assert record["rewrite_mode"] == "local"
    assert len(record["variants"]) > 1
    assert any("转化率" in v for v in record["variants"])
    assert record["retry"]["triggered"] is True
    assert record["retry"]["gate_after"] > record["retry"]["gate_before"]


# ── 2. 读回与 CLI ───────────────────────────────────────────────────


def test_read_audit_tail_limits_records(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="taillkb")
    pipeline.add_texts([DOC], source="doc.md")
    for index in range(5):
        pipeline.search(f"授信额度 {index}", top_n=1)
    assert len(read_audit(pipeline.audit.audit_dir, pipeline.kb, tail=2)) == 2
    assert read_audit(pipeline.audit.audit_dir, "不存在的库", tail=5) == []


def test_format_audit_records_is_human_readable(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="formatkb")
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度上限", top_n=1)
    text = format_audit_records(read_records(pipeline))
    assert "授信额度上限" in text
    assert "source" in text or "doc.md" in text
    assert "指纹" in text
    assert format_audit_records([]) == "(没有审计记录)"


def test_cli_audit_reads_back_records(capsys, monkeypatch, tmp_path, hash_embedder):
    import app.core.config as core_config
    import app.rag.cli as cli
    import app.rag.config as rag_config

    data_dir = tmp_path / "_rag_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = RagConfig(
        data_dir=data_dir,
        default_kb="cliaudit",
        embedding_provider="hash",
        query_rewrite="off",
    )
    seed = RagPipeline(kb="cliaudit", config=cfg, embedder=hash_embedder)
    seed.add_texts([DOC], source="doc.md")
    seed.search("授信额度上限", top_n=1, filter={"部门": "风控"})
    clear_pipeline_cache()

    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)

    code = cli.main(["audit", "-k", "cliaudit", "--tail", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "审计目录:" in out
    assert "授信额度上限" in out
    assert "风控" in out

    code = cli.main(["audit", "-k", "cliaudit", "--tail", "5", "--json"])
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert code == 0
    assert payloads and payloads[-1]["kb"] == "cliaudit"


def test_cli_audit_on_missing_kb_is_empty(capsys, monkeypatch, tmp_path, hash_embedder):
    import app.core.config as core_config
    import app.rag.cli as cli
    import app.rag.config as rag_config

    data_dir = tmp_path / "_rag_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)

    code = cli.main(["audit", "-k", "nokb", "--tail", "5"])
    out = capsys.readouterr().out
    assert code == 0
    assert "(没有审计记录)" in out


# ── 3. 降级与配置 ───────────────────────────────────────────────────


def test_audit_disabled_writes_nothing(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditoff", audit_enabled=False)
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度", top_n=1)
    assert pipeline.audit.enabled is False
    assert not pipeline.audit.path.exists()


def test_audit_write_failure_does_not_break_search(tmp_path, hash_embedder, caplog):
    """审计目录被一个文件占住 → 写入失败只 warning,检索必须照常返回。"""
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="auditbroken")
    pipeline.add_texts([DOC], source="doc.md")
    # 把审计目录路径替换成一个文件
    blocker = tmp_path / "blocker"
    blocker.write_text("我是文件不是目录", encoding="utf-8")
    pipeline.audit = AuditLogger(audit_dir=blocker / "sub", kb=pipeline.kb, enabled=True)

    results = pipeline.search("授信额度", top_n=1)
    assert results, "审计写失败不能影响检索结果"
    assert pipeline.audit.failures == 1
    assert pipeline.audit.writes == 0


def test_audit_default_dir_follows_data_dir(tmp_path, hash_embedder):
    """默认审计目录跟随数据根(默认即 rag_data/.audit/),测试/自测用临时目录时不写真实 rag_data。"""
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="dirkb")
    assert pipeline.audit.audit_dir == tmp_path / "_rag_data" / ".audit"
    assert audit_path(pipeline.audit.audit_dir, "dirkb").name == "dirkb.jsonl"


def test_audit_explicit_dir_overrides_default(tmp_path, hash_embedder):
    custom = tmp_path / "custom_audit"
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="customkb", audit_dir=str(custom))
    pipeline.add_texts([DOC], source="doc.md")
    pipeline.search("授信额度", top_n=1)
    assert (custom / "customkb.jsonl").exists()


def test_stats_exposes_audit_state(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="statkaudit")
    stats = pipeline.stats()
    assert stats["audit_enabled"] is True
    assert stats["audit"]["path"].endswith("statkaudit.jsonl")


def test_empty_store_search_is_audited(tmp_path, hash_embedder):
    pipeline = make_pipeline(tmp_path, hash_embedder, kb="emptyaudit")
    assert pipeline.search("任意问题") == []
    record = read_records(pipeline)[-1]
    assert record["kind"] == "search"
    assert record["results"] == []


def test_audit_logger_is_append_only(tmp_path):
    logger = AuditLogger(audit_dir=tmp_path / "audit", kb="appendkb", enabled=True)
    logger.record({"kind": "search", "query": "第一条"})
    logger.record({"kind": "search", "query": "第二条"})
    records = read_audit(tmp_path / "audit", "appendkb", tail=10)
    assert [item["query"] for item in records] == ["第一条", "第二条"]
    assert records[0]["ts"] and records[0]["kb"] == "appendkb"
