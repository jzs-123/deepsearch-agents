"""app/rag/metadata.py 与检索期元数据过滤:契约与回归。

覆盖三层:

1. **抽取**:文件级元数据 + Markdown YAML frontmatter;非法/缺失/未闭合都必须"只 warning 不报错";
2. **匹配**:等值、列表值(命中其一)、缺字段=不匹配、大小写/空白归一、空列表=不可满足;
3. **接入**:元数据随 chunk 落盘、`meta.json` 记录可过滤字段、`search(filter=...)` 在「召回后、重排前」过滤、
   CLI `--filter` 与 Agent 工具 `filter_json` 的可见性与错误提示。

另锁一条向后兼容:**不传 filter 时 search 行为与引入该功能前逐值一致**(按历史实现手工算一遍再比对)。
"""

from __future__ import annotations

import json

import pytest

from app.rag.config import RagConfig
from app.rag.metadata import (
    RESERVED_KEYS,
    InvalidFilterError,
    apply_filter,
    canonical_filter,
    extract_metadata,
    file_metadata,
    filter_fields_of,
    format_filter_hint,
    matches_filter,
    normalize_filter,
    parse_filter_json,
    split_frontmatter,
)
from app.rag.pipeline import RagPipeline, clear_pipeline_cache, get_pipeline

FRONTMATTER_DOC = """---
部门: 风控
类型: 制度
密级: 内部
版本: 2
---

# 风控额度制度

新客首单的授信额度上限为 500 元。
"""

PLAIN_DOC = "# 财务报销制度\n\n出差住宿费标准为每晚上限 600 元。\n"

MARKET_DOC = """---
部门: 市场
类型: 方案
---

# 市场活动方案

全年新增会员 50 万。
"""


# ── 1. frontmatter 抽取 ─────────────────────────────────────────────


def test_frontmatter_is_parsed_and_stripped_from_body():
    result = split_frontmatter(FRONTMATTER_DOC)
    assert result.frontmatter == {"部门": "风控", "类型": "制度", "密级": "内部", "版本": 2}
    assert result.stripped is True
    assert result.warnings == []
    assert result.body.startswith("# 风控额度制度")
    assert "部门:" not in result.body, "frontmatter 必须从正文里剥掉(否则污染检索)"


def test_plain_document_has_no_frontmatter():
    result = split_frontmatter(PLAIN_DOC)
    assert result.frontmatter == {}
    assert result.stripped is False
    assert result.body == PLAIN_DOC
    assert result.warnings == []


def test_frontmatter_without_closing_line_is_not_treated_as_metadata():
    """很多 Markdown 文章开头就是一条分隔线,不能把正文当 frontmatter 吃掉。"""
    text = "---\n\n# 标题\n\n正文内容。\n"
    result = split_frontmatter(text)
    assert result.frontmatter == {}
    assert result.body == text
    assert result.warnings and "未找到结束行" in result.warnings[0]


def test_invalid_yaml_frontmatter_warns_and_degrades():
    text = "---\n部门: 风控\n类型: [未闭合\n---\n\n# 标题\n"
    result = split_frontmatter(text)
    assert result.frontmatter == {}
    assert result.body == text, "解析失败时不能剥掉正文"
    assert any("解析失败" in item for item in result.warnings)


def test_frontmatter_must_be_a_mapping():
    text = "---\n- 甲\n- 乙\n---\n\n正文\n"
    result = split_frontmatter(text)
    assert result.frontmatter == {}
    assert any("不是键值对" in item for item in result.warnings)


def test_reserved_keys_in_frontmatter_are_ignored():
    """file/path/suffix 这类系统字段不能被文档里的同名值覆盖,否则过滤语义会乱。"""
    text = "---\nfile: 假的.md\n部门: 风控\n---\n\n正文\n"
    result = split_frontmatter(text)
    assert "file" not in result.frontmatter
    assert result.frontmatter["部门"] == "风控"
    assert any("同名" in item for item in result.warnings)


def test_oversized_frontmatter_is_rejected():
    text = "---\n" + ("字段" * 5000) + "\n---\n\n正文\n"
    result = split_frontmatter(text)
    assert result.frontmatter == {}
    assert any("超过" in item for item in result.warnings)


def test_too_many_lines_before_closing_marker_is_not_frontmatter():
    body = "\n".join(f"字段{i}: 值{i}" for i in range(200))
    text = f"---\n{body}\n---\n\n正文\n"
    result = split_frontmatter(text)
    assert result.frontmatter == {}
    assert any("未找到结束行" in item for item in result.warnings)


def test_file_metadata_fields(tmp_path):
    path = tmp_path / "制度.md"
    path.write_text("内容", encoding="utf-8")
    fields = file_metadata(path, base_dir=tmp_path)
    assert fields["file"] == "制度.md"
    assert fields["path"] == "制度.md"
    assert fields["suffix"] == ".md"
    assert fields["size"] > 0
    assert fields["mtime"].startswith("20")


def test_extract_metadata_merges_file_and_frontmatter(tmp_path):
    path = tmp_path / "风控.md"
    path.write_text(FRONTMATTER_DOC, encoding="utf-8")
    result = extract_metadata(path, FRONTMATTER_DOC, base_dir=tmp_path)
    assert result.fields["file"] == "风控.md"
    assert result.fields["部门"] == "风控"
    assert result.filter_fields == ["密级", "版本", "类型", "部门"]  # 不含系统字段
    assert RESERVED_KEYS[0] == "file"


# ── 2. 过滤匹配 ─────────────────────────────────────────────────────


def test_normalize_filter_accepts_scalar_and_list():
    assert normalize_filter({"部门": "风控"}) == {"部门": ["风控"]}
    assert normalize_filter({"类型": ["制度", "规范"]}) == {"类型": ["制度", "规范"]}
    assert normalize_filter(None) == {}


def test_normalize_filter_rejects_non_mapping():
    with pytest.raises(InvalidFilterError):
        normalize_filter(["部门"])  # type: ignore[arg-type]
    with pytest.raises(InvalidFilterError):
        normalize_filter({"": "风控"})


@pytest.mark.parametrize(
    "metadata,spec,expected",
    [
        ({"部门": "风控"}, {"部门": "风控"}, True),
        ({"部门": "风控"}, {"部门": "市场"}, False),
        ({"部门": "市场"}, {"部门": ["风控", "市场"]}, True),      # 列表值命中其一
        ({"部门": "法务"}, {"部门": ["风控", "市场"]}, False),
        ({"部门": "风控"}, {"密级": "内部"}, False),                # 缺字段 = 不匹配
        ({"标签": ["甲", "乙"]}, {"标签": "乙"}, True),             # 元数据是列表
        ({"部门": " 风控 "}, {"部门": "风控"}, True),               # 空白归一
        ({"部门": "Risk"}, {"部门": "risk"}, True),                 # 大小写归一
        ({"部门": "风控"}, {"部门": []}, False),                    # 空列表 = 不可满足
        ({"部门": "风控"}, {}, True),                               # 空 filter = 不过滤
        (None, {"部门": "风控"}, False),
    ],
)
def test_matches_filter_matrix(metadata, spec, expected):
    assert matches_filter(metadata, normalize_filter(spec)) is expected


def test_apply_filter_on_retrieved_chunks(pipeline, make_chunk):
    from app.rag.types import RetrievedChunk

    items = [
        RetrievedChunk(chunk=make_chunk(text="甲", source="a.md"), score=1.0, rank=1),
        RetrievedChunk(chunk=make_chunk(text="乙", source="b.md"), score=0.9, rank=2),
    ]
    items[0].chunk.metadata = {"部门": "风控"}
    items[1].chunk.metadata = {"部门": "市场"}
    kept = apply_filter(items, normalize_filter({"部门": "风控"}))
    assert [item.chunk.source for item in kept] == ["a.md"]
    assert len(apply_filter(items, {})) == 2, "空过滤必须原样返回"


def test_canonical_filter_is_order_insensitive():
    assert canonical_filter({"部门": "风控", "类型": "制度"}) == canonical_filter(
        {"类型": "制度", "部门": "风控"}
    )
    assert canonical_filter({"类型": ["制度", "规范"]}) == canonical_filter(
        {"类型": ["规范", "制度"]}
    )
    assert canonical_filter(None) == "{}"


def test_parse_filter_json():
    assert parse_filter_json("") == {}
    assert parse_filter_json(None) == {}
    assert parse_filter_json({"部门": "风控"}) == {"部门": "风控"}
    assert parse_filter_json('{"部门": "风控"}') == {"部门": "风控"}
    with pytest.raises(InvalidFilterError, match="合法 JSON"):
        parse_filter_json("{不是 JSON}")
    with pytest.raises(InvalidFilterError, match="JSON 对象"):
        parse_filter_json("[1, 2]")


def test_filter_fields_of_dedupes_and_skips_reserved(make_chunk):
    chunks = [make_chunk(text="甲"), make_chunk(text="乙")]
    chunks[0].metadata = {"部门": "风控", "file": "a.md"}
    chunks[1].metadata = {"部门": "风控", "类型": "制度"}
    assert filter_fields_of(chunks) == ["类型", "部门"]


def test_format_filter_hint_lists_available_fields():
    hint = format_filter_hint(normalize_filter({"部门": "法务"}), ["密级", "部门"])
    assert "没有符合过滤条件的片段" in hint
    assert "密级" in hint and "部门" in hint


# ── 3. 入库:元数据落盘与字段清单 ────────────────────────────────────


def _write_docs(root):
    (root / "风控额度制度.md").write_text(FRONTMATTER_DOC, encoding="utf-8")
    (root / "市场活动方案.md").write_text(MARKET_DOC, encoding="utf-8")
    (root / "财务报销制度.md").write_text(PLAIN_DOC, encoding="utf-8")
    return root


def test_ingest_attaches_metadata_and_strips_frontmatter(pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_docs(docs)
    report = pipeline.ingest(docs)
    assert report.files_indexed == 3

    chunks = [c for c in pipeline.store.all_chunks() if c.source == "风控额度制度.md"]
    assert chunks, "风控文档应入库"
    assert chunks[0].metadata["部门"] == "风控"
    assert chunks[0].metadata["file"] == "风控额度制度.md"
    assert all("部门:" not in c.text for c in chunks), "frontmatter 不应留在正文里"

    plain = [c for c in pipeline.store.all_chunks() if c.source == "财务报销制度.md"]
    assert plain and "部门" not in plain[0].metadata, "没有 frontmatter 的文档不应有部门字段"


def test_ingest_records_filter_fields_in_meta_json(pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_docs(docs)
    pipeline.ingest(docs)

    meta = json.loads((pipeline.kb_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["filter_fields"] == ["密级", "版本", "类型", "部门"]
    assert pipeline.filter_fields == ["密级", "版本", "类型", "部门"]


def test_ingest_warns_on_invalid_frontmatter_without_failing(pipeline, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "坏.md").write_text("---\n类型: [未闭合\n---\n\n正文内容。\n", encoding="utf-8")
    report = pipeline.ingest(docs)
    assert report.files_indexed == 1, "元数据非法不能导致文件被跳过"
    assert any("解析失败" in w for w in report.warnings)


def test_add_texts_gets_file_level_metadata(pipeline):
    pipeline.add_texts(["风控准确率为 91.5%。"], source="manual.md")
    chunk = pipeline.store.all_chunks()[0]
    assert chunk.metadata["file"] == "manual.md"
    assert chunk.metadata["path"] == "manual.md"
    assert chunk.metadata["document_key"] == "manual.md"
    assert chunk.metadata["revision_id"]
    assert chunk.metadata["content_hash"]


# ── 4. 检索过滤 ─────────────────────────────────────────────────────


def _meta_pipeline(tmp_path, embedder, **overrides):
    params = dict(
        data_dir=tmp_path / "_rag_data",
        default_kb="meta",
        embedding_provider="hash",
        chunk_size=500,
        chunk_overlap=80,
        query_rewrite="off",
    )
    params.update(overrides)
    cfg = RagConfig(**params)
    pipeline = RagPipeline(kb="meta", config=cfg, embedder=embedder)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_docs(docs)
    pipeline.ingest(docs)
    return pipeline


def test_search_filter_equality_excludes_non_matching_sources(tmp_path, hash_embedder):
    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    results = pipeline.search("额度上限", top_n=5, filter={"部门": "风控"})
    assert results
    assert {item.chunk.source for item in results} == {"风控额度制度.md"}
    assert all(item.chunk.metadata["部门"] == "风控" for item in results)


def test_search_filter_list_value_matches_any(tmp_path, hash_embedder):
    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    results = pipeline.search("制度", top_n=5, filter={"类型": ["制度", "方案"]})
    assert results
    assert {item.chunk.metadata.get("类型") for item in results} <= {"制度", "方案"}


def test_search_filter_with_unknown_value_returns_empty_and_reports(tmp_path, hash_embedder):
    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    results = pipeline.search("额度上限", top_n=5, filter={"部门": "法务"})
    assert results == []
    debug = pipeline.last_search_debug
    assert debug["filter"] == {"部门": "法务"}
    assert debug["filtered_empty"] is True
    assert debug["filtered_out"] > 0, "过滤掉的候选数必须可审计"
    assert debug["filter_fields"] == ["密级", "版本", "类型", "部门"]


def test_filter_is_applied_before_rerank(tmp_path, hash_embedder):
    """过滤必须发生在重排之前:否则 top_n 名额会被过滤不掉的候选占满。"""
    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    results = pipeline.search("制度", top_n=2, filter={"部门": "市场"})
    assert results
    assert {item.chunk.source for item in results} == {"市场活动方案.md"}


def test_filter_combines_with_rewrite_and_keeps_audit(tmp_path, hash_embedder):
    pipeline = _meta_pipeline(tmp_path, hash_embedder, query_rewrite="local")
    results = pipeline.search("成交比例", top_n=3, filter={"部门": "风控"})
    debug = pipeline.last_search_debug
    assert debug["filter"] == {"部门": "风控"}
    assert debug["rewrite_mode"] == "local"
    for item in results:
        assert item.chunk.metadata.get("部门") == "风控"


def test_empty_filter_dict_is_same_as_no_filter(tmp_path, hash_embedder):
    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    without = pipeline.search("额度上限", top_n=3)
    with_empty = pipeline.search("额度上限", top_n=3, filter={})
    assert [i.chunk.chunk_id for i in without] == [i.chunk.chunk_id for i in with_empty]
    assert [i.rank for i in without] == [i.rank for i in with_empty]
    assert [round(i.score, 6) for i in without] == [round(i.score, 6) for i in with_empty]


def test_search_without_filter_matches_legacy_path(tmp_path, hash_embedder):
    """★ 向后兼容:不传 filter 时,顺序/分数/通道与历史实现逐值一致。"""
    from app.rag.hybrid import hybrid_search
    from app.rag.rerank import build_reranker

    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    question = "额度上限"
    query_vector = pipeline.embedder.encode_one(question, is_query=True)
    dense_hits = [(hit.chunk, hit.score) for hit in pipeline.store.search(query_vector, 10)]
    sparse_hits = pipeline._bm25_index().search(question, 10)
    candidates = hybrid_search(dense_hits, sparse_hits, top_k=10, rrf_k=pipeline.config.rrf_k).fused
    expected = build_reranker("lexical").rerank(question, candidates, 3)

    got = pipeline.search(question, top_n=3)
    signature = lambda items: [  # noqa: E731
        (i.chunk.chunk_id, i.rank, round(i.score, 6), tuple(i.channels), i.cached) for i in items
    ]
    assert signature(got) == signature(expected)
    assert all(item.cached is False for item in got)


# ── 5. CLI 与 Agent 工具可见性 ──────────────────────────────────────


def _seed_cli_kb(tmp_path, hash_embedder, kb="climeta"):
    data_dir = tmp_path / "_rag_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = RagConfig(
        data_dir=data_dir,
        default_kb=kb,
        embedding_provider="hash",
        query_rewrite="off",
    )
    seed = RagPipeline(kb=kb, config=cfg, embedder=hash_embedder)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    _write_docs(docs)
    seed.ingest(docs)
    clear_pipeline_cache()
    return data_dir


def _patch_settings(monkeypatch, data_dir):
    import app.core.config as core_config
    import app.rag.config as rag_config

    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)


def test_cli_search_with_filter_prints_only_matching(capsys, monkeypatch, tmp_path, hash_embedder):
    import app.rag.cli as cli

    data_dir = _seed_cli_kb(tmp_path, hash_embedder)
    _patch_settings(monkeypatch, data_dir)

    code = cli.main(
        [
            "search", "-k", "climeta", "--embedding-provider", "hash",
            "-q", "额度上限", "-n", "5", "--filter", '{"部门": "风控"}',
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "元数据过滤:" in out
    assert "风控额度制度.md" in out
    assert "市场活动方案.md" not in out


def test_cli_search_filter_miss_prints_filter_hint(capsys, monkeypatch, tmp_path, hash_embedder):
    import app.rag.cli as cli

    data_dir = _seed_cli_kb(tmp_path, hash_embedder, kb="climeta2")
    _patch_settings(monkeypatch, data_dir)
    code = cli.main(
        [
            "search", "-k", "climeta2", "--embedding-provider", "hash",
            "-q", "额度上限", "-n", "5", "--filter", '{"部门": "法务"}',
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "没有符合过滤条件的片段" in out


def test_cli_search_invalid_filter_returns_code_2(capsys, monkeypatch, tmp_path, hash_embedder):
    import app.rag.cli as cli

    data_dir = _seed_cli_kb(tmp_path, hash_embedder, kb="climeta3")
    _patch_settings(monkeypatch, data_dir)
    code = cli.main(
        ["search", "-k", "climeta3", "--embedding-provider", "hash", "-q", "x", "--filter", "{坏}"]
    )
    assert code == 2


def test_cli_list_shows_filter_fields(capsys, monkeypatch, tmp_path, hash_embedder):
    import app.rag.cli as cli

    data_dir = _seed_cli_kb(tmp_path, hash_embedder, kb="climeta4")
    _patch_settings(monkeypatch, data_dir)
    code = cli.main(["list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "可过滤字段" in out
    assert "部门" in out


def test_agent_search_tool_filter_filters_results(tmp_path, hash_embedder, monkeypatch):
    from app.tools import rag_tools

    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline)

    out = rag_tools.search_knowledge_base.invoke(
        {"question": "额度上限", "kb_name": "meta", "filter_json": '{"部门": "风控"}'}
    )
    assert "风控额度制度.md" in out
    assert "市场活动方案.md" not in out


def test_agent_search_tool_filter_miss_message(tmp_path, hash_embedder, monkeypatch):
    from app.tools import rag_tools

    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline)

    out = rag_tools.search_knowledge_base.invoke(
        {"question": "额度上限", "kb_name": "meta", "filter_json": '{"部门": "法务"}'}
    )
    assert "没有符合过滤条件的片段" in out
    assert "当前可用于过滤的字段" in out


def test_agent_tool_rejects_invalid_filter_json_without_raising(tmp_path, hash_embedder, monkeypatch):
    from app.tools import rag_tools

    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline)

    out = rag_tools.search_knowledge_base.invoke(
        {"question": "额度上限", "kb_name": "meta", "filter_json": "{不是 JSON}"}
    )
    assert "过滤条件不合法" in out


def test_agent_ask_tool_accepts_filter_json(tmp_path, hash_embedder, monkeypatch):
    from app.tools import rag_tools

    pipeline = _meta_pipeline(tmp_path, hash_embedder)
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline)

    out = rag_tools.ask_knowledge_base.invoke(
        {"question": "额度上限", "kb_name": "meta", "filter_json": '{"部门": "风控"}'}
    )
    assert isinstance(out, str)
    assert "风控额度制度.md" in out


def test_agent_tools_expose_filter_json_argument():
    from app.tools import rag_tools

    for tool in (rag_tools.search_knowledge_base, rag_tools.ask_knowledge_base):
        properties = tool.args_schema.model_json_schema()["properties"]
        assert "filter_json" in properties
        assert "filter_json" in tool.description, "docstring 必须说明过滤字段与格式,否则模型不会用"


def test_list_tool_shows_filter_fields(capsys, tmp_path, hash_embedder, monkeypatch):
    from app.tools import rag_tools

    data_dir = _seed_cli_kb(tmp_path, hash_embedder, kb="listmeta")
    import app.core.config as core_config
    import app.rag.config as rag_config

    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)
    clear_pipeline_cache()

    out = rag_tools.list_knowledge_bases.invoke({})
    assert "可过滤字段" in out
    assert "部门" in out


def test_get_pipeline_still_uses_meta_filter_fields(tmp_path, hash_embedder, monkeypatch):
    """可过滤字段要能通过 meta.json 读回来(进程重启后仍然可用),不依赖内存里的旧对象。"""
    data_dir = _seed_cli_kb(tmp_path, hash_embedder, kb="persistmeta")
    import app.core.config as core_config
    import app.rag.config as rag_config

    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)
    clear_pipeline_cache()

    pipeline = get_pipeline(kb="persistmeta")
    assert pipeline.filter_fields == ["密级", "版本", "类型", "部门"]
