"""app/rag/rewrite.py + pipeline 的查询改写接入:契约与回归。

分四层锁:

1. **改写器契约**:`rewrite(question) -> [原始查询, 变体…]`,原始查询永远第一(保证"只增不减召回");
   变体数量受 `max_variants` 约束;任何异常都不向上抛。
2. **本地规则**:口语归一化、同义词/领域词扩展、专有名词与数字强化、配置补充词表(脏配置只忽略不报错)。
3. **大模型级**:复用 `ChatClient`;不可用/超时/返回乱码时**必须降级回本地变体**,不得让查询失败。
   测试内一律用 `stub_client` 替换,绝不联网(conftest 也清空了 `OPENAI_API_KEY`)。
4. **检索链路接入**:改写关闭时 `search()` 与引入改写前的行为**逐值一致**(顺序/分数/通道);
   低分重试用 `grounded_score()`(向量余弦)判定;多变体融合按 chunk_id 去重、通道带变体来源;
   审计信息默认可见(`last_search_debug` / CLI / Agent 工具)。
"""

from __future__ import annotations

import pytest

from app.rag.hybrid import hybrid_search
from app.rag.pipeline import format_search_audit, get_pipeline
from app.rag.rerank import build_reranker
from app.rag.rewrite import (
    REWRITE_MODES,
    QueryRewriter,
    build_rewriter,
    parse_synonyms,
    rewrite_query,
)

COLLOQUIAL = "请问一下夜间要是没人回咋办"
DOMAIN_DOC = (
    "# 交易指标\n\n"
    "高客单价订单的支付转化率为 18.6%；平台整体的转化率为 3.2%。\n\n"
    "会员用户的复购率为 42.3%。\n"
)
TITLE_DOC = "# 风控指标\n\n目标完成率要求达到 95% 以上。风控模型 A13 的准确率为 91.5%。\n"


# ── 1. 改写器契约 ───────────────────────────────────────────────────


def test_off_mode_returns_only_the_original_query():
    rewriter = build_rewriter(mode="off")
    assert rewriter.rewrite("成交比例") == ["成交比例"]


def test_original_query_is_always_first():
    """改写只能"加",不能"换" —— 原始查询丢了就可能把本来能召回的片段弄丢。"""
    for mode in ("local", "llm", "local+llm"):
        rewriter = build_rewriter(mode=mode)
        variants = rewriter.rewrite(COLLOQUIAL)
        assert variants[0] == COLLOQUIAL, f"mode={mode} 没有把原始查询放在第一位"


def test_variant_count_is_capped():
    variants = build_rewriter(mode="local", max_variants=2).rewrite("请问一下成交比例是多少呢")
    assert len(variants) <= 2
    variants3 = build_rewriter(mode="local", max_variants=3).rewrite("请问一下成交比例是多少呢")
    assert len(variants3) <= 3


def test_max_variants_must_be_positive():
    with pytest.raises(ValueError, match="max_variants"):
        build_rewriter(mode="local", max_variants=0)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="改写模式"):
        QueryRewriter(mode="magic")


@pytest.mark.parametrize("mode", REWRITE_MODES)
def test_empty_question_is_safe(mode):
    assert build_rewriter(mode=mode).rewrite("") == [""]


def test_rewrite_query_never_raises_on_bad_mode():
    """便捷函数:即使参数非法,也只能返回原始查询,不能把异常抛给检索链路。"""
    assert rewrite_query("成交比例", mode="not-a-mode") == ["成交比例"]


def test_variants_are_deduplicated():
    variants = build_rewriter(mode="local").rewrite("成交比例 成交比例")
    assert len(variants) == len(set(variants))


# ── 2. 本地规则 ─────────────────────────────────────────────────────


def test_colloquial_normalization_drops_noise_words():
    variants = build_rewriter(mode="local").rewrite(COLLOQUIAL)
    normalized = [v for v in variants if "请问" not in v and "咋办" not in v and "一下" not in v]
    assert normalized, f"应产出剥掉口语成分的变体,实际:{variants}"
    assert any("夜间" in v for v in normalized)


def test_synonym_expansion_adds_document_words():
    """用词与文档不一致时,扩展出文档里的正式说法 —— 这是改写收益的主要来源。"""
    variants = build_rewriter(mode="local").rewrite("成交比例")
    assert any("转化率" in v for v in variants), variants

    variants2 = build_rewriter(mode="local").rewrite("达标率")
    assert any("目标完成率" in v or "完成率" in v for v in variants2), variants2


def test_proper_noun_variant_keeps_models_and_ratios():
    variants = build_rewriter(mode="local", max_variants=4).rewrite("A13 型号准确率 91.5% 达标吗")
    assert any("A13" in v for v in variants), variants
    assert any("91.5" in v for v in variants), variants


def test_bare_number_query_gets_context_window():
    """单个裸数字信息量太低,变体里要带上它周围的字,而不是孤零零一个数字。"""
    variants = build_rewriter(mode="local", max_variants=3).rewrite("响应时限 30")
    assert any(v != "30" and "30" in v for v in variants), variants


def test_config_supplied_synonyms_are_merged():
    rewriter = build_rewriter(mode="local", synonym_spec="客诉=投诉|工单投诉")
    variants = rewriter.rewrite("客诉处理")
    assert any("投诉" in v for v in variants), variants


def test_builtin_synonyms_are_not_lost_when_config_adds_more():
    rewriter = build_rewriter(mode="local", synonym_spec="客诉=投诉")
    assert any("转化率" in v for v in rewriter.rewrite("成交比例"))


def test_bad_synonym_spec_is_ignored_with_warning():
    merged, warnings = parse_synonyms("这条没有等号;=没有key;key=")
    assert len(warnings) == 3
    assert "成交比例" in merged  # 内置词表仍在


# ── 3. 大模型级:解析与降级 ─────────────────────────────────────────


def test_llm_variants_are_parsed_and_original_stays_first(stub_client):
    stub = stub_client(reply="转化率是多少\n- 成交比例的换算口径\n2. 交易转化率指标")
    result = build_rewriter(mode="llm", max_variants=4, llm_client=stub).rewrite_detailed("成交比例")
    assert result.variants[0] == "成交比例"
    assert result.llm_used is True
    assert any("成交比例的换算口径" in v for v in result.llm_variants)
    assert len(result.variants) <= 4


def test_hyde_answer_line_becomes_a_variant(stub_client):
    """轻量 HyDE:模型给出的"可能的答案片段"也要能当检索变体用。"""
    stub = stub_client(reply="答案：平台整体的转化率为 3.2%")
    result = build_rewriter(mode="llm", llm_client=stub).rewrite_detailed("成交比例")
    assert any("转化率为 3.2" in v for v in result.variants), result.variants


def test_llm_error_degrades_to_local_variants(stub_client):
    stub = stub_client(error=RuntimeError("网关超时"))
    result = build_rewriter(mode="llm", llm_client=stub).rewrite_detailed("成交比例")
    assert result.llm_used is False
    assert "网关超时" in result.llm_error
    assert result.degraded is True
    assert any("转化率" in v for v in result.variants), "降级后必须仍有本地变体"
    assert result.variants[0] == "成交比例"


def test_llm_unavailable_client_degrades_to_local(stub_client):
    stub = stub_client(available=False)
    result = build_rewriter(mode="llm", llm_client=stub).rewrite_detailed("成交比例")
    assert result.llm_used is False
    assert result.degraded is True
    assert any("转化率" in v for v in result.variants)


def test_llm_garbage_output_degrades_to_local(stub_client):
    stub = stub_client(reply="   \n  ")
    result = build_rewriter(mode="llm", llm_client=stub).rewrite_detailed("成交比例")
    assert result.llm_used is False
    assert "无法解析" in result.llm_error
    assert any("转化率" in v for v in result.variants)


def test_local_plus_llm_merges_both_sources(stub_client):
    stub = stub_client(reply="平台转化率指标口径")
    result = build_rewriter(mode="local+llm", max_variants=4, llm_client=stub).rewrite_detailed("成交比例")
    assert result.local_variants and result.llm_variants
    assert any("转化率" in v for v in result.local_variants)
    assert any("平台转化率指标口径" in v for v in result.llm_variants)


def test_no_llm_client_does_not_call_anything(stub_client):
    """配置了 llm 模式但没有可用客户端时,不应产生任何大模型调用。"""
    stub = stub_client()
    result = build_rewriter(mode="llm", llm_client=None).rewrite_detailed("成交比例")
    assert stub.calls == []
    assert result.llm_error


# ── 4. 检索链路接入 ─────────────────────────────────────────────────


def _signature(items) -> list[tuple]:
    return [
        (item.chunk.chunk_id, item.rank, round(item.score, 6), round(item.rrf_score, 6),
         None if item.dense_score is None else round(item.dense_score, 6),
         None if item.sparse_score is None else round(item.sparse_score, 6),
         tuple(item.channels))
        for item in items
    ]


def test_rewrite_off_search_is_identical_to_legacy_path(pipeline):
    """★ 向后兼容硬指标:改写关闭时,结果顺序/分数/通道与历史实现逐值一致。

    这里不"看着像"就算过 —— 直接按历史实现(hybrid_search + reranker)手工算一遍再逐项比对。
    """
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    question = "转化率是多少"

    query_vector = pipeline.embedder.encode_one(question, is_query=True)
    dense_hits = [(hit.chunk, hit.score) for hit in pipeline.store.search(query_vector, 10)]
    sparse_hits = pipeline._bm25_index().search(question, 10)
    candidates = hybrid_search(dense_hits, sparse_hits, top_k=10, rrf_k=pipeline.config.rrf_k).fused
    expected = build_reranker("lexical").rerank(question, candidates, 2)

    got = pipeline.search(question, top_n=2, rewrite=False)
    assert _signature(got) == _signature(expected)
    assert all("@" not in channel for item in got for channel in item.channels)


def test_rewrite_off_audit_reports_no_retry(pipeline):
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("转化率是多少", top_n=2, rewrite=False)
    debug = pipeline.last_search_debug
    assert debug["rewrite_mode"] == "off"
    assert debug["variants"] == ["转化率是多少"]
    assert debug["retry"]["triggered"] is False
    assert debug["rewrite"] is None


def test_config_off_and_explicit_false_are_equivalent(pipeline):
    """数据类默认 off:默认路径与显式 rewrite=False 必须完全一致。"""
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    default = pipeline.search("转化率是多少", top_n=2)
    explicit = pipeline.search("转化率是多少", top_n=2, rewrite=False)
    assert _signature(default) == _signature(explicit)


def test_low_score_triggers_retry_and_improves_gate(build_pipeline):
    """★ 低分重试:短查询/异词查询首轮判定分不足 → 用改写变体重试 → 判定分提升。"""
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")

    results = pipeline.search("成交比例", top_n=3)
    debug = pipeline.last_search_debug
    retry = debug["retry"]

    assert retry["triggered"] is True, "首轮判定分应为低分并触发重试"
    assert retry["gate_before"] < pipeline.config.min_score
    assert retry["gate_after"] > retry["gate_before"], "重试后判定分必须真的提升"
    assert retry["improved"] is True
    assert debug["gate_score"] == retry["gate_after"], "审计里的最终判定分要与返回结果一致"
    assert len(debug["variants"]) > 1
    assert any("转化率" in v for v in debug["variants"])
    assert any("转化率" in item.chunk.text for item in results), "重试后应召回含转化率的片段"


def test_low_score_retry_demo_with_colloquial_query(build_pipeline):
    """口语化查询的重试:判定分从远低于阈值提升到阈值之上(与 docs 里的实测数字同源)。"""
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline.add_texts(["# 客服规范\n\n非工作时间实行无人值守，访客留言须在次日跟进处理。\n"], source="客服.md")
    pipeline.search("请问一下夜间要是没人回咋办", top_n=3)
    retry = pipeline.last_search_debug["retry"]
    assert retry["triggered"] is True
    assert retry["gate_before"] < 0.2 <= retry["gate_after"]
    assert retry["improved"] is True


def test_retry_uses_vector_cosine_not_rerank_score(build_pipeline):
    """★ 判定分口径:重试前后比较的必须是向量余弦(grounded_score),不能是重排分。"""
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    results = pipeline.search("成交比例", top_n=3)
    assert pipeline.last_search_debug["gate_score"] == round(pipeline.grounded_score(results), 4)
    assert results[0].dense_score is not None


def test_retry_can_be_switched_off(build_pipeline):
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=False)
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("成交比例", top_n=3)
    debug = pipeline.last_search_debug
    assert debug["retry"]["triggered"] is False
    assert debug["retry"]["enabled"] is False
    assert debug["variants"] == ["成交比例"]


def test_high_score_does_not_trigger_retry(build_pipeline):
    """判定分本来就够时不该多花一次检索(低分重试是异常路径,不是默认路径)。"""
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("转化率是多少", top_n=3)
    debug = pipeline.last_search_debug
    assert debug["retry"]["gate_before"] >= pipeline.config.min_score
    assert debug["retry"]["triggered"] is False
    assert debug["variants"] == ["转化率是多少"]


def test_rewrite_always_uses_variants_from_the_first_attempt(build_pipeline):
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("成交比例", top_n=3, rewrite=True)
    debug = pipeline.last_search_debug
    assert len(debug["variants"]) > 1
    assert debug["retry"]["triggered"] is False
    assert "首轮即改写" in debug["retry"]["reason"]


def test_rewrite_true_overrides_config_off(build_pipeline):
    """配置为 off 时,显式 rewrite=True 仍应做本地改写(CLI 的 --rewrite-always 依赖它)。"""
    pipeline = build_pipeline(query_rewrite="off")
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("成交比例", top_n=3, rewrite=True)
    assert pipeline.last_search_debug["rewrite_mode"] == "local"
    assert len(pipeline.last_search_debug["variants"]) > 1


def test_multi_variant_results_are_deduplicated_and_channel_tagged(build_pipeline):
    """多变体融合:按 chunk_id 去重,通道名带变体来源(可审计),且仍保留 dense/sparse 分数。"""
    pipeline = build_pipeline(query_rewrite="local", query_variants=3)
    pipeline.add_texts([DOMAIN_DOC, TITLE_DOC], source="doc.md")
    results = pipeline.search("成交比例", top_n=3, rewrite=True)
    ids = [item.chunk.chunk_id for item in results]
    assert len(ids) == len(set(ids)), "融合结果不得出现重复 chunk"

    debug = pipeline.last_search_debug
    assert debug["variant_count"] > 1
    assert len(debug["variant_hits"]) == debug["variant_count"]
    multi = [item for item in results if any("@" in ch for ch in item.channels)]
    assert multi, f"多变体路径的通道名应带变体来源,实际:{[item.channels for item in results]}"
    assert multi[0].dense_score is not None, "多变体路径也要保留向量余弦(判定分不能退化)"
    assert any(item.sparse_score is not None for item in multi), "稀疏分也应保留,便于排查"


def test_rewrite_does_not_lose_original_recall(build_pipeline):
    """多召回的候选集合必须是原始查询的超集(改写"只增不减"的实测保证)。"""
    pipeline = build_pipeline(query_rewrite="local", query_variants=3, top_k=10)
    pipeline.add_texts([DOMAIN_DOC, TITLE_DOC], source="doc.md")
    baseline = {item.chunk.chunk_id for item in pipeline.search("成交比例", top_n=10, rewrite=False)}
    rewritten = {item.chunk.chunk_id for item in pipeline.search("成交比例", top_n=10, rewrite=True)}
    assert baseline <= rewritten, f"改写后丢失了原有召回:{baseline - rewritten}"


def test_variants_are_visible_in_debug_and_audit_text(build_pipeline):
    pipeline = build_pipeline(query_rewrite="local")
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("成交比例", top_n=3, rewrite=True)
    audit = format_search_audit(pipeline.last_search_debug)
    assert "查询改写:local" in audit
    assert "变体「" in audit
    assert "低分重试" in audit


def test_audit_reports_state_even_without_variants(pipeline):
    """单变体/关闭改写时,审计只给一行状态(CLI 默认就要能看到),但不引入任何噪声字段。"""
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    pipeline.search("转化率是多少", top_n=2, rewrite=False)
    audit = format_search_audit(pipeline.last_search_debug)
    assert "查询改写:off" in audit
    assert "低分重试:已关闭" in audit
    assert format_search_audit(None) == ""
    assert format_search_audit({}) == ""


def test_empty_store_audit_is_safe(pipeline):
    assert pipeline.search("任意问题") == []
    assert pipeline.last_search_debug["empty_store"] is True
    assert format_search_audit(pipeline.last_search_debug) == ""


def test_query_debug_carries_rewrite_information(build_pipeline):
    pipeline = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    answer = pipeline.query("成交比例")
    assert answer.debug["rewrite_mode"] == "local"
    assert answer.debug["retry_triggered"] is True
    assert answer.debug["variants"][0] == "成交比例"
    assert "gate_before" in answer.debug


def test_stats_reports_rewrite_configuration(build_pipeline):
    pipeline = build_pipeline(query_rewrite="local+llm", query_variants=4, retry_on_low_score=False)
    stats = pipeline.stats()
    assert stats["query_rewrite"] == "local+llm"
    assert stats["query_variants"] == 4
    assert stats["retry_on_low_score"] is False


# ── 5. Agent 工具层可见性 ───────────────────────────────────────────


def test_search_tool_appends_audit_when_rewrite_is_active(build_pipeline, monkeypatch):
    from app.tools import rag_tools

    pipeline_obj = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline_obj.add_texts([DOMAIN_DOC], source="doc.md")
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline_obj)

    out = rag_tools.search_knowledge_base.invoke({"question": "成交比例", "kb_name": "test"})
    assert "【检索过程】" in out
    assert "查询改写:local" in out
    assert "低分重试:已触发" in out


def test_search_tool_has_no_audit_block_when_rewrite_is_off(pipeline, monkeypatch):
    from app.tools import rag_tools

    pipeline.add_texts([DOMAIN_DOC], source="doc.md")
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline)

    out = rag_tools.search_knowledge_base.invoke({"question": "转化率是多少", "kb_name": "test"})
    assert "【检索过程】" not in out


def test_ask_tool_appends_audit_when_rewrite_is_active(build_pipeline, monkeypatch):
    from app.tools import rag_tools

    pipeline_obj = build_pipeline(query_rewrite="local", retry_on_low_score=True)
    pipeline_obj.add_texts([DOMAIN_DOC], source="doc.md")
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *a, **kw: pipeline_obj)

    out = rag_tools.ask_knowledge_base.invoke({"question": "成交比例", "kb_name": "test"})
    assert "【检索过程】" in out


# ── 6. CLI 可见性 ───────────────────────────────────────────────────


def test_cli_search_prints_variants_and_retry_state(capsys, monkeypatch, rag_data_dir, hash_embedder):
    """默认输出就要能看到"用了哪些变体 / 是否重试",不能藏在 --json 或 debug 开关后面。"""
    import app.core.config as core_config
    import app.rag.cli as cli
    import app.rag.config as rag_config
    from app.rag.config import RagConfig
    from app.rag.pipeline import RagPipeline, clear_pipeline_cache

    cfg = RagConfig(
        data_dir=rag_data_dir,
        default_kb="clikb",
        embedding_provider="hash",
        min_score=0.2,
        query_rewrite="off",
    )
    seed = RagPipeline(kb="clikb", config=cfg, embedder=hash_embedder)
    seed.add_texts([DOMAIN_DOC], source="doc.md")
    clear_pipeline_cache()

    monkeypatch.setenv("RAG_DATA_DIR", str(rag_data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)

    # ① 关闭改写:输出里应明确写"已关闭"
    code = cli.main(
        ["search", "-k", "clikb", "--embedding-provider", "hash", "-q", "成交比例", "-n", "1", "--no-rewrite"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "查询改写:off" in out
    assert "低分重试:已关闭" in out

    # ② 始终改写:应能看到变体清单、各变体命中条数与判定分
    clear_pipeline_cache()
    code = cli.main(
        ["search", "-k", "clikb", "--embedding-provider", "hash", "-q", "成交比例", "-n", "1",
         "--rewrite-mode", "local", "--rewrite-always"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "查询改写:local" in out
    assert "变体「" in out
    assert "向量命中" in out and "关键词命中" in out
    assert "首轮即改写" in out


def test_cli_search_json_includes_rewrite_and_retry(capsys, monkeypatch, rag_data_dir, hash_embedder):
    import json

    import app.core.config as core_config
    import app.rag.cli as cli
    import app.rag.config as rag_config
    from app.rag.config import RagConfig
    from app.rag.pipeline import RagPipeline, clear_pipeline_cache

    cfg = RagConfig(
        data_dir=rag_data_dir,
        default_kb="clikb2",
        embedding_provider="hash",
        min_score=0.2,
        query_rewrite="off",
    )
    seed = RagPipeline(kb="clikb2", config=cfg, embedder=hash_embedder)
    seed.add_texts([DOMAIN_DOC], source="doc.md")
    clear_pipeline_cache()

    monkeypatch.setenv("RAG_DATA_DIR", str(rag_data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)

    code = cli.main(
        ["search", "-k", "clikb2", "--embedding-provider", "hash", "-q", "成交比例", "-n", "1",
         "--rewrite-mode", "local", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["variants"][0] == "成交比例"
    assert payload["retry"]["triggered"] is True
    assert payload["retry"]["improved"] is True
    assert payload["variant_hits"][0]["variant"] == "成交比例"
    assert payload["results"][0]["channels"]


def test_cli_eval_reports_rewrite_stats(capsys, monkeypatch, rag_data_dir, tmp_path, hash_embedder):
    import json

    import app.core.config as core_config
    import app.rag.cli as cli
    import app.rag.config as rag_config
    from app.rag.config import RagConfig
    from app.rag.pipeline import RagPipeline, clear_pipeline_cache

    cfg = RagConfig(
        data_dir=rag_data_dir,
        default_kb="clikb3",
        embedding_provider="hash",
        min_score=0.2,
        query_rewrite="off",
    )
    seed = RagPipeline(kb="clikb3", config=cfg, embedder=hash_embedder)
    seed.add_texts([DOMAIN_DOC], source="doc.md")
    clear_pipeline_cache()

    eval_set = tmp_path / "eval.json"
    eval_set.write_text(
        json.dumps(
            {
                "items": [
                    {"question": "成交比例", "expected_sources": ["doc.md"]},
                    {"question": "转化率是多少", "expected_sources": ["doc.md"]},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_DATA_DIR", str(rag_data_dir))
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)

    code = cli.main(
        ["eval", "-k", "clikb3", "--embedding-provider", "hash", "-e", str(eval_set), "-n", "2",
         "--rewrite-mode", "local", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["rewrite_mode"] == "local"
    assert payload["rewrite_stats"]["retry_triggered"] >= 1
    assert payload["rewrite_stats"]["multi_variant_queries"] >= 1
    assert payload["per_query"][0]["variants"]


def test_get_pipeline_still_works_with_rewrite_config(tmp_path, hash_embedder):
    """改写配置不应影响流水线缓存键的既有语义(同 kb 仍复用同一实例)。"""
    import app.core.config as core_config
    import app.rag.config as rag_config

    (tmp_path / "_rag_data4").mkdir(parents=True, exist_ok=True)
    monkeypatch_env = None
    try:
        import os

        monkeypatch_env = os.environ.get("RAG_DATA_DIR")
        os.environ["RAG_DATA_DIR"] = str(tmp_path / "_rag_data4")
        core_config.get_settings.cache_clear()
        rag_config.settings = core_config.get_settings()
        from app.rag.pipeline import clear_pipeline_cache

        clear_pipeline_cache()
        first = get_pipeline(kb="cachekb")
        second = get_pipeline(kb="cachekb")
        assert first is second
        assert first.config.query_rewrite == "local"  # 应用级默认开启
        clear_pipeline_cache()
    finally:
        import os

        if monkeypatch_env is None:
            os.environ.pop("RAG_DATA_DIR", None)
        else:
            os.environ["RAG_DATA_DIR"] = monkeypatch_env
        core_config.get_settings.cache_clear()
