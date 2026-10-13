"""app/tools/rag_tools.py：三个 Agent 工具的契约。

这是 Agent 与自研 RAG 之间的**唯一接口层**，所以契约要单独锁死：

* 三个工具都是 `@tool` 装饰出来的 **StructuredTool**：直接 `tool(args)` 会
  `TypeError: 'StructuredTool' object is not callable`，测试与调用方都必须用 `.invoke({...})`。
* **所有错误分支都返回中文字符串，绝不抛异常** —— 子智能体拿到的是可读提示，
  而不是让整条 Agent 链路崩掉。这是本文件的主要断言对象。

另锁一条刚修好的缺陷（audit 的 A3）：
`search_knowledge_base` 原先用 `results[0].score`（排序/重排分）与 `min_score`（按向量余弦标定）
比较，属于 docs/rag-self-built.md 5.10 那类"排序分与判定分混用"。现已改为
`pipeline.grounded_score(results)`。见 `test_search_tool_gate_uses_vector_cosine`。
"""

from __future__ import annotations

import pytest

from app.rag.config import load_rag_config
from app.rag.vectorstore import EmbeddingMismatchError
from app.tools import rag_tools

KB_DOC = (
    "# 风控\n\n"
    "风控模型的准确率为 91.5%。\n\n"
    "## 授信\n\n"
    "授信规则要求 A13 型号必须复核，编号 3.2.1。"
)
RELEVANT_Q = "风控模型的准确率是多少"
IRRELEVANT_Q = "公司食堂本周的菜单"


# ── 工具形态 ────────────────────────────────────────────────────────


def test_tools_are_langchain_structured_tools():
    from langchain_core.tools import BaseTool

    for tool in (rag_tools.list_knowledge_bases, rag_tools.search_knowledge_base, rag_tools.ask_knowledge_base):
        assert isinstance(tool, BaseTool)
        assert type(tool).__name__ == "StructuredTool"


def test_tool_names_are_stable():
    assert rag_tools.list_knowledge_bases.name == "list_knowledge_bases"
    assert rag_tools.search_knowledge_base.name == "search_knowledge_base"
    assert rag_tools.ask_knowledge_base.name == "ask_knowledge_base"


def test_tools_expose_docstrings_for_the_model():
    """工具描述是模型决定"要不要调、怎么传参"的唯一依据，不能为空。"""
    for tool in (rag_tools.list_knowledge_bases, rag_tools.search_knowledge_base, rag_tools.ask_knowledge_base):
        assert tool.description and len(tool.description.strip()) > 10


def test_search_tool_argument_schema():
    properties = rag_tools.search_knowledge_base.args_schema.model_json_schema()["properties"]
    assert {"question", "kb_name", "top_n"} <= set(properties)


def test_ask_tool_argument_schema():
    properties = rag_tools.ask_knowledge_base.args_schema.model_json_schema()["properties"]
    assert {"question", "kb_name"} <= set(properties)


def test_tools_are_not_directly_callable():
    """历史/审计要点：StructuredTool 必须走 .invoke()，直接调用会 TypeError。"""
    with pytest.raises(TypeError):
        rag_tools.list_knowledge_bases()  # type: ignore[operator]


# ── list_knowledge_bases ────────────────────────────────────────────


def test_list_returns_chinese_hint_when_no_knowledge_base():
    out = rag_tools.list_knowledge_bases.invoke({})
    assert isinstance(out, str)
    assert "当前没有任何知识库" in out
    assert "python -m app.rag.cli ingest" in out  # 给出可执行的补救命令


def test_list_reports_knowledge_base_and_chunk_count(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    out = rag_tools.list_knowledge_bases.invoke({})
    assert "可用知识库" in out
    assert "test" in out
    assert "个片段" in out
    assert f"默认知识库:{load_rag_config().default_kb}" in out


def test_list_marks_empty_knowledge_base(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    # 再开一个空库
    from app.rag.pipeline import get_pipeline

    get_pipeline(kb="emptykb")
    out = rag_tools.list_knowledge_bases.invoke({})
    assert "空库，暂不可检索" in out


def test_list_never_raises_on_broken_config(monkeypatch):
    """任何异常都必须被收敛成中文提示，而不是冒泡给 Agent。"""

    def boom(*args, **kwargs):
        raise RuntimeError("配置炸了")

    monkeypatch.setattr(rag_tools, "load_rag_config", boom)
    out = rag_tools.list_knowledge_bases.invoke({})
    assert "查询知识库列表异常" in out


# ── search_knowledge_base ───────────────────────────────────────────


def test_search_returns_fragments_with_source_and_score(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    out = rag_tools.search_knowledge_base.invoke({"question": RELEVANT_Q, "kb_name": "test", "top_n": 3})
    assert "检索到" in out
    assert "【片段 1】" in out
    assert "匹配度" in out
    assert "来源:" in out


def test_search_emits_structured_sources_for_frontend(pipeline):
    from app.api.monitor import monitor

    pipeline.add_texts([KB_DOC], source="doc.md")
    events = []
    token = monitor.bind_event_collector(events)
    try:
        rag_tools.search_knowledge_base.invoke(
            {"question": RELEVANT_Q, "kb_name": "test", "top_n": 2}
        )
    finally:
        monitor.reset_event_collector(token)

    source_event = next(event for event in events if event["event"] == "rag_sources")
    assert source_event["data"]["kb"] == "test"
    assert source_event["data"]["sources"][0]["source"] == "doc.md"
    assert source_event["data"]["sources"][0]["text"]


def test_search_on_existing_but_empty_kb(pipeline):
    from app.rag.pipeline import get_pipeline

    get_pipeline(kb="emptykb_search")  # 目录存在，但没有任何 chunk
    out = rag_tools.search_knowledge_base.invoke({"question": "任意", "kb_name": "emptykb_search"})
    assert "是空的" in out
    assert "没有可检索的文档" in out


def test_search_tool_gate_uses_vector_cosine(pipeline, make_chunk, monkeypatch):
    """★ A3 回归锁：门槛必须与 `pipeline.grounded_score()`（向量余弦）同口径。

    刻意造一个「排序分 0.95 越过阈值、向量余弦 0.01 远低于阈值」的候选：
    若实现退回 `results[0].score`，工具会把无关片段当成依据返回；
    正确实现必须报"资料不足"，且提示里的分数应是 **0.010（余弦）**。
    """
    from app.rag.types import RetrievedChunk

    pipeline.add_texts([KB_DOC], source="doc.md")
    crafted = [
        RetrievedChunk(
            chunk=make_chunk(text="无关内容"), score=0.95, dense_score=0.01, rank=1, channels=["dense"]
        )
    ]
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: crafted)
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *args, **kwargs: pipeline)

    out = rag_tools.search_knowledge_base.invoke({"question": "任意问题", "kb_name": "test"})
    assert "没有与该问题足够相关的资料" in out
    assert "0.010" in out, "门槛判定的分数必须是向量余弦 0.01，而不是排序分 0.95"
    assert "0.950" not in out


def test_search_tool_returns_fragments_when_cosine_is_high(pipeline, make_chunk, monkeypatch):
    """反向对照：余弦高时必须正常返回片段（防止上一条因为恒拒答而假通过）。"""
    from app.rag.types import RetrievedChunk

    pipeline.add_texts([KB_DOC], source="doc.md")
    crafted = [
        RetrievedChunk(
            chunk=make_chunk(text="风控准确率为 91.5%。"),
            score=0.9,
            dense_score=0.9,
            rank=1,
            channels=["dense"],
        )
    ]
    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: crafted)
    monkeypatch.setattr(rag_tools, "get_pipeline", lambda *args, **kwargs: pipeline)

    out = rag_tools.search_knowledge_base.invoke({"question": "风控准确率", "kb_name": "test"})
    assert "【片段 1】" in out
    assert "匹配度:0.900" in out


def test_search_never_raises_on_embedding_mismatch(monkeypatch):
    def boom(*args, **kwargs):
        raise EmbeddingMismatchError("向量由 [hash:512] 生成，当前配置为 [hash:64]")

    monkeypatch.setattr(rag_tools, "get_pipeline", boom)
    out = rag_tools.search_knowledge_base.invoke({"question": "任意", "kb_name": "test"})
    assert "向量配置不匹配" in out
    assert "需要重新入库" in out


def test_search_never_raises_on_unexpected_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("索引文件损坏")

    monkeypatch.setattr(rag_tools, "get_pipeline", boom)
    out = rag_tools.search_knowledge_base.invoke({"question": "任意", "kb_name": "test"})
    assert "知识库检索失败" in out
    assert "索引文件损坏" in out


# ── ask_knowledge_base ──────────────────────────────────────────────


def test_ask_returns_answer_plus_citations(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    out = rag_tools.ask_knowledge_base.invoke({"question": RELEVANT_Q, "kb_name": "test"})
    assert isinstance(out, str)
    # 本机没有配置大模型 → 走降级分支：明确提示 + 仍给出引用
    assert "大模型调用不可用" in out
    assert "引用来源" in out
    assert "[1]" in out


def test_ask_on_existing_but_empty_kb(pipeline):
    from app.rag.pipeline import get_pipeline

    get_pipeline(kb="emptykb_ask")
    out = rag_tools.ask_knowledge_base.invoke({"question": "任意", "kb_name": "emptykb_ask"})
    assert "是空的" in out


def test_ask_refuses_irrelevant_question_without_llm(pipeline):
    """无依据时必须明说"知识库中没有"，不能编造。"""
    pipeline.add_texts([KB_DOC], source="doc.md")
    out = rag_tools.ask_knowledge_base.invoke({"question": IRRELEVANT_Q, "kb_name": "test"})
    assert "知识库中没有找到" in out


def test_ask_never_raises_on_embedding_mismatch(monkeypatch):
    def boom(*args, **kwargs):
        raise EmbeddingMismatchError("不匹配")

    monkeypatch.setattr(rag_tools, "get_pipeline", boom)
    out = rag_tools.ask_knowledge_base.invoke({"question": "任意", "kb_name": "test"})
    assert "向量配置不匹配" in out


def test_ask_never_raises_on_unexpected_error(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("生成层炸了")

    monkeypatch.setattr(rag_tools, "get_pipeline", boom)
    out = rag_tools.ask_knowledge_base.invoke({"question": "任意", "kb_name": "test"})
    assert "知识库问答失败" in out


# ── A4：不存在的 kb 不得凭空创建目录（captain 已修，这里锁死新行为）──


def _kb_dirs() -> set[str]:
    data_path = load_rag_config().data_path
    if not data_path.exists():
        return set()
    return {p.name for p in data_path.iterdir() if p.is_dir()}


def test_search_on_missing_kb_reports_and_creates_nothing(pipeline):
    """★ A4 回归锁：库不存在时必须明确报"不存在"，且**不得在磁盘上留下任何痕迹**。

    早期实现用 `get_pipeline(create=True)`：模型一旦传了幻觉库名（或用户口误），
    就会在数据目录下建出一个空目录，之后所有查询都返回「库是空的」，
    把「库不存在」这个真实原因掩盖掉，还留下垃圾目录。
    """
    pipeline.add_texts([KB_DOC], source="doc.md")  # 让 "test" 成为可用库
    before = _kb_dirs()

    out = rag_tools.search_knowledge_base.invoke(
        {"question": "随便问问", "kb_name": "definitely_not_exist_kb"}
    )

    assert "不存在" in out
    assert "未创建也不会自动创建" in out
    assert "definitely_not_exist_kb" in out
    assert "test" in out, "提示里应列出可用知识库，帮助模型自我纠正"
    assert _kb_dirs() == before, f"不应新建任何知识库目录，新增了 {_kb_dirs() - before}"
    assert not (load_rag_config().data_path / "definitely_not_exist_kb").exists()


def test_ask_on_missing_kb_reports_and_creates_nothing(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    before = _kb_dirs()

    out = rag_tools.ask_knowledge_base.invoke({"question": "随便问问", "kb_name": "ghost_kb_ask"})

    assert "不存在" in out
    assert _kb_dirs() == before
    assert not (load_rag_config().data_path / "ghost_kb_ask").exists()


def test_missing_kb_hint_lists_available_knowledge_bases(pipeline):
    pipeline.add_texts([KB_DOC], source="doc.md")
    out = rag_tools.search_knowledge_base.invoke({"question": "x", "kb_name": "nope"})
    assert "当前可用的知识库" in out
    assert "test" in out


def test_get_pipeline_create_false_raises_instead_of_creating(tmp_path):
    """底层契约：create=False 时打开不存在的库必须抛 FileNotFoundError，目录不得出现。"""
    from app.rag.config import RagConfig
    from app.rag.pipeline import RagPipeline

    cfg = RagConfig(data_dir=tmp_path / "_rag_data", default_kb="default", embedding_provider="hash")
    with pytest.raises(FileNotFoundError):
        RagPipeline(kb="nope", config=cfg, create=False)
    assert not cfg.kb_dir("nope").exists()


# ── 离线性守门 ──────────────────────────────────────────────────────


def test_suite_runs_offline_without_llm_credentials():
    """守门用例：整套测试必须在不配置大模型的前提下跑通。

    conftest 会清空 OPENAI_API_KEY。若这条失败，说明有用例会真的联网调用大模型
    —— 那既不确定、又要花钱，还违背「离线可复现」的验收要求。
    """
    from app.rag.generator import ChatClient

    cfg = load_rag_config()
    assert cfg.llm_api_key == "", "测试环境里不得配置大模型 key"
    client = ChatClient(model=cfg.llm_model, api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
    assert client.available is False
