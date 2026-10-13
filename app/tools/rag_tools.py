"""
自研 RAG 知识库工具模块

替代原 app/tools/ragflow_tools.py:原来是把问题转发给外部 RAGFlow 服务,现在直接调用
本项目的 app.rag 流水线,检索过程完全在本地可控。

给子智能体暴露三个工具,刻意把「检索」和「生成」拆开:
1. list_knowledge_bases —— 先看清有哪些库、各自多少文档,避免向空库提问;
2. search_knowledge_base —— 只做检索,返回带分数的原文片段。复杂问题可以让模型多轮换角度检索,
   自己判断信息够不够(Agentic RAG 的基础),而且检索结果可解释、可引用;
3. ask_knowledge_base —— 一步到位的检索 + 生成带引用答案,适合简单事实型问题,省一轮交互。

为什么不像 RAGFlow 那样「创建会话再提问」:自研检索是无状态的纯函数调用,每次检索独立、
不存在跨轮上下文污染,所以不需要会话语义,也不需要「用完即删」的资源清理。

只读语义:查询类工具一律以 create=False 打开知识库。用户或模型传了不存在的库名时,
必须报「库不存在」并列出可用库,而不是悄悄建出一个空库(见 _open_pipeline_for_query)。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.api.monitor import monitor
from app.rag.config import load_rag_config
from app.rag.metadata import (
    InvalidFilterError,
    format_filter_hint,
    normalize_filter,
    parse_filter_json,
)
from app.rag.pipeline import RagPipeline, format_search_audit, get_pipeline
from app.rag.vectorstore import EmbeddingMismatchError, list_knowledge_bases as _list_kbs


def _with_audit(text: str, pipeline: RagPipeline) -> str:
    """把本次检索的改写/重试审计信息追加到工具返回文本后。

    为什么要带出来:改写一旦生效,"召回变好/变差"和"改写做了什么"必须能对上账,
    否则调参只能靠猜。只在**真的发生了改写或重试**时追加,避免每轮都往模型上下文里
    塞一行无关的状态(工具返回值是给大模型看的,噪声会挤占预算)。
    """
    debug = getattr(pipeline, "last_search_debug", None) or {}
    variant_count = debug.get("variant_count", len(debug.get("variants") or []))
    retry_triggered = bool((debug.get("retry") or {}).get("triggered"))
    llm_error = (debug.get("rewrite") or {}).get("llm_error")
    if variant_count <= 1 and not retry_triggered and not llm_error:
        return text
    audit = format_search_audit(debug)
    if not audit:
        return text
    return f"{text}\n\n【检索过程】\n{audit}"


def _open_pipeline_for_query(kb_name: str) -> RagPipeline:
    """只读地打开知识库:库不存在时抛 FileNotFoundError,而不是静默新建空库。

    背景:早期实现用 get_pipeline(create=True),模型一旦传了不存在的库名(幻觉出的库名、
    或用户口误),就会在 rag_data/ 下建出一个空目录,之后所有查询都返回「库是空的」,
    把「库不存在」这个真实原因掩盖掉,还留下垃圾目录。这里显式关闭创建。
    """
    return get_pipeline(kb=kb_name or None, create=False)


def _missing_kb_hint(kb_name: str) -> str:
    """库不存在时给出可用库清单,帮助模型自我纠正。"""
    try:
        bases = _list_kbs(load_rag_config().data_path)
        names = [item["name"] for item in bases if item["chunk_count"] > 0]
    except Exception:
        names = []
    available = "、".join(names) if names else "(当前没有任何已入库的知识库)"
    return (
        f"知识库 {kb_name!r} 不存在,未创建也不会自动创建。"
        f"当前可用的知识库:{available}。"
        f"请改用已有知识库名称,或先由管理员执行入库命令。"
    )


@tool
def list_knowledge_bases() -> str:
    """
    列出当前可用的知识库,以及每个知识库的文档规模与向量配置

    作用:让模型先确认「有哪些内部资料可以查」,再决定调用 search_knowledge_base 还是
    ask_knowledge_base,以及是否需要换关键词多轮检索。
    :return: 知识库名称、chunk 数量、更新时间;没有知识库或异常时返回中文提示
    """
    monitor.report_tool(tool_name="自研RAG知识库列表查询工具：list_knowledge_bases")

    try:
        config = load_rag_config()
        bases = _list_kbs(config.data_path)
        if not bases:
            return (
                "当前没有任何知识库。请先由管理员执行入库命令:"
                "python -m app.rag.cli ingest -k default -p <文档目录>"
            )
        lines = ["可用知识库："]
        for item in bases:
            state = "（空库，暂不可检索）" if item["chunk_count"] == 0 else ""
            fields = item.get("filter_fields") or []
            fields_text = "、".join(fields) if fields else "无"
            lines.append(
                f"- {item['name']}: {item['chunk_count']} 个片段"
                f" | 向量模型 {item['embedding'] or '未初始化'}"
                f" | 更新时间 {item['updated_at'] or '-'}"
                f" | 可过滤字段 {fields_text}{state}"
            )
        lines.append(f"\n默认知识库:{config.default_kb}")
        lines.append(
            "提示:过滤字段来自文档头部 YAML frontmatter(如 部门/类型/密级/版本)与系统字段"
            "(file/path/suffix);调用检索/问答工具时可用 filter_json 传 JSON(如 {\"部门\": \"风控\"})。"
        )
        return "\n".join(lines)
    except Exception as exc:
        return f"查询知识库列表异常:{exc}"


def _parse_tool_filter(filter_json: str) -> tuple[dict, str]:
    """解析工具层的 filter_json;返回 (filter, 错误提示)。非法输入不抛异常,交给模型自我纠正。"""
    if not filter_json:
        return {}, ""
    try:
        return parse_filter_json(filter_json), ""
    except InvalidFilterError as exc:
        return {}, f"过滤条件不合法:{exc}"


def _filter_miss_message(kb: str, filter_spec: dict, pipeline) -> str:
    """过滤把候选清空时的专用提示:必须指向"过滤条件",而不是笼统的"没检索到"。"""
    try:
        hint = format_filter_hint(normalize_filter(filter_spec), pipeline.filter_fields)
    except Exception:  # noqa: BLE001 - 提示本身不能再抛异常
        hint = f"没有符合过滤条件 {filter_spec} 的片段。"
    return f"知识库 {kb} 中{hint}"


def _emit_search_sources(question: str, pipeline: RagPipeline, results: list, gate_score: float) -> None:
    """把检索原文作为结构化事件交给前端，不依赖模型转述。"""
    monitor.report_rag_sources(
        {
            "kb": pipeline.kb,
            "question": question,
            "grounded": gate_score >= pipeline.config.min_score,
            "gate_score": round(gate_score, 4),
            "threshold": pipeline.config.min_score,
            "sources": [
                {
                    "rank": item.rank,
                    "source": item.chunk.source,
                    "source_path": item.chunk.source_path,
                    "chunk_index": item.chunk.chunk_index,
                    "heading": item.chunk.heading,
                    "score": round(item.score, 4),
                    "dense_score": round(item.dense_score, 4) if item.dense_score is not None else None,
                    "text": item.chunk.text[:1200],
                }
                for item in results
            ],
        }
    )


def _emit_answer_sources(question: str, pipeline: RagPipeline, answer) -> None:
    """上报实际送入答案生成器的上下文片段。"""
    gate_score = float(answer.debug.get("gate_score", 0.0) or 0.0)
    monitor.report_rag_sources(
        {
            "kb": pipeline.kb,
            "question": question,
            "grounded": bool(answer.citations) and gate_score >= pipeline.config.min_score,
            "gate_score": round(gate_score, 4),
            "threshold": pipeline.config.min_score,
            "sources": [
                {
                    "rank": context.get("index", index),
                    "source": context.get("source", ""),
                    "source_path": "",
                    "chunk_index": context.get("chunk_index", 0),
                    "heading": context.get("heading", ""),
                    "score": context.get("score", 0),
                    "dense_score": None,
                    "text": str(context.get("text", ""))[:1200],
                }
                for index, context in enumerate(answer.contexts, start=1)
            ],
        }
    )


@tool
def search_knowledge_base(
    question: str, kb_name: str = "", top_n: int = 4, filter_json: str = ""
) -> str:
    """
    在企业内部知识库中做混合检索,返回最相关的原文片段(不做总结、不生成答案)

    适用场景:需要自己判断信息是否充分、需要多角度反复检索、或需要引用原文精确数字时。
    :param question: 检索问题或关键词,尽量包含专有名词(型号、编号、指标名)以提升命中率
    :param kb_name: 知识库名称,留空则使用默认知识库
    :param top_n: 返回片段数量,默认 4
    :param filter_json: 可选的元数据过滤,JSON 字符串,例如 {"部门": "风控"} 或 {"类型": ["制度", "规范"]}。
        语义是"白名单":元数据缺该字段或取值不匹配的片段会被剔除;值为列表时命中其一即可。
        可用字段先用 list_knowledge_bases 查看(每行末的"可过滤字段"),常见字段来自文档头部的
        YAML frontmatter(如 部门/类型/密级/版本)以及系统字段 file/path/suffix。留空表示不过滤。
    :return: 带来源编号和相似度的原文片段;检索不到、库不存在或过滤后为空时返回中文提示
    """
    monitor.report_tool(
        tool_name="自研RAG知识库检索工具：search_knowledge_base",
        args={"question": question, "kb_name": kb_name, "top_n": top_n, "filter": filter_json},
    )

    filter_spec, filter_error = _parse_tool_filter(filter_json)
    if filter_error:
        return filter_error

    try:
        pipeline = _open_pipeline_for_query(kb_name)
        if not len(pipeline.store):
            return f"知识库 {pipeline.kb} 是空的,没有可检索的文档。"

        results = pipeline.search(
            question, top_n=max(1, min(int(top_n), 10)), filter=filter_spec
        )
        if not results:
            if filter_spec:
                return _filter_miss_message(pipeline.kb, filter_spec, pipeline)
            return "知识库中没有检索到相关内容。"

        # 门槛判定必须用 grounded_score()(向量余弦),而不是 results[0].score:
        # 后者在开启重排后是「词面覆盖率/大模型打分」,尺度与 min_score 完全不同,
        # 混用会让无关问题越过阈值(同类问题见 docs/rag-self-built.md 5.10)。
        gate_score = pipeline.grounded_score(results)
        _emit_search_sources(question, pipeline, results, gate_score)
        if gate_score < pipeline.config.min_score:
            return (
                f"知识库 {pipeline.kb} 中没有与该问题足够相关的资料"
                f"(最高匹配度 {gate_score:.3f},低于阈值 {pipeline.config.min_score})。"
            )

        blocks = [
            f"【片段 {item.rank}】来源:{item.chunk.source}#{item.chunk.chunk_index}"
            f" 标题:{item.chunk.heading or '-'} 匹配度:{item.score:.3f}\n{item.chunk.text}"
            for item in results
        ]
        out = f"在知识库 {pipeline.kb} 中检索到 {len(results)} 个相关片段:\n\n" + "\n\n".join(blocks)
        # 改写/重试的审计信息随结果一起返回:模型与人都能看出"这次检索经历了什么"。
        # 只在真的有变体/重试时追加,保持既有输出格式与断言不被破坏。
        return _with_audit(out, pipeline)
    except FileNotFoundError:
        return _missing_kb_hint(kb_name or load_rag_config().default_kb)
    except EmbeddingMismatchError as exc:
        return f"知识库向量配置不匹配,需要重新入库:{exc}"
    except Exception as exc:
        return f"知识库检索失败:{exc}"


@tool
def ask_knowledge_base(question: str, kb_name: str = "", filter_json: str = "") -> str:
    """
    向企业内部知识库提问,直接返回带引用来源的答案(内部完成检索 + 大模型生成)

    适用场景:明确的事实型问题,一次提问即可得到结论。
    复杂问题建议改用 search_knowledge_base 分多轮检索,信息更可控。
    :param question: 用户问题
    :param kb_name: 知识库名称,留空则使用默认知识库
    :param filter_json: 可选的元数据过滤,JSON 字符串,例如 {"部门": "风控"} 或 {"密级": ["公开", "内部"]}。
        语义是"白名单":元数据缺该字段或取值不匹配的片段会被剔除;值为列表时命中其一即可。
        可用字段见 list_knowledge_bases 返回的"可过滤字段"(多来自文档头部 YAML frontmatter)。
    :return: 答案 + 引用列表;无依据时明确说明知识库中没有相关资料
    """
    monitor.report_tool(
        tool_name="自研RAG知识库问答工具：ask_knowledge_base",
        args={"question": question, "kb_name": kb_name, "filter": filter_json},
    )

    filter_spec, filter_error = _parse_tool_filter(filter_json)
    if filter_error:
        return filter_error

    try:
        pipeline = _open_pipeline_for_query(kb_name)
        if not len(pipeline.store):
            return f"知识库 {pipeline.kb} 是空的,没有可检索的文档。"

        answer = pipeline.query(question, filter=filter_spec)
        _emit_answer_sources(question, pipeline, answer)
        if filter_spec and not answer.citations:
            return _filter_miss_message(pipeline.kb, filter_spec, pipeline)
        lines = [answer.answer]
        if answer.citations:
            lines.append("")
            lines.append("引用来源：")
            for cite in answer.citations:
                lines.append(
                    f"[{cite['index']}] {cite['source']}#{cite['chunk_index']}"
                    f"（标题:{cite['heading'] or '-'},匹配度:{cite['score']}）"
                )
        if not answer.grounded:
            lines.append(f"\n（检索调试信息:{json.dumps(answer.debug, ensure_ascii=False)}）")
        # 与 search 工具一致:改写/重试过程默认可见(不可观测的机制无法调优)
        return _with_audit("\n".join(lines), pipeline)
    except FileNotFoundError:
        return _missing_kb_hint(kb_name or load_rag_config().default_kb)
    except EmbeddingMismatchError as exc:
        return f"知识库向量配置不匹配,需要重新入库:{exc}"
    except Exception as exc:
        return f"知识库问答失败:{exc}"


__all__ = ["list_knowledge_bases", "search_knowledge_base", "ask_knowledge_base"]
