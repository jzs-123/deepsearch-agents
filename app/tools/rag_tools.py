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
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from app.api.monitor import monitor
from app.rag.config import load_rag_config
from app.rag.pipeline import get_pipeline
from app.rag.vectorstore import EmbeddingMismatchError, list_knowledge_bases as _list_kbs


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
            lines.append(
                f"- {item['name']}: {item['chunk_count']} 个片段"
                f" | 向量模型 {item['embedding'] or '未初始化'}"
                f" | 更新时间 {item['updated_at'] or '-'}{state}"
            )
        lines.append(f"\n默认知识库:{config.default_kb}")
        return "\n".join(lines)
    except Exception as exc:
        return f"查询知识库列表异常:{exc}"


@tool
def search_knowledge_base(question: str, kb_name: str = "", top_n: int = 4) -> str:
    """
    在企业内部知识库中做混合检索,返回最相关的原文片段(不做总结、不生成答案)

    适用场景:需要自己判断信息是否充分、需要多角度反复检索、或需要引用原文精确数字时。
    :param question: 检索问题或关键词,尽量包含专有名词(型号、编号、指标名)以提升命中率
    :param kb_name: 知识库名称,留空则使用默认知识库
    :param top_n: 返回片段数量,默认 4
    :return: 带来源编号和相似度的原文片段;检索不到时返回中文提示
    """
    monitor.report_tool(
        tool_name="自研RAG知识库检索工具：search_knowledge_base",
        args={"question": question, "kb_name": kb_name, "top_n": top_n},
    )

    try:
        pipeline = get_pipeline(kb=kb_name or None)
        if not len(pipeline.store):
            return f"知识库 {pipeline.kb} 是空的,没有可检索的文档。"
        results = pipeline.search(question, top_n=max(1, min(int(top_n), 10)))
        if not results:
            return "知识库中没有检索到相关内容。"
        # 低于阈值的结果不作为依据返回,避免把无关内容塞给模型造成幻觉。
        # 注意必须用 grounded_score()(向量余弦)而不是 results[0].score:
        # 后者在开启重排后是「词面覆盖率/大模型打分」,尺度与 min_score 完全不同,
        # 混用会让无关问题越过阈值(同类问题见 docs/rag-self-built.md 5.10)。
        gate_score = pipeline.grounded_score(results)
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
        return f"在知识库 {pipeline.kb} 中检索到 {len(results)} 个相关片段:\n\n" + "\n\n".join(blocks)
    except EmbeddingMismatchError as exc:
        return f"知识库向量配置不匹配,需要重新入库:{exc}"
    except Exception as exc:
        return f"知识库检索失败:{exc}"


@tool
def ask_knowledge_base(question: str, kb_name: str = "") -> str:
    """
    向企业内部知识库提问,直接返回带引用来源的答案(内部完成检索 + 大模型生成)

    适用场景:明确的事实型问题,一次提问即可得到结论。
    复杂问题建议改用 search_knowledge_base 分多轮检索,信息更可控。
    :param question: 用户问题
    :param kb_name: 知识库名称,留空则使用默认知识库
    :return: 答案 + 引用列表;无依据时明确说明知识库中没有相关资料
    """
    monitor.report_tool(
        tool_name="自研RAG知识库问答工具：ask_knowledge_base",
        args={"question": question, "kb_name": kb_name},
    )

    try:
        pipeline = get_pipeline(kb=kb_name or None)
        if not len(pipeline.store):
            return f"知识库 {pipeline.kb} 是空的,没有可检索的文档。"
        answer = pipeline.query(question)
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
        return "\n".join(lines)
    except EmbeddingMismatchError as exc:
        return f"知识库向量配置不匹配,需要重新入库:{exc}"
    except Exception as exc:
        return f"知识库问答失败:{exc}"


__all__ = ["list_knowledge_bases", "search_knowledge_base", "ask_knowledge_base"]
