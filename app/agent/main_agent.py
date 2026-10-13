"""
主智能体组装与异步执行模块(企业级重构版)

负责把模型、主提示词、文件类工具、记忆工具和三个专家子智能体组装成 DeepAgent,
并提供 run_deep_agent 作为 API 层调用的统一入口。相比教学版,本模块接入:

1. 持久化长期记忆(PersistentMemoryStore 注入 create_deep_agent 的 store)
2. 上下文装配(ContextManager:长期记忆召回 + 历史重放 + Token 预算裁剪)
3. 结构化日志与指标埋点
4. 身份/租户上下文(供记忆工具隔离用户数据)
"""

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlparse

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from app.agent.result import AgentResult, ArtifactRef, StructuredResultError, retry_structure, structure_retry_budget
from app.agent.sql_query import DatabaseRun, database_run
from app.core.config import settings
from app.utils.path_utils import resolve_path

from app.agent.llm import model
from app.agent.prompts import main_agent_content
from app.agent.subagents.database_query_agent import database_query_agent
from app.agent.subagents.knowledge_base_agent import knowledge_base_agent
from app.agent.subagents.network_search_agent import network_search_agent
from app.api.context import (
    get_session_context,
    reset_session_context,
    set_session_context,
    set_thread_context,
)
from app.api.monitor import monitor
from app.context.manager import get_context_manager
from app.core.logging import bind_thread_id, get_logger, reset_thread_id
from app.memory.store import get_memory_store
from app.memory.tools import forget_memory, recall_memory, save_memory
from app.observability.metrics import mark_task_cancelled, mark_task_failed, task_timer
from app.security.auth import (
    Identity,
    reset_identity_context,
    set_identity_context,
)

# 文件类工具由主智能体直接掌握,负责读取上传附件和生成最终交付文档
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content
from app.utils.history_manager import save_message, upsert_session

logger = get_logger("agent.main")

# 当前文件位于 app/agent/main_agent.py,parents[1] 即 app 目录(工作根目录,
# output/updated/data/skills 等都落在其下)
project_root_path = Path(__file__).parents[1].resolve()

# Skill 通过 FilesystemBackend 从 app/skills 目录按需加载(渐进式披露):
# 先注入各技能的 name/description,模型判断触发后再读取完整 SKILL.md。
# virtual_mode=True 把内置文件工具约束在当前会话目录内,避免越界读写。
def file_backend(_runtime):
    session_dir = get_session_context()
    if not session_dir:
        raise RuntimeError("Agent 文件访问缺少会话上下文")
    return FilesystemBackend(root_dir=session_dir, virtual_mode=True)

# 主智能体是调度中心:
# 1. tools 放文件交付工具 + 长期记忆工具(模型可主动记忆/回忆/遗忘)
# 2. subagents 放网络、数据库、自研 RAG 知识库三类信息获取助手
# 3. store 注入持久化长期记忆,checkpointer 仅用于单次运行的运行态
# 4. backend+skills 让主智能体按需加载 app/skills 下的领域方法论
main_model = model
if urlparse(settings.openai_base_url).hostname == "api.deepseek.com":
    # DeepSeek 思考模式拒绝 ToolStrategy 的 required 工具选择。
    main_model = model.model_copy(update={
        "extra_body": {**(getattr(model, "extra_body", None) or {}), "thinking": {"type": "disabled"}},
    })
main_agent = create_deep_agent(
    model=main_model,
    system_prompt=main_agent_content["system_prompt"],
    tools=[generate_markdown, convert_md_to_pdf, read_file_content, save_memory, recall_memory, forget_memory],
    checkpointer=InMemorySaver(),
    store=get_memory_store(),
    backend=file_backend,
    skills=["skills"],
    subagents=[database_query_agent, network_search_agent, knowledge_base_agent],
    response_format=ToolStrategy(AgentResult, handle_errors=retry_structure),
)


async def run_deep_agent(task_query: str, session_id: str, identity: Identity | None = None):
    """
    异步流式执行主智能体

    API 层为每次任务传入用户问题和 session_id。本函数负责准备会话目录、复制上传
    文件、装配上下文(长期记忆 + 历史重放 + 预算裁剪),并在流式执行过程中把关键
    事件上报给前端。

    :param task_query: 前端提交的原始任务问题
    :param session_id: 当前任务 ID,用于输出目录和 WebSocket 定向推送
    :param identity: 当前用户身份,用于长期记忆隔离;None 时退化为匿名
    """
    identity = identity or Identity(user_id="anonymous", tenant_id="default")
    logger.info("开始执行会话 session_id=%s user=%s", session_id, identity.user_id)

    # 每个会话独立使用 output/session_{session_id},避免不同用户的产物互相覆盖
    session_dir = project_root_path / "output" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_root_path / "skills", session_dir / "skills", dirs_exist_ok=True)

    session_dir_str = str(session_dir).replace("\\", "/")
    relative_session_dir_str = "/"

    # 上传文件先落在 updated/session_{session_id},执行前复制到本次 output 工作目录
    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    updated_info_prompt = ""
    if updated_dir_path.exists():
        files = [f.name for f in updated_dir_path.iterdir() if f.is_file()]
        if files:
            for filename in files:
                shutil.copy2(updated_dir_path / filename, session_dir / filename)
            updated_info_prompt = (
                "\n    [已上传文件] 已加载到工作目录:\n"
                + "\n".join([f"    - {f}" for f in files])
                + "\n    请优先使用工具(read_file_content)读取并参考这些文件。"
            )

    # 工作环境指令是运行时动态补充的,约束模型只在当前会话目录读写文件
    path_instruction = f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """

    # ★ 上下文装配:长期记忆召回 + 会话历史重放 + 预算裁剪(在保存本次用户消息之前,
    #   历史里不含当前问题,避免重复)
    context_manager = get_context_manager()
    input_messages, context_report = context_manager.assemble(
        session_id, task_query, identity, suffix=path_instruction
    )

    # 持久化:创建/更新会话记录,保存用户问题(供下次运行重放)
    upsert_session(session_id, task_query[:80], identity.user_id, identity.tenant_id)
    save_message(
        session_id,
        "user",
        task_query,
        "[]",
        identity.user_id,
        identity.tenant_id,
    )

    # 每次运行使用独立的运行态 thread_id,历史由上方显式注入,避免与 checkpointer 累积重复
    run_thread_id = f"{session_id}:{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": run_thread_id}, "recursion_limit": settings.agent_recursion_limit}

    collected_events: list[dict] = []
    final_result: AgentResult | None = None
    # 在准备成功后绑定上下文，确保所有绑定都由 finally 清理。
    session_dir_token = set_session_context(session_dir_str)
    session_id_token = set_thread_context(session_id)
    identity_token = set_identity_context(identity)
    log_token = bind_thread_id(session_id)
    event_collector_token = monitor.bind_event_collector(collected_events)
    retry_token = structure_retry_budget.set([2])
    db_scope = DatabaseRun(task_query)
    db_token = database_run.set(db_scope)

    try:
        monitor.report_session_dir(session_dir_str)
        monitor._emit("context_report", f"上下文装配: 记忆{context_report['memories']}条 历史{context_report['history_turns']}条 裁剪{context_report['trimmed']}条 tokens={context_report['tokens_after']}/{context_report['tokens_before']}")
        with task_timer():
            async for chunk in main_agent.astream({"messages": input_messages}, config=config):
                for node_name, state in chunk.items():
                    if not state:
                        continue
                    if state.get("structured_response") is not None:
                        final_result = AgentResult.model_validate(state["structured_response"])
                    if "messages" not in state:
                        continue
                    messages = state["messages"]
                    if messages and isinstance(messages, list):
                        last_msg = messages[-1]
                        if node_name == "model":
                            if getattr(last_msg, "tool_calls", None):
                                for tool_call in last_msg.tool_calls:
                                    if tool_call["name"] == "task":
                                        subagent_type = tool_call["args"]["subagent_type"]
                                        subagent_desc = tool_call["args"]["description"]
                                        monitor.report_assistant(subagent_type, {"description": subagent_desc})
            if not final_result:
                raise StructuredResultError("Agent 未提交符合 Schema 的最终结果")
            if db_scope.result:
                other_assistants = any(event["event"] == "assistant_call" and event["data"].get("assistant_name") != database_query_agent["name"] for event in collected_events)
                # 直接取数由程序呈现完整原始数据，防止主模型再漏行、改数或声称字段不存在。
                if not other_assistants and db_scope.response_mode == "table" and all(Path(artifact.path).name == db_scope.artifact for artifact in final_result.artifacts):
                    final_result.answer = db_scope.answer
                    final_result.status = db_scope.status
                if db_scope.artifact and not any(Path(artifact.path).name == db_scope.artifact for artifact in final_result.artifacts):
                    final_result.artifacts.append(ArtifactRef(path=db_scope.artifact, title="数据库查询原始结果"))
            for artifact in final_result.artifacts:
                path = Path(resolve_path(artifact.path, session_dir_str))
                if not path.is_file():
                    raise StructuredResultError("Agent 引用的生成文件不存在")
                artifact.path = str(path)

    except asyncio.CancelledError:
        mark_task_cancelled()
        save_message(
            session_id,
            "system",
            "任务执行已停止",
            json.dumps(collected_events, ensure_ascii=False, default=str),
            identity.user_id,
            identity.tenant_id,
        )
        logger.info("会话 %s 已被取消", session_id)
        raise
    except Exception:
        mark_task_failed()
        error_msg = "任务执行失败，请稍后重试"
        save_message(
            session_id,
            "system",
            error_msg,
            json.dumps(collected_events, ensure_ascii=False, default=str),
            identity.user_id,
            identity.tenant_id,
        )
        monitor._emit("error", error_msg)
        logger.exception("会话 %s 执行异常", session_id)
        raise
    else:
        # 正常完成:保存助手最终响应(带真实内容,供历史重放使用)
        save_message(
            session_id,
            "assistant",
            final_result.answer,
            json.dumps(collected_events, ensure_ascii=False, default=str),
            identity.user_id,
            identity.tenant_id,
        )
        monitor.report_task_result(final_result)
        return final_result
    finally:
        database_run.reset(db_token)
        monitor.reset_event_collector(event_collector_token)
        structure_retry_budget.reset(retry_token)
        reset_session_context(session_dir_token, session_id_token)
        reset_identity_context(identity_token)
        reset_thread_id(log_token)


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        run_deep_agent("从网络查询机器人信息，并生成Markdown文件", "test_session_001")
    )
