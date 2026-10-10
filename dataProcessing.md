# 深度研搜 — 前端提问到结果交付，完整数据处理流程

本文档梳理从前端用户提问到最终拿到结果的**完整数据流链路**，涵盖 FastAPI、DeepAgents 多智能体调度、WebSocket 实时推送和文件交付全流程。

---

## 全景流程图

```
前端 React 页面
  │
  ├─① POST /api/task  {query: "...", thread_id: "xxx"}
  │
  ▼
FastAPI (server.py)
  │
  ├─② 创建 asyncio.Task → 立即返回 {"status":"started", "thread_id"}
  │
  ├─③ 前端同时建立 WebSocket /ws/{thread_id}
  │
  ▼
run_deep_agent(query, thread_id)  [main_agent.py]
  │
  ├─④ 创建会话目录 output/session_{thread_id}
  ├─⑤ 复制上传文件 updated/ → output/
  ├─⑥ 设置 ContextVar (thread_id + session_dir)
  ├─⑦ monitor.report_session_dir() ──────────────► 前端收到"工作目录已创建"
  ├─⑧ upsert_session() + save_message() 持久化用户问题
  │
  ├─⑨ main_agent.astream(用户问题 + 路径指令)
  │     │
  │     ▼
  │   DeepAgents 主智能体（Orchestrator）
  │     │
  │     ├── 分析任务 → 决定调度策略
  │     │
  │     ├── 需要搜索? → 分派给 network_search_agent ──► Tavily API
  │     ├── 需要查库? → 分派给 database_query_agent ──► MySQL
  │     ├── 需要知识库? → 分派给 knowledge_base_agent ─► RAGFlow
  │     ├── 需要读附件? → 自己调 read_file_content
  │     │
  │     ├── 汇总所有来源信息
  │     │
  │     └── 生成交付物 → generate_markdown / convert_md_to_pdf
  │
  ├─⑩ 流式过程中，逐 chunk 检查：
  │     ├── tool_call.name == "task" → monitor.report_assistant() ─► 前端
  │     └── 最终文本内容 → monitor.report_task_result() ──────────► 前端
  │
  ├─⑪ save_message() 持久化助手回复
  └─⑫ reset_session_context() 清理 ContextVar
```

---

## 逐步详解

### ① 前端发起请求

用户在 React 页面输入问题（比如「搜索 2026 年 AI 电商趋势，生成 PDF」），前端调用：

```
POST /api/task
Body: { "query": "搜索2026年AI电商趋势，生成PDF", "thread_id": "uuid-xxx" }
```

### ② 后端秒回，不等待

[server.py:98-118](app/api/server.py#L98-L118) — `run_task()` 接口：

```python
task = asyncio.create_task(run_deep_agent(request.query, thread_id))
return {"status": "started", "thread_id": thread_id}
```

关键设计：`create_task` 把耗时 Agent 执行丢给事件循环后台跑，HTTP 接口立刻返回。用户不需要等 30 秒才看到响应。

### ③ 前端建立 WebSocket 长连接

前端拿到 `thread_id` 后，立刻连接：

```
WebSocket /ws/{thread_id}
```

[server.py:333-362](app/api/server.py#L333-L362) — `manager.connect()` 把这个 WebSocket 按 `thread_id` 存起来，后续 monitor 所有事件只推给这个连接。循环中的 `receive_text` 用于接收前端心跳（ping/pong），避免连接空闲断开。

### ④⑤⑥ 准备会话环境

[main_agent.py:63-93](app/agent/main_agent.py#L63-L93)：

| 步骤 | 做什么 | 为什么 |
|---|---|---|
| ④ | 创建 `output/session_{thread_id}/` | 每个任务独立目录，产物不互相覆盖 |
| ⑤ | 把 `updated/` 下用户上传的文件复制到 `output/` | 统一工作目录，工具只在一个地方读写 |
| ⑥ | `set_session_context(dir)` + `set_thread_context(id)` | 用 `ContextVar` 存进协程上下文，深层工具无需层层传参就能拿到会话信息 |

### ⑦⑧ 通知前端 + 持久化

- `monitor.report_session_dir()` → 前端立刻看到「工作目录已创建」，知道任务开始了
- `upsert_session()` + `save_message()` → 把用户问题写入 SQLite 历史

### ⑨ 核心：DeepAgents 主智能体执行

[main_agent.py:126-128](app/agent/main_agent.py#L126-L128)：

```python
async for chunk in main_agent.astream(
    {"messages": [{"role": "user", "content": task_query + path_instruction}]},
    config={"configurable": {"thread_id": session_id}},
):
```

#### 主智能体的组成（[main_agent.py:40-46](app/agent/main_agent.py#L40-L46)）

```
主智能体 (Orchestrator)
├── 模型: OpenAI 兼容接口 (DeepSeek/Qwen)，通过 .env 配置
├── 系统提示词: prompts.yml 中的 main_agent.system_prompt
├── 直接工具 (tools):
│   ├── read_file_content    — 读取用户上传的 PDF/Word/Excel/Markdown
│   ├── generate_markdown    — 生成 Markdown 报告
│   └── convert_md_to_pdf    — Markdown 转 PDF
├── 子智能体 (subagents):
│   ├── network_search_agent — 调 Tavily 搜互联网公开资料
│   ├── database_query_agent — 查 MySQL 结构化业务数据
│   └── knowledge_base_agent — 问 RAGFlow 私有知识库
└── 记忆: InMemorySaver (按 thread_id 隔离，同一会话复用上下文)
```

#### 执行过程

模型拿到用户问题后，自己判断需要哪些信息：

- 「这个任务涉及最新行业趋势」→ 调度 `network_search_agent` 去搜 Tavily
- 「用户提到了数据库里的药品库存」→ 调度 `database_query_agent` 去查 MySQL
- 「需要参考内部文档」→ 调度 `knowledge_base_agent` 去问 RAGFlow
- 汇总所有返回的信息，调用 `generate_markdown` 或 `convert_md_to_pdf` 生成交付物

每个子智能体都是独立的 "mini-agent"：有自己的 `name`、`description`、`system_prompt` 和 `tools`。主智能体通过 `description` 判断什么时候该派给谁。

### ⑩ 流式推送进度

[main_agent.py:126-166](app/agent/main_agent.py#L126-L166) — `astream` 每吐出一个 chunk，代码就检查：

```python
for node_name, state in chunk.items():
    messages = state["messages"]
    last_msg = messages[-1]

    if node_name == "model":
        if last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                if tool_call["name"] == "task":
                    # DeepAgents 调用子智能体 → 推送
                    monitor.report_assistant(subagent_type, ...)
        elif last_msg.content:
            # 模型产出最终文本 → 推送最终结果
            monitor.report_task_result(last_msg.content)
```

此时 [monitor.py:38-77](app/api/monitor.py#L38-L77) 的 `_emit()` 触发三级推送：

```
monitor._emit()
  ├── WebSocket → 推给前端（实时事件流）
  ├── stream_writer → 写入 DeepAgents 调试流
  └── print() → 控制台保底输出
```

**WebSocket 跨循环投递**：后台 Agent 任务可能不在 FastAPI 主事件循环里，[monitor.py:79-100](app/api/monitor.py#L79-L100) 做了智能判断——如果在同一个循环就直接 `create_task`，否则用 `run_coroutine_threadsafe` 线程安全投递。

### ⑪⑫ 收尾

| 步骤 | 正常完成 | 被取消 | 异常 |
|---|---|---|---|
| 保存历史 | `save_message(session_id, "assistant", ...)` | `save_message(session_id, "system", "任务已被用户取消")` | `save_message(session_id, "system", error_msg)` |
| 清理上下文 | `reset_session_context()` | `reset_session_context()` | `reset_session_context()` |

---

## 核心设计思想

> **HTTP 只做轻量调度（立刻返回），WebSocket 做实时推送（不等 Agent 跑完），DeepAgents 做主从分工（一个 Orchestrator + 三个专家 Worker），ContextVar 做协程级上下文隔离（深层工具无感传参）。**

整条链路的核心关键词：

| 概念 | 作用 |
|---|---|
| `thread_id` | 贯穿全链路的会话 ID，用于 WebSocket 路由、会话目录隔离、checkpointer 记忆隔离 |
| `ContextVar` | 协程级上下文变量，让深层工具无需显式传参就能拿到 `thread_id` 和 `session_dir` |
| `monitor` | 单例全局监控器，统一包装工具调用/子智能体调用/结果/异常事件，自动选择推送通道 |
| `asyncio.create_task` | HTTP 接口把 Agent 执行丢给后台，实现"请求秒回 + 后台慢慢跑" |
| `astream` | DeepAgents 流式执行，每步决策实时产出 chunk，前端可以逐步看到进度 |
| `InMemorySaver` | LangGraph 检查点，同一 `thread_id` 下的对话上下文可复用 |
| `ConnectionManager.loop` | 保存 FastAPI 主事件循环引用，确保 WebSocket 消息跨线程安全投递 |
