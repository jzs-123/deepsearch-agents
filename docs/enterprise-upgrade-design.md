# 「深度研搜」企业级升级设计说明

> 本文档面向 `deepsearch-agents` 项目从「教学/演示级」到「企业级」的架构升级,重点回答三个问题:
> **记忆模块怎么设计、上下文模块怎么设计、还有哪些其它方面需要补齐**,并给出与现有代码一一对应的落地方案。
> 文中所有模块均已在 `app/` 下实现,可直接对照代码阅读。

---

## 1. 现状评估(问题清单)

对照企业级标准,当前项目存在以下缺口:

| 维度 | 现状 | 企业级要求 |
| --- | --- | --- |
| 记忆 | `InMemorySaver` 进程重启即丢失;`sessions.db` 只存文本历史,无跨会话语义记忆;无记忆生命周期管理 | 分层记忆(情节/语义/程序),持久化,可检索,可衰减 |
| 上下文 | `ContextVar` 只传 `thread_id`/`session_dir`,无 Token 预算、无裁剪、无摘要压缩 | 上下文装配流水线 + 预算管理 + 自动压缩 |
| 配置 | 各模块散落 `os.getenv`,无类型、无默认值、无集中校验 | 集中式类型化配置(pydantic-settings) |
| 日志 | 全项目 `print`,无级别、无链路、无结构化 | 结构化日志 + 请求/会话关联 |
| 安全 | `db_tools` 表名拼接、SQL 未限制只读;无认证、无租户隔离 | SQL 注入防护 + 身份/租户隔离 + 越权防护 |
| 可观测 | 无指标、无健康检查、无链路追踪 | 指标暴露、健康检查、任务统计 |
| 并发/治理 | `active_tasks` 裸字典,无并发上限、无限流 | 并发信号量 + 限流 + 幂等 |
| 交付 | 无测试、无 CI 约束、无版本化 API | 评测集、回归测试、API 规范 |

## 2. 目标与原则

1. **渐进式,不推倒重来**:保留 DeepAgents 主链路与前端契约(`thread_id`、WebSocket 事件类型不变),在既有结构上做增量。
2. **分层解耦**:新增 `core` / `memory` / `context` / `security` / `observability` 五层,互不反向依赖。
3. **可替换存储**:记忆与历史底层用 SQLAlchemy + SQLite 起步,连接串可一键切换 Postgres/MySQL。
4. **默认安全**:SQL 只读校验、路径越界校验、身份隔离默认开启,配置开关控制是否需要鉴权。

## 3. 总体架构

```text
前端 (React)
   │  HTTP / WebSocket
   ▼
FastAPI 层 (server.py)
   ├─ 请求ID / 限流 / 鉴权 / 并发信号量   ← observability + security
   ├─ 健康检查 / 指标 / 会话 CRUD
   └─ run_deep_agent(task, session_id, identity)
          │
          ▼
   ┌───────────────────────────── 执行层 ─────────────────────────────┐
   │  ContextManager.assemble()   ← 上下文模块:记忆召回 + 历史重放 + 裁剪 │
   │        │                                                        │
   │        ▼                                                        │
   │  main_agent.astream()  (DeepAgents)                            │
   │   ├─ 持久化 store  → PersistentMemoryStore (语义记忆)          │
   │   ├─ checkpointer → InMemorySaver (单次运行态)                  │
   │   └─ 内置 SummarizationMiddleware (超预算自动压缩)              │
   └──────────────────────────────────────────────────────────────────┘
          │
          ▼
   MemoryManager (记忆模块)          History (episodic, SQLite)
```

## 4. 记忆模块设计(核心)

记忆采用**三层分级**,对应业界通行的 episodic / semantic / procedural 划分:

| 层级 | 名称 | 内容 | 生命周期 | 存储 | 本项目实现 |
| --- | --- | --- | --- | --- | --- |
| L1 | 情节记忆 | 单会话对话历史 | 随会话 | 会话级 | `history_manager`(SQLite,会话历史) |
| L2 | 语义记忆 | 跨会话事实/偏好/项目知识 | 跨会话、可衰减 | 用户/租户级 | `PersistentMemoryStore` + `MemoryManager` |
| L3 | 程序记忆 | 静态指令、技能、约束 | 长期稳定 | 全局 | 提示词 + DeepAgents `memory`(AGENTS.md) |

### 4.1 持久化存储 `PersistentMemoryStore`

- 继承 LangGraph `BaseStore`,可直接注入 `create_deep_agent(store=...)`,与 DeepAgents 内置 `StoreBackend` / `memory` 特性无缝衔接。
- 底层 SQLAlchemy(SQLite 起步,`MEMORY_DB_PATH` 可换成 Postgres 连接串)。
- 命名空间 `(维度, 租户/用户, 类别)` 采用分隔符拼接 + 前缀 `LIKE` 检索,天然支持租户/用户隔离。
- 每条记录字段:`namespace / key / value(JSON) / created_at / updated_at / access_count / last_access_at`。
- 检索策略:`query` 为空 → 按更新时间倒序;`query` 非空 → 词面重合度打分(中文按二元组切分 + 英文分词);`filter` → 元数据过滤。预留 `embed_fn` 钩子,可平滑升级为向量召回。

### 4.2 记忆管理器 `MemoryManager`

- `remember(identity, kind, content, importance, tags)` 写入语义记忆。
- `recall(identity, query, kinds, limit)` 按当前用户/租户命名空间召回,按 `importance × 新鲜度 × 访问次数` 综合排序。
- `forget(identity, key)` 显式遗忘。
- `decay()` 衰减:长时间未访问的记忆降低 `importance`,低于阈值自动淘汰(可选,默认关闭)。
- 类别 `kind` 约定:`fact`(事实)、`preference`(偏好)、`project`(项目知识)、`procedure`(流程)。

### 4.3 与 Agent 的两种集成方式

1. **显式工具**:`save_memory` / `recall_memory` / `forget_memory` 三个 LangChain 工具交给主智能体,模型可在对话中主动记忆/回忆。
2. **自动注入**:每次运行前 `MemoryManager.recall()` 把相关记忆拼成上下文块注入消息,模型无需主动调用即可获得长期记忆。

## 5. 上下文模块设计(核心)

### 5.1 上下文装配流水线 `ContextManager`

每次任务启动前,按序装配最终送入模型的消息列表:

```text
[长期记忆块(注入)]  →  [历史消息重放(裁剪后)]  →  [当前用户问题 + 工作目录指令]
```

- **历史重放**:从 `history_manager` 读取该会话的历史 user/assistant 消息,重建上下文,解决 `InMemorySaver` 重启丢状态的问题(重启后模型仍能看到前文)。
- **Token 预算**:`CONTEXT_MAX_TOKENS` 控制总预算,超预算时从最旧的历史消息开始裁剪,始终保留「长期记忆块」与「当前问题」。
- **可观测**:装配过程返回 `context_report`(历史条数、召回记忆数、裁剪前后 Token、裁剪条数),经 `monitor` 推送给前端。

### 5.2 Token 计数 `TokenCounter`

- 优先使用 `tiktoken`(`cl100k_base`),失败时回退 `len(text)/4` 启发式。
- 统一提供 `count(text)` / `count_messages(messages)`,供预算裁剪与日志复用。

### 5.3 摘要压缩(超长会话)

- **运行内自动压缩**:DeepAgents 内置 `SummarizationMiddleware`(项目已默认装配),按 Token 阈值触发,把旧消息摘要化并卸载到 `conversation_history/{thread_id}.md`,避免上下文爆炸。本项目通过 `create_summarization_middleware` 暴露并配置阈值。
- **入口前裁剪**:`ContextManager` 在历史重放阶段先做硬预算裁剪,作为压缩前的第一道防线。

## 6. 其它企业级能力

### 6.1 集中配置 `app/core/config.py`

- 基于 `pydantic-settings` 的 `Settings`,集中管理 LLM / Tavily / RAGFlow / MySQL / 存储路径 / 记忆 / 上下文 / 安全 / 限流 / 日志等全部配置。
- 类型化 + 默认值 + 环境变量自动绑定,`get_settings()` 单例缓存。

### 6.2 结构化日志 `app/core/logging.py`

- 统一 `app.*` 命名空间 logger,文本/JSON 两种格式(`LOG_FORMAT`)。
- 通过 `contextvars` 注入 `request_id` / `thread_id`,实现全链路关联。

### 6.3 认证与多租户 `app/security/auth.py`

- `Identity{user_id, tenant_id, roles, authenticated}` 从请求头解析(`X-User-Id` / `X-Tenant-Id` / `Authorization: Bearer`)。
- 记忆/历史按 `(tenant, user)` 命名空间隔离;`AUTH_ENABLED` 开启后强制校验 API Key,未开启时退化为匿名(不破坏现有前端)。
- `scope_thread_id()` 提供线程 ID 的多租户前缀,防止跨租户串扰。

### 6.4 安全加固 `app/utils/sql_guard.py`

- `validate_identifier()` 表名/列名白名单校验,杜绝 `db_tools` 表名拼接注入。
- `assert_readonly_sql()` 仅允许 `SELECT/SHOW/DESCRIBE/EXPLAIN/WITH`,拦截 `INSERT/UPDATE/DELETE/DROP/...`。
- 文件下载/列表已有 `is_relative_to` 越界防护,保持。

### 6.5 可观测性 `app/observability/metrics.py`

- 进程内指标:任务开始/完成/失败/取消计数、耗时统计、活跃任务数。
- `GET /api/metrics` 输出 Prometheus 文本格式;`GET /api/health` 健康检查。

### 6.6 并发与限流

- `MAX_CONCURRENT_TASKS` 信号量限制全局并发任务;同一 `thread_id` 仍保持单任务(取消旧任务)。
- 进程内令牌桶限流 `RATE_LIMIT_PER_MINUTE`。

### 6.7 API 规范

- 请求 ID 中间件为每个请求/WS 连接生成 `X-Request-ID`。
- 统一错误结构、健康检查、指标端点,为后续版本化(`/api/v1`)与网关接入预留。

## 7. 目录结构(新增)

```text
app/
├── core/
│   ├── config.py            # 集中配置
│   └── logging.py           # 结构化日志
├── memory/
│   ├── schemas.py           # 记忆数据结构
│   ├── store.py             # PersistentMemoryStore (BaseStore)
│   ├── manager.py           # 记忆管理器(召回/写入/衰减)
│   └── tools.py             # save_memory / recall_memory / forget_memory
├── context/
│   ├── token_counter.py     # Token 计数
│   ├── manager.py           # 上下文装配 + 预算裁剪
│   └── compaction.py        # 摘要压缩中间件封装
├── security/
│   └── auth.py              # 身份/租户隔离
├── observability/
│   └── metrics.py           # 指标与健康检查
└── utils/
    └── sql_guard.py         # SQL 注入防护
```

## 8. 实施计划与迁移

1. **Phase 1(本批次)**:配置/日志/记忆/上下文/安全/可观测六层落地,主流程接入,SQL 加固。
2. **Phase 2(后续)**:Postgres/Redis 存储切换、向量化语义召回、评测集与回归测试、任务队列(如 Celery/Arq)、分布式链路追踪(OpenTelemetry)、灰度发布。
3. **迁移注意**:新增配置均有默认值,`AUTH_ENABLED=false`、`CONTEXT_MAX_TOKENS=30000` 等默认值保证旧前端与旧流程开箱可用;旧 `os.getenv` 读取保留兼容。

## 9. 能力边界与演进路线

本批次解决「记忆持久化 + 上下文管理 + 安全/可观测/治理」三大主线,仍未覆盖:分布式任务队列、多副本一致性、完整评测体系、复杂协同编辑。这些属于后续横向扩展方向,本文档第 8 节已列明路线。
