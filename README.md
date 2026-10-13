<div align="center">
  <h1>深度研搜 · DeepSearch Agents</h1>
  <p>面向多来源信息分析与报告交付的对话式多智能体研究系统</p>

![Agent](https://img.shields.io/badge/AI-Multi--Agent-2563eb)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Frontend-React-149eca?logo=react&logoColor=white)
![RAG](https://img.shields.io/badge/Retrieval-Hybrid_RAG-7c3aed)

</div>

DeepSearch Agents 将公开网页、业务数据库、内部知识库与上传附件连接到统一的研究流程。用户以自然语言提出任务，主 Agent 按需调度专业子 Agent，收集证据、综合分析，并交付回答、Markdown 或 PDF 报告。

项目围绕三个目标构建：**多来源协作、可追溯的知识检索、可观察的长任务执行**。

```text
结合我上传的行业报告、内部知识库和最新公开资料，
分析行业趋势，标明资料来源，并生成一份 PDF 报告。
```

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 多 Agent 协作 | 主 Agent 负责规划与汇总，网络搜索、数据库查询和知识库子 Agent 分别获取信息 |
| 多来源信息获取 | 集成 Tavily、MySQL、自研 RAG，以及 PDF、Word、Excel、Markdown 等附件读取 |
| 知识库管理 | 支持前端上传入库、检索问答、元数据过滤和逻辑文档版本替换 |
| 混合检索 | BGE 向量召回与 BM25 结合，经过 RRF 融合、重排和可配置查询改写 |
| 依据与溯源 | RAG 返回来源、标题和原文片段；低分问题直接拒答，前端可查看检索证据 |
| 报告交付 | 支持生成 Markdown、转换 PDF，以及文件列表和下载 |
| 会话与记忆 | 会话历史和长期记忆分别持久化，上下文按预算组织 |
| 执行过程 | 通过 WebSocket 展示工具调用、子 Agent 调用、结果与异常，支持任务取消 |
| 基础治理 | 提供 API Key 鉴权配置、SQL 查询规则、并发限制和会话上下文管理 |
| 评测与观测 | 提供检索评测、Ragas 适配、审计日志、健康检查及基础运行指标 |

## 系统架构

```text
React 研究工作台
  ├─ HTTP：任务、附件、知识库、历史与文件
  └─ WebSocket：实时工具与任务事件
                         │
                    FastAPI 服务
                         │
                    主 Agent
        ┌────────────────┼────────────────┐
   网络搜索子 Agent  数据库子 Agent   知识库子 Agent
        │                │                │
      Tavily           MySQL          自研 RAG
                         │
                汇总回答 / Markdown / PDF

持久化：SQLite（会话与记忆）
知识索引：NumPy 或 Qdrant + BM25 + 文档与版本元数据
```

主 Agent 使用 DeepAgents/LangGraph 组织执行，通过工具获取数据和生成文件。子 Agent 按任务需要调用，不要求每次都访问所有数据源。

### 自研 RAG

```text
文件 → 文本提取 / 必要时 OCR → 结构化切块 → 版本元数据 → 向量化与索引
问题 → 向量 + BM25 召回 → RRF → 重排 → 依据判断 → 带引用回答 / 拒答
```

- **检索后端**：默认 NumPy 精确检索，可配置 Qdrant 本地或远程后端。
- **向量模型**：支持本地 BGE、OpenAI 兼容 embedding 接口和 hash 离线兜底；hash 不具备训练模型的语义能力。
- **文档更新**：用稳定的 `document_key` 关联同一逻辑文档，更新时替换旧版本片段；`revision_id` 标识版本。
- **兼容校验**：索引记录 embedding 指纹，避免不同模型生成的向量混用。
- **质量控制**：支持困难负例、检索与生成分层评测；拒答阈值需要根据具体语料和模型验证。
- **证据输出**：提供来源、chunk、标题、文本与分数，便于定位召回与回答问题。

设计与实现详见 [自研 RAG 文档](docs/rag-self-built.md)。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| Agent 编排 | DeepAgents、LangGraph、LangChain |
| 模型接入 | OpenAI 兼容 API |
| API 与任务 | FastAPI、Uvicorn、asyncio、WebSocket |
| 前端 | React、TypeScript、Vite、Ant Design、Tailwind CSS |
| 检索 | sentence-transformers、NumPy、jieba/BM25、Qdrant（可选） |
| 数据存储 | SQLite、SQLAlchemy、MySQL |
| 文件处理 | pypdf、python-docx、pandas、ReportLab |
| 依赖与验证 | uv、pnpm、pytest、前端单元测试、Ragas（可选） |

## 快速开始

### 1. 环境与安装

- Python **3.12**、uv。
- Node.js **20.19+（20.x）或 22.12+**、pnpm。
- 可用的 OpenAI 兼容模型服务。
- 网络搜索需要 Tavily Key；数据库查询需要 MySQL。Docker 仅用于可选的本地数据库环境。

```bash
git clone https://github.com/jzs-123/deepsearch-agents.git
cd deepsearch-agents

# 安装应用依赖与本地语义向量模型依赖
uv sync --extra local-embedding
```

复制 `.env.example` 为 `.env`，然后填写模型和需要启用的数据源配置。

```bash
# macOS / Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

最小配置示例：

```dotenv
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your-model-api-key
LLM_QWEN_MAX=qwen-max

# 使用网络搜索时填写
TAVILY_API_KEY=your-tavily-api-key

# 使用本地语义检索
RAG_EMBEDDING_PROVIDER=local
RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RAG_DEFAULT_KB=default
```

本地模型优先从缓存加载；首次没有缓存时需要下载模型。离线环境可提前准备完整模型文件。后续命令使用 `--no-sync` 复用已安装环境；调整依赖后先重新执行 `uv sync` 并带上所需 extras。

### 2. 启动后端与前端

在项目根目录启动后端：

```bash
uv run --no-sync uvicorn app.api.server:app --host 127.0.0.1 --port 8000 --reload
```

在另一个终端启动前端：

```bash
cd frontend
pnpm install
pnpm dev
```

打开 **http://localhost:5173**。API 文档位于 **http://localhost:8000/docs**。

前端默认连接 `http://localhost:8000`，WebSocket 地址随 API 地址推导。如需连接其他服务地址，在 `frontend/.env.local` 中设置：

```dotenv
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_BASE_URL=ws://localhost:8000
```

### 3. 准备知识库

可以在前端上传文件入库，也可以在项目根目录通过 CLI 导入文档：

```bash
# 导入资料；替换路径即可使用自己的文档
uv run --no-sync python -m app.rag.cli ingest -k default -p docs/knowledge_base

# 检索原文，不调用生成模型
uv run --no-sync python -m app.rag.cli search -k default -q "2025年货币政策的总体基调" -n 5

# 基于证据生成带引用回答
uv run --no-sync python -m app.rag.cli ask -k default -q "2025年货币政策的总体基调"

# 查看规模与索引配置
uv run --no-sync python -m app.rag.cli stats -k default
```

入库和检索必须使用一致的 embedding 模型。切换模型或索引后端时，应保留原数据并重新建立对应索引，不能直接混用既有向量。

同一逻辑文档更新可显式指定身份：

```bash
uv run --no-sync python -m app.rag.cli ingest -k default -p ./documents/policy.md --document-key school-dismissal-policy --revision-id v2
```

再次使用相同 `document_key` 会替换旧版本片段。业务生效时间与上传时间不同；`effective_from` 当前用于保存元数据，完整历史时点查询与预约生效尚需扩展。

### 4. 接入数据库（可选）

使用已有 MySQL 时，在 `.env` 中配置 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD` 和 `MYSQL_DATABASE`。

也可启动仓库提供的本地数据库及样例数据：

```bash
docker compose --env-file .env -f docker/docker-compose.yaml up -d
```

默认映射端口为 `3307`，初始化脚本在首次创建数据卷时执行。部署时使用受限的只读账号，不沿用本地数据库默认凭证。

### 5. 开始研究任务

```text
请先读取我上传的行业报告，再结合公开资料整理研究摘要，并列出依据。
```

```text
查询数据库中的库存情况，整理异常项并生成 Markdown 报告。
```

```text
从知识库检索有关货币政策的内容，展示原文依据；资料不足时明确说明。
```

## 配置与访问控制

完整配置见 [.env.example](.env.example)。常用项包括：

| 配置 | 用途 |
| --- | --- |
| `RAG_DATA_DIR` / `RAG_DEFAULT_KB` | 知识库目录与默认库 |
| `RAG_TOP_K` / `RAG_TOP_N` | 召回候选与最终上下文数量 |
| `RAG_RERANK_MODE` / `RAG_QUERY_REWRITE` | 重排与查询改写策略 |
| `RAG_MIN_SCORE` | 依据判断阈值，需按语料验证 |
| `RAG_VECTORSTORE` / `RAG_QDRANT_URL` | 向量存储后端 |
| `CONTEXT_MAX_TOKENS` | 上下文预算 |
| `MAX_CONCURRENT_TASKS` | 进程内并发任务上限 |
| `AUTH_ENABLED` / `AUTH_API_KEY` | 配置 API Key 鉴权入口 |

默认是本地匿名模式。当前 API Key 配置不等于完整的多用户资源授权；多人部署前需补齐会话归属、文件访问与知识库作用域检查，并使用 HTTPS。配置、记忆与上下文设计见 [工程设计文档](docs/enterprise-upgrade-design.md)。

## API 概览

| 接口 | 用途 |
| --- | --- |
| `POST /api/task` | 提交研究任务 |
| `POST /api/task/{thread_id}/cancel` | 请求取消会话任务 |
| `GET /api/sessions` / `GET /api/sessions/{thread_id}` | 会话列表与历史 |
| `POST /api/upload` | 上传会话附件 |
| `GET /api/knowledge-bases` | 查看可访问知识库 |
| `POST /api/knowledge-bases/ingest` | 上传文档入库 |
| `GET /api/files` / `GET /api/download` | 文件列表与下载 |
| `GET /api/health` / `GET /api/metrics` | 健康状态与基础指标 |
| `WebSocket /ws/{thread_id}` | 实时执行事件 |

参数与请求格式以运行服务的 `/docs` 为准。

## 测试与评测

```bash
# 安装测试依赖，同时保留本地向量模型依赖
uv sync --extra local-embedding --extra test
uv run --no-sync pytest

# 前端生产构建（在 frontend 目录）
pnpm build
```

RAG 评测分为检索质量、生成质量与负例拒答。评测集应与目标语料匹配；Ragas 需要额外安装 `ragas`、`datasets` 并配置评测模型。离线近似检查与 Ragas 指标分别报告。

- [RAG 评测指南](docs/rag-eval-guide.md)
- [检索与集成测试说明](docs/rag-testing-guide.md)
- [查询改写与对照实验](docs/rag-query-rewrite.md)

## 项目结构

```text
app/
├── agent/          # 主 Agent、子 Agent、模型与提示组装
├── api/            # HTTP、WebSocket、知识库上传与任务调度
├── context/        # 历史、记忆注入与上下文预算
├── core/           # 配置与日志
├── memory/         # 持久化长期记忆与相关工具
├── rag/            # 解析、切块、索引、召回、融合、重排、生成与评测
├── security/       # 基础身份与鉴权配置
├── tools/          # 搜索、SQL、RAG、附件与报告工具
└── utils/          # 历史、路径与文件处理
frontend/           # React 研究工作台
rag_data/           # 运行时知识索引与元数据
docker/             # 本地数据库环境与初始化脚本
docs/               # 架构、配置、评测与设计文档
examples/           # 独立调用样例与评测数据
tests/              # 后端回归测试
```

## 运行边界与后续计划

当前后台任务与执行连接在进程内管理；会话历史持久化不代表进行中的任务可在重启后续跑。默认 NumPy 检索适用于小规模知识库，大规模场景需要结合 ANN、过滤与并发实测。

后续工作包括会话资源隔离、任务与事件持久化、前端断线恢复、严格最终结果 Schema、按知识库绑定拒答标定配置、异步入库与规模压测、执行预算、统一来源证据和完整链路观测。

## 致谢

项目在 [didilili/deepsearch-agents](https://github.com/didilili/deepsearch-agents) 的基础上持续扩展，新增自研 RAG、知识版本管理、评测及会话历史改进。感谢上游项目，以及 DeepAgents、LangGraph、LangChain 等开源生态。
