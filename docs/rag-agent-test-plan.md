# 自研 RAG × DeepAgents 集成与测试方案（审计报告 + 三条并行任务的接口约定）

> 版本：v1（审计基线 = 本地 main `64ea4a1`，工作区根目录 `G:\deepsearch-agents-main`）
> 作者：integ（t1 / audit），供 t2（测试套件）、t3（环境搭建）、t4（集成）直接引用
> 状态：**审计完成**；本文所有结论均附实际命令与关键输出，命令在 2026-10-10 于本机实测
> 交付物路径：`docs/rag-agent-test-plan.md`

---

## 0. 阅读范围声明（避免"声称读过未打开的文件"）

本文用 `read` 工具**逐行读过**的文件：

```
docs/rag-self-built.md
app/agent/main_agent.py
app/agent/prompts.py
app/agent/llm.py
app/agent/subagents/knowledge_base_agent.py
app/agent/subagents/database_query_agent.py
app/agent/subagents/network_search_agent.py
app/prompt/prompts.yml
app/tools/rag_tools.py
app/tools/ragflow_tools.py
app/core/config.py
app/api/monitor.py
app/skills/deep-research/SKILL.md
app/rag/__init__.py  app/rag/types.py  app/rag/config.py  app/rag/loaders.py  app/rag/chunking.py
app/rag/embedding.py  app/rag/vectorstore.py  app/rag/bm25.py  app/rag/hybrid.py  app/rag/rerank.py
app/rag/generator.py  app/rag/pipeline.py  app/rag/eval.py  app/rag/cli.py  app/rag/selftest.py
app/rag/vendor.py
rag_data/README.md
pyproject.toml
```

只用 `grep`/`Get-Content`/`git` 看过、**没有**逐行阅读的文件（本文只引用其中的匹配片段）：`app/api/server.py`（仅 grep 到入口与路由行）、`README.md`（仅 grep 匹配行）、`.gitignore`、`uv.lock`、`requirements.txt`（仅见 grep 片段）、`app/tools/db_tools.py`、`app/tools/tavily_tool.py`、`app/memory/*`、`examples/*.py`、`docs/enterprise-upgrade-design.md`（未打开）。

---

## 1. 结论摘要（TL;DR）

| # | 问题 | 结论 | 关键依据 |
| --- | --- | --- | --- |
| 1 | **默认 Python 3.10 能否端到端跑通 Agent 链路** | **不能。** 3.10 下 `import app.agent.main_agent` 直接 `ModuleNotFoundError: No module named 'deepagents'`；但 3.10 **可以**跑 RAG 子系统（`python -m app.rag.selftest` → 18 通过 / 0 失败） | 见 §2.1 证据 E1/E2 |
| 2 | 哪个解释器能跑通 Agent 链路 | **仓库内 `.venv`（Python 3.12.11，uv 0.7.20）**：已装 `deepagents 0.5.7 + langchain 1.2.17 + langgraph 1.1.10 + langchain-core 1.3.3 + langchain-openai 1.2.1`，实测 `import app.agent.main_agent` 成功并编译出 `CompiledStateGraph` | 证据 E3 |
| 3 | 三个工具能否被 DeepAgents/LangChain 识别 | **能**：都是 `StructuredTool`，名称/docstring/args schema 正确；`create_deep_agent(subagents=[...])` 装配成功即为最强证据（工具对象在装配期被绑定） | 证据 E4/E5 |
| 4 | 改名后是否有遗漏的 `ragflow` 引用 | **prompts.yml 与装配代码已干净**（`sub_agents` 键 = `['db','rag','tavily']`，无 `ragflow` 键，无 `from app.tools.ragflow_tools import`）。**但 `app/skills/deep-research/SKILL.md` 仍是 RAGFlow 版**，会指示主智能体调用已不存在的 `get_assistant_list` | 证据 E6/E7 |
| 5 | `app/tools/ragflow_tools.py` 是否在装配路径上 | **不在**：全仓库无任何 import 语句引用它；仅在文件自身、注释、文档与 `README.md` 教程链接中出现 | 证据 E7 |
| 6 | RAG 模块是否有 pytest 测试 | **没有**：无 `tests/` 目录，无 `pytest.ini`/`setup.cfg`/`tox.ini`/`conftest.py`，`pyproject.toml` 无 `[tool.pytest.*]`；`python -c "import pytest"` 在 3.10 与 `.venv` 均为 `False`，只有 conda 环境有 `pytest 9.0.3` | 证据 E8 |
| 7 | 本机测不了什么 | 真实 BGE 语义向量（无 `sentence-transformers`）、真实生成质量（需付费大模型调用）、数据库子智能体端到端（`127.0.0.1:3307` 未监听）、`uv`/`pip` 装包路径（沙箱内不可用/未验证） | §7 风险清单 |
| 8 | 最大的隐形坑 | 沙箱下 **CPython 3.12/3.13 建 `0o700` 目录后连自己都无法访问**，导致 `tmp_path`/`--basetemp`/`TemporaryDirectory` 全部崩，且目录**永久不可删**；`app/rag/selftest.py` 因此在 3.12/3.13 下 0 断言就崩（3.10 正常） | 证据 E9/E10 |

---

## 2. 集成链路图与装配核查

### 2.1 证据 E1/E2：默认 Python 3.10 的边界

```
$ python -c "import sys; print(sys.version); print(sys.executable)"
3.10.9 (tags/v3.10.9:1dd9be6, Dec  6 2022, 20:01:21) [MSC v.1934 64 bit (AMD64)]
C:\Users\lenovo\AppData\Local\Programs\Python\Python310\python.exe

$ python -c "import app.agent.main_agent"
ModuleNotFoundError: No module named 'deepagents'          # exit=1
    File "G:\deepsearch-agents-main\app\agent\main_agent.py", line 19, in <module>
      from deepagents import create_deep_agent

$ python -m app.rag.selftest
──── 自测结果:18 通过 / 0 失败 ────                          # exit=0

$ python -c "import importlib.util as u; print({n: bool(u.find_spec(n)) for n in ['deepagents','langchain','langgraph','langchain_core','pytest','numpy','jieba','pypdf','sentence_transformers']})"
{'deepagents': False, 'langchain': True, 'langgraph': True, 'langchain_core': True,
 'pytest': False, 'numpy': True, 'jieba': True, 'pypdf': False, 'sentence_transformers': False}
```

**结论**：Agent 链路在 3.10 上断在 `deepagents` 缺失（且 3.10 违反 `requires-python = ">=3.12,<3.13"`）；RAG 子系统在 3.10 上可测（`pypdf` 走 `vendor/pypdf 6.20.0` 兜底）。

### 2.2 集成链路图（谁调用谁）

```
┌─ 入口（Web）────────────────────────────────────────────────────────────┐
│ app/api/server.py  POST /api/task  →  _run_task_safely()               │
│                       → run_deep_agent(query, thread_id, identity)      │
│   （仅 grep 看过 server.py：L38 `from app.agent.main_agent import ...`，│
│     L189 `await run_deep_agent(...)`；未逐行阅读）                      │
└────────────────────────────────────────────────────────────────────────┘
                              │                           状态
                              ▼
app/agent/main_agent.py  ── 模块级装配 ───────────────────  ✅ 已接通（.venv 实测）
  ├── model            ← app/agent/llm.py（init_chat_model, OpenAI 兼容, .env）  ✅
  ├── system_prompt    ← app/agent/prompts.py: main_agent_content["system_prompt"] ✅
  ├── tools            = [generate_markdown, convert_md_to_pdf, read_file_content,
  │                       save_memory, recall_memory, forget_memory]              ✅
  ├── checkpointer     = InMemorySaver()                                           ✅
  ├── store            = get_memory_store()   ← app/data/memory.db（import 期即初始化）✅
  ├── backend          = FilesystemBackend(app/, virtual_mode=True)                ✅
  ├── skills           = ["skills"]（← app/skills/*/SKILL.md）                     ⚠️ 见 2.4
  └── subagents        = [database_query_agent, network_search_agent,
                          knowledge_base_agent]                                    ✅
                              │
        ┌─────────────────────┼──────────────────────────────┐
        ▼                     ▼                              ▼
  tavily 子智能体        db 子智能体                 knowledge_base_agent（自研 RAG）✅
  prompts.yml[tavily]   prompts.yml[db]             prompts.yml["rag"]  ✅（改名已生效）
  internet_search       3 个 db_tools               3 个自研工具 ↓
        │                     │                    ┌───────────────────────────────┐
        ▼                     ▼                    │ app/tools/rag_tools.py        │
   tavily_tool.py        db_tools.py               │  list_knowledge_bases()       │
（未逐行阅读）         （未逐行阅读）              │  search_knowledge_base(q,kb,n)│
   MySQL localhost:3307 ← 未监听（3306 在监听）     │  ask_knowledge_base(q,kb)     │
                                                   └───────────────┬───────────────┘
                                                                   ▼ 进程内调用
                            app/rag/（自研流水线，16 文件）
                            config → loaders → chunking → embedding → vectorstore
                                   → bm25 → hybrid(RRF) → rerank → generator
                                   pipeline（ingest/search/query, get_pipeline 缓存）
                                   ← eval / cli / selftest
                                                                   ▼
                                        rag_data/<kb>/（chunks.jsonl / vectors.npy / meta.json）
                                        现有唯一知识库：fin（238 chunks，hash:512）

  ✗ app/tools/ragflow_tools.py（get_assistant_list / create_ask_delete）——不在任何装配路径上
  ✗ app/ragflow/*（SDK 示例）——不在装配路径上
```

### 2.3 逐项核查结论（点 1 的四个子问题）

1. **main_agent 如何组装 subagents**：`create_deep_agent(..., subagents=[database_query_agent, network_search_agent, knowledge_base_agent])`（`app/agent/main_agent.py:75`），三个都是**字典式子智能体**（`name`/`description`/`system_prompt`/`tools`）。
2. **三个 RAG 工具能否被识别**：能。`.venv` 下装配成功（证据 E3），三个工具对象的 `name`/`description`/`args` 均正确（证据 E4）。
3. **prompts.py 的键名与 prompts.yml 是否一致**：一致。`sub_agents_content` 键 = `['db','rag','tavily']`，`knowledge_base_agent` 读的就是 `sub_agents_content["rag"]`（证据 E6）。**没有遗留 `ragflow` 键**。
4. **ragflow_tools 是否在装配路径**：不在（证据 E7）。

### 2.4 ⚠️ 存疑/缺陷点（装配层）

| 编号 | 位置 | 问题 | 影响 |
| --- | --- | --- | --- |
| A1 | `app/skills/deep-research/SKILL.md` L3/L16/L18-21/L29 | 仍是 RAGFlow 文案，明确要求"先查助手列表 `get_assistant_list`"、"三种信息源（网络/数据库/RAGFlow）" | 主智能体会按技能加载路径读到它，被指示调用**已不存在**的工具；属于改名遗漏（t4 应修） |
| A2 | `app/agent/prompts.py:38` | `print(sub_agents_content)` 残留在模块级 | 任何 `import app.agent.prompts` 都向 stdout 打印整段配置（实测输出见 E3 中夹带的 dict 文本）；污染服务日志/stdio |
| A3 | `app/tools/rag_tools.py:86` | `if results[0].score < pipeline.config.min_score:` —— 这里 `score` 是**重排后的排序分**，而 `pipeline.grounded_score()`（`app/rag/pipeline.py:287-303`）、`cli.py:205`、`eval.py:134` 三处判定统一用**向量余弦** | 与 `docs/rag-self-built.md` §5.10 记录的 bug 属同一类；`search_knowledge_base` 的"相关性不足"提示与 `ask_knowledge_base` 的拒答会**不一致**（同一问题一个说"有资料"、一个说"没依据"） |
| A4 | `app/tools/rag_tools.py:79,121` | `get_pipeline(kb=kb_name or None)` → `RagPipeline` → `VectorStore.open(..., create=True)` | **传入不存在的知识库名会真的在 `rag_data/` 下创建空目录**（实测见 E5）；模型幻觉出的 kb 名会污染数据目录 |

---

## 3. 三个工具的行为契约（点 2）

### 3.1 身份与 schema（实测，`.venv`）

证据 E4：

```
list_knowledge_bases | StructuredTool | {}
search_knowledge_base | StructuredTool | {"question": {...string...}, "kb_name": {default "", string}, "top_n": {default 4, integer}}
ask_knowledge_base | StructuredTool | {"question": {...string...}, "kb_name": {default "", string}}
```

### 3.2 **关键测试约束：必须 `.invoke()`，不能直接调用**

证据 E5（`.venv` 实测）：

```
>>> m.list_knowledge_bases()
TypeError: 'StructuredTool' object is not callable
>>> m.list_knowledge_bases.invoke({})
'当前没有任何知识库。请先由管理员执行入库命令:python -m app.rag.cli ingest -k default -p <文档目录>'
```

> `langchain_core.tools.StructuredTool`（langchain-core 1.3.3）**不支持 `__call__`**。所有用例必须走 `.invoke({...})`；同理 `create_ask_delete` 旧文档里的 `get_assistant_list.invoke({})` 写法在这里是对的。
> 另外每次 `.invoke()` 都会调用 `monitor.report_tool()` → 向 stdout 打印 `[Monitor:tool_start] ...`（`app/api/monitor.py:77`），测试里用 `capsys` 捕获或忽略，不要断言 stdout 干净。

### 3.3 分支矩阵（全部实测，`.venv`，数据目录 = 临时副本）

| 分支 | 输入 | 实测输出（截断） | 备注 |
| --- | --- | --- | --- |
| 无任何知识库 | `list_knowledge_bases.invoke({})` | `'当前没有任何知识库。请先由管理员执行入库命令:python -m app.rag.cli ingest -k default -p <文档目录>'` | 无异常 |
| 有知识库 | 同上 | `'可用知识库：\n- fin: 238 个片段 \| 向量模型 hash:512 \| 更新时间 2026-10-10 12:08:25\n- ghost: 0 个片段 \| 向量模型 未初始化 \| 更新时间 -（空库，暂不可检索）\n\n默认知识库:default'` | 空库会显式标注 |
| 知识库不存在 | `search_knowledge_base.invoke({"question":"随便问问","kb_name":"ghost"})` | `'知识库 ghost 是空的,没有可检索的文档。'`，**且 `rag_data/ghost/` 目录被真实创建**（`ghost 目录是否被创建: True`） | ⚠️ 缺陷 A4 |
| 知识库不存在 | `ask_knowledge_base.invoke({"question":"随便问问","kb_name":"ghost"})` | 同上一行（不调用大模型） | 空库短路 |
| 正常检索 | `search_knowledge_base.invoke({"question":"货币政策","kb_name":"fin","top_n":2})` | 返回 `【片段 1】来源:...pdf#0 ... 匹配度:1.000` + 原文 | 与 `cli search` 行为一致 |
| 向量指纹不匹配 | 库 `hash:512`，查询配置 `hash:256` | `'知识库向量配置不匹配,需要重新入库:知识库 fin 的向量由 [hash:512] 生成,当前配置为 [hash:256]。向量空间不可比,请统一向量模型后重新入库(python -m app.rag.cli reset -k fin)。'`（`search` 与 `ask` 同样文案） | 走 `except EmbeddingMismatchError` 分支，**不抛异常** |
| 大模型不可用 | `ask_knowledge_base` 且 `ChatClient.available=False` | 返回 `LLM_UNAVAILABLE_ANSWER`（"（大模型调用不可用,以下仅返回检索到的原文依据…）"）+ 引用 + `debug.reason=llm_unavailable` | 由 `app/rag/generator.py:198-203` 决定；无异常 |
| 大模型报错 | 调用抛错 | 返回 `（生成回答时调用大模型失败:...）\n\n已检索到以下依据，请参考引用。`，`debug.reason=llm_error` | `generator.py:205-215` |
| 阈值不足 | 排序分 < `min_score` | `ask` 走 `NO_EVIDENCE_ANSWER`（拒答，`reason=score_below_threshold`）；`search` 走"没有与该问题足够相关的资料(最高匹配度 x, 低于阈值 y)" | ⚠️ 两者用的分数不同源，见 A3 |
| 参数名写错 | `search_knowledge_base.invoke({"query":"..."})` | pydantic 校验报错（缺 `question`） | 结构化校验由 `StructuredTool` 保证 |

> **契约要点**：三个工具**永不抛异常**给模型，一律返回中文字符串。因此错误分支的测试必须**断言返回字符串内容**（例如 `"向量配置不匹配" in out`），不能 `pytest.raises`。

---

## 4. selftest 覆盖盘点 与 现有测试基础设施（点 3）

### 4.1 `app/rag/selftest.py` 已覆盖的点（18 项断言，实测 18/0）

`python -m app.rag.selftest`（3.10）输出：

```
[1] 文本入库成功 3 chunks / chunk 数量随切块参数增长
[2] 重复入库后总量不变(整篇替换)
[3] chunk 带标题路径 / chunk 长度受控 max=118 / 长文档切出多个 chunk
[4] 检索命中相关片段 / 召回结果带分数 / 记录了命中通道 ['dense','sparse']
[5] BM25 能精确命中型号 A13
[6] 重排返回指定条数 / 重排把包含答案的片段排到第一
[7] 无依据时不编造答案
[8] 维度不一致会显式报错
[9] 重新加载 chunk 数一致 / 重新加载后 BM25 可用
[10] 目录入库成功 2 个文件 / 新文档可被检索到
──── 自测结果:18 通过 / 0 失败 ────
```

### 4.2 selftest **未**覆盖的点（→ 全部是 t2 的 pytest 增量）

| 类别 | 未覆盖点 |
| --- | --- |
| loaders | PDF 逐页抽取与页码伪标题、空文本页 warning、`normalize_text`（`\xa0`/零宽/中文字间空格/多空行）、GBK 中文 txt 读取、`docx` 表格转 Markdown、`csv` 逐行成句、不支持后缀报错、`iter_supported_files` 跳过隐藏/`~$` 文件 |
| chunking | 中文编号标题与正文编号列表的区分（`_is_cn_title`）、代码块围栏内不切标题、三级标题路径拼接 `A > B > C`、句边界重叠（`_with_overlap`）、无标点长文硬切 `_hard_split`、`max_chunks=2000` 上限、空文档返回 `[]` |
| embedding | `HashEmbedder` 确定性/`dim` 生效/L2 归一化、零向量不产生 NaN、`build_embedder(provider=...)` 三后端分派、**local 降级到 hash 的触发条件与 warning**、`allow_fallback=False` 必须抛错、未知 provider 抛 `ValueError`、`fingerprint` 文本格式 |
| vectorstore | `EmbeddingMismatchError` 的**两条路径**（open 校验 / add 追加校验）、空库不做指纹校验、`chunks.jsonl` 与 `vectors.npy` 行数撕裂截断、`delete_document` 整篇替换、`clear`/`save` 原子替换、`list_knowledge_bases` 在无 meta/坏 JSON 时的容错 |
| config | `chunk_overlap >= chunk_size`、`top_n > top_k`、非法 provider/rerank_mode、`with_kb`/`kb_dir`、`ensure_within` 路径越界（`../`） |
| bm25 | `tokenize` 型号/编号整体保留（`A13`、`3.2.1`）、停用词过滤、jieba 缺失退化路径（单字+二元组）、`_idf` 平滑非负、`explain()` 贡献分解 |
| hybrid | RRF 只用名次（量纲无关）、`DEFAULT_CHANNEL_WEIGHTS`、双通道 overlap 统计、`top_k` 截断、`rrf_score` 与 `score` 关系 |
| rerank | `off`/`lexical`/`llm` 三种模式分派、llm 无 client 时回退 lexical、`LLMReranker._parse_order` 对乱序/重复/越界编号的鲁棒性、重排后 `rank` 重算、同分稳定排序 |
| generator | `build_context` 的 `max_chars=8000` 整条截断、`grounded_score` 与排序分分离（**A3 回归测试的落点**）、`_REFUSAL_MARKERS` 出口识别、`ChatClient.available` 语义 |
| pipeline | `ingest` 路径不存在报 `FileNotFoundError`、空目录 warning、`files_skipped` 计数、`add_texts` 与文件入库共用切块路径、`reset()` 删文件、`stats()`、`get_pipeline` 进程内缓存与 `clear_pipeline_cache`、`chunk_size` 变更告警 |
| eval | `load_eval_set` 支持 JSON/JSONL、`_norm_source` 去目录+小写、假失败/假通过回归、`expect_no_answer` 拒答统计、`format_report` |
| cli | `list/create/ingest/search/ask/stats/reset` 各子命令返回码、`search` 空库返回 1、`EmbeddingMismatchError` 返回 2、`reset` 需 `--yes`/确认 |
| Agent 工具层 | 三个工具的 `.invoke()` 契约全部分支（§3.3）、**工具对象类型必须是 `StructuredTool`**、A4（不存在 kb 不得创建目录）回归 |
| Agent 装配 | `sub_agents_content` 键集合 = prompts.yml 键集合（改名防回归）、三个子智能体字典的必需字段、`knowledge_base_agent["tools"]` 名称集合 = `{list_knowledge_bases, search_knowledge_base, ask_knowledge_base}` |
| 端到端 | `run_deep_agent` 流式事件、知识库子智能体真实分派（需大模型，付费/非确定性 → 标记 `e2e`，默认跳过） |

### 4.3 现有测试基础设施现状（实测）

```
$ foreach ($f in @('pytest.ini','setup.cfg','tox.ini','conftest.py')) {...}   # 全部 False
$ Select-String pyproject.toml -Pattern 'tool\.(pytest|coverage|mypy|ruff)'    # 无输出
$ Get-ChildItem -Recurse -Directory -Filter 'test*' | ...                     # 只有 examples\test_docs
$ python        -c "import importlib.util as u; print(bool(u.find_spec('pytest')))"   # False (3.10)
$ .\.venv\Scripts\python.exe -c "import importlib.util as u; print(bool(u.find_spec('pytest')))"  # False
$ C:\Users\lenovo\.conda\envs\langchain1.2\python.exe -m pytest --version   # pytest 9.0.3
```

**结论**：项目**零 pytest 基础设施**；当前唯一可用的 pytest 在 conda 环境 `langchain1.2`（Python 3.13.12）。

### 4.4 向量后端现状（点 4）

`app/rag/embedding.py` 的 `local` 路径是**完整**的（`LocalEmbedder._load` 懒加载 → `SentenceTransformer`；`encode` 走 `normalize_embeddings=True`；BGE 查询前缀；`dim` 取模型真实维度；`fingerprint = local:<model>:<dim>`）。**降级逻辑的静默风险**在 `build_embedder(..., allow_fallback=True)`（L249-259）：

```python
try:
    embedder = LocalEmbedder(...)
    embedder.encode(["预热"], is_query=False)   # 触发真实加载
    return embedder
except Exception as exc:
    if not allow_fallback: raise
    logger.warning("本地向量模型不可用(%s),降级为 hash 离线向量。检索质量会明显下降。", exc)
    return HashEmbedder(dim=dim)
```

实测（`.venv`，无 `sentence-transformers`）：

```
WARNING app.rag.embedding: 本地向量模型不可用(缺少 sentence-transformers 依赖,请执行: pip install sentence-transformers,
或把 RAG_EMBEDDING_PROVIDER 改为 hash 使用离线兜底),降级为 hash 离线向量。检索质量会明显下降。
```

**会静默掩盖问题的情况（重要）**：

1. `except Exception` 捕获的是**所有异常**：模型下载失败（网络/磁盘）、OOM、transformers 版本不兼容、`torch` DLL 加载失败等，一律只留一条 `WARNING` 后转 hash → 使用者可能长期以为在用语义向量。
2. **空库/新库时指纹校验不触发**：`VectorStore.open` 只在 `store.chunks` 非空时比对指纹（`vectorstore.py:92`）。所以"先装失败降级建了空库 → 后续修复后入库写入完全不同的向量空间"这条路上，只有**非空库**才会报 `EmbeddingMismatchError`；空库会静默采纳新指纹。
3. 降级只在 `logger.warning` 层面可见；`WARNING` 若未配置 handler，Python 的 `lastResort` 会写 stderr，但 Web 服务日志里很容易被淹没。**建议**：在 `RagConfig`/`stats()` 或工具返回文本里显式暴露 `embedder.fingerprint`（`search_knowledge_base` 已输出 `pipeline.kb`，可再加后端指纹）。
4. 历史事实（`docs/rag-self-built.md` §5.8 + 本次实测）：本机默认 Python 3.10 装不上 `sentence-transformers`（pip 空转、系统 site-packages 不可写），`pypdf` 靠 `vendor/`（`vendor/pypdf` = 6.20.0）兜底；**因此 `rag_data/fin`（238 chunks）是 `hash:512` 建的**，一切"语义质量"结论都不成立。

---

## 5. 环境可行性（点 5）

### 5.1 解释器与包实测

| 解释器 | 版本 | deepagents | langchain/langgraph | pytest | numpy/jieba/pypdf | sentence-transformers |
| --- | --- | --- | --- | --- | --- | --- |
| 默认 `python` | 3.10.9 | ❌ | ✅（langchain/langgraph/langchain_core/openai） | ❌ | numpy✅ jieba✅ pypdf❌ | ❌ |
| 仓库 `.venv`（uv） | **3.12.11** | ✅ **0.5.7** | ✅ **1.2.17 / 1.1.10**（core 1.3.3, openai 1.2.1, anthropic 1.4.3, google-genai 4.2.2） | ❌ | numpy✅ jieba❌ pypdf 6.11.0✅ | ❌ |
| conda `langchain1.2` | 3.13.12 | ❌ | ✅ 1.2.12 / 1.1.2（core 1.2.18, openai 1.1.11, anthropic 1.3.4） | ✅ **9.0.3** | numpy✅ jieba❌ pypdf 6.10.2✅ | ❌（有 torch 2.11.0 / transformers 5.3.0） |
| miniconda base | 3.13.12 | — | — | — | — | — |

证据 E3（`.venv` 装配成功）：

```
$ .\.venv\Scripts\python.exe -c "import app.agent.main_agent as m; print('MAIN_AGENT_OK'); print(type(m.main_agent))"
... (langgraph 弃用告警、prompts.py 打印的 dict)
MAIN_AGENT_OK
<class 'langgraph.graph.state.CompiledStateGraph'>      # exit=0
```

### 5.2 `deepagents==0.5.7` 与已装版本的冲突分析（读 METADATA）

```
deepagents-0.5.7 METADATA:
  Requires-Python: <4.0,>=3.11
  Requires-Dist: langchain-core<2.0.0,>=1.3.2
  Requires-Dist: langsmith>=0.8.0
  Requires-Dist: langchain<2.0.0,>=1.2.17
  Requires-Dist: langchain-anthropic<2.0.0,>=1.4.3
  Requires-Dist: langchain-google-genai<5.0.0,>=4.2.2
  Requires-Dist: wcmatch
langgraph-1.1.10 METADATA: Requires-Python >=3.10, langchain-core>=1.3.0, langgraph-checkpoint>=2.1.0 ...
```

| 目标环境 | 是否满足 `deepagents==0.5.7` | 说明 |
| --- | --- | --- |
| `.venv`（3.12.11） | ✅ **完全满足**（langchain 1.2.17 / core 1.3.3 / anthropic 1.4.3 / google-genai 4.2.2 均达标） | 与 `pyproject.toml` 声明完全一致（`langchain==1.2.17`、`langgraph==1.1.10`、`langchain-openai==1.2.1`、`deepagents==0.5.7`） |
| conda `langchain1.2`（3.13.12） | ❌ 需强升：`langchain 1.2.12→>=1.2.17`、`langchain-core 1.2.18→>=1.3.2`、`langchain-anthropic 1.3.4→>=1.4.3`，并**新增 `langchain-google-genai>=4.2.2`** | 且 3.13.12 **违反项目 `requires-python = ">=3.12,<3.13"`**。结论：**不要把 deepagents 装进这个共享 conda 环境**（会连带升级整套 langchain 栈，污染其它项目） |

### 5.3 结论：用哪个解释器、装在哪

- **Agent 链路与测试套件统一用仓库内 `.venv`（Python 3.12.11）**：唯一同时满足"`deepagents` 已装 + 版本符合 pyproject + `requires-python` 合规"的环境，且**已实测装配成功**。
- **conda `langchain1.2` 仅作为"当前唯一有 pytest 的解释器"**，用于跑纯 RAG 层用例（已实测可 import `app.tools.rag_tools` 与 `app.rag.pipeline`）；不往里装 deepagents。
- **`.venv` 需要补的包（按优先级）**：`pytest`（必需）→ `jieba`（强烈建议：缺失时 BM25 与词面重排退化为单字+二元组，检索质量下降）→ `sentence-transformers`+`torch`（**不建议在本沙箱尝试**，见 R3）。
- **不要执行 `uv sync`**：`uv.lock` 是旧的（仍钉 `ragflow-sdk`；`grep` 显示 lock 内 `ragflow-sdk` 条目，且没有 `jieba`/`sentence-transformers`），会按 pyproject 拉 `sentence-transformers → torch`，正是 `docs/rag-self-built.md` §5.8 记录的卡死路径。

### 5.4 装包通道实测（重要：通道本身被沙箱打断）

```
$ uv pip install --python ".venv\Scripts\python.exe" --dry-run pytest jieba
error: failed to open file `C:\Users\lenovo\AppData\Local\uv\cache\sdists-v9\.git`: 拒绝访问。 (os error 5)

$ $env:UV_CACHE_DIR="G:\deepsearch-agents-main\.uv-cache"    # 改成工作区缓存后
$ uv pip install --python ".venv\Scripts\python.exe" --dry-run pytest jieba
error: Failed to inspect Python interpreter from provided path at `.venv\Scripts\python.exe`
Caused by: Failed to query Python interpreter at `.venv\Scripts\python.exe`
Caused by: 拒绝访问。 (os error 5)          # uv 用管道读子进程 stdout，被沙箱拒绝

$ C:\...\langchain1.2\python.exe -m pip --python ".venv\Scripts\python.exe" install --dry-run pytest jieba
（>4 分钟无任何输出，与 docs/rag-self-built.md §5.8 的 "pip 空转" 现象一致，已 kill）
```

→ **装包通道在当前沙箱内属于"未验证可行"**（见 R2/R3），t3 必须先解决通道再谈补包；t2 的设计必须允许"pytest 进不了 `.venv`"这一情况（用 `importorskip` 让装配用例优雅跳过）。

---

## 6. 三条并行任务的接口约定（点 c）

### 6.1 t2 测试套件：确切文件布局 + 确切运行命令

**文件布局（`tests/` 为新建目录，19 个文件）**：

```
tests/conftest.py                       # 沙箱安全 tmp_path、hash embedder、隔离的数据目录
tests/test_rag_types.py                 # Chunk/RetrievedChunk/Answer 序列化、fingerprint、chunk_id 稳定性
tests/test_rag_config.py                # 非法参数校验、with_kb/kb_dir、ensure_within
tests/test_rag_loaders.py               # normalize_text / 多编码 txt / md 标题 / pdf 页码与空页 / docx 表格 / csv / 不支持后缀
tests/test_rag_chunking.py              # 结构切块、中文编号标题 vs 正文列表、代码块、重叠、硬切、max_chunks
tests/test_rag_embedding.py             # hash 后端性质、build_embedder 分派、local 降级与 allow_fallback=False、未知 provider
tests/test_rag_vectorstore.py           # 持久化/幂等/整篇替换/指纹两条路径/撕裂截断/clear/list_knowledge_bases 容错
tests/test_rag_bm25.py                  # tokenize 型号与编号、停用词、jieba 退化、idf、explain
tests/test_rag_hybrid.py                # RRF 名次融合、权重、overlap_count、top_k
tests/test_rag_rerank.py                # off/lexical/llm 分派、llm 回退 lexical、_parse_order 鲁棒性、rank 重算
tests/test_rag_generator.py             # build_context 截断、拒答、llm 不可用/报错、refusal marker、grounded_score 分离
tests/test_rag_pipeline.py              # ingest/add_texts/search/query/reset/stats、get_pipeline 缓存、chunk_size 告警
tests/test_rag_eval.py                  # load_eval_set、_norm_source、指标与拒答统计、format_report
tests/test_rag_cli.py                   # main(argv) 各子命令返回码与输出（capsys）
tests/test_rag_selftest.py              # subprocess 冒烟（仅 3.10 解释器可跑，否则 skip）
tests/test_agent_prompts_contract.py    # prompts.yml 键集合 == sub_agents_content 键集合；knowledge_base_agent 三工具名
tests/test_agent_rag_tools.py           # 三工具 .invoke() 全分支契约（§3.3）；不存在的 kb 不得创建目录（A4 回归）
tests/test_agent_assembly.py            # pytest.importorskip("deepagents") → import app.agent.main_agent 并断言 CompiledStateGraph
tests/test_integration_rag_agent.py     # @pytest.mark.e2e，仅 RAG_E2E=1 时执行（真大模型/真分派）
```

**确切运行命令（当前环境即可跑，纯 RAG + 工具层 + 契约层）**：

```powershell
# 1) 准备沙箱安全的临时根（pytest 自带 tmp_path 在本沙箱不可用，见 6.1.1）
New-Item -ItemType Directory -Path tmpdir\pytest-tmp -Force | Out-Null
$env:RAG_TEST_TMP          = (Resolve-Path tmpdir\pytest-tmp).Path
$env:PYTHONIOENCODING      = 'utf-8'
$env:RAG_EMBEDDING_PROVIDER = 'hash'      # 避免每次构造 pipeline 都尝试加载本地模型

# 2) 跑测试（conda 解释器 = 当前唯一有 pytest 的环境）
& 'C:\Users\lenovo\.conda\envs\langchain1.2\python.exe' -m pytest tests -q -p no:tmpdir -p no:cacheprovider
```

**装配层/E2E（需要 `.venv`）**：

```powershell
# 需要 .venv 里有 pytest（t3 负责；没有则用下面的兜底单行命令）
& '.\.venv\Scripts\python.exe' -m pytest tests -q -p no:tmpdir -p no:cacheprovider -k "assembly or integration"

# 兜底（不依赖 pytest 进 .venv）——装配是否成立的最小验证
& '.\.venv\Scripts\python.exe' -c "import app.agent.main_agent as m; print('ASSEMBLY_OK', type(m.main_agent).__name__)"
```

**E2E（默认跳过，显式开启）**：

```powershell
$env:RAG_E2E = '1'
& '.\.venv\Scripts\python.exe' -m pytest tests\test_integration_rag_agent.py -q -m e2e -p no:tmpdir -p no:cacheprovider
```

**建议写进 `pyproject.toml`（t2/t4 决定）**，让标准命令无需额外参数：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-p no:tmpdir -p no:cacheprovider"
markers = ["e2e: 需要真实大模型/网络/已入库知识库的端到端用例（默认跳过）"]
```

#### 6.1.1 `tests/conftest.py` 必须包含的三件事（实测验证过）

1. **沙箱安全 `tmp_path`**（根因见 R1：3.12/3.13 下 `os.mkdir(path, 0o700)` 之后自己都读不了，pytest 自带 tmpdir 插件、`--basetemp`、`tempfile.TemporaryDirectory/mkdtemp` 全部中招）：

```python
# tests/conftest.py
import os, uuid
from pathlib import Path
import pytest

ROOT = Path(os.environ.get(
    "RAG_TEST_TMP",
    str(Path(__file__).resolve().parents[1] / "tmpdir" / "pytest-tmp"),
))

@pytest.fixture
def tmp_path():
    d = ROOT / f"t{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)     # 不带 mode → 沙箱可读写
    return d
```

  配套：命令行必须带 `-p no:tmpdir`（否则 pytest 仍会建 `0o700` basetemp 并在 `sessionfinish → cleanup_dead_symlinks(basetemp)` 崩溃）。实测：`2 passed in 0.31s`，exit=0。

2. **数据目录隔离**：`app.core.config.settings` 是 `lru_cache` 模块级单例，**import 之后再改 `os.environ["RAG_DATA_DIR"]` 无效**（实测：先 import 工具再设环境变量，`list_knowledge_bases` 仍指向真实 `rag_data/`）。测试必须：
   - 用显式参数路径构造：`load_rag_config(data_dir=<tmp>, default_kb="test")` → `RagPipeline(kb="test", config=cfg, embedder=HashEmbedder())`；或
   - 在 conftest 顶层（任何 `app.*` import 之前）`os.environ.setdefault("RAG_DATA_DIR", ...)`；或
   - 用 `monkeypatch.setattr(app.rag.config.settings, "rag_data_dir", str(tmp))` + `app.rag.pipeline.clear_pipeline_cache()`。
   否则用例会写进真实 `rag_data/`（实测后果：凭空生成 `rag_data/ghost/` 空库目录）。

3. **`get_pipeline` 缓存清理**：任何改配置/注入 embedder 的用例前后都要 `app.rag.pipeline.clear_pipeline_cache()`；注入 embedder 的路径本身不缓存（`pipeline.py:365`）。

**t2 交付验收建议命令**（可直接照抄）：

```powershell
& 'C:\Users\lenovo\.conda\envs\langchain1.2\python.exe' -m pytest tests -q -p no:tmpdir -p no:cacheprovider
# 期望：全部通过；test_agent_assembly / test_integration_rag_agent 在无 deepagents 时 skip
```

**路线 3（纯 RAG 用例，最省事，已由我与 envi 各自实测）**：用**系统 Python 3.10**（免疫 `0o700` 问题，无需 `-p no:tmpdir`，也不需要自建 `tmp_path`），把 `.uv-cache/pytest-deps` 里的 pytest 通过 `PYTHONPATH` 借来：

```powershell
New-Item -ItemType Directory -Path tmpdir\<新的basetemp名字> -Force | Out-Null
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests -q -p no:cacheprovider --basetemp="G:\deepsearch-agents-main\tmpdir\<新的basetemp名字>"
# 实测：4 passed in 1.42s（exit=0）
```

注意：路线 3 **只适用于纯 RAG/工具层用例**——3.10 没有 `deepagents`，`test_agent_assembly.py` / E2E 必须回到 `.venv`（即 `importorskip` 会跳过）。另外 `--basetemp` 目录名要**每次换新**：pytest 对已存在的 basetemp 会 `rm_rf` 重建（`0o700`），被毁过的目录永久不可用（见 R1）。

### 6.2 t3 环境搭建：解释器路径与依赖安装命令

**目标环境（唯一推荐）**：

```
解释器：G:\deepsearch-agents-main\.venv\Scripts\python.exe      (Python 3.12.11, uv 0.7.20)
已满足：deepagents 0.5.7 / langchain 1.2.17 / langgraph 1.1.10 / langchain-core 1.3.3 /
        langchain-openai 1.2.1 / langchain-anthropic 1.4.3 / langchain-google-genai 4.2.2 /
        fastapi 0.136.1 / uvicorn 0.46.0 / pypdf 6.11.0 / numpy 2.4.4 / pyyaml / pydantic-settings /
        mysql-connector / tavily-python 0.7.24 / reportlab / python-docx / pandas / tiktoken
缺失（t3 需要补）：pytest  ← 必需
                   jieba   ← 强烈建议（BM25/词面重排质量）
                   sentence-transformers+torch ← 可选，风险高（见 R3/R7）
```

**安装命令（按可行性排序，前两条需要在沙箱外或用其它通道验证）**：

```powershell
# 通道 A：给 .venv 装 pip，再用它装（uv 创建的 venv 默认无 pip）
& '.\.venv\Scripts\python.exe' -m ensurepip --upgrade
& '.\.venv\Scripts\python.exe' -m pip install pytest jieba

# 通道 B：用 conda 环境的 pip 指定目标解释器（pip>=22.3 支持 --python）
& 'C:\Users\lenovo\.conda\envs\langchain1.2\python.exe' -m pip --python '.\.venv\Scripts\python.exe' install pytest jieba

# 通道 C（docs §5.8 已验证可用的逃生通道）：urllib 直连 PyPI 下载纯 Python wheel，
#   解压到 .venv\Lib\site-packages（或项目根 vendor/ 由 app.rag.vendor 兜底）
#   pytest 依赖链：pytest → pluggy, iniconfig, packaging, colorama(win)
#   jieba 为纯 Python 包

# 通道 D（不装包也能跑装配验证）：直接用 -c 单行命令，见 6.1
```

**注意（t3 必须遵守）**：
- 不要 `uv sync`（`uv.lock` 陈旧，会拉 torch，见 5.3）。
- 不要在 conda `langchain1.2` 里装 deepagents（会强升整套 langchain 栈，见 5.2）。
- `uv pip install` 在本沙箱不可用（E11）。
- 保持 `.env` 不加 `RAG_EMBEDDING_PROVIDER=local`；测试与演示统一 `hash`。
- 若坚持要真实 BGE：必须在**沙箱外**新建独立环境（例如 `uv venv .venv-agent --python 3.12`），装完 `sentence-transformers` 后**必须重新入库**或把 `RAG_EMBEDDING_PROVIDER=hash` 固定，否则 `fin` 库会被指纹校验拦死（R6）。

### 6.3 t4 集成任务：需要改动的确切文件

| 优先级 | 文件:行 | 改什么 | 为什么（依据） |
| --- | --- | --- | --- |
| P0 | `app/tools/rag_tools.py:86` | 阈值判定改用 `pipeline.grounded_score(results)`（或删除该提前返回，交给模型判断） | 与 `cli.py:205`、`eval.py:134`、`pipeline.grounded_score()` 统一；当前用的是重排分（A3，§5.10 同类 bug） |
| P0 | `app/tools/rag_tools.py:79-81, 121-123` | 知识库不存在时**不要创建目录**：先判断 `(config.data_path / kb_name).exists()`，或给 `get_pipeline`/`RagPipeline` 加 `create=False`，返回"知识库 X 不存在,可用:`list_knowledge_bases`" | 实测 `search_knowledge_base(kb_name='ghost')` 会创建 `rag_data/ghost/`（A4） |
| P0 | `app/agent/prompts.py:38` | 删除模块级 `print(sub_agents_content)` | stdout 污染（A2） |
| P0 | `app/skills/deep-research/SKILL.md:3,16,18-21,29` | 把 RAGFlow 段改为自研 RAG：`list_knowledge_bases` → `search_knowledge_base`/`ask_knowledge_base`，删掉 `get_assistant_list` 与"RAGFlow"字样 | 主智能体的技能目录里仍指示调用已删除的工具（A1，实测 grep） |
| P1 | `app/rag/selftest.py:60` | `TemporaryDirectory()` 改为使用 `RAG_SELFTEST_TMP`（默认 `tmpdir/selftest-tmp`）且用 `mkdir()` 不带 mode；或直接改为 pytest 用例并保留脚本入口 | 3.12/3.13 下 0 断言即崩（R1/E10）；当前只有 3.10 能跑 |
| P1 | `pyproject.toml` | 加 `[tool.pytest.ini_options]`（见 6.1）；`uv.lock` 要么刷新要么在文档里声明"不可用 `uv sync`" | 无 pytest 配置；lock 陈旧到会拉 torch |
| P2 | `app/rag/rerank.py:54-58` | 修正 docstring 里的权重（写的是 0.45/0.30/0.15/0.10，代码 `WEIGHTS` 是 0.40/0.25/0.20/0.10/0.05，且多一项 `char`） | 文档与代码不一致（仅注释） |
| P2 | `README.md:213-217` | RAGFlow 环境变量段落标注"仅回滚保留"，并补 `RAG_EMBEDDING_PROVIDER=hash` 的示例 | 当前示例会让人以为仍需 RAGFlow |
| P2 | `rag_data/README.md:32-52` | 示例命令补 `-k fin`（`default` 库并不存在）；说明"不传 kb_name 会创建空库" | 现有唯一库是 `fin`；`default` 不存在（E12） |
| — | `app/agent/subagents/knowledge_base_agent.py` | **不需要改**（已核对正确，且是回滚目标） | 见 §2.3 |
| — | `app/tools/ragflow_tools.py`、`app/core/config.py` 的 `ragflow_*` | **保持原样**（回滚路径），可选在文件头加 `# 已下线,不在 Agent 装配路径` 注释 | `docs/rag-self-built.md` §8 |

---

## 7. 风险与阻塞清单（点 6）

| ID | 等级 | 现象（实测） | 影响 | 规避方案 |
| --- | --- | --- | --- | --- |
| **R1** | blocker | 沙箱下 CPython 3.12.11/3.13.12 执行 `os.mkdir(p, 0o700)` 后**自己都无法 listdir/write**；3.10.9 正常。`pytest` 的 basetemp（`--basetemp` 也会先 `rm_rf` 再 `mkdir(0o700)`）、`tmp_path`、`tempfile.TemporaryDirectory/mkdtemp` 全部触发；`sessionfinish → cleanup_dead_symlinks` 还会让整个 pytest 进程以 exit=1 结束。**被碰过的目录永久不可访问、`Remove-Item` 也删不掉** | t2/t4 的测试与 `app/rag/selftest.py` 在 3.12/3.13 下不可用；`tmpdir/` 残留一堆不可删目录（我和 envi 各有若干） | 命令带 `-p no:tmpdir -p no:cacheprovider`；conftest 自建 `tmp_path`（默认 mode）；测试里禁用 `TemporaryDirectory`。实测 `2 passed`，exit=0 |
| **R2** | blocker | `uv pip install/list --python .venv\Scripts\python.exe` → `Failed to query Python interpreter ... 拒绝访问 (os error 5)`；`uv` 默认缓存目录也 `拒绝访问` | 无法用 uv 补包/查包 | 用 `UV_CACHE_DIR=<工作区>` 只能解决缓存；解释器查询仍被拒 → 见 R3 通道 |
| **R3** | blocker | `conda python -m pip --python .venv\Scripts\python.exe install --dry-run pytest jieba` 运行 >4 分钟无输出（与 `docs/rag-self-built.md` §5.8 的 "pip 空转" 一致） | `.venv` 补 `pytest` 的通道未验证 | 优先通道 A（`ensurepip`）；不行则通道 C（urllib 下 wheel 解压，§5.8 已验证网络可用、纯 Python 包可解压）；最差也用「`-c` 单行装配验证 + conda 跑纯 RAG 用例」的两段式 |
| **R4** | high | 写权限矩阵（envi 实测）：shell 可写＝工作区根、`tmpdir/`、`vendor/`、`output/`、`rag_data/`；**不可写**＝`app/`、`docs/`、`examples/`、`frontend/`、`docker/`、`.venv/`。但 `write`/`edit` 工具能写 `docs/`、`app/` | 用 shell 重定向写源码/文档会失败 | 代码与文档一律用 `write`/`edit` 工具落盘；shell 重定向只用于 `tmpdir/` |
| **R5** | high | `rag_data/` 只有 `fin`（238 chunks, hash:512）；`config.default_kb = "default"` 而 **`default` 库不存在** | `ask_knowledge_base(question)` 不带 `kb_name` → 命中"空库"提示（还会创建空目录）；E2E 断言会莫名失败 | E2E/演示一律显式 `kb_name="fin"`；或在 t4 里先 `python -m app.rag.cli ingest -k default -p docs/knowledge_base --embedding-provider hash`（注意 PDF 入库需 pypdf，3.10 下只有 `vendor/` 兜底） |
| **R6** | high | `RAG_EMBEDDING_PROVIDER` 默认 `local`；`.env` 未覆盖。今天靠"没装 sentence-transformers → 静默降级 hash"恰好与 `fin` 的 `hash:512` 一致 | 一旦 t3 装上 `sentence-transformers`：默认 provider 变 local → 三个工具全部返回"向量配置不匹配"，**端到端链路突然全断**且看起来像坏了 | t3/t4 在 `.env` 或测试环境显式固定 `RAG_EMBEDDING_PROVIDER=hash`；要试真实模型则新建库（不要动 `fin`）并重标定 `RAG_MIN_SCORE`（`docs/rag-self-built.md` §7.2 局限 3） |
| **R7** | medium | 无 `sentence-transformers`；`torch` 只存在于 conda 环境（未在 `.venv`）。hash 向量无语义能力 | 语义检索质量、拒答阈值标定都无法验收 | 把"语义质量"列为明确不在本期验收范围；用 `examples/rag_eval_set*.json` 只做**链路级**回归（词面重合可命中） |
| **R8** | medium | `127.0.0.1:3307` 未监听（`Test-NetConnection` False），`3306` 在监听且 `mysql` 服务 Running；`.env: MYSQL_PORT=3307` | 数据库子智能体端到端不可用 | E2E 若要覆盖 db 助手，先统一端口（`MYSQL_PORT=3306`）并确认 `deepsearch_db` 有数据；否则 E2E 只走知识库助手 |
| **R9** | medium | `dashscope.aliyuncs.com:443 / api.tavily.com:443 / pypi.org:443` 均可达（`Test-NetConnection` True），`.env` 有 `OPENAI_API_KEY`(35)/`LLM_DEEPSEEK`/`TAVILY_API_KEY` | 真模型调用可用但**非确定性 + 有成本** | E2E 用例只在 `RAG_E2E=1` 时跑，只断言结构（无异常、引用非空、`debug` 字段），不断言自然语言内容 |
| **R10** | medium | Windows PowerShell 管道与中文编码互踩（`docs/rag-self-built.md` §5.9；本次 stdout 中文显示为乱码） | 误判"测试失败" | `$env:PYTHONIOENCODING='utf-8'`；中文输出重定向到 UTF-8 文件再读 |
| **R11** | low | 队友并发写同一工作区（我观察到 `pytest-install.log`、`tmpdir/__probe2.py`、根目录 `pytest-of-lenovo/`）；`tmpdir/` 残留不可删 | 误删/覆盖别人产物 | 各自用独立子目录（如 `tmpdir/audit_integ`）；提交前 `git status` 核对 |
| **R12** | low | `PYTHONPATH` 未配置：从仓库根跑才 import 得动 | 从其它 cwd 运行会 `ModuleNotFoundError` | 统一在仓库根执行；或用 `python -m pytest`（rootdir 自动进 `sys.path`，需 `tests/` 存在 `__init__.py` 或用 rootdir 插入） |
| **R13** | low | `app/agent/prompts.py` 的 `print` 会在 import 期向 stdout 输出整段 YAML 配置 | 日志噪声；若上层用 stdio 协议通信还可能被污染 | 已在 t4 清单（P0） |

---

## 8. 附录：本次审计使用的命令与关键输出（可复现）

```powershell
# E1 默认解释器与 Agent import
python -c "import sys;print(sys.version, sys.executable)"
python -c "import app.agent.main_agent"                       # → ModuleNotFoundError: deepagents
# E2 RAG 子系统在 3.10
python -m app.rag.selftest                                    # → 18 通过 / 0 失败
# E3 .venv 装配
.\.venv\Scripts\python.exe -c "import app.agent.main_agent as m; print(type(m.main_agent))"
                                                              # → <class 'langgraph.graph.state.CompiledStateGraph'>
# E4 工具身份
.\.venv\Scripts\python.exe -c "import app.tools.rag_tools as m; [print(t.name, type(t).__name__, t.args) for t in (m.list_knowledge_bases, m.search_knowledge_base, m.ask_knowledge_base)]"
# E5 工具行为分支（.invoke vs 直接调用；空库/不存在/指纹不匹配/正常）
.\.venv\Scripts\python.exe C:\Users\lenovo\AppData\Local\Temp\audit_probe_tools2.py tmpdir\audit_integ\a tmpdir\audit_integ\b
# E6 键名一致性
.\.venv\Scripts\python.exe -c "from app.agent.prompts import sub_agents_content as s; print(sorted(s))"
                                                              # → ['db', 'rag', 'tavily']
# E7 旧链路是否在装配路径
rg -n "from app\.tools\.ragflow_tools|import ragflow_tools" app   # → 无匹配
rg -n "get_assistant_list|create_ask_delete" app                  # → 仅命中 app/tools/ragflow_tools.py 自身 + app/skills/deep-research/SKILL.md:20
# E8 pytest 基础设施
Get-Command pytest; foreach ($f in 'pytest.ini','setup.cfg','tox.ini','conftest.py') { Test-Path $f }   # 全 False
Select-String pyproject.toml -Pattern 'tool\.pytest'                                                    # 无
# E9 0o700 目录矩阵
python                                        C:\...\Temp\audit_mode_probe.py <base>   # 3.10 全 ok
.\.venv\Scripts\python.exe                    C:\...\Temp\audit_mode_probe.py <base>   # d_700 listdir/write 全 PermissionError
G:\miniconda3\python.exe                      C:\...\Temp\audit_mode_probe.py <base>   # 同上
# E10 pytest 绕过验证
New-Item -ItemType Directory -Path tmpdir\pytest-tmp -Force
$env:RAG_TEST_TMP=(Resolve-Path tmpdir\pytest-tmp).Path
& C:\Users\lenovo\.conda\envs\langchain1.2\python.exe -m pytest <diag> -q -p no:tmpdir -p no:cacheprovider   # → 2 passed in 0.31s
# E11 uv 通道
uv pip install --python ".venv\Scripts\python.exe" --dry-run pytest jieba    # → os error 5
# E12 现有知识库与默认库
.\.venv\Scripts\python.exe -m app.rag.cli list        # → 仅 fin: 238 chunks | hash:512
.\.venv\Scripts\python.exe -m app.rag.cli stats -k fin
# E13 版本约束
Get-Content .venv\Lib\site-packages\deepagents-0.5.7.dist-info\METADATA | Select-String '^Requires-'
```

**本次审计的验证命令（任务契约要求）结果**：

```
$ python -c "import app.tools.rag_tools as m; print(m.list_knowledge_bases.name, m.search_knowledge_base.name, m.ask_knowledge_base.name)"
list_knowledge_bases search_knowledge_base ask_knowledge_base        # exit=0

$ python -m app.rag.selftest
──── 自测结果:18 通过 / 0 失败 ────                                   # exit=0
```

---

## 9. 给三条并行任务的一句话交接

- **t2（测试套件）**：按 §6.1 的 19 个文件落地；命令必须带 `-p no:tmpdir -p no:cacheprovider` 与自建 `tmp_path`；工具用例一律 `.invoke({...})` 且断言返回字符串；A3/A4 两个缺陷写成失败用例（作为 t4 的验收靶子）。
- **t3（环境搭建）**：认定 `.venv`（3.12.11）为 Agent 环境，目标只是补 `pytest`（+`jieba`）；先解通道（§6.2 通道 A/B/C），**不要 `uv sync`、不要动 conda 环境、不要碰 torch/sentence-transformers**；给 t2 一个能跑的命令行。
- **t4（集成）**：按 §6.3 表格改，P0 四项（阈值分数口径、不存在 kb 不建目录、删 print、清 SKILL.md 的 RAGFlow 残留）是硬要求；改完用 §6.1 的三条命令自证（再交由独立验证任务复核）。

---

## 10. 补充（队员交叉验证后的增补，均为实测）

### 10.1 `.venv` 缺 `jieba` 对"知识库助手答得不好"的归因顺序

```
$ .\.venv\Scripts\python.exe -c "import importlib.util as u; print('jieba:', bool(u.find_spec('jieba')), '| ragflow_sdk:', bool(u.find_spec('ragflow_sdk')), '| deepagents:', bool(u.find_spec('deepagents')))"
jieba: False | ragflow_sdk: True | deepagents: True

$ .\.venv\Scripts\python.exe -m app.rag.cli search -k fin -q "2025年货币政策的总体基调" -n 2
WARNING app.rag.bm25: 未安装 jieba,中文将退化为单字切分。建议: pip install jieba
WARNING app.rag.embedding: 本地向量模型不可用(缺少 sentence-transformers 依赖...),降级为 hash 离线向量。
命中 2 条(向量后端:hash:512,重排:lexical)
```

因为 RAG 工具是**进程内调用**，用 `.venv` 跑 Agent 时 BM25 与 `lexical` 重排都会走 `app/rag/bm25.py:70-87` 的退化路径（单字 + 相邻二元组）。**检索不会报错、链路仍然通**，但中文词召回与重排质量下降。因此：

- 端到端结果不理想时，**先看 stderr 有没有上面两条 WARNING**，再怀疑接线；
- `.venv` 里补 `jieba`（纯 Python 包，无编译产物）是性价比最高的一步；
- **本次审计未安装任何包**（遵守任务范围）。若要补：优先 `vendor/` 或一个被忽略目录 + `PYTHONPATH` 的离线 wheel 解压路线（`docs/rag-self-built.md` §5.8 的逃生通道），并在 `t3` 的交付里写明"装了/没装 + 证据"。

### 10.2 澄清：3.12/3.13 上"项目代码本身没问题"

R1 的根因**只**发生在「显式带 mode 的 `os.mkdir(d, 0o700)`」与「`tempfile.mkdtemp/TemporaryDirectory`（内部就是 0700）」上。项目源码里创建目录都用默认 mode（`VectorStore.open` → `path.mkdir(parents=True, exist_ok=True)`、`cli`/`pipeline` 同理），所以：

- `.venv` 3.12.11 下 `app.rag.cli ingest/search/stats` 与工具层调用**都能正常工作**（本次实测 `cli list/stats/search -k fin` 全部成功）；
- 只有"测试脚手架/自测脚本"里的临时目录用法会踩雷（`app/rag/selftest.py:60` 用的正是 `TemporaryDirectory`，所以它在 3.12/3.13 下 0 断言即崩，在 3.10 下 18/0 —— 这是**脚本问题，不是 RAG 代码问题**）。

### 10.3 `ragflow_sdk` 的干扰排除

`.venv` 里确实预装了 `ragflow_sdk 0.25.2`（见 `dist-info` 列表），但 §2.3/E7 已用 grep 证明：**没有任何装配路径 import 它**。排查接线时不要因为它存在就认为旧链路还在生效；真正的装配入口只有 `app/agent/main_agent.py:26` 的 `knowledge_base_agent` → `app/tools/rag_tools.py`。

### 10.4 残留物说明（环境副作用，不影响交付）

`tmpdir/` 下已存在一批**不可删除**的目录（`pt*`、`pytest-of-lenovo`、`pdir1`、以及本次审计产生的 `audit_tmp*`、`audit_mode`、`audit_pytest`），成因是 R1；`git check-ignore` 确认 `tmpdir/` 已被 `.gitignore` 忽略（`.gitignore:17`），`.uv-cache/`（`line 39`）、`.agent-teams/`（`line 19`）同样被忽略，因此**不污染仓库**。后续任务请使用**新的** basetemp 目录名。
