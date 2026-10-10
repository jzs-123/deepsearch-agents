# RAG 工具接入原 Agent 项目：集成与端到端验证报告

> 版本：v1（t7 / integration）
> 环境：仓库根 `G:\deepsearch-agents-main`，解释器 `.venv\Scripts\python.exe`（CPython 3.12.11，uv 0.7.20）
> 前置依赖：t1 审计（`docs/rag-agent-test-plan.md`）、t2 测试套件（`docs/rag-test-report.md`）、t3 环境（`docs/rag-agent-env.md`）
> 本报告只记录**实跑**结果；所有结论附命令与关键输出。

---

## 1. 结论（TL;DR）

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | **RAG 工具层已真正接入原 Agent 项目**：主智能体 → `task(subagent_type=知识库助手)` → 子智能体 → `list_knowledge_bases / search_knowledge_base / ask_knowledge_base` → `app/rag` 流水线 → `rag_data/fin`，**三次真实调用全部走通** | §3、§4 |
| 2 | **官方入口也跑通了**：`app/api/server.py` 调用的 `run_deep_agent()`（含 ContextManager 装配、monitor 事件、`save_message` 持久化、流式执行）端到端成功，26.7s，最终回答带来源与片段编号 | §4.3 |
| 3 | t1 审计的 4 个集成缺陷 **A1–A4 已全部修复并提交**，t2 发现的 D1（`normalize_text` 吞换行）也已修复；修复后 `pytest tests` = **292 passed**，`python -m app.rag.selftest` = 18 通过 / 0 失败 | §2.2、§5 |
| 4 | **两处沙箱适配是本机特有、生产不需要**：① `run_deep_agent` 会在 `app/output/` 下建会话目录（`app/` 在本沙箱不可写）→ 需把 `main_agent.project_root_path` 指到可写根；② `history_manager._conn` 在 import 时就绑定了 `app/sessions.db`（不可写）→ 需换成临时库。未适配时分别报 `PermissionError [WinError 5]` 与 `OperationalError: attempt to write a readonly database` | §4.2 |
| 5 | 本次验证**验证的是"接线"，不是"语义检索质量"**：`.venv` 缺 `jieba`（BM25/词面重排退化为单字+二元组）、`fin` 库是 `hash:512` 兜底向量、`sentence-transformers` 未安装。答案正确是因为语料词面重合度高，不能据此宣称语义质量 | §6 |

---

## 2. 接入状态

### 2.1 装配链路（复核，与 t1 审计一致）

```
app/api/server.py  POST /api/task
  └─ run_deep_agent()            app/agent/main_agent.py:79
       ├─ ContextManager.assemble()（长期记忆 + 历史重放 + 预算裁剪）
       ├─ monitor 事件（session_created / context_report / assistant_call / tool_start / task_result）
       └─ main_agent.astream({"messages": [...]})      ← create_deep_agent(...) 编译产物
            ├─ tools: generate_markdown / convert_md_to_pdf / read_file_content / save_memory / recall_memory / forget_memory
            ├─ skills: app/skills（FilesystemBackend, virtual_mode）
            └─ subagents: [database_query_agent, network_search_agent, knowledge_base_agent]
                 └─ knowledge_base_agent            app/agent/subagents/knowledge_base_agent.py
                      ├─ name/description/system_prompt ← prompts.yml["sub_agents"]["rag"]
                      └─ tools: [list_knowledge_bases, search_knowledge_base, ask_knowledge_base]
                           └─ app/tools/rag_tools.py
                                └─ app/rag（config → loaders → chunking → embedding → vectorstore
                                          → bm25 → hybrid(RRF) → rerank → generator → pipeline）
                                     └─ rag_data/fin（248 chunks，embedding_fingerprint=hash:512）
```

`app/tools/ragflow_tools.py` 与 `app/ragflow/` 仍在仓库中，但**不在任何装配路径上**（t1 已用 grep 证明，本次复核未发现新增引用）。

### 2.2 t1 审计缺陷的修复状态（已提交，本次逐项复核）

| 编号 | 缺陷 | 修复提交 | 复核证据（本次实跑） |
| --- | --- | --- | --- |
| A1 | `app/skills/deep-research/SKILL.md` 等 4 个技能仍写 RAGFlow / 已删除的 `get_assistant_list` | `39b4b91` | `Select-String app\skills\deep-research\SKILL.md` → 第 3/18/22/23/27/33 行已是「自研 RAG」+ `list_knowledge_bases`/`ask_knowledge_base`/`search_knowledge_base`；全文无 `RAGFlow`/`get_assistant_list` |
| A2 | `app/agent/prompts.py:38` 模块级 `print(sub_agents_content)` | `39b4b91` | `import app.agent.main_agent` 的 stdout 中不再出现整段 YAML 配置（§4.1 输出仅含 monitor 事件） |
| A3 | `search_knowledge_base` 用重排分比 `min_score`（口径错误） | `676c0f6` | `app/tools/rag_tools.py:118` 已是 `gate_score = pipeline.grounded_score(results)`；`pytest tests` 中锁定该行为的用例通过 |
| A4 | 传不存在的 `kb_name` 会在 `rag_data/` 下建空目录 | `39b4b91` | 实测：`search_knowledge_base.invoke({"kb_name":"ghost"})` → `"知识库 'ghost' 不存在,未创建也不会自动创建。当前可用的知识库:fin。…"`；调用前后 `rag_data` 目录集合完全相同（`before==after: True`），`ghost created: False` |
| D1 | `app/rag/loaders.py` 的 `normalize_text` 用 `\s+` 吞掉中文之间的换行（t2 发现） | `c811a96` | `inspect.getsource(normalize_text)` → `re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)`；`normalize_text("甲\n\n乙") == '甲\n\n乙'` |

> 说明：A1–A4/D1 由 captain 直接提交修复，**t7 未修改任何 `app/` 业务代码**（见 §7）。

---

## 3. 端到端真实调用：环境与前置

```powershell
$env:PYTHONIOENCODING      = 'utf-8'
$env:RAG_EMBEDDING_PROVIDER = 'hash'   # 与 fin 的向量指纹 hash:512 一致，避免走 local→hash 的降级噪声
$env:RAG_DEFAULT_KB         = 'fin'    # 默认库 "default" 并不存在，工具不传 kb_name 时会命中"库不存在"提示
& '.\.venv\Scripts\python.exe' <脚本>
```

运行期实测环境（三次运行一致）：

```
[env] python=3.12.11 exe=G:\deepsearch-agents-main\.venv\Scripts\python.exe
[env] rag_data_dir=G:\deepsearch-agents-main\rag_data default_kb=fin provider=hash
[env] llm_model='deepseek-v4-pro' llm_api_key=set
```

（`llm_model` 来自 `.env` 的 `LLM_DEEPSEEK=deepseek-v4-pro`，经 `OPENAI_BASE_URL` 走 OpenAI 兼容协议；`fin` 库本次实测 248 个片段、`updated_at=2026-10-10 17:07:35`。）

---

## 4. 端到端运行记录

### 4.1 运行 1 —— 主智能体图（`main_agent.astream_events`）

问题：*"请使用「知识库助手」查询企业内部知识库：2025年第四季度中国货币政策执行报告里，中国人民银行实施了什么类型的货币政策？…"*

tool 调用序列（实测）：

```
[model] tool_calls=['task']
[tool_start] task | {"subagent_type": "知识库助手", ...}          ← 主智能体正确路由
[model] tool_calls=['list_knowledge_bases']
[tool_end] list_knowledge_bases | 可用知识库：
- fin: 248 个片段 | 向量模型 hash:512 | 更新时间 2026-10-10 17:07:35
默认知识库:fin
[model] tool_calls=['search_knowledge_base' × 4]
[tool_end] search_knowledge_base | 在知识库 fin 中检索到 5 个相关片段:
【片段 1】来源:中国人民银行2025年第四季度中国货币政策执行报告63页.pdf#1 匹配度:0.883 …
[elapsed] 41.5s
```

最终回答（节选）：

```
## 结论
《2025年第四季度中国货币政策执行报告》中，**中国人民银行实施的是「适度宽松的货币政策」**…
## 来源文件与片段编号
| #1（内容摘要） | "实施适度宽松的货币政策……" |
| #5 | "继续实施好适度宽松的货币政策。…" |
| #16（第一部分 货币信贷概况） | "实施好适度宽松的货币政策，加大逆周期和跨周期调节力度" |
| #35（第二部分 货币政策操作） | "实施好适度宽松的货币政策……" |
| #128（下一阶段货币政策主要思路） | "…继续实施好适度宽松的货币政策…" |
```

> 该次运行的 PowerShell 包装器报了 `exit=1`。已定位为**包装器伪影**：命令以「输出到 stderr 的原生命令 + `2>&1 |` 管道」作为**最后一条语句**时，pwsh 会因 `NativeCommandError` 记录把作业退出码置 1（对照实验：后续补一条 `Write-Output` 即回到 0，同一命令改用 `*>` 重定向也是 0）。脚本自身完整执行并写出结果文件（末行 `[written] …`），后续运行均改用 `*>` 并确认 exit=0。

### 4.2 运行 2 —— 主智能体图，第二个问题（可复现性）

问题：*"…该报告对「下一阶段货币政策主要思路」是如何表述的？…"*

```
[model] tool_calls=['task'] → subagent_type=知识库助手
[model] tool_calls=['list_knowledge_bases']
[model] tool_calls=['search_knowledge_base' × 3]   → #11 目录定位 / #49 / #0 / #1
[model] tool_calls=['search_knowledge_base' × 4]   → #8 / #131 / #130 / #128
[model] tool_calls=['search_knowledge_base' × 2]   → #132
[elapsed] 96.8s     exit=0
最终回答：结论 + 来源文件 + 片段编号表 #128–#133（并注明"目录定位依据片段 #11"）
```

`tool_start` 序列共 11 次（1 次 `task` + 1 次 `list_knowledge_bases` + 9 次 `search_knowledge_base`），命中片段编号与回答中的引用完全对应。

### 4.3 运行 5 —— **官方入口 `run_deep_agent`**（本任务的核心验收）

```powershell
& '.\.venv\Scripts\python.exe' tmpdir\integ_e2e\run_e2e_entrypoint2.py <out> t7-entrypoint-3
```

输出（实测，节选）：

```
[Monitor:session_created] 工作目录已创建: …/projroot/output/session_t7-entrypoint-3
[Monitor:context_report] 上下文装配: 记忆0条 历史0条 裁剪0条 tokens=260/260
[Monitor:assistant_call] 正在调用助手: 知识库助手
[Monitor:tool_start] 开始执行工具: 自研RAG知识库列表查询工具：list_knowledge_bases
[Monitor:tool_start] 开始执行工具: 自研RAG知识库检索工具：search_knowledge_base
[Monitor:tool_start] 开始执行工具: 自研RAG知识库检索工具：search_knowledge_base
[Monitor:tool_start] 开始执行工具: 自研RAG知识库检索工具：search_knowledge_base
[Monitor:task_result] 任务执行完成
[elapsed] 26.7s
[db_message] role=user len=101
[db_message] role=assistant len=178
===== 最终回答 =====
结论：一是保持金融总量合理增长。继续实施好适度宽松的货币政策，发挥增量政策和存量政策集成效应……持续营造适宜的货币金融环境。
来源文件名：中国人民银行2025年第四季度中国货币政策执行报告63页.pdf
片段编号：#128
```

这证明官方入口的完整链路（会话目录 → 上下文装配 → 子智能体分派 → RAG 工具 → 生成 → 消息持久化）全部走通。

### 4.4 官方入口在沙箱下的两次失败与适配（诚实记录）

| 运行 | 适配程度 | 结果 |
| --- | --- | --- |
| 运行 3 | 无适配 | `[EXCEPTION] PermissionError: [WinError 5] 拒绝访问。: 'G:\deepsearch-agents-main\app\output\session_t7-entrypoint-1'`，0.0s。原因：`main_agent.project_root_path = Path(__file__).parents[1] = app/`，`run_deep_agent` 在其下创建 `app/output/session_<id>/`，而**本沙箱 `app/` 不可写**（envi 的权限矩阵） |
| 运行 4 | 仅把 `main_agent.project_root_path` 指到 `tmpdir/integ_e2e/projroot` | 会话目录建好、上下文装配成功，随后 `[EXCEPTION] OperationalError: attempt to write a readonly database`（4.6s）。原因：`app/utils/history_manager.py:62` 在 **import 时** 就把 `_conn = _get_conn()` 绑到 `app/sessions.db`；`upsert_session`/`save_message` 用的是这个模块级连接，写 `app/` 被拒 |
| 运行 5 | 再把 `history_manager._conn` 换成临时库（并 `_init_db`） | **成功**，见 §4.3 |

> 这两处适配**只在"整个工作区被沙箱限制为不可写 `app/`"的本机成立**；正常机器上 `app/output/` 可写、`app/sessions.db` 可写，`run_deep_agent` 无需任何改动（`app/output` 本身也在 `.gitignore` 里，属设计内产物目录）。
> 另注：本次刻意**没有**让 `app/sessions.db`、`app/data/memory.db` 被写入（历史库与 `MEMORY_DB_PATH` 都重定向到 `tmpdir/`），运行后 `git status` 无新增改动。

### 4.5 第三个工具 `ask_knowledge_base` 的真实调用

Agent 循环里子智能体只用 `list_knowledge_bases` + `search_knowledge_base`（符合它 system_prompt 里「复杂问题多轮检索」的约束），因此单独用真实大模型跑了一次 `ask_knowledge_base`：

```
$ .\.venv\Scripts\python.exe -c "… m.ask_knowledge_base.invoke({'question':'2025年第四季度货币政策执行报告中，中国人民银行实施了什么类型的货币政策？','kb_name':'fin'})"
[Monitor:tool_start] 开始执行工具: 自研RAG知识库问答工具：ask_knowledge_base
2025年第四季度货币政策执行报告显示，中国人民银行实施了适度宽松的货币政策[2][4]。
引用来源：
[1] 中国人民银行2025年第四季度中国货币政策执行报告63页.pdf#0（标题:-,匹配度:0.7063）
[2] …#1（标题:-,匹配度:0.6976）
[3] …#41（标题:四、继续完善宏观审慎制度和管理框架,匹配度:0.4974）
[4] …#35（标题:第二部分货币政策操作,匹配度:0.4771）
exit=0
```

三个工具现在都有真实调用证据（`list`/`search` 在 Agent 循环内，`ask` 直连工具层）。

### 4.6 运行汇总

| 运行 | 入口 | 问题 | 关键调用 | 结论 | 耗时 | exit |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `main_agent.astream_events` | 货币政策类型 | task → 知识库助手 → list + 4×search | 「适度宽松」+ #1/#5/#16/#35/#128 | 41.5s | 1（包装器伪影，§4.1） |
| 2 | `main_agent.astream_events` | 下一阶段主要思路 | task → 知识库助手 → list + 9×search | 表述 + #128–#133（#11 定位） | 96.8s | 0 |
| 3 | `run_deep_agent`（无适配） | — | — | PermissionError（app/ 不可写） | 0.0s | 0（异常被脚本捕获） |
| 4 | `run_deep_agent`（仅根重定向） | — | session_created → context_report | OperationalError（历史库只读） | 4.6s | 0（同上） |
| 5 | `run_deep_agent`（根 + 临时历史库） | 「一是」是什么 | assistant_call → list + 3×search | 结论 + #128，消息已持久化 | 26.7s | **0** |
| tool | `ask_knowledge_base.invoke` | 货币政策类型 | 流水线内检索+生成 | 答案 + 4 条带分数引用 | ~10s | 0 |

---

## 5. 回归验证（集成后整套复跑）

```powershell
# 1) 全套测试（conda 解释器，见 docs/rag-test-report.md）
& 'C:\Users\lenovo\.conda\envs\langchain1.2\python.exe' -m pytest tests -q
# → 292 passed in 4.16s            exit=0

# 2) Agent 工具层 + 提示词契约（与本次集成直接相关）
& 'C:\Users\lenovo\.conda\envs\langchain1.2\python.exe' -m pytest tests\test_agent_rag_tools.py tests\test_agent_prompts_contract.py -q
# → 36 passed in 2.00s             exit=0

# 3) 离线自测（系统 Python 3.10；3.12/3.13 下会因沙箱 0o700 临时目录问题崩，见 t1 §R1）
python -m app.rag.selftest
# → ──── 自测结果:18 通过 / 0 失败 ────   exit=0

# 4) A4 修复复核（集成后的工具层行为）
# → "知识库 'ghost' 不存在,未创建也不会自动创建。当前可用的知识库:fin。…"
# → before==after: True | ghost created: False
```

---

## 6. 未覆盖与残留风险（本次验证的边界）

| # | 事项 | 现状 | 影响 |
| --- | --- | --- | --- |
| 1 | `.venv` 缺 `jieba` | 每次 RAG 调用都会打 `WARNING app.rag.bm25: 未安装 jieba,中文将退化为单字切分`；BM25 与 `lexical` 重排走单字+二元组退化路径 | 检索**能工作**（本次三次运行都命中正确片段），但中文召回/重排质量下降。若结果不理想，先查这条警告再看接线（t1 §10.1） |
| 2 | `fin` 是 `hash:512` 兜底向量 | `WARNING app.rag.embedding: 本地向量模型不可用…降级为 hash 离线向量`（未显式设 `RAG_EMBEDDING_PROVIDER=hash` 时） | 无真正语义能力；本次问答依赖词面重合。**语义检索质量不在本次验收范围** |
| 3 | `default` 知识库不存在 | 不设 `RAG_DEFAULT_KB` 时，工具不传 `kb_name` 会返回「知识库 'default' 不存在…当前可用：fin」（A4 修复后的正确提示，不会再建空库） | 演示/接入时应显式用 `kb_name="fin"`，或设 `RAG_DEFAULT_KB=fin`，或先 `ingest -k default` |
| 4 | `app/` 在本沙箱不可写 | `run_deep_agent` 需要两处运行期重定向（§4.4） | **仅本机沙箱**；正常机器无需任何改动 |
| 5 | 真模型依赖 | 三次运行都真实调用了 `deepseek-v4-pro`（`OPENAI_BASE_URL` 指向的兼容端点），有网络与费用 | 端到端用例不宜进默认 CI；建议保留 `RAG_E2E=1` 门控 |
| 6 | `sentence-transformers`/`torch` 未安装 | 本地语义向量路径未验证（t3 记录为欠账） | 若要切真实 BGE：必须新建库或重入库，并重标定 `RAG_MIN_SCORE`（t1 §R6） |
| 7 | 数据库子智能体 | `127.0.0.1:3307` 未监听（MySQL 在 3306） | 本次未涉及；若做全链路多助手 e2e 需先对齐端口 |
| 8 | 沙箱临时目录残留 | `tmpdir/` 下有若干 0o700 造成、永久不可删的目录 | 已被 `.gitignore` 忽略（`tmpdir/`），不污染仓库 |

---

## 7. 改动清单（本次 t7）

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 新增（交付物） | `docs/rag-agent-integration.md` | 本报告 |
| 未改动 | `app/**` | 集成缺陷 A1–A4/D1 已由 captain 在 `676c0f6` / `39b4b91` / `c811a96` 修复，t7 复核并实跑，未改任何业务代码 |
| 未改动 | `tests/**`、`pyproject.toml`（t2 的 `[tool.pytest.ini_options]`） | 只在复跑时使用 |
| 运行脚本（gitignored） | `tmpdir/integ_e2e/run_e2e.py`、`run_e2e_entrypoint2.py`、`e2e_run*.txt` | 复现用；`tmpdir/` 已被 `.gitignore` 忽略 |
| 未污染 | `app/sessions.db`、`app/data/memory.db`、`rag_data/fin/**` | 本次运行重定向了历史库与记忆库；`rag_data/` 只读访问（A4 复核断言目录集合不变） |

---

## 8. 复现步骤（照抄即可）

```powershell
# ── A. 主智能体图直连（真实大模型）──
$env:PYTHONIOENCODING = 'utf-8'
$env:RAG_EMBEDDING_PROVIDER = 'hash'
$env:RAG_DEFAULT_KB = 'fin'
& '.\.venv\Scripts\python.exe' tmpdir\integ_e2e\run_e2e.py `
    "G:\deepsearch-agents-main\tmpdir\integ_e2e\e2e_repro.txt" `
    "请使用「知识库助手」查询企业内部知识库：2025年第四季度中国货币政策执行报告里，中国人民银行实施了什么类型的货币政策？请给出结论与来源片段编号。"
Write-Output ("LASTEXITCODE=" + $LASTEXITCODE)   # 期望 0

# ── B. 官方入口 run_deep_agent（需两处沙箱适配，脚本已内置）──
& '.\.venv\Scripts\python.exe' tmpdir\integ_e2e\run_e2e_entrypoint2.py `
    "G:\deepsearch-agents-main\tmpdir\integ_e2e\e2e_repro_entry.txt" "t7-repro"
Write-Output ("LASTEXITCODE=" + $LASTEXITCODE)   # 期望 0

# ── C. 回归 ──
& 'C:\Users\lenovo\.conda\envs\langchain1.2\python.exe' -m pytest tests -q            # 292 passed
python -m app.rag.selftest                                                            # 18 通过 / 0 失败
```

> 注意：`run_e2e_entrypoint2.py` 里两行"适配"（`ma.project_root_path = WRITABLE_ROOT`、把 `history_manager._conn` 换成临时库）是沙箱专用；在正常机器上应把它们删掉，直接用 `app/api/server.py` 或 `run_deep_agent` 即可。
