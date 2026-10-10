# t8 独立验证报告：RAG 接入与测试体系的结论是否成立

> 任务：t8「独立验证 RAG 接入与测试体系的结论是否成立」（验证与评审工程师 verif，attempt 09ab14bf）
> 方法：**不采信上游自述**，对 t1/t2/t3/t7 的每条关键结论各自选一条**能被证伪的命令或探针**重跑；
> 对回归锁另做**变异（回退）测试**判断断言是否真的敏感；对技能系统做**零大模型花费的运行时取证**。
> 结论：**t1/t2/t3/t7 的核心结论全部成立（可复现）**；发现 2 条低严重度遗留（均不影响结论）与 3 条证据等级说明。
> 本报告只新增本文件，未改任何业务代码、未安装任何包。所有复现脚本与原始输出在 `tmpdir/verif_t8/`（已被 `.gitignore` 忽略）。

---

## 0. 一句话结论

| 被验证的结论 | 独立复现结果 | 判定 |
| --- | --- | --- |
| 默认 python 3.10 跑不了 Agent 链路，只能跑 RAG 子系统 | `python -c "import app.agent.main_agent"` → `ModuleNotFoundError: No module named 'deepagents'`（exit=1）；`python -m app.rag.selftest` → 18 通过/0 失败（exit=0） | ✅ 成立 |
| 唯一可用 Agent 环境是仓库 `.venv`（3.12.11 + deepagents 0.5.7） | 三条验收命令 + `import app.agent.main_agent` → `CompiledStateGraph`，全部 exit=0 | ✅ 成立 |
| pytest 套件 292 passed / 0 failed 且可重入 | 我独立跑 **2 次**：`292 passed in 8.04s` / `292 passed in 7.31s`，exit=0；零 skip | ✅ 成立 |
| A1–A4 与 D1 五个缺陷已修且被测试锁死 | 静态核对 + 运行时探针 + **回退变异**：五条回归锁在回退后全部转红 | ✅ 成立 |
| t7 的「真实大模型端到端接线跑通」 | 我用自己的脚本、**不同的问题**复跑官方入口 `run_deep_agent`：路由到知识库助手、7 次 RAG 工具调用、回答引用 `#130`，且 `#130` 文本经语料核对确实含该要点 | ✅ 成立 |
| 「技能系统在运行中生效」 | 打桩模型捕获真实 system prompt：含 skills 段，列出全部 7 个技能与 `SKILL.md` 路径 | ✅ **直接证据**（t7 未提供，本次补上） |
| 语义检索质量 | 未做、也无法做（`sentence-transformers`/`jieba` 缺失、`fin` 是 `hash:512`） | ⚠️ 所有文档均已如实标注为欠账，**未见夸大** |

---

## 1. 复现环境与命令（全部实跑）

```powershell
cd G:\deepsearch-agents-main
# 1) 离线测试套件（系统 3.10.9 + 工作区内的 pytest 离线副本）
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests -q                       # → 292 passed / exit=0（跑两次同结果）
# 2) RAG 子系统自测
python -m app.rag.selftest                      # → 18 通过 / 0 失败 / exit=0
# 3) Agent 环境三条验收
& '.\.venv\Scripts\python.exe' -c "import deepagents; print(deepagents.__version__)"          # → 0.5.7
& '.\.venv\Scripts\python.exe' -c "from deepagents import create_deep_agent; print('ok')"     # → ok
& '.\.venv\Scripts\python.exe' -c "import app.agent.prompts as p; print(sorted(p.sub_agents_content))"
                                                                                              # → ['db','rag','tavily']
& '.\.venv\Scripts\python.exe' -c "import app.agent.main_agent as m; print(type(m.main_agent).__name__)"
                                                                                              # → CompiledStateGraph
# 4) 我自己的端到端复跑（真实大模型 + 真实 fin 库，51.3s，exit=0）
& '.\.venv\Scripts\python.exe' tmpdir\verif_t8\run_verif_e2e.py tmpdir\verif_t8\e2e_out.txt
# 5) 变异/回退敏感度 + 技能注入取证
python tmpdir\verif_t8\probe_falsify.py
& '.\.venv\Scripts\python.exe' tmpdir\verif_t8\probe_skills.py
```

原始输出文件：`tmpdir/verif_t8/e2e_out.txt`（端到端）、`tmpdir/verif_t8/skills_evidence.txt`（技能取证）；
变异探针与 pytest 输出见本报告正文（已转录关键行）。

---

## 2. 测试体系（t2）独立复现

### 2.1 全量套件：两次同结果

```
$ python -m pytest tests -q                # RAG_TEST_TMP=tmpdir/pytest-tmp-verif-t8
292 passed in 8.04s                        # exit=0
$ python -m pytest tests -q                # RAG_TEST_TMP=tmpdir/pytest-tmp-verif-t8b（换目录）
292 passed in 7.31s                        # exit=0
```

* **0 skipped / 0 xfailed**：`test_rag_embedding.py` 里 3 处 `skipif`（依赖缺失分支）与所有 skip 分支在
  本机条件下均未触发 → 与 t2 自述的「转硬断言后 292 passed」一致；`xfail` 已全部移除，套件不再有静默吞断言的口子。
* **不写真实数据**：跑完后 `Get-ChildItem rag_data` 仍只有 `fin`、`README.md`；`git status --porcelain`
  与跑测试前逐字相同（`M pyproject.toml` + 4 个未跟踪 docs + `?? tests/`）。
* **离线**：守门用例 `tests/test_agent_rag_tools.py::test_suite_runs_offline_without_llm_credentials`
  存在且通过（断言 `llm_api_key == ""` 且 `ChatClient.available is False`）；conftest 的 autouse 夹具
  在 import 前 setdefault 环境、并清 `OPENAI_API_KEY`。

### 2.2 三条回归锁 + A4 锁：定向运行 + 变异测试

定向运行（6/6 PASSED，1.25s）：

```
tests/test_rag_loaders.py::test_normalize_preserves_blank_line_between_cjk_paragraphs PASSED
tests/test_rag_loaders.py::test_markdown_structure_survives_normalisation              PASSED
tests/test_agent_rag_tools.py::test_search_tool_gate_uses_vector_cosine                PASSED
tests/test_agent_rag_tools.py::test_search_on_missing_kb_reports_and_creates_nothing   PASSED
tests/test_agent_rag_tools.py::test_suite_runs_offline_without_llm_credentials         PASSED
tests/test_rag_eval.py::test_refusal_uses_vector_cosine_not_rank_score                 PASSED
```

**变异测试**（`probe_falsify.py`，只在内存里 monkeypatch 回退到修复前行为，不改仓库代码）：

```
[FIXED]        title='风控报告'  chunks=2  headings=['风控报告','风控报告 > 授信']
[FIXED]        asserts: title==ok True; len>=2 True; all(heading) True
[REVERTED-D1]  title='风控报告风控模型的准确率为 91.5%。' chunks=1 headings=['']
[REVERTED-D1]  asserts: title==ok False; len>=2 False; all(heading) False

[A3 FIXED]     refusal_present=True  has_0.010=True  has_0.950=False
               → 知识库 test 中没有与该问题足够相关的资料(最高匹配度 0.010,低于阈值 0.2)。
[A3 REVERTED]  refusal_present=False has_0.010=False has_0.950=True
               → 风控模型的准确率为 91.5%。
=> A3 锁敏感: True
```

判定：两条锁**真的敏感**，不是空断言；而且回退后**精确复现了 t2 描述的 D1 症状**
（标题吞正文 → `chunks=1`、`heading` 全空），说明 t2 对根因-症状链的描述属实。
A3 回退后工具会把「排序分 0.95 / 余弦 0.01」的无关片段当依据返回，也精确复现了旧缺陷。

### 2.3 代码级核对（非采信文档）

| 项 | 位置 | 实测 |
| --- | --- | --- |
| A1 技能文案 | `app/skills/deep-research/SKILL.md` | 已是「自研 RAG」文案；恰提到 `list_knowledge_bases` / `ask_knowledge_base` / `search_knowledge_base` 三个真实工具；无 `get_assistant_list` |
| A2 模块级 print | `app/agent/prompts.py` | 全文无 `print(`（`grep print\( app/**/*.py` 无该文件命中） |
| A3 门槛口径 | `app/tools/rag_tools.py:118` | `gate_score = pipeline.grounded_score(results)`；提示用 `gate_score:.3f`，不再用 `results[0].score` |
| A4 幽灵库 | `app/tools/rag_tools.py:39` | `get_pipeline(create=False)`；实测传 `ghost_verif_t8` → 提示「不存在,未创建也不会自动创建」，`before==after: True`、`ghost created: False` |
| D1 换行 | `app/rag/loaders.py:65` | `re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)`；实测 `normalize_text("甲\n\n乙") == "甲\n\n乙"` |
| D1 索引重建 | `rag_data/fin` | `all_chunks()` 实测 **248** 条，与 t7/t3 声称一致 |

工具契约独立复核：三工具均为 `StructuredTool`，`callable(tool) is False`（直接调用会 TypeError），
必须 `.invoke({...})`；错误分支返回中文字符串不抛异常 —— 与 t1 §4 结论一致。

---

## 3. 集成接线（t7）独立复现 —— 我自己的端到端运行

**与 t7 的差异**（避免「复述同一份证据」）：换问题（问「二是」而不是「一是」）、自写运行器
（`tmpdir/verif_t8/run_verif_e2e.py`）、自己读回 sqlite 消息。

```
[Monitor:session_created] …/projroot/output/session_verif-t8-e2e-1
[Monitor:context_report]  上下文装配: 记忆0条 历史0条 裁剪0条 tokens=276/276
[Monitor:assistant_call]  正在调用助手: 知识库助手
[Monitor:tool_start] ×7   1×list_knowledge_bases + 6×search_knowledge_base（全部为「自研RAG」工具）
[Monitor:task_result]     任务执行完成        [elapsed] 51.3s     exit=0
[db_message] role=user len=110 / role=assistant len=315
最终回答（从临时 sessions.db 读回）：
  **结论**：二是「充分发挥货币信贷政策导向作用」——扎实做好金融“五篇大文章”…
  **来源文件名**：中国人民银行2025年第四季度中国货币政策执行报告63页.pdf
  **片段编号**：#130（完整“二是”内容；该条起始部分另见 #129）
独立断言: tool_start_count=7  rag_tool_count=7  routed_to_kb_agent=True  answer_nonempty=True
          answer_has_fragment_number=True  run_deep_agent_raised=False
```

**引用可核对性（关键的抗幻觉检查）**——我回到语料里核对 `#130`：

```
total_chunks: 248
HIT chunk_index=130 source=中国人民银行2025年第四季度中国货币政策执行报告63页.pdf
    heading='二、下一阶段货币政策主要思路'
    text head: 二是充分发挥货币信贷政策导向作用。扎实做好金融“五篇大文章”，引导金融机构加力支持扩大内需、
               科技创新、中小微企业等重点领域…
```

模型回答与该片段文本一致（有压缩改写，无凭空新增结论）→ 该次回答**确有语料依据**，
不是「靠大模型先验知识凑答案」。同时 t7 声称的 `#128`（「一是」）与我的 `#129/#130` 落在同一 heading 下，互相自洽。

**沙箱适配的合规性核对**：`git diff --stat` = `pyproject.toml | 15 +++`（仅 pytest 配置），
`git status --porcelain -- app` 无输出 → 全程**未改 `app/` 业务代码**；适配只发生在 t7/我的运行脚本里。
`app/data/memory.db` 的 mtime 仍是 `2026-08-13 16:43`（本次未写库）。

---

## 4. 技能系统是否真的生效 —— t7 的唯一证据缺口，本次补齐

captain 的疑问点：t7 在 **import 之后**才改 `ma.project_root_path`，而
`file_backend = FilesystemBackend(root_dir=project_root_path)` 是 **import 时**构造的，
所以后端根并未被重定向。我实测确认：

```
[backend] type=FilesystemBackend root_dir=None cwd=G:\deepsearch-agents-main\app
```

即 `file_backend` 仍以 `app/` 为根（`root_dir` 属性为 None、真实根在 `cwd`）。
但这**恰好**让 `sources=['skills']` 继续相对 `app/` 解析 → 技能正常加载。

**直接证据（零大模型花费）**：

1. 构造级：`SkillsMiddleware(backend=ma.file_backend, sources=["skills"]).before_agent(...)`
   → `skills_loaded=7`、`skills_load_errors=None`，路径形如 `/skills/deep-research/SKILL.md`；
2. **运行级**：把 `ChatOpenAI._agenerate/_generate` 打桩（不联网），真跑一次 `run_deep_agent`，
   捕获实际发给模型的 messages：

```
[B] captured_model_calls=1
[B] call#0: messages=2 system_chars=11280
          skills_mentioned=['code-reviewer','deep-research','file-conversion','industry-analysis',
                            'memory-policy','report-authoring','sql-analytics']
          has_SKILL_md=True
```

system prompt 里确实出现了技能注入段（含全部 7 个技能名与 `SKILL.md` 路径），
且与 `app/agent/prompts.py` 的主提示 + deepagents 的 `SKILLS_SYSTEM_PROMPT` 拼接位置一致。

**判定：技能系统在端到端运行中确实生效（证据等级：直接证据 / 运行时捕获，非推断）。**
唯一仍未直接观测的是「模型是否在某一轮真的 `read_file` 了某个 `SKILL.md`」——
本次真实运行的 7 次工具调用全是 RAG 工具，没有读技能文件（符合「渐进式披露：按需读取」的设计），
这一点如实记为「按需触发，未在本轮观测到触发」。

---

## 5. 密钥与工作区卫生

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| `.env` 是否被跟踪 | `git ls-files \| Select-String '\.env$'` | 无（`.gitignore` 含 `.env`） |
| 已跟踪文件是否含密钥 | `git grep -nE "sk-…\|ghp_…\|ragflow-…\|AKIA…" HEAD` | 无命中 |
| **新增文档**是否含密钥 | 对 `docs/*.md` 正则搜 `sk-`/`ghp_`/`OPENAI_API_KEY=` 等 | 无命中 |
| 工作区是否被测试污染 | 跑完 pytest 后 `git status --porcelain` | 与跑之前逐字相同 |

---

## 6. 边界与遗留（均不阻塞，按证据等级如实分级）

| 编号 | 严重度 | 问题 | 证据 | 建议 |
| --- | --- | --- | --- | --- |
| **F1** | low | **pytest 执行通道不自足**：`pyproject.toml` 未声明任何 pytest 依赖，`.venv` 内无 pytest（实测 `import pytest` → ModuleNotFoundError），套件实际依赖 `.gitignore` 掉的 `.uv-cache/pytest-deps` 离线副本（从 conda 拷来）。干净检出（clean checkout）上「怎么装 pytest」这一步没有落进仓库 | `.venv\Scripts\python.exe -c "import pytest"` exit=1；`pyproject.toml` 无 test 依赖组；`git check-ignore` 覆盖 `.uv-cache/` | 在 `pyproject.toml` 增加 dev 依赖组（或 `[project.optional-dependencies].test = ["pytest>=9"]`），并在复现清单里补一句 `pip install pytest`；本沙箱的 `-p no:tmpdir` 规避作为脚注保留 |
| **F2** | low | `pyproject.toml` 声明了 `e2e` marker，但 `tests/` 内**没有任何用例使用** `@pytest.mark.e2e`（`RAG_E2E` 开关也无引用）→ 属未接线的死配置，不影响当前 292 条结果 | `Select-String -Path tests\*.py -Pattern "mark.e2e\|RAG_E2E"` 无命中 | 要么补一个被 marker 标住的 e2e 冒烟用例，要么删掉 marker 声明，避免读者以为有 e2e 门禁 |
| **F3** | info（证据等级说明） | t7 文档 §4.4 写「把 `main_agent.project_root_path` 指到可写根」是准确的（会话目录确实改到了 tmpdir），但**未写明** `file_backend` 的根在 import 时已绑定为 `app/`，因此「可写根重定向」并不覆盖文件工具。行为上无影响：技能仍正常注入（§4），当次运行也没生成文件 | 本报告 §4 的 `cwd=…\app` 运行级观测 | 可在 t7 文档补一句「backend 根未重定向；本次未触发文件写入，故不影响结论」 |
| **F4** | info | 文档自报行数普遍**低于**实际：测试计划 500→实测 631 行；环境报告 469→484；集成报告 223→275 | `[System.IO.File]::ReadAllText() -split "\`n"` 计数 | 无需处理（只是低估），但后续引用行数时以实测为准 |

**明确不判为缺陷的已披露边界**（同意 captain 的定调：如实披露 ≠ 隐瞒）：
`.venv` 缺 `jieba`（BM25/词面重排退化为单字+二元组）、`fin` 是 `hash:512` 兜底向量、
`sentence-transformers`/`torch` 未装、`0o700`/ACL 沙箱限制与 `-p no:tmpdir` 规避、两处运行期适配。
我逐一核对了 4 份新文档的相关段落（`rag-agent-env.md` §7.2/§7.3、`rag-agent-integration.md` §1/§6、
`rag-agent-test-plan.md` §7/R7、`rag-test-report.md` §9），**没有任何一处把 hash 兜底向量说成「语义检索已验证」**，
`docs/rag-embedding-validation.md` 确实不存在（`Test-Path` → False），欠账被反复明确标注。

---

## 7. 本轮**未**验证 / 无法验证的项（不要当成已通过）

1. **语义检索质量与阈值标定**：无真实 BGE 向量（缺 `sentence-transformers`/`torch`），
   `RAG_MIN_SCORE=0.2` 仍是按 hash 兜底向量标定的值 → **不能**沿用为生产口径。
2. **jieba 在场时的检索质量**：本机无 jieba，退化路径可跑通、真实分词路径未跑。
3. **数据库子智能体端到端**：`127.0.0.1:3307` 未监听（t1 已记），本轮同样未验证。
4. **`ask_knowledge_base` 在 Agent 循环内被调用**：我的运行里子智能体只用了 list+search（与 t7 相同），
   `ask` 的真实调用只有「直连工具层」的证据（t7 §4.5）。
5. **干净检出的一键复现**：受 F1 限制，本机复现依赖 `.uv-cache/pytest-deps` 存在。

---

## 8. 原始证据落点

| 文件 | 内容 |
| --- | --- |
| `tmpdir/verif_t8/e2e_out.txt` | 我自己的端到端运行完整输出（monitor 事件、sqlite 读回、独立断言） |
| `tmpdir/verif_t8/run_verif_e2e.py` | 我自写的官方入口运行器（含两处沙箱适配及理由） |
| `tmpdir/verif_t8/probe_falsify.py` | D1/A3 回退变异测试（内存 monkeypatch，不改仓库） |
| `tmpdir/verif_t8/skills_evidence.txt` | 技能注入的构造级 + 运行级取证 |
| `tmpdir/verif_t8/probe_skills.py` | 打桩模型捕获 system prompt 的取证脚本 |
| `tmpdir/verif_t8/probe_d1.md` | D1 下游症状样例（标题被正文粘住） |
| `tmpdir/verif_t8/memory_verif-t8-*.db`、`sessions_verif-t8-*.db` | 运行期重定向的临时记忆库/历史库（未写 `app/`） |

> 全部临时产物位于 `.gitignore` 覆盖的 `tmpdir/` 下；本报告是本次唯一新增的仓库文件。
