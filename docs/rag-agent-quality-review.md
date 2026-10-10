# t9 质量评审报告：RAG 接入原 Agent 项目的整体质量与遗留风险

> 任务：t9「评审 RAG 接入原 Agent 项目的整体质量与遗留风险」（评审人 verif，attempt f80f6d2b）
> 评审对象：`t1`（审计/测试计划）、`t2`（pytest 套件）、`t3`（环境）、`t7`（集成与端到端）四份交付 + 其对应的代码改动
> （`676c0f6` / `39b4b91` / `c811a96` / `6162141`）与 `pyproject.toml` 的 pytest 配置。
> 评审方法：读交付物 + 读代码 + **独立复跑**（不采信自述）；对「可复现」类主张另做**重建实验**。
> **判定：verdict = pass**（团队目标的四类验收全部达成，无阻塞项）。发现 7 条遗留，其中 1 条 medium、6 条 low（严重度口径对齐任务契约的 low/medium/high/blocker，不使用 info），均已给出 requiredFix，均不影响本轮结论成立。
> 本报告只新增本文件；未改任何业务代码、未装任何包。证据落点见 §7。

---

## 1. 团队目标 → 证据 → 判定

| 团队目标 | 证据（本次评审亲自跑/读） | 判定 |
| --- | --- | --- |
| 把自研 RAG 真正接入主智能体 + 知识库子智能体 | `app/api/server.py:38,189,194`：`POST /api/task` → `run_deep_agent` → 主图 → `task(subagent_type=知识库助手)`；`app/agent/subagents/knowledge_base_agent.py:24` 挂 `list/search/ask` 三个**自研 RAG** 工具；`app/prompt/prompts.yml:52-77` 的 rag 段只提这三个真工具、无 RAGFlow 残留；`app/skills/deep-research/SKILL.md` 已改自研 RAG 文案。真实大模型端到端：t7 三次 + **我自己的独立一次（换问题、51.3s、exit=0，引用 `#130` 并回语料核对属实）** | ✅ 达成 |
| 建立可复现的测试体系 | `python -m pytest tests`：**292 collected**；两条独立通道均复跑通过——系统 3.10.9 + 工作区离线副本 → `292 passed`（8.04s / 7.31s 两次一致）；conda 3.13 + 自带 pytest 9.0.3 → `292 passed`（4.11s）。离线守门用例通过；跑后 `rag_data` 与 `git status` 无变化；`-p no:tmpdir` 规避已落进 `pyproject.toml` | ✅ 达成（附带 F1 打包缺口） |
| 端到端集成验证 | t7 §4 五次运行记录 + 我独立复跑一次；两处沙箱适配只存在于运行脚本（`git diff --stat` 仅 `pyproject.toml +15`，`app/` 零改动） | ✅ 达成 |
| 独立验证与评审 | t8 独立验证报告（回退变异证明回归锁敏感、技能注入直接证据）；本 t9 评审 | ✅ 达成 |
| 交付物内部一致性（不夸大、不隐瞒） | 4 份文档对 `hash:512` 兜底向量 / 缺 `jieba` / 缺 `sentence-transformers` / 沙箱限制**逐条标注**，未出现「语义检索已验证」式表述；未验证项被显式列出 | ✅ 达成 |

---

## 2. 本次评审新增的独立证据（不是复述上游）

### 2.1 演示知识库 `fin` 可复现（重建实验）

```powershell
$env:RAG_DATA_DIR = '...\tmpdir\verif_t9\rag_data'      # 隔离目录，不碰真实 rag_data/
python -m app.rag.cli ingest -k fincheck -p "docs\knowledge_base\金融行业" --embedding-provider hash
```

```
知识库:fincheck(累计 248 chunks)   文件:3/3 入库,跳过 0,新增 chunk 248,替换旧 chunk 0,用时 7.03s
  + 中国人民银行2025年第四季度中国货币政策执行报告63页.pdf(46647 字符 -> 134 chunks)
  + 贝莱德BlackRock2026年全球投资展望报告17页.pdf(21267 字符 -> 53 chunks)
  + 长江商学院2025年投资者情绪问卷调查CKISS四季度报告63页.pdf(24944 字符 -> 61 chunks)
```

与仓库内 `rag_data/fin/meta.json`（`chunk_count: 248`、`embedding_fingerprint: hash:512`）以及 `fin` 库内
逐文件分布（134 / 53 / 61）**完全一致** → 演示语料可以从仓库里的 `docs/knowledge_base/金融行业` 原样重建，
「248 chunks」不是一次性手工产物，D1 修复后的切块结果可复现。

### 2.2 两条测试通道都能跑（不是只有一条路）

| 通道 | 命令 | 结果 |
| --- | --- | --- |
| 系统 3.10.9 + `.uv-cache/pytest-deps` 离线副本 | `python -m pytest tests -q` | `292 passed`（8.04s、7.31s 两次） |
| conda `langchain1.2`（3.13，自带 pytest 9.0.3） | `& $conda -m pytest tests -q` | `292 passed in 4.11s`，exit=0 |

`tests/__pycache__` 里同时存在 `cpython-310`/`cpython-313` 字节码，与两条通道的实际使用一致。

### 2.3 回归锁的敏感性（已在 t8 用回退变异证明）

* 退回 D1 旧行为 → `title` 粘正文、`chunks=1`、`heading` 全空（三条断言同时转红）；
* 退回 A3 旧口径 → 工具把「排序分 0.95 / 余弦 0.01」的无关片段当依据返回。
  → 结论：这些用例不是「照着当前实现写死」的空断言，而是真的在对行为做约束。

### 2.4 技能系统确实注入（t7 未给直接证据，t8 补齐）

`file_backend` 根在 import 时绑定为 `app/`（实测 `cwd=G:\deepsearch-agents-main\app`），因此
`sources=['skills']` 照常解析：`SkillsMiddleware.before_agent` → `skills_loaded=7`、`errors=None`；
打桩模型捕获真实 system prompt（11280 字符）→ 含全部 7 个技能名与 `SKILL.md` 路径。

---

## 3. 发现清单（均为非阻塞，含 requiredFix 与证据）

| ID | 严重度 | 问题 | 证据 | requiredFix |
| --- | --- | --- | --- | --- |
| **R1** | medium | **依赖声明与运行环境漂移**：`pyproject.toml` 声明 `jieba`、`sentence-transformers`，但 `.venv` 两者都没有，`uv.lock` 仍是旧的（含 `ragflow-sdk`，缺 `jieba`/`sentence-transformers`/`torch`）。一旦有人执行 `uv sync`/`uv lock`：轻则直奔 torch 下载（本机已知空转路径），重则装上 `sentence-transformers` 后默认 `RAG_EMBEDDING_PROVIDER=local` 与 `fin` 的 `hash:512` 指纹不符 → `VectorStore.open` 抛 `EmbeddingMismatchError` → 三个 RAG 工具**全部**只回「向量配置不匹配,需要重新入库」，交付物看起来"坏了" | `pyproject.toml:35-36`；`.venv` `import jieba/sentence_transformers` 均失败；`rag_data/fin/meta.json` = `hash:512`；`app/rag/vectorstore.py:90-97` 指纹校验；`app/rag/embedding.py:156-158` `local:BAAI/bge-small-zh-v1.5:512`；t1 §R6、t3 §7.4、t7 §6#6 均有记载 | 在**沙箱外**刷新 `uv.lock`（或至少把 `sentence-transformers` 降为可选 extra）；同时在 `.env.example` 的 RAG 段**取消注释并保留** `RAG_EMBEDDING_PROVIDER=hash` + `RAG_DEFAULT_KB=fin` 示例，并在 README「自研 RAG 配置」处写明「若安装了本地向量模型必须重新入库或固定 hash」。本轮已用文档兜底，属"已知并记录"，故不判阻塞 |
| **R2** | low | **测试依赖未声明**：仓库内没有任何 pytest 依赖声明（无 dev group / extra），`.venv` 内也无 pytest；套件实际依赖被 `.gitignore` 忽略的 `.uv-cache/pytest-deps` 离线副本（从 conda 复制而来）。干净检出上「装 pytest」这一步没有落进仓库，复现清单也没有这一步 | `.venv\Scripts\python.exe -c "import pytest"` → ModuleNotFoundError；`pyproject.toml` 全文无 `pytest`；`git check-ignore .uv-cache/` 命中 | 增加 `[dependency-groups] dev = ["pytest>=9"]`（或 `[project.optional-dependencies] test = ["pytest>=9"]`），并在 `docs/rag-test-report.md` 的复现清单里加一行 `pip install pytest`（沙箱内的离线副本路径保留为脚注） |
| **R3** | low | **`e2e` 门控名存实亡**：`pyproject.toml` 声明了 `e2e` marker，t7 §6#5 也建议「保留 `RAG_E2E=1` 门控」，但 `tests/` 里没有任何 `@pytest.mark.e2e` 用例、也没有任何 `RAG_E2E` 引用 → 文档描述的端到端门禁实际不存在 | `Select-String -Path tests\*.py -Pattern "mark.e2e\|RAG_E2E"` 零命中；`pyproject.toml:56-58` | 要么补一条被 marker 标住、默认 skip 的 e2e 冒烟用例（复用 t7/t8 的运行器思路），要么删掉 marker 声明并同步修正 t7 §6#5 的措辞 |
| **R4** | low | **无代码覆盖率度量**：t2 的「7 类测试点覆盖」靠映射表说明是**结构性论证**，不是行/分支覆盖率数字；环境里没有 `pytest-cov` | `pyproject.toml` 无 cov 依赖；文档中亦未出现覆盖率数字（未夸大，但缺少定量底座） | 若要在沙箱外做，装 `pytest-cov` 后补一条 `pytest --cov=app/rag --cov=app/tools/rag_tools` 的基线数字写进 `docs/rag-test-report.md`；沙箱内不可得则维持"结构性覆盖"的表述 |
| **R5** | low | **旧 RAGFlow 死代码/死配置仍在仓库**：`app/tools/ragflow_tools.py`、`app/ragflow/*.py` 在 **import 时**就构造 `RAGFlow` 客户端（模块级副作用），`app/core/config.py:46-47` 保留 `ragflow_api_url/api_key`，`.env`/`.env.example` 保留 `RAGFLOW_*`。当前**无任何文件 import 它们**（已 grep 确认），但有"误 import 即触发外部客户端构造"的隐患 | `app/tools/ragflow_tools.py:12-19`；`app/ragflow/knowledge_demo.py:11-17`；全仓库无 `from app.tools.ragflow_tools import` | 建议后续单独清理：删除或把模块级客户端构造改为函数内懒加载，并给两个文件加显式 `DEPRECATED` 头；`app/core/config.py` 的 ragflow 字段标注"仅回滚保留" |
| **R6** | low | **文档/注释级别的残留不一致**（t1 §11 的 P2 项）仍在：① `app/rag/rerank.py:55-56` docstring 写权重 `0.45/0.30/…`，实际 `WEIGHTS` 是 `0.40/0.25/0.20/0.10/0.05`（多一项 `char`）；② `rag_data/README.md:36-51` 仍用 `-k default` 举例、并保留"不传 kb_name 会创建空库"的旧告警（A4 已修，现会明确报"不存在"）；③ 各文档自报行数低于实测（500→631 / 469→484 / 223→275） | `app/rag/rerank.py:55,63`；`rag_data/README.md:36-51`；行数用 `ReadAllText().Split("\n")` 实测 | 修 docstring 权重；把 `rag_data/README.md` 的示例改为 `-k fin` 并把"幽灵库"告警替换为 A4 修复后的行为说明；后续引用行数以实测为准 |
| **R7** | low | **沙箱专用规避被误读为代码问题**：`app/rag/selftest.py:60` 用 `tempfile.TemporaryDirectory()` → 在本沙箱 3.12/3.13 下 0o700 自毁，只有 3.10 能跑；`pyproject.toml` 的 `-p no:tmpdir` 同理。**正常机器上都不需要**，文档（t1 §R1/E10、t3 §0o700 对照表、t7 §5#3）已如实标注 | `app/rag/selftest.py:60`；t3/t1 文档 | 无需改代码；若要跨沙箱一致，可给 selftest 加 `RAG_SELFTEST_TMP` 覆盖（t1 P1 建议），优先级低于 R1/R2 |

### 3.1 为什么不判 `needs_revision`

* R1 是**仓库既有的依赖/lock 漂移**（本团队未引入）且被三份文档显式记载并给出规避方式，当前环境实测完全可用；
* R2–R4 是**交付完整度**问题（打包与度量），不影响「套件在本机可复现、结论可验证」这一验收事实（两条独立通道各自实测通过）；
* R5–R7 是清理与措辞问题，均不影响功能与结论。
* 反向检查：**没有发现任何"声称已修但实际未修"的项**（A1–A4、D1 逐项复核并有变异测试佐证），也**没有发现夸大表述**（§1 最后一行）。

---

## 4. 明确不判为缺陷的已披露边界

`.venv` 缺 `jieba`（BM25/词面重排退化）、`fin` 为 `hash:512` 兜底向量、`sentence-transformers`/`torch` 未装、
沙箱 `app/` 不可写导致的两处运行期重定向、`0o700` 临时目录与 ACL 现象、`ragflow-sdk` 仍在 `.venv` 但不在装配路径上。
这些在 `docs/rag-agent-env.md` §7、`docs/rag-agent-integration.md` §6、`docs/rag-agent-test-plan.md` §7/R7、
`docs/rag-test-report.md` §9 中均被如实标注 —— 按 captain 口径，**如实披露 ≠ 缺陷**。

---

## 5. 本轮仍未验证（不得当作已通过）

1. 语义检索质量与 `RAG_MIN_SCORE` 重标定（`docs/rag-embedding-validation.md` 不存在，欠账成立）；
2. `jieba` 在场时的 BM25/词面重排真实路径；
3. 数据库子智能体端到端（MySQL 端口未对齐）；
4. `ask_knowledge_base` 在 Agent 循环内被调用（只有工具层直连证据）；
5. 干净检出的一键复现（受 R2 限制）；
6. 代码覆盖率数字（受 R4 限制）。

---

## 6. 结论

* 团队目标四项验收**全部达成**，交付物内部一致、证据可复现、边界披露诚实。
* 遗留 7 条（R1 medium、R2–R7 low）已按严重度分级并逐条给出 requiredFix，**不阻塞**本轮交付；
  其中 R1/R2 建议在进入生产或下次迭代前优先处理（各为一行配置级改动）。
* **verdict = pass**。

---

## 7. 证据落点

| 文件 | 内容 |
| --- | --- |
| `tmpdir/verif_t9/rag_data/fincheck/meta.json` | §2.1 重建实验产物（248 chunks，与 `fin` 逐文件一致） |
| `tmpdir/verif_t9/pytest_conda.txt` | conda 3.13 通道 `292 passed in 4.11s` 原始输出 |
| `docs/rag-agent-independent-verification.md` | t8 独立验证报告（变异测试、技能注入、独立 e2e） |
| `tmpdir/verif_t8/*` | t8 的运行器、探针与原始输出 |

> 所有临时产物均在 `.gitignore` 覆盖的 `tmpdir/` 下；本次评审唯一新增的仓库文件是本报告。
