# Agent 运行环境说明（t3 交付物）

> 任务：`t3 [envtask] 搭建可运行的 Agent 环境（Python 3.12 隔离环境 + deepagents 依赖）`
> 作者：envi（t3 环境工程师）｜日期：2026-10-10｜工作区根目录：`G:\deepsearch-agents-main`
> 上游输入：`docs/rag-agent-test-plan.md`（t1 审计报告，本文多处引用其证据编号 E1–E13）

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 |
| --- | --- |
| Agent 链路用哪个解释器？ | **`G:\deepsearch-agents-main\.venv\Scripts\python.exe`（Python 3.12.11，uv 0.7.20 管理，隔离环境）** |
| 环境是否已就绪？ | **已就绪。三条验收命令全部通过（真实输出见 §3）**，无需新建环境、无需安装任何包 |
| t3 是否新建了环境？ | **没有。** 经 captain 批准，t3 口径改为「复用现成 `.venv` + 核实 + 出文档」 |
| 是否放宽了版本？ | **没有放宽任何版本。** `pyproject.toml` 的钉版本全部原生满足（见 §5） |
| 本环境能否装包？ | **本沙箱内不能。** `uv` 全部子命令与解释器探测被拒（os error 5），`.venv\Lib\site-packages` 对子进程不可写（见 §6） |
| 环境是否"完整"？ | **不完整，不要这样写。** 适合跑 Agent 链路；**不适合**跑 RAG 的本地语义向量路径（缺 `sentence-transformers`/`torch`），且缺 `jieba` 会让 BM25 与词面重排质量下降（见 §7） |

**下游可直接照用的一条命令**（装配链路自证）：

```powershell
& '.\.venv\Scripts\python.exe' -c "import app.agent.main_agent as m; print('ASSEMBLY_OK', type(m.main_agent).__name__)"
# → ASSEMBLY_OK CompiledStateGraph
```

---

## 1. 环境身份

| 项 | 值 |
| --- | --- |
| 解释器绝对路径 | `G:\deepsearch-agents-main\.venv\Scripts\python.exe` |
| Python 版本 | **3.12.11**（`main, Jul 8 2025`，MSC v.1944 64-bit） |
| CPython 来源 | uv 托管发行版 `C:\Users\lenovo\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none` |
| 创建者 | `uv 0.7.20`（工作区外 `C:\Users\lenovo\.local\bin\uv.exe`） |
| 隔离性 | `include-system-site-packages = false`（**不**继承系统 site-packages） |
| 已安装分发包总数 | **104** |
| 环境目录创建时间 | 2026-08-03 21:55（**早于本次团队任务**，属项目既有资产） |
| 与 `requires-python` 的关系 | `pyproject.toml` 声明 `>=3.12,<3.13`，`.python-version` = `3.12`，本环境 3.12.11 → **合规** |
| 是否被 git 忽略 | 是：`git check-ignore -v .venv` → `.gitignore:38:.venv/` |

`.venv\pyvenv.cfg` 原文（提供 provenance 证据）：

```
home = C:\Users\lenovo\AppData\Roaming\uv\python\cpython-3.12.11-windows-x86_64-none
implementation = CPython
uv = 0.7.20
version_info = 3.12.11
include-system-site-packages = false
prompt = deepsearch-agents
```

> `deepagents 0.5.7` 的 `Requires-Python: <4.0,>=3.11` → 满足验收标准「存在一个 Python>=3.11 的隔离环境」。

---

## 2. 为什么复用现成 `.venv` 而不新建环境

1. **它已经满足全部验收标准**，且是**唯一**同时满足三条硬约束的环境（详见 §3、§4、§8）。
2. **本沙箱内无法创建/维护环境**：`uv venv --python 3.12` 实测失败（`Failed to query Python interpreter ... os error 5`，连 uv 自己的托管解释器都探测不了）；`uv pip`、`uv lock` 同样被拒（§6）。新建环境在本沙箱内不是"更干净"，而是"做不到"。
3. **conda 环境被明确禁止改动**：往 `langchain1.2`（3.13.12）装 `deepagents` 会强升 `langchain 1.2.12→>=1.2.17`、`langchain-core 1.2.18→>=1.3.2`、`langchain-anthropic`，并新增 `langchain-google-genai`，且违反 `requires-python<3.13`（t1 §5.2）。
4. **`.venv/` 已被 `.gitignore:38` 忽略**，复用它不污染仓库；反之任务原建议的 `.venv-agent/` **不在** `.gitignore` 里（`.gitignore` 只覆盖 `.venv/`(38)、`venv/`(40)、`env/`(41)）——若真按原方案建 `.venv-agent/`，反而会污染 `git status`。

---

## 3. 验收命令与真实输出（全部实测，exit=0）

> 运行约定：在**仓库根目录** `G:\deepsearch-agents-main` 下执行；`$env:PYTHONIOENCODING='utf-8'` 可避免中文乱码。

### 3.1 验收命令 1

```powershell
& 'G:\deepsearch-agents-main\.venv\Scripts\python.exe' -c "import deepagents; print('deepagents ok')"
```

真实输出：

```
deepagents ok
EXIT=0
```

（stderr 有一条无关的第三方告警：`LangChainPendingDeprecationWarning: The default value of 'allowed_objects' will change...`，来自 `langgraph/checkpoint/serde/encrypted.py:5`，不影响结果。）

### 3.2 验收命令 2

```powershell
& 'G:\deepsearch-agents-main\.venv\Scripts\python.exe' -c "from deepagents import create_deep_agent; print('create_deep_agent ok')"
```

真实输出：

```
create_deep_agent ok
EXIT=0
```

### 3.3 验收命令 3

任务契约里该行末尾的引号/括号本身残缺（`...print(list(p.sub_agents_content))` 少了收尾），以下是**补全后**的实测命令：

```powershell
# 方式 A（推荐，从仓库根执行即可）
& 'G:\deepsearch-agents-main\.venv\Scripts\python.exe' -c "import app.agent.prompts as p; print(list(p.sub_agents_content))"

# 方式 B（从任意 cwd 执行，显式给 PYTHONPATH）
$env:PYTHONPATH='G:\deepsearch-agents-main'
& 'G:\deepsearch-agents-main\.venv\Scripts\python.exe' -c "import app.agent.prompts as p; print(list(p.sub_agents_content))"
```

真实输出（**方式 A 与方式 B 均实测通过**，`EXIT=0`）：

```
['tavily', 'db', 'rag']
```

> 注意：该命令会先输出约 **1739 字符**的配置字典——这是 t1 审计记录的缺陷 **A2**（`app/agent/prompts.py:38` 残留模块级 `print(sub_agents_content)`），不是本环境的故障。抑制噪声后的等价检查（`tmpdir/envi_t3_keys.py`）输出：
>
> ```
> sub_agents_content keys: ['tavily', 'db', 'rag']
> suppressed stdout chars (audit defect A2 noise): 1739
> main_agent_content keys: ['system_prompt']
> ```

### 3.4 超额验证（装配链路，比验收要求更强）

```powershell
& '.\.venv\Scripts\python.exe' -c "import app.rag.pipeline, app.tools.rag_tools, app.agent.main_agent as m; print('assembly:', type(m.main_agent).__name__)"
```

真实输出（末行）：

```
assembly: CompiledStateGraph
EXIT=0
```

即 `app.rag.pipeline`、`app.tools.rag_tools`（三个自研 RAG 工具）、`app.agent.main_agent` 均可在本环境 import 并成功装配主智能体。与 t1 审计证据 E3 一致。

---

## 4. 依赖版本表

### 4.1 关键包（实测于本环境）

| 包 | 版本 | 包 | 版本 |
| --- | --- | --- | --- |
| **deepagents** | **0.5.7** | **langchain** | **1.2.17** |
| langchain-core | 1.3.3 | langchain-openai | 1.2.1 |
| langchain-anthropic | 1.4.3 | langchain-google-genai | 4.2.2 |
| langchain-community | 0.4.1 | langchain-text-splitters | 1.1.2 |
| **langgraph** | **1.1.10** | langgraph-checkpoint | 4.0.3 |
| langgraph-prebuilt | 1.0.13 | langgraph-sdk | 0.3.14 |
| langsmith | 0.8.2 | pydantic | 2.13.4 |
| pydantic-settings | 2.14.1 | python-dotenv | 1.2.2 |
| PyYAML | 6.0.3 | openai | 2.35.1 |
| tiktoken | 0.12.0 | SQLAlchemy | 2.0.49 |
| fastapi | 0.136.1 | uvicorn | 0.46.0 |
| python-multipart | 0.0.28 | aiofiles | 25.1.0 |
| mysql-connector-python | 9.7.0 | tavily-python | 0.7.24 |
| python-docx | 1.2.0 | pypdf | 6.11.0 |
| Markdown | 3.10.2 | reportlab | 4.5.1 |
| requests | 2.33.1 | numpy | 2.4.4 |
| pandas | 3.0.3 | openpyxl | 3.1.5 |
| httpx | 0.28.1 | | |

### 4.2 缺失 / 需注意的包

| 包 | 状态 | 影响 |
| --- | --- | --- |
| `pytest` | **ABSENT** | `.venv` 内跑不了 pytest；测试套件用工作区的 `.uv-cache/pytest-deps` + 系统 3.10 运行，确切命令见 §7.4 |
| `jieba` | **ABSENT** | 中文 BM25 / 词面重排退化为单字+二元组（§7.2） |
| `sentence-transformers` | **ABSENT** | 本地语义向量不可用，`build_embedder` 降级为 `HashEmbedder`（§7.3） |
| `torch` | **ABSENT** | 同上（未尝试安装，见 §6.4） |
| `ragflow-sdk` | **0.25.2（预装）** | 旧链路已下线；t1 §2.3/E7 已用 grep 证明**没有任何装配路径 import 它**。非本任务安装，建议保留不动 |

> 本环境**没有 pip**（uv 创建的 venv 默认不装 pip），因此 `pip freeze` 无法直接执行。等价的冻结命令见 §9.2；`§9.3` 给出完整 104 行清单。

---

## 5. 版本放宽说明

**本任务未放宽任何版本。** 任务步骤 2 列出的钉版本全部原生满足：

```
deepagents==0.5.7          ✅ 0.5.7
langchain==1.2.17          ✅ 1.2.17
langchain-openai==1.2.1    ✅ 1.2.1
langgraph==1.1.10          ✅ 1.1.10
langchain-core (>=0.2.0)   ✅ 1.3.3（pyproject 要求 >=0.2.0）
```

`deepagents==0.5.7` 的依赖约束（t1 §5.2 读取其 METADATA）：`langchain-core>=1.3.2`、`langchain>=1.2.17`、`langchain-anthropic>=1.4.3`、`langchain-google-genai>=4.2.2`、`langsmith>=0.8.0` —— 本环境分别提供 `1.3.3 / 1.2.17 / 1.4.3 / 4.2.2 / 0.8.2`，**全部达标且无冲突**，故无需以"能让 deepagents import"为理由放宽任何包。

`tavily-python==0.7.24`、`python-dotenv==1.2.2`（pyproject 钉版本）亦精确命中。

> 任务步骤 2 明确要求"**不要安装 ragflow-sdk**"。本任务**未安装**它；它是 `.venv` 既有内容（§4.2），保留不动、也不删除。

---

## 6. 安装/复现命令 与 失败尝试

### 6.1 历史命令：仓库内无记录（如实说明）

`.venv` 创建于 2026-08-03，**早于本次任务**，仓库内没有任何构建脚本或日志记录其确切安装命令。可获得的 provenance 证据只有 `pyvenv.cfg`（uv 0.7.20 + uv 托管 CPython 3.12.11）与"已装集合与 `pyproject.toml` 钉版本完全一致"这一事实。

因此下面给出的 §6.2 是**按 pyproject 钉版本复原的重建配方**，标注了每一项的验证状态。

### 6.2 重建配方（在**有网络、非受限**的终端执行）

```powershell
# 0) 前置：uv 0.7.20（本机路径 C:\Users\lenovo\.local\bin\uv.exe）
uv --version

# 1) 创建隔离环境（Python 3.12，符合 pyproject requires-python ">=3.12,<3.13"）
uv venv .venv --python 3.12

# 2) 安装与本环境一致的依赖（显式钉版本，避免解析器顺手拉起 torch）
uv pip install --python .venv\Scripts\python.exe `
  "deepagents==0.5.7" "langchain==1.2.17" "langchain-openai==1.2.1" "langgraph==1.1.10" `
  "langchain-core>=0.2.0" "langchain-community>=0.2.0" "pydantic>=2.0.0" "pydantic-settings>=2.0.0" `
  "python-dotenv==1.2.2" "pyyaml>=6.0" "openai>=1.0.0" "tiktoken>=0.6.0" "sqlalchemy>=2.0.0" `
  "fastapi>=0.100.0" "uvicorn[standard]>=0.20.0" "python-multipart>=0.0.6" "aiofiles>=23.0.0" `
  "mysql-connector-python>=8.0.0" "tavily-python==0.7.24" "python-docx>=1.0.0" "pypdf>=3.0.0" `
  "markdown>=3.5" "reportlab>=4.0.0" "requests>=2.31.0"

# 3) 自证（三条验收命令，见 §3）
```

| 步骤 | 本沙箱内状态 | 证据 |
| --- | --- | --- |
| `uv --version` | ✅ 可执行 | `uv 0.7.20 (251420396 2025-07-09)` |
| `uv venv --python 3.12` | ❌ **不可执行** | `× Failed to inspect Python interpreter from managed installations at ...cpython-3.12.11...` → `拒绝访问。 (os error 5)` |
| `uv pip install --python ...` | ❌ **不可执行** | `Failed to query Python interpreter at '.venv\Scripts\python.exe'` → `拒绝访问。 (os error 5)` |
| 三条验收命令 | ✅ 全部通过（§3） | 环境已存在，无需安装 |

> **可复现性的诚实边界**：本沙箱**无法**执行 §6.2 的步骤 1–2（原因见 6.3，属沙箱限制而非配方问题）；**已在沙箱内验证的是**"环境已就绪且满足全部验收"。请需要真正重建环境的人在普通终端执行 §6.2。

### 6.3 失败尝试 1：`uv` 在本沙箱完全不可用（含 UV_CACHE_DIR 假设已排除）

```
$ uv pip list --python .venv\Scripts\python.exe
error: Failed to inspect Python interpreter from provided path at `.venv\Scripts\python.exe`
Caused by: Failed to query Python interpreter at `.venv\Scripts\python.exe`
Caused by: 拒绝访问。 (os error 5)                                  # exit=2
```

**已排除的假设**：曾怀疑是 uv 缓存目录在工作区外所致。按建议显式改为工作区内缓存后**仍然失败**：

```
$ $env:UV_CACHE_DIR='G:\deepsearch-agents-main\.uv-cache\uv'
$ uv pip list --python .venv\Scripts\python.exe
uv 0.7.20 (251420396 2025-07-09)
error: Failed to inspect Python interpreter from provided path at `.venv\Scripts\python.exe`
Caused by: 拒绝访问。 (os error 5)                                  # exit=2
```

结论：**不是缓存问题**。失败发生在 uv 通过管道探测解释器（读子进程 stdout）这一步，被沙箱拒绝。受影响的子命令已实测包括：

| uv 子命令 | 结果 |
| --- | --- |
| `uv pip list --python <venv>` | ❌ os error 5 |
| `uv pip install --python <venv>` | ❌ os error 5 |
| `uv venv --python 3.12` | ❌ os error 5 |
| `uv lock --check` | ❌ os error 5 |

即 `uv` 在本沙箱**既不能管理既有环境，也不能创建新环境**。

### 6.4 失败尝试 2：pip 通道空转；site-packages 不可写

- 用 conda 的 pip 指定目标解释器（`pip --python .venv\Scripts\python.exe install --dry-run pytest jieba`）**>4 分钟无任何输出**，已终止（t1 审计 R3；与 `docs/rag-self-built.md` §5.8 记录的 "pip 装 torch 系依赖空转" 现象一致）。
- `.venv\Lib\site-packages` 对**子进程不可写**（实测 `PermissionError: [Errno 13] Permission denied`）→ 即便有 pip 也装不进去。
- **本任务未安装 `sentence-transformers`/`torch`**：captain 已决定不装，且本机两次实测 pip 在装 torch 系依赖时空转（CPU 上升、无网络连接）。故 torch 一条记为**"按决策未尝试"**，而非"尝试失败"。

### 6.5 失败尝试 3：`uv sync` —— 明确禁止执行

**这是给后来维护者的重要警告。**

- `pyproject.toml` 现在包含 `sentence-transformers>=3.0.0,<6.0.0` 与 `jieba>=0.42.1`；
- 但 `uv.lock` 是**旧的**：`grep '^name = ...'` 显示 lock 内仍有 `ragflow-sdk`，而**没有** `jieba` / `sentence-transformers` / `torch` 条目；
- 因此运行裸 `uv sync` 会判定 lock 过期并重新解析，直奔 `sentence-transformers → torch` 的下载与安装——正是本机已知的**卡死/空转路径**。

```powershell
# ✗ 不要执行
uv sync
```

若必须重新同步，请先显式排除重依赖，例如 `uv sync --no-install-package sentence-transformers`（**未在本沙箱验证**，仅作方向建议），或按 §6.2 显式钉版本安装。

### 6.6 已知环境限制（**非阻塞**）：子进程写权限、`0o700` 自毁 与 跨会话删除失败

**不要把它写成"缺少完全权限的阻塞项"**——t3 验收与 pytest 都已有可用的规避方案。

实测的**子进程**写权限矩阵（`write`/`edit` 工具不受此限，可正常写 `docs/`、`app/`）：

| 路径 | 子进程可写 |
| --- | --- |
| 工作区根、`tmpdir/`、`vendor/`、`output/`、`rag_data/` | ✅ |
| `app/`、`docs/`、`examples/`、`frontend/`、`docker/`、`.venv/` | ❌（含已存在文件） |

本环境存在**两个真实并存、互不否定**的现象：

**现象 ①：`0o700` 目录自毁（按解释器区分）** —— **3.12/3.13 下使用 0o700 建目录的 API（pytest 的 tmpdir 插件、`tempfile.mkdtemp`）在本环境的多数会话中会自毁：目录建立后连创建者都无法访问；3.10 免疫。** 实测对照（同进程内「建 0o700 → 立刻 listdir / 写文件」）：

| 解释器 | 默认模式 `mkdir` | `os.mkdir(d, 0o700)` 后 listdir | `tempfile.mkdtemp()` |
| --- | --- | --- | --- |
| 系统 3.10.9 | ok | ok | ok |
| `.venv` 3.12.11 | ok | **WinError 5** | **WinError 5** |
| conda 3.13.12 | ok | **WinError 5** | **WinError 5** |

**现象 ②：跨会话删除失败（三个版本都会出现）**

删除**其他会话**创建的目录会返回 `WinError 5`，**3.10 / 3.12 / 3.13 均会出现**，与 ACL 相关；本会话按 skill 规定记为 **undiagnosed 并停止**，未做修复、未手工修改任何权限。

> **两点措辞要求（避免误导后来者）**
> 1. **不要**写成"3.12+ 一律不能用 `mkdtemp`"这类普适定律：项目自身 RAG 代码在 3.12 下**完全正常**（`VectorStore` 用默认 mode 的 `mkdir`；captain 已在 `.venv` 3.12.11 下实测入库+检索 `chunks_added=1, hits=1`），出问题的只是"用 0o700 建目录的测试脚手架"。
> 2. **也不要**弱化"在 3.12 上可稳定复现"这一事实：tester 在其会话中未复现现象 ①，很可能是路径/所有权上下文不同，**不构成反证**。
>
> **规避方案**：**测试套件已用 `-p no:tmpdir` 从设计上规避整类问题**，不依赖 basetemp 技巧（确切运行命令见 §7.4）。已毁目录（`tmpdir\pt`、`ptbase`、`pdir1`、`ptfresh`、`ptconda`、`tmpdir\pytest-of-lenovo`）请勿复用。

---

## 7. 能力边界（**不要把本环境写成"已完整"**）

### 7.1 ✅ 可用：Agent 链路与 RAG 进程内检索

- `deepagents` / `create_deep_agent` / `app.agent.prompts` 全部通过（§3）；
- `app.rag.pipeline` + 三个 RAG 工具 + `app.agent.main_agent` 装配成功（`CompiledStateGraph`，§3.4）；
- t1 审计另已实测：`.venv` 下 `python -m app.rag.cli list/stats/search -k fin` 均成功。

### 7.2 ⚠️ 质量下降：缺 `jieba` → BM25 与词面重排退化

实测（`app/rag/bm25.py` 的分词退化路径）：

```
$ .\.venv\Scripts\python.exe -c "import app.rag.bm25 as b; print('tokens:', b.tokenize('型号A13 准确率为 3.2.1'))"
未安装 jieba,中文将退化为单字切分。建议: pip install jieba
tokens: ['型', '号', '型号', 'a13', '准', '确', '率', '准确', '确率', '率为', '3.2.1']
```

可见中文退化为「单字 + 相邻二元组」：`准确率` 没有被切成词（只得到 `准确`/`确率`/`率为` 这类伪词），**专有名词之外的词召回与重排质量会下降**。检索**不会报错、链路仍然通**，但 `lexical` 重排的判别力变弱。

### 7.3 ❌ 未验证：RAG 的本地语义向量路径

- 缺 `sentence-transformers` 与 `torch` → `build_embedder(..., allow_fallback=True)` 会捕获异常并降级为 `HashEmbedder`，只留一条 `WARNING app.rag.embedding: 本地向量模型不可用(...),降级为 hash 离线向量。检索质量会明显下降。`（t1 §4.4 实测）；
- 现存唯一知识库 `rag_data/fin` 是 **`hash:512`** 建的（`meta.json` 实测：`embedding_provider: "hash"`、`chunk_count: 248`、`updated_at 2026-10-10 17:07:35`）→ 一切"语义检索质量"结论在本轮**不成立**；
- 该库已因切块缺陷 **D1 修复**而重建（提交 `c811a96`：`normalize_text` 不再用 `\s+` 吞掉中文之间的换行，此前会导致"标题吞正文、整篇退化成 1 个 chunk"；chunk 数 238→248，指标无回退）。**但引擎仍是 hash 向量，与语义质量无关**——D1 是切块/解析层修复，不能据此认为语义向量路径已可用；
- 因此本环境**不能用来说明语义向量链路可用**，该路径属**已知欠账**。

### 7.4 ⚠️ `.venv` 内没有 pytest（测试用工作区自带的 pytest 副本运行）

`.venv` 与系统 Python 3.10 均无 pytest。测试套件（**t2 已完成，其报告为 290 passed / 2 xpassed / 0 failed**）落在**仓库根 `tests/`**，由 captain 复制到工作区的 `.uv-cache/pytest-deps/` 提供 pytest 9.0.3；`pyproject.toml` 的 `addopts` 已含 `-p no:tmpdir`，**不需要 `--basetemp`，也不需要改 TEMP**（本文件仅同步该结论，未独立复跑该套件；独立复跑属验证任务范围）：

```powershell
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
python -m pytest tests -q
```

（另一条等价路线是只读借用 conda `langchain1.2` 的 pytest 9.0.3。装配层/E2E 用例需要 `deepagents` → 必须用 `.venv`。）

### 7.5 一句话边界

> **`.venv` = Agent 链路可用；RAG 检索链路可跑但质量打折（无 jieba）；本地语义向量路径未验证（无 sentence-transformers/torch）。**

---

## 8. 证据：用户 conda 环境**未被修改**

本任务**从未**在 conda 环境中安装、升级或删除任何包。基线对照（与 t1 审计 §5.1 记录的数值一致）：

```
$ G:\miniconda3\Scripts\conda.exe env list
langchain1.2             C:\Users\lenovo\.conda\envs\langchain1.2
                         G:\anaconda3
base                     G:\miniconda3

$ C:\Users\lenovo\.conda\envs\langchain1.2\python.exe -c "<打印关键包版本>"
python                 3.13.12
langchain              1.2.12      ← 与审计基线一致（未被升级到 1.2.17）
langchain-core         1.2.18
langgraph              1.1.2
fastapi                0.135.1
langchain-openai       1.1.11
torch                  2.11.0
transformers           5.3.0
numpy                  2.5.2
pypdf                  6.10.2
deepagents             ABSENT      ← 未安装（按约束不装）
sentence-transformers  ABSENT
jieba                   ABSENT
pytest                 9.0.3
```

> 该环境在本次任务中仅被**只读**使用（读取版本号）。未执行 `conda install` / `pip install` / `conda create --clone` 等任何写操作。

---

## 9. 复现与验证命令清单

### 9.1 三条验收 + 装配自证

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$py = 'G:\deepsearch-agents-main\.venv\Scripts\python.exe'

& $py -c "import deepagents; print('deepagents ok')"                                    # → deepagents ok
& $py -c "from deepagents import create_deep_agent; print('create_deep_agent ok')"       # → create_deep_agent ok
& $py -c "import app.agent.prompts as p; print(list(p.sub_agents_content))"              # → ['tavily','db','rag']
& $py -c "import app.agent.main_agent as m; print('ASSEMBLY_OK', type(m.main_agent).__name__)"  # → ASSEMBLY_OK CompiledStateGraph
```

### 9.2 冻结清单（本环境无 pip，用 importlib.metadata 等价替代）

```powershell
& 'G:\deepsearch-agents-main\.venv\Scripts\python.exe' -c "
import importlib.metadata as md
for d in sorted(md.distributions(), key=lambda x: (x.metadata['Name'] or '').lower()):
    print('%s==%s' % (d.metadata['Name'], d.version))
"
```

### 9.3 完整包清单（104 项，实测输出）

```
aiofiles==25.1.0            aiohappyeyeballs==2.6.1     aiohttp==3.13.5
aiosignal==1.4.0            annotated-doc==0.0.4        annotated-types==0.7.0
anthropic==0.100.0          anyio==4.13.0               attrs==26.1.0
beartype==0.22.9            bracex==2.6                 certifi==2026.4.22
cffi==2.0.0                 charset-normalizer==3.4.7  click==8.3.3
colorama==0.4.6             cryptography==48.0.0        dataclasses-json==0.6.7
deepagents==0.5.7           distro==1.9.0               docstring_parser==0.18.0
et_xmlfile==2.0.0           fastapi==0.136.1            filetype==1.2.0
frozenlist==1.8.0           google-auth==2.50.0         google-genai==1.75.0
greenlet==3.5.0             h11==0.16.0                 httpcore==1.0.9
httptools==0.7.1            httpx==0.28.1               httpx-sse==0.4.3
idna==3.13                  jiter==0.14.0               jsonpatch==1.33
jsonpointer==3.1.1          langchain==1.2.17           langchain-anthropic==1.4.3
langchain-classic==1.0.7    langchain-community==0.4.1  langchain-core==1.3.3
langchain-google-genai==4.2.2  langchain-openai==1.2.1  langchain-protocol==0.0.15
langchain-text-splitters==1.1.2  langgraph==1.1.10      langgraph-checkpoint==4.0.3
langgraph-prebuilt==1.0.13  langgraph-sdk==0.3.14       langsmith==0.8.2
lxml==6.1.0                 Markdown==3.10.2            marshmallow==3.26.2
multidict==6.7.1            mypy_extensions==1.1.0      mysql-connector-python==9.7.0
numpy==2.4.4                openai==2.35.1              openpyxl==3.1.5
orjson==3.11.9              ormsgpack==1.12.2           packaging==26.2
pandas==3.0.3               pillow==12.2.0              propcache==0.5.2
pyasn1==0.6.3               pyasn1_modules==0.4.2       pycparser==3.0
pydantic==2.13.4            pydantic-settings==2.14.1   pydantic_core==2.46.4
pypdf==6.11.0               python-dateutil==2.9.0.post0  python-docx==1.2.0
python-dotenv==1.2.2        python-multipart==0.0.28    PyYAML==6.0.3
ragflow-sdk==0.25.2         regex==2026.4.4             reportlab==4.5.1
requests==2.33.1            requests-toolbelt==1.0.0    six==1.17.0
sniffio==1.3.1              SQLAlchemy==2.0.49          starlette==1.0.0
tavily-python==0.7.24       tenacity==9.1.4             tiktoken==0.12.0
tqdm==4.67.3                typing-inspect==0.9.0       typing-inspection==0.4.2
typing_extensions==4.15.0   tzdata==2026.2              urllib3==2.6.3
uuid_utils==0.14.1          uvicorn==0.46.0             watchfiles==1.1.1
wcmatch==10.1               websockets==16.0            xxhash==3.7.0
yarl==1.23.0                zstandard==0.25.0
```

### 9.4 原始证据文件（被 `.gitignore` 忽略的 `tmpdir/`，可直接复用）

| 文件 | 内容 |
| --- | --- |
| `tmpdir/envi_t3_evidence.txt` | 解释器身份、`pyvenv.cfg`、三条验收命令与真实输出、装配自证、关键包版本、jieba 退化证据 |
| `tmpdir/envi_t3_evidence2.txt` | conda 基线对照、完整 104 包清单、uv 失败证据、`uv.lock` 陈旧证据、`.gitignore` 覆盖情况 |
| `tmpdir/envi_t3_versions.py` | 打印关键包版本的脚本 |
| `tmpdir/envi_t3_keys.py` | 抑制 A2 噪声后打印 `sub_agents_content` 键 |
| `tmpdir/repro700.py` | `os.mkdir(...,0o700)` 根因复现脚本（三个解释器通用） |

---

## 10. 给后来维护者的注意事项

1. **不要执行裸 `uv sync`** —— `uv.lock` 陈旧（含 `ragflow-sdk`、缺 `jieba`/`sentence-transformers`），会直奔 `torch` 拉取（§6.5）。
2. **不要往 conda `langchain1.2` 装 `deepagents`** —— 会强升整套 langchain 栈并违反 `requires-python<3.13`（§2）。
3. **不要因为看到 `ragflow_sdk 0.25.2` 就以为旧链路还在生效** —— 它不在任何装配路径上（t1 §2.3/E7）。
4. **若给 `.venv` 装上 `sentence-transformers`，必须同时处理指纹问题**：`RAG_EMBEDDING_PROVIDER` 默认 `local`，装上后默认后端变化会让三个工具全部返回"向量配置不匹配"，端到端链路看起来像"坏了"（t1 R6）。要么固定 `RAG_EMBEDDING_PROVIDER=hash`，要么新建知识库、不要动 `fin`。
5. **`.venv-agent/` 不在 `.gitignore`**（只覆盖 `.venv/`、`venv/`、`env/`）——要新建环境请用 `.venv/` 或 `venv/`，或先补 `.gitignore`。
6. **本沙箱内装包不可行**：uv 全系子命令被拒、`.venv\Lib\site-packages` 对子进程不可写、pip 空转（§6.3/6.4）。需要补包请在有网络、非受限的终端操作，或走 `docs/rag-self-built.md` §5.8 的离线 wheel 解包通道。
7. **`.venv` 无 pip**，请勿假设 `python -m pip` 可用；冻结清单用 §9.2 的命令。
