# 自研 RAG 测试指南（自己动手验证能不能跑通）

面向「想知道这套 RAG 到底能不能跑、怎么自己验证」的使用者。按**从易到难**分 5 层，
每层都独立可跑，**上一层不过就别看下一层**。

> 设计原则：**第 1~2 层不装任何第三方包、不联网、不花钱**。
> 只有第 4 层（大模型生成）需要 API Key，第 5 层（真实向量模型）需要额外装依赖。

---

## 0. 先看结论：当前环境实测状态

| 能力 | 当前状态 | 依据 |
| --- | --- | --- |
| RAG 核心链路（解析/切块/检索/重排/评测） | ✅ 可跑 | `selftest` 18/18 通过 |
| 离线向量（hash 兜底） | ✅ 可跑 | 无需任何依赖 |
| 本地语义向量（BGE） | ⚠️ 未启用 | 缺 `sentence-transformers`，自动降级为 hash |
| 中文分词 BM25 | ✅ 可跑 | 系统 Python 已有 `jieba` |
| PDF 解析 | ✅ 可跑 | `pypdf`（系统包或 `vendor/` 兜底） |
| 大模型生成答案 | ✅ 需 API Key | 读 `.env` 的 `OPENAI_*` |
| 扫描件 OCR | ❌ 未实现 | 见 `docs/rag-self-built.md` 第六节 |

---

## 1. 冒烟自测：18 项断言（最快，约 1 秒）

```powershell
cd G:\deepsearch-agents-main
python -m app.rag.selftest
```

**期望输出**（结尾）：
```
──── 自测结果:18 通过 / 0 失败 ────
```

**它验证什么**：入库、切块结构、重复入库幂等、混合检索、BM25 专有名词命中、重排生效、
无依据拒答、向量指纹校验、持久化重载、多格式文件入库。

**安全**：它在 `tempfile` 临时目录里建库，**不会改动你的 `rag_data/`**。

**不通过怎么办**：先确认在项目根目录执行、且用的是 Python 3.10+（`python -V`）。
若报 `ModuleNotFoundError`，缺的是可选依赖，不是链路坏了——看下一层怎么判断。

---

## 2. 命令行走一遍真实链路（不调大模型、不花钱）

### 2.1 看有哪些知识库

```powershell
python -m app.rag.cli list
```
期望能看到仓库自带的示例库：
```
- fin: 248 chunks | 向量:hash:512 | 更新:...
```

### 2.2 入库一批文档

```powershell
# 支持 .pdf / .docx / .md / .txt / .csv，目录会递归
python -m app.rag.cli ingest -k mykb -p docs/knowledge_base/金融行业 --embedding-provider hash
```
期望：`文件:3/3 入库` + `新增 chunk N`。
`--embedding-provider hash` 是**显式指定离线向量**，避免依赖缺失带来的歧义。

### 2.3 只做检索（这一步最能看出检索质量）

```powershell
python -m app.rag.cli search -k mykb --embedding-provider hash -q "2025年货币政策的总体基调" -n 3
```
期望看到 3 条片段，每条都带：
- `score` / `rrf` / `dense` / `sparse` 四个分数
- `通道=['dense','sparse']`（说明两路都命中了）
- 来源文件名 + 片段编号 + 标题路径

**关键判读**：`dense` 是向量余弦、`sparse` 是 BM25 分、`rrf` 是融合分。
若 `通道` 只有 `['dense']`，说明关键词路没命中（可试试把专有名词写进查询）。

### 2.4 检索评测（拿数字，而不是凭感觉）

```powershell
python -m app.rag.cli eval -k mykb --embedding-provider hash -e examples/rag_eval_set_finance.json -n 5
```
期望（用仓库示例语料时）：
```
Hit@5: 100.00%   MRR: 1.0000   Recall@5: 100.00%   拒答准确率: 100.00%
```

> ⚠️ **注意这份指标的含义**：它是 **hash 兜底向量**下的结果，且评测集与语料用词高度重合，
> 说明的是「链路通、引用对、拒答准」，**不能证明语义检索质量好**。详见 `docs/rag-self-built.md` 7.2。

### 2.5 验证「无依据会拒答」

```powershell
python -m app.rag.cli search -k mykb --embedding-provider hash -q "公司食堂本周的菜单"
```
期望顶部出现提示：`向量余弦 X.XXXX 低于拒答阈值 0.2，ask 会判定为「知识库无依据」并拒答`。

---

## 3. 跑单元测试套件（292 用例）

### 3.1 首选：本机现成的离线 pytest 副本

```powershell
cd G:\deepsearch-agents-main
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
python -m pytest tests -q
```
**期望**：`292 passed`（约 8 秒）。

### 3.2 正规方式（干净检出的机器）

```powershell
uv sync --extra test      # 或 pip install -e ".[test]"
python -m pytest tests -q
```

### 3.3 如果你在别的机器上跑，遇到 pytest 报错

`pyproject.toml` 里已经配了 `addopts = "-p no:tmpdir -p no:cacheprovider"`，
**这是刻意的**：pytest 自带的 tmpdir 插件会用 `0o700` 建临时目录，在某些沙箱里会导致目录
自毁（详见 `docs/rag-agent-env.md`）。所以：

- **不要**再传 `--basetemp`，也**不要**设 `TEMP` 去"修"它；
- 需要临时目录的用例由 `tests/conftest.py` 自己用默认 mode 提供。

---

## 4. 端到端问答（这一步会调用大模型，会花钱）

```powershell
python -m app.rag.cli ask -k mykb --embedding-provider hash -q "2025年第四季度央行实施了什么类型的货币政策"
```
期望：一段**带 `[编号]` 引用**的回答 + 引用列表（文件、片段号、标题、匹配度）+ 调试信息。

**调试信息怎么读**：
```json
{"gate_score": 0.2678, "rank_score": 0.8579, "used_chunks": 3, "elapsed": 11.3}
```
- `gate_score` = **向量余弦**，用于「是否有依据」的判定；
- `rank_score` = **重排分**，用于排序；
- 两者**故意不同**：混用会导致拒答失效（这是踩过的坑，见 `docs/rag-self-built.md` 5.10）。

**不花钱的替代**：只跑第 2.3 节的 `search`，检索能力完全一样，只是不生成答案。

---

## 5. 启用真实语义向量（可选，需要装依赖）

```powershell
uv sync --extra local-embedding     # 装 sentence-transformers（会牵动 torch，体积大）
python -m app.rag.cli reset -k mykb --yes
python -m app.rag.cli ingest -k mykb -p docs/knowledge_base/金融行业
python -m app.rag.cli eval -k mykb -e examples/rag_eval_set_finance.json
```

> ⚠️ **两个必须知道的点**：
> 1. **换向量后端必须重新入库**。`meta.json` 里记了指纹（后端+模型+维度），
>    指纹不一致会抛 `EmbeddingMismatchError` 并给出重入库命令——这是**有意的保护**，
>    不是 bug（不同模型的向量空间不可比，混用只会得到无意义结果）。
> 2. **`RAG_MIN_SCORE` 需要重新标定**。它比较的是向量余弦，而 hash 与 BGE 的分数分布不同。
>    做法：用 `eval` 看「可答问题的最低分」与「不可答问题的最高分」，把阈值放在两者之间。

---

## 6. 你自己加文档时怎么测

```powershell
# 放进任意目录，递归入库
python -m app.rag.cli ingest -k mydocs -p D:\我的文档 --embedding-provider hash
python -m app.rag.cli stats  -k mydocs
python -m app.rag.cli search -k mydocs --embedding-provider hash -q "你的问题"
```

**入库报告怎么读**：
- `files_indexed / files_total`：成功/总数
- `files_skipped` + `warnings`：**跳过原因写在这里**（例如缺 `pypdf`），不要只看成功数
- `chunks_added` / `chunks_replaced`：新增 / 整篇替换的片段数

**调参入口**（`.env` 或 CLI 覆盖）：

| 参数 | 作用 | 调整建议 |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | 片段字符数 | 中文 400~600；答案常被切断就调大 |
| `RAG_CHUNK_OVERLAP` | 相邻片段重叠 | 80 起步 |
| `RAG_TOP_K` / `RAG_TOP_N` | 粗排候选数 / 精排后送模型条数 | 召回不够加 K，噪声多加 N 减小 |
| `RAG_RERANK_MODE` | `lexical` / `llm` / `off` | 用 `eval --no-rerank` 对比收益 |

---

## 7. 常见故障对照表

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'deepagents'` | 系统 Python 3.10 装不了 Agent 框架（要求 ≥3.11） | 只测 RAG 用系统 3.10 即可；测 Agent 链路用 `.venv\Scripts\python.exe` |
| 日志出现「降级为 hash 离线向量」 | 没装 `sentence-transformers` | 正常兜底，链路仍可用；想用真实向量见第 5 节 |
| 日志出现「未安装 jieba」 | 当前解释器缺 jieba | BM25/词面重排退化为单字，装上即可（纯 Python 包） |
| `EmbeddingMismatchError` | 查询用的向量后端与建库时不一致 | 按报错里的命令重入库，或把 `RAG_EMBEDDING_PROVIDER` 改回建库时的值 |
| `知识库 'x' 不存在,未创建也不会自动创建` | 传了不存在的库名 | 用 `list` 看可用库名（此行为是修复过的，见 `docs/rag-self-built.md`） |
| `缺少 pypdf 依赖` | 当前解释器没有 pypdf | `pip install pypdf`；本项目 `vendor/` 也已放了纯 Python 轮子作兜底 |
| `pytest` 报 tmpdir 相关 `WinError 5` | 解释器 ≥3.12 且用了 0o700 建目录 | 用第 3.1 节的命令（`addopts` 已规避），**不要**加 `--basetemp` |

---

## 8. 最小验证清单（复制即可）

```powershell
cd G:\deepsearch-agents-main
python -m app.rag.selftest                                  # 1) 期望 18 通过 / 0 失败
python -m app.rag.cli list                                   # 2) 期望看到 fin: 248 chunks
python -m app.rag.cli search -k fin --embedding-provider hash -q "2025年货币政策" -n 3   # 3) 期望 3 条带分数的片段
$env:PYTHONPATH='G:\deepsearch-agents-main\.uv-cache\pytest-deps'
python -m pytest tests -q                                    # 4) 期望 292 passed
python -m app.rag.cli eval -k fin --embedding-provider hash -e examples/rag_eval_set_finance.json -n 5  # 5) 期望 Hit@5 100%
```

五条全过 = 这套 RAG 在你这台机器上完全可用。

---

## 9. 尚未验证的能力（别当作已通过）

来自独立验证报告的明确清单，避免误读：

- **语义检索质量与 `RAG_MIN_SCORE` 重标定** —— 所有指标都基于 hash 兜底向量
- **jieba 在场时的 BM25 表现** —— 当前一直处于单字退化态
- **数据库子智能体的端到端** —— 只验证了知识库这条链路
- **`ask_knowledge_base` 在 Agent 循环内被调用** —— 只验证了 `search` 被循环调用
- **扫描件 OCR** —— 本轮未实现（Windows 内置 OCR 已探明可用，但尚未接入代码）
