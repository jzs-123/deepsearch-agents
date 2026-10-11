# 上传自己的文档 + 自己提问：操作手册

面向「我要用自己的资料做问答」的使用者。三种方式，**从最简单到最灵活**，任选其一。
所有命令都在项目根目录执行。

```powershell
cd G:\deepsearch-agents-main
```

> 前置：不需要装任何第三方包。向量后端用 `hash`（离线兜底）即可全流程跑通；
> 想要更好的语义效果，见本文末尾「换成真实语义向量」。

---

## 方式一：命令行（最常用，已实测）

### 1. 建库并上传文档

```powershell
python -m app.rag.cli ingest -k mydocs -p "D:\我的资料" --embedding-provider hash
```

- `-k mydocs`：知识库名字，随便起（英文/数字更稳），后续提问都用它
- `-p "D:\我的资料"`：**文件或目录都行**，目录会递归扫描
- `--embedding-provider hash`：显式用离线向量，避免依赖缺失带来的歧义

**支持格式**：`.pdf`、`.docx`、`.md`、`.markdown`、`.txt`、`.csv`

**实测输出**（拿一份 411 字的中文规范试的）：
```
知识库:mydocs(累计 3 chunks)
文件:1/1 入库,跳过 0,新增 chunk 3,替换旧 chunk 0,用时 0.02s
  + 客服运营规范.md(411 字符 -> 3 chunks)
```

**报告怎么读**：

| 字段 | 含义 |
| --- | --- |
| `文件:1/1 入库` | 成功数 / 总数 |
| `跳过 N` + 下面的 `! 警告` | **重点看这里**，跳过原因都写在这（例如缺 `pypdf`、扫描件无文本层） |
| `新增 chunk N` | 切出来的检索片段数 |
| `替换旧 chunk N` | 同名文档重复上传时会**整篇替换**，不会留下过期版本 |

### 2. 看库状态

```powershell
python -m app.rag.cli stats -k mydocs --embedding-provider hash
```

### 3. 提问（两种问法）

**问法 A：只检索，不调用大模型**（快、免费、能看原文依据）

```powershell
python -m app.rag.cli search -k mydocs --embedding-provider hash -q "夜间时段的响应时限是多久" -n 3
```

实测输出：
```
问题:夜间时段的响应时限是多久
命中 2 条(向量后端:hash:512,重排:lexical)

[1] score=0.7597 rrf=0.0295 dense=0.3632 sparse=3.7246 通道=['dense', 'sparse']
    来源:客服运营规范.md#0 2026 年客服中心运营规范（内部） > 一、响应时限
    内容:在线客服首次响应时限为 30 秒，超时需系统自动提醒。...每日 22:00 至次日 08:00
         为夜间时段，响应时限放宽至 4 小时。
```

**问法 B：让大模型组织成答案**（需要 `.env` 里配好 `OPENAI_API_KEY`，会花钱）

```powershell
python -m app.rag.cli ask -k mydocs --embedding-provider hash -q "夜间时段的响应时限是多久"
```

会输出「答案（带 `[编号]` 引用）+ 引用列表 + 调试信息」。

### 4. 后续增补文档

**直接再传一次即可**，同一个库会自动合并：

```powershell
python -m app.rag.cli ingest -k mydocs -p "D:\我的资料\新增制度.docx" --embedding-provider hash
```

- 新文件：新增片段
- 改过的同名文件：**整篇替换**旧片段（不会新旧并存）
- 完全相同的文件：自动去重，不重复入库

---

## 方式二：Python 代码（适合集成到自己程序里）

```python
from pathlib import Path
from app.rag.config import load_rag_config
from app.rag.pipeline import RagPipeline

config = load_rag_config(data_dir=Path("G:/deepsearch-agents-main/rag_data"))
pipe = RagPipeline(kb="mydocs", config=config)      # 默认向量后端（缺模型会自动降级 hash）

# 1) 上传：文件或目录
report = pipe.ingest(r"D:\我的资料")
print(report.files_indexed, "个文件入库，新增", report.chunks_added, "片段")

# 2) 上传：直接给一段文本（不落文件）
pipe.add_texts(["值班经理热线为 400-000-1234，仅工作日 9:00-18:00 接听。"], source="人工补充.md")

# 3) 检索（不调大模型，拿到结构化结果）
for hit in pipe.search("值班经理热线是多少", top_n=3):
    print(hit.rank, round(hit.score, 4), hit.chunk.source, hit.chunk.text[:50])

# 4) 带大模型的问答（返回 Answer 对象：答案 + 引用 + 调试信息）
answer = pipe.query("夜间时段的响应时限是多久")
print(answer.answer)
for c in answer.citations:
    print(f"[{c['index']}] {c['source']}#{c['chunk_index']} 匹配度={c['score']}")
```

实测（第 1、3 步）：
```
新增片段: 1
  [rank=1 score=0.8156] 人工补充.md#0 :: 值班经理热线为 400-000-1234，仅在工作日 9:00-18:00 接听
  [rank=2 score=0.5721] 客服运营规范.md#1 :: 满足以下任一条件必须升级至二线：...
```

> 指定离线向量（不依赖任何模型）时加一句：
> `from app.rag.embedding import HashEmbedder` → `RagPipeline(..., embedder=HashEmbedder(dim=512))`

---

## 方式三：让 Agent 助手去查（问答会带引用）

如果你的目的是「像聊天一样问」，而不是自己敲命令：

```powershell
# 让默认知识库指向你的库，然后走 Agent 入口
$env:RAG_DEFAULT_KB = "mydocs"
```

Agent 侧的三个工具（`app/tools/rag_tools.py`）：
- `list_knowledge_bases`：看有哪些库
- `search_knowledge_base`：只检索，返回原文+来源+匹配度
- `ask_knowledge_base`：检索 + 生成带引用答案

启动后端后，在对话框里直接问即可（例如「请让知识库助手查一下夜间时段响应时限」）。
注意 Agent 链路需要 Python ≥3.11：用 `.venv\Scripts\python.exe`。

---

## 常见问题

| 症状 | 原因与处理 |
| --- | --- |
| `跳过 xxx.pdf:缺少 pypdf 依赖` | 当前解释器没装 pypdf。`pip install pypdf`；本项目 `vendor/` 已放纯 Python 轮子作兜底 |
| `跳过 xxx.pdf:整份 PDF 未提取到任何文本` | **扫描件**，没有文本层，需要 OCR（当前版本尚未实现，见文末） |
| 检索结果不相关 | ① 用词与文档差太多（hash 向量只匹配词面）→ 换真实语义向量；② 查询里加上专有名词（型号/编号/指标名），关键词路更敏感 |
| 答案说「知识库中没有找到相关信息」 | 触发了拒答阈值（向量余弦低于 `RAG_MIN_SCORE`）。可以先用 `search` 确认是否真的检索不到 |
| `EmbeddingMismatchError` | 查询与建库用了不同向量后端。按报错里的命令重入库，或改回建库时的 `RAG_EMBEDDING_PROVIDER` |
| 文档更新后检索到旧内容 | 不会发生——同名文档是**整篇替换**；若怀疑，用 `stats` 看 chunk 数是否变化 |
| 中文被切成单字 | 日志出现「未安装 jieba」：`pip install jieba`（纯 Python 包） |

---

## 调参：让检索更准

在 `.env` 里改，或用 CLI 参数临时覆盖：

| 参数 | 默认 | 什么时候调 |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | 500 | 答案常被切断 → 调大（600~800）；噪声多 → 调小（300~400） |
| `RAG_CHUNK_OVERLAP` | 80 | 答案常跨片段 → 调大 |
| `RAG_TOP_K` | 10 | 召不回 → 调大（15~20） |
| `RAG_TOP_N` | 4 | 送给大模型的条数；噪声多 → 调小 |
| `RAG_RERANK_MODE` | `lexical` | 用 `eval --no-rerank` 对比，看重排是否有收益 |

**调完必须重建库**（切块参数变了，旧片段仍是老切法）：
```powershell
python -m app.rag.cli reset -k mydocs --yes
python -m app.rag.cli ingest -k mydocs -p "D:\我的资料" --embedding-provider hash
```

---

## 换真实语义向量（效果更好，需装依赖）

hash 向量只匹配**词面**，所以「转化率」问不出「成交比例」。要真正的语义检索：

```powershell
uv sync --extra local-embedding                # 装 sentence-transformers（会牵动 torch，体积大）
python -m app.rag.cli reset -k mydocs --yes
python -m app.rag.cli ingest -k mydocs -p "D:\我的资料"     # 不带 --embedding-provider，用默认 local
python -m app.rag.cli search -k mydocs -q "成交比例"          # 现在能召回「转化率」相关内容
```

**两个必读警告**：
1. **换后端必须重新入库**——`meta.json` 记了指纹，不一致会抛 `EmbeddingMismatchError`（有意的保护）。
2. **`RAG_MIN_SCORE` 需要重新标定**——它比的是向量余弦，hash 与 BGE 的分数分布不同。
   做法：用 `eval` 看「可答问题的最低分」和「不可答问题的最高分」，把阈值放在两者中间。

---

## 当前版本的能力边界（别误读）

- **扫描件/图片型 PDF 不支持**：没有文本层的 PDF 会被跳过。Windows 内置 OCR 已探明可用
  （含简体中文语言包），但**尚未接入代码**，属下一版计划。
- **纯图片文件（jpg/png）不支持**：入库只处理上表那六种格式。
- **PDF 表格是线性读取**：跨页表头会丢，数值问答不可靠（应走数据库/SQL）。
- **大规模数据**：向量检索是 numpy 全量点积，万级片段毫秒级；十万级以上需要换 ANN 索引。
