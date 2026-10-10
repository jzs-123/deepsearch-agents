# rag_data —— 自研 RAG 知识库数据目录

这个目录存放 `app/rag` 自研检索增强模块的知识库数据,**属于运行时数据,不属于源码**。
每个子目录是一个独立知识库,目录名就是知识库名(`-k` / `RAG_DEFAULT_KB` 的值)。

## 目录结构

```
rag_data/
├── default/                 # 默认知识库
│   ├── chunks.jsonl         # 每个检索片段一行 JSON(可 grep,可增量追加)
│   ├── vectors.npy          # 与 chunks 顺序严格对应的 (n, dim) float32 向量矩阵
│   └── meta.json            # 向量后端指纹、维度、切块参数、更新时间
└── <你的知识库名>/
```

## 为什么用 numpy 全量检索而不是 FAISS

万级片段规模内,一次 numpy 矩阵乘法的耗时在毫秒级,且结果是精确的(没有 ANN 召回损失)。
少一个二进制依赖在 Windows 上更省事。如果环境里恰好装了 `faiss-cpu`,代码会自动改用
`IndexFlatIP` 加速(见 `app/rag/vectorstore.py::_build_index`),行为与结果一致。

## 当前内容

| 目录 | 说明 |
| --- | --- |
| `fin` | 已入库的真实研报示例（`docs/knowledge_base/金融行业` 下 3 份 PDF，143 页 → **248 个片段**，用 hash 兜底向量建的，可直接用于检索演示） |

> 这些是运行时数据，可随时删除重建。注意**向量后端要与建库时一致**：`fin` 是用 `hash` 后端建的，
> 用 `local`/`openai` 后端查询会被指纹校验拦下——这正是设计意图，防止跨向量空间检索产生无意义结果。
> 想直接用示例库，把默认知识库设为它即可：`RAG_DEFAULT_KB=fin`。
>
> 248 这个数字是可复现的：用当前代码重跑
> `python -m app.rag.cli ingest -k fin -p docs/knowledge_base/金融行业 --embedding-provider hash`
> 会得到同样结果（逐文件 134 / 53 / 61）。

## 常用命令

```bash
# 入库(支持 .pdf / .docx / .md / .txt / .csv,可传目录递归)
python -m app.rag.cli ingest -k fin -p docs/knowledge_base

# 查看知识库状态
python -m app.rag.cli stats -k fin

# 只做检索(不调用大模型,便于调参)
python -m app.rag.cli search -k fin -q "风控策略有哪些要求"

# 检索 + 生成带引用的答案
python -m app.rag.cli ask -k fin -q "风控策略有哪些要求"

# 用评测集评估检索质量
python -m app.rag.cli eval -k fin -e examples/rag_eval_set.json

# 清空某个知识库(不可恢复)
python -m app.rag.cli reset -k fin --yes
```

## 注意事项

- **换向量模型必须重新入库**。`meta.json` 里记录了指纹(后端 + 模型 + 维度),加载时会校验:
  指纹不一致会直接报错,而不是静默返回错误结果(不同模型的向量空间不可比)。
- 该目录可以直接删除,删除后重新 `ingest` 即可重建;也可以整体拷贝到别的机器复用。
- 默认路径可以用环境变量覆盖:`RAG_DATA_DIR=/path/to/rag_data`。
