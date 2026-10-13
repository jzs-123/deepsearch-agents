# Agent + RAG 生产使用清单

## 1. 从前端上传知识库

启动前后端后，点击左侧 **上传知识库**：

1. 选择已有知识库，或输入新名称；
2. 选择 PDF、DOCX、Markdown、TXT、CSV 或图片；
3. 单文件更新时可填写稳定的“逻辑文档标识”，例如 `school/放学时间`；
4. 点击“上传并建立索引”。

同一个 `document_key` 再次入库会先完成解析和向量化，再删除旧版本，避免新版本失败时把可用旧版本删掉。未手工填写时，文件名就是逻辑身份。

接口也可直接调用：

```text
GET  /api/knowledge-bases
POST /api/knowledge-bases/ingest
```

上传限制为单文件 50MB；多个文件一起上传时，每个文件使用自己的文件名作为逻辑身份。

## 2. 文档变多后的检索配置

按 `chunk_count` 选后端：

| 规模 | 建议 |
| --- | --- |
| 少于 1 万片段 | 默认 `numpy`，保留 BM25 混合检索 |
| 1 万到 10 万片段 | `numpy` 可继续使用；安装 FAISS 后本地向量检索更快 |
| 10 万片段以上 | 使用远程 Qdrant 的 ANN 索引；并发高时横向扩展 Qdrant |

十万级配置示例：

```dotenv
RAG_VECTORSTORE=qdrant
RAG_QDRANT_URL=http://localhost:6333
RAG_SPARSE_ENABLED=false
RAG_TOP_K=20
RAG_TOP_N=4
```

`RAG_SPARSE_ENABLED=false` 会关闭 BM25，避免远程 Qdrant 第一次查询时把所有 chunk 拉回进程建索引。含大量编号、专有名词的中小知识库应保持 `true`，因为 BM25 对这类查询有价值。

重复问题很多时可以设置 `RAG_CACHE_ENABLED=true`。缓存键带知识库数据版本，任何入库更新都会让旧缓存失效。入库已批量向量化；线上应复用常驻 API 进程，不要每次查询都重新启动 CLI 和加载模型。

本地 BGE 加载会先使用 Hugging Face 缓存，只有首次缺少模型时才联网下载。`Loading weights` 表示从磁盘把权重载入内存，不等于重新下载；常驻 API 进程只加载一次。

## 3. 用负面样例标定拒答

运行包含可答和应拒答问题的评测集：

```powershell
.venv\Scripts\python.exe -m app.rag.cli eval `
  -k fin-bge `
  -e examples/rag_eval_set_finance.json `
  -n 5
```

报告中的 `拒答阈值标定` 会基于正负样例的实际向量余弦给出 `recommended_threshold` 和平衡准确率。把建议值写入 `RAG_MIN_SCORE` 后重新跑一次评测。更换向量模型或语料后必须重新标定。

也可以先不改环境变量，用命令行临时复验建议值：

```powershell
.venv\Scripts\python.exe -m app.rag.cli eval `
  -k fin-bge `
  -e examples/rag_eval_set_finance.json `
  --min-score 0.6127
```

负面样例应至少覆盖：完全无关、相近领域但实体错误、时间版本错误、资料没有的隐私字段、错误前提和要求模型猜测的提示注入。`examples/rag_eval_set_finance_negatives.json` 提供了独立的困难负面集。

线上拒答发生在生成模型之前：`grounded_score < RAG_MIN_SCORE` 时直接返回无依据文案，不调用大模型。

## 4. 前端检索溯源

RAG 工具每次检索都会推送 `rag_sources` WebSocket 事件，字段包含知识库、问题、判定分、阈值、来源文件、chunk 序号、标题、排序分和原文片段。前端在回答下方展示“检索依据”，历史会话也会从持久化事件中恢复。

## 5. Agent 稳定 JSON

传输层不让模型拼 JSON。后端把模型文本包装成固定结构：

```json
{
  "schema_version": "1.0",
  "status": "success",
  "answer": "模型最终文本"
}
```

该对象位于 `task_result.data.output`，旧字段 `task_result.data.result` 继续保留供现有前端使用。这样即使模型正文不是 JSON，客户端收到的协议仍然稳定。

如果业务需要固定字段，例如合同审核结果，应再定义对应的 Pydantic 模型，并用模型的 structured output / tool calling 生成后做服务端校验；校验失败时重试一次或返回明确错误。不要只在提示词里写“请输出 JSON”。
