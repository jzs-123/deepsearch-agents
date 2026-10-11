# 向量存储后端：numpy → Qdrant

说明向量"存在哪"、为什么默认用 numpy、如何切换到 Qdrant、以及怎么验证两边结果一致。

---

## 一、现状：numpy 本地文件

默认后端 `numpy`，每个知识库一个目录 `rag_data/<kb>/`：

| 文件 | 内容 |
| --- | --- |
| `chunks.jsonl` | 每个检索片段一行 JSON（正文/来源/标题/元数据/chunk_id） |
| `vectors.npy` | 与 chunks 逐行对应的 `(N, dim)` float32 向量矩阵 |
| `meta.json` | 向量后端指纹、维度、切块参数、数据版本号 |

检索方式：打开知识库时把 `vectors.npy` 整个 load 进内存，做**全量矩阵点积**取 Top-K。

**为什么默认这样**：万级 chunk 以内毫秒级、结果精确（无 ANN 召回损失）、零依赖、且能当换索引后的「正确性基准」。代价是**十万级以上全量点积会慢**，也没有数据库的并发/管理能力。

## 二、Qdrant 后端（可插拔）

新增 `app/rag/qdrant_store.py::QdrantStore`，实现与 `VectorStore` 完全相同的接口，pipeline 通过 `open_vectorstore(backend, ...)` 工厂选择，无需改其它代码。

**两种模式**：

| 模式 | 配置 | 数据落点 | 适用 |
| --- | --- | --- | --- |
| 本地嵌入 | `RAG_QDRANT_URL` 留空 | `rag_data/<kb>/qdrant/` | 单机、免 Docker |
| 远程 server | `RAG_QDRANT_URL=http://localhost:6333` | Qdrant server | 多机、并发、生产 |

**关键设计**：

1. **存自己算好的向量**，绝不走 Qdrant 的 embedding（本地模式要 embedding 需另装 fastembed，本项目不需要，也不该让它擅自重新向量化）。collection 用 `Distance.COSINE` 声明"输入已就绪，按余弦算相似度"。
2. **sidecar `meta.json` 复用**：指纹 / dim / data_version / chunk_size / filter_fields 仍写 `<kb>/meta.json`，与 numpy 版一致——`list_knowledge_bases`、指纹校验、缓存失效（data_version）零改动可用。
3. **本地模式每个知识库一个独立存储目录**：`reset` 删目录即清库，数据仍在 `rag_data/` 下。
4. **numpy 仍是默认 + golden baseline**：换 qdrant 后两边 Top-K 应当一致（向量都 L2 归一化，余弦=内积），可对拍。

## 三、安装

```powershell
cd G:\deepsearch-agents-main

# 本地嵌入模式（免 Docker）
uv pip install "qdrant-client[local]" --python .venv\Scripts\python.exe

# 远程 server 模式（先 docker 起 qdrant，再只装客户端）
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
uv pip install qdrant-client --python .venv\Scripts\python.exe
```

验证：

```powershell
.\.venv\Scripts\python.exe -c "import qdrant_client; print(qdrant_client.__version__)"
```

## 四、切换到 Qdrant

在 `.env` 里：

```bash
RAG_VECTORSTORE=qdrant
# RAG_QDRANT_URL=http://localhost:6333   # 远程才需要；本地留空
```

**必须重新入库**（不同后端之间不自动迁移）：

```powershell
python -m app.rag.cli reset -k fin --yes
python -m app.rag.cli ingest -k fin -p docs/knowledge_base/金融行业 --embedding-provider hash
```

之后 `search` / `ask` / `eval` / 缓存 / 审计 / 元数据过滤都照常工作。

## 五、对拍验证（确认 qdrant 结果与 numpy 一致）

这是最重要的验收：换库后检索结果应与 numpy 完全一致（同 embedding 下余弦=内积）。

```powershell
# 1) numpy 建库，导出检索结果
$env:RAG_VECTORSTORE="numpy"
python -m app.rag.cli ingest -k cmp -p docs/knowledge_base/金融行业 --embedding-provider hash
python -m app.rag.cli search -k cmp --embedding-provider hash -q "2025年货币政策基调" -n 5 --json > numpy_top.json

# 2) 切换到 qdrant，重入库，再检索
$env:RAG_VECTORSTORE="qdrant"
python -m app.rag.cli reset -k cmp --yes
python -m app.rag.cli ingest -k cmp -p docs/knowledge_base/金融行业 --embedding-provider hash
python -m app.rag.cli search -k cmp --embedding-provider hash -q "2025年货币政策基调" -n 5 --json > qdrant_top.json

# 3) 比对两边的 (chunk_id, score) 序列是否一致
```

预期：两边 Top-K 的 chunk_id 与 score 逐项一致（浮点允许极小误差）。

## 六、已知限制与边界

- **不自动迁移**：numpy 和 qdrant 是两套存储，切换要手动 reset + ingest（有意的，避免"静默迁移"出错）。
- **指纹校验照常**：换 embedding 模型仍会被 `EmbeddingMismatchError` 拦住。
- **本仓库示例 `rag_data/fin` 是 numpy 建的**：直接设 `RAG_VECTORSTORE=qdrant` 后查 `fin` 会报"知识库不存在"（因为 qdrant 数据在别的目录），需重入库。
- **mock 测试覆盖逻辑、真实 qdrant 端到端在用户机器验证**：`tests/test_qdrant_store.py` 用假 client 验证了幂等/指纹/删除/检索/版本号等逻辑，真实 qdrant 的端到端见第五节对拍。
- **规模**：qdrant 用 HNSW（默认）做 ANN，十万级以上明显快于 numpy 全量点积；但 ANN 有召回损失，所以 numpy 保留为精确对照。
