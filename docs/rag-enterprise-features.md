# 自研 RAG 三项企业级能力：元数据过滤、缓存、审计日志

> 版本：v1（t13）
> 代码：`app/rag/metadata.py`（元数据与过滤）、`app/rag/cache.py`（三级缓存）、`app/rag/audit.py`（审计日志）
> 接入：`app/rag/pipeline.py`（`search/query/ingest/reset/stats`）、`app/rag/cli.py`、`app/tools/rag_tools.py`
> 测试：`tests/test_rag_metadata.py`（50）、`tests/test_rag_cache.py`（18）、`tests/test_rag_audit.py`（18）
> 演示语料：`examples/test_docs_metadata/`（含 frontmatter / 无 frontmatter / 非法 frontmatter 三类）
> 前置文档：`docs/rag-self-built.md`（5.10 判定分口径、向量指纹）、`docs/rag-query-rewrite.md`（改写与重试）

---

## 1. 结论摘要

| # | 能力 | 一句话说明 | 默认状态 | 关键证据 |
| --- | --- | --- | --- | --- |
| 1 | **元数据过滤** | 入库时给每个 chunk 打上「文件级 + frontmatter」元数据；检索时按 `{"部门": "风控"}` 这类条件做白名单过滤，**过滤发生在召回之后、重排之前** | 元数据抽取始终开启；过滤由调用方按需传 `filter` | §3.3~§3.5（过滤出/过滤掉/过滤空 三种真实输出） |
| 2 | **缓存** | embedding / 查询结果 / 生成结果三级缓存，键里带**向量指纹 + 数据版本号**，库一变整体失效；命中时结果标注 `cached=True` | `RAG_CACHE_ENABLED=false`（默认关，避免陈旧答案） | §4.2（冷缓存 0/12 命中 → 热缓存 12/12）、§4.3（miss→hit→库变更后 miss） |
| 3 | **审计日志** | 每次 `search`/`query` 追加一条 JSONL：时间/查询/过滤/改写变体/是否命中缓存/是否低分重试/结果摘要（**不含原文**）/是否调用大模型与耗时/拒答原因/向量指纹 | `RAG_AUDIT_ENABLED=true`（默认开；写失败只 warning） | §5.2（真实记录 + `cli audit` 读回） |
| 4 | **向后兼容** | 不传 `filter`、缓存关闭时，`search()/query()` 与引入这三项能力前**逐值一致**（顺序/分数/通道/`cached` 都比对过） | — | §6（快照比对用例）、`pytest tests -q` 516 passed |

---

## 2. 三者为什么必须一起设计

它们共享同一条「查询 → 检索 → 生成」链路，任何一环没设计好，另两环就会失真：

```
query ──► search ───────────────────────────────────────────────► (可选) 生成
  │          │                                                          │
  │          ├─ ① 查询缓存查找(键: query + filter + 指纹 + 数据版本)      ├─ ③ 生成缓存查找
  │          │      命中 → 直接返回并标注 cached=True                     │      命中 → 跳过整次大模型调用
  │          ├─ ② 召回(稠密 + 稀疏, 多查询 RRF)                          │
  │          │      └─ 元数据过滤(白名单,重排前)                        │
  │          ├─ 重排                                                     │
  │          └─ 写查询缓存(带数据版本号)                                 │
  └────────────────────────► 审计日志:一次 search/query 追加一条(含上面每一步的结果) ◄──┘
```

- **过滤必须能被审计**：只按 `filter` 过滤却不记录「过滤掉了多少条」，事后无法解释"为什么没查到"；
- **缓存必须能失效**：缓存键里不放**数据版本号**，库更新后就会返回过期答案——这正是默认关闭的原因；
- **审计必须包含缓存与过滤状态**：否则"这次结果是缓存来的还是重算的"永远说不清（审计里 `cache.hit` 明确标注）。

---

## 3. 一、元数据与检索过滤

### 3.1 抽取规则

| 来源 | 字段 | 说明 |
| --- | --- | --- |
| 文件级（自动） | `file` | 文件名（不含目录） |
| | `path` | 相对入库基准目录的路径（`/` 分隔，便于跨平台过滤） |
| | `suffix` | 小写后缀（`.md` / `.pdf` …） |
| | `size` | 字节数 |
| | `mtime` | 修改时间（UTC ISO8601，秒精度） |
| 内容级（Markdown frontmatter） | 字段名随文档自定 | 例：`部门` / `类型` / `密级` / `版本`（值可以是字符串、数字、布尔或列表） |

frontmatter 判定是**严格**的：文件第一行必须是 `---`（允许 BOM 与尾随空白），且在前 100 行内能找到收尾的 `---`/`...`；
命中后**从正文里剥掉**（它是元数据，留在正文里会污染切块与向量）。以下情况**只 warning、不报错、不跳过文件**：

| 情况 | 行为 |
| --- | --- |
| 没有 frontmatter | 只有文件级元数据 |
| 只有起始 `---`、没有收尾（很多 Markdown 开头就是分隔线） | 不当作 frontmatter，正文原样保留 + warning `未找到结束行` |
| YAML 语法错（如 `类型: [未闭合`） | 不当作 frontmatter，正文原样保留 + warning `解析失败(...)` |
| 解析出来不是键值对（如 YAML 列表） | 忽略元数据 + warning `不是键值对` |
| frontmatter 里出现 `file/path/suffix/mtime/size` | 该字段被忽略 + warning `与系统字段同名`（避免文档覆盖系统字段） |
| frontmatter 过大（>100 行或 >8192 字符） | 不解析 + warning |

> 兜底：环境里没有 PyYAML 时会退化为"只认 `key: value` 与 `- item`"的极简解析器（本仓库已依赖 PyYAML，这条只是保险）。

### 3.2 存储位置

- 每个 chunk 的元数据随 `Chunk.metadata` 写入 `rag_data/<kb>/chunks.jsonl`（一行一个 chunk，可 grep）；
- `rag_data/<kb>/meta.json` 增加 `filter_fields`（自动去重排序的可过滤字段清单）与 `data_version`（数据版本号，缓存失效用）。

### 3.3 过滤语义（filter JSON 写法）

| 写法 | 含义 | 示例 |
| --- | --- | --- |
| `{"部门": "风控"}` | 等值匹配（忽略大小写与首尾空白） | 只保留 `部门=风控` 的片段 |
| `{"类型": ["制度", "规范"]}` | 多值：**命中其一即可** | 保留 `类型` 为「制度」或「规范」的片段 |
| `{"部门": "风控", "密级": "内部"}` | 多字段：**全部满足**（AND） | 两个条件都要成立 |
| `{"标签": "乙"}` | 元数据值是列表时按"任一元素命中" | 文档里 `标签: [甲, 乙]` 也会命中 |
| 缺字段 | **不匹配**（白名单语义） | 没有 `部门` 字段的文档在 `{"部门": "风控"}` 下被剔除 |
| `{"部门": []}` | 该字段不可满足 | 结果必然为空（用于"故意查空"） |
| `{}` / 不传 | 不过滤 | 与引入过滤前行为一致 |

实现位置：`app/rag/metadata.py::normalize_filter/matches_filter/apply_filter`；
调用点：`pipeline._search_once()` 在 `_retrieve_variants()` 之后、`_apply_rerank()` 之前过滤。

### 3.4 真实命令与输出

```powershell
# 入库(6 份文档:3 份带 frontmatter、1 份无、2 份非法/未闭合)
python -m app.rag.cli ingest -k meta_demo -p examples/test_docs_metadata --embedding-provider hash
```

入库报告里的 warning（**不报错、不跳过文件**）：

```
  ! 分隔线示例.md:检测到疑似 frontmatter 起始行,但未找到结束行(---),已按普通正文处理
  ! 非法frontmatter示例.md:frontmatter 解析失败(ParserError: while parsing a flow sequence ...),
    已按普通正文处理
```

`list` 展示**可过滤字段**：

```
$ python -m app.rag.cli list
  - fin: 248 chunks | 向量:hash:512 | 更新:2026-10-10 17:07:35 | 数据版本:0 | 可过滤字段:(无)
  - meta_demo: 12 chunks | 向量:hash:512 | 更新:2026-10-11 17:58:16 | 数据版本:1 | 可过滤字段:密级、版本、类型、部门
```

① **不过滤**（3 条命中，其中含无元数据的「财务报销制度.md」）：

```
$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "准确率" -n 3
命中 3 条(向量后端:hash:512,重排:lexical)
    来源:风控模型操作规范.md#1 ... 元数据:{"file": "风控模型操作规范.md", ..., "部门": "风控", "类型": "规范", "密级": "内部", "版本": 1}
    来源:风控模型操作规范.md#0 ...
    来源:财务报销制度.md#0 ... 元数据:{"file": "财务报销制度.md", "suffix": ".md", "path": "财务报销制度.md", "size": 264, "mtime": "..."}
```

② **等值过滤**（只留风控，且明确写出过滤掉多少候选）：

> ⚠️ **PowerShell 5.1 注意**：下面命令里的内层 JSON 在 PowerShell 5.1 中会被剥掉双引号，导致"过滤条件错误"退出码 2。在 Windows PowerShell 里请把内层双引号用反引号转义，例如 `--filter '{\"部门\": \"风控\"}'`，或在 `bash`/`WSL` 里直接运行原命令。

```
$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "准确率" -n 3 --filter '{"部门": "风控"}'
命中 2 条
  · 元数据过滤:{'部门': '风控'}(过滤掉 8 条候选)
    来源:风控模型操作规范.md#1 ... / 风控模型操作规范.md#0 ...
```

③ **列表值过滤**（命中其一）：

```
$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "制度要求" -n 5 --filter '{"类型": ["制度", "规范"]}'
  · 元数据过滤:{'类型': ['制度', '规范']}(过滤掉 7 条候选)
    来源:风控额度制度.md#0 / 风控额度制度.md#1 / 风控模型操作规范.md#0
```

④ **过滤把候选清空**：必须明确说"是过滤条件的问题"，而不是笼统的"没检索到"：

```
$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "准确率" -n 3 --filter '{"部门": "法务"}'
  · 元数据过滤:{'部门': '法务'}(过滤掉 10 条候选)
没有符合过滤条件的片段(过滤条件:部门=['法务'])。当前可用于过滤的字段:密级、版本、类型、部门。
请检查字段名与取值,或先执行 python -m app.rag.cli list 查看各库的过滤字段。
```

⑤ **非法 filter**：退出码 2（与 `EmbeddingMismatchError` 同级的"用法错误"，不是崩溃）：

```
$ python -m app.rag.cli search -k meta_demo -q "x" --filter '{不是 JSON}'
过滤条件错误:filter 不是合法 JSON:Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
$ echo $LASTEXITCODE
2
```

### 3.5 Agent 工具侧

`search_knowledge_base` / `ask_knowledge_base` 新增可选参数 `filter_json`（JSON 字符串）：

```
search_knowledge_base(question, kb_name="", top_n=4, filter_json="")
  filter_json 示例: {"部门": "风控"} 或 {"类型": ["制度", "规范"]}
  可用字段见 list_knowledge_bases 输出里的"可过滤字段"
```

- 过滤后为空 → 返回「没有符合过滤条件的片段…当前可用于过滤的字段:…」；
- `filter_json` 不是合法 JSON → 返回「过滤条件不合法:…」而**不抛异常**（工具层铁律：任何错误都是中文提示）；
- `list_knowledge_bases` 的输出每行末增加「可过滤字段 …」，并附一行用法提示。

---

## 4. 二、缓存

### 4.1 三类缓存与失效策略

| 缓存 | 文件 | 键 | 失效方式 | 默认 |
| --- | --- | --- | --- | --- |
| Embedding | `rag_data/<kb>/.cache/embeddings.jsonl` | `SHA1(文本) + 向量后端指纹` | 键含指纹 → 换后端自然不命中；`reset()` 清空；维度不一致时忽略 | 受总开关控制 |
| 查询结果 | `rag_data/<kb>/.cache/query_cache.jsonl` | `SHA1(query + 归一化 filter + 指纹 + 数据版本号)` | 库内 chunk 增删改 → 版本号 +1 → **整体失效**；`reset()` 清空 | **默认关闭** |
| 生成结果 | `rag_data/<kb>/.cache/generation_cache.jsonl` | `SHA1(query + filter + 命中 chunk_id 集合 + 模型名 + 指纹 + 数据版本号)` | 同上 | **默认关闭** |

**为什么默认关闭**：查询/生成缓存一旦陈旧就是"返回过期答案"，这种风险不该由默认配置承担。Embedding 缓存虽然是内容寻址（同一文本 + 同一后端 = 同一向量，语义上不可能陈旧），但一起受 `RAG_CACHE_ENABLED` 控制，保持"一个开关解释全部行为"。开启方式：

```powershell
$env:RAG_CACHE_ENABLED='true'      # 或单次命令加 --cache
```

**数据版本号**（`meta.json: data_version`）由 `VectorStore.add/delete_document` 递增、`save()` 落盘，因此进程重启后依然有效；`reset()` 会把版本归零并**删除 `.cache/` 目录下的缓存文件**（目录本身保留）。

### 4.2 真实证据：embedding 缓存冷/热对比

```
$ python -m app.rag.cli ingest -k meta_demo -p examples/test_docs_metadata --embedding-provider hash --cache
文件:6/6 入库,跳过 0,新增 chunk 12,替换旧 chunk 0,用时 0.05s
  · 向量化缓存:命中 0/12 条(未命中 12 条需要重新编码)      ← 冷缓存

$ python -m app.rag.cli ingest -k meta_demo -p examples/test_docs_metadata --embedding-provider hash --cache
文件:6/6 入库,跳过 0,新增 chunk 12,替换旧 chunk 12,用时 0.05s
  · 向量化缓存:命中 12/12 条(未命中 0 条需要重新编码)      ← 热缓存:完全跳过向量化
```

单元测试里用「带调用计数的 embedder」证明得更硬：第二次入库时 `encode()` 的调用次数**没有增加**（`tests/test_rag_cache.py::test_embedding_cache_hits_on_reingest_without_reencoding`）。

### 4.3 真实证据：查询缓存 miss → hit → 失效

```
$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "授信额度上限" -n 2 --cache
命中 2 条
  · 缓存:query 未命中(数据版本 1)                       ← 第一次:未命中并写入
[1] score=0.9725 dense=0.4824 通道=['dense', 'sparse']
[2] score=0.4994 dense=0.1722 通道=['dense', 'sparse']

$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "授信额度上限" -n 2 --cache
命中 2 条
  · 缓存:query 命中(数据版本 1)                          ← 第二次:命中(未重新检索/重排)
[1] score=0.9725 dense=0.4824 通道=['dense', 'sparse'] [缓存]
[2] score=0.4994 dense=0.1722 通道=['dense', 'sparse'] [缓存]

# 入库一份新文档(数据版本 1 → 3)后,同一查询再次未命中 → 缓存整体失效
$ python -m app.rag.cli ingest -k meta_demo -p tmpdir/t13_extra --embedding-provider hash
$ python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "授信额度上限" -n 2 --cache
  · 缓存:query 未命中(数据版本 3)
```

`reset()` 清理：

```
$ python -m app.rag.cli reset -k meta_demo --yes
$ Get-ChildItem rag_data/meta_demo/.cache -File
(已清空)
```

生成缓存的效果在 `tests/test_rag_cache.py::test_generation_cache_skips_second_llm_call` 里锁定：
同一个问题第二次 `query()` 时 **stub 大模型只被调用 1 次**，且 `answer.debug["cache_hit"] is True`。

### 4.4 降级（缓存坏了不能拖垮查询）

| 情况 | 行为 |
| --- | --- |
| 缓存文件里有半截 JSON（写入中断） | 跳过该行，其余键照常命中 |
| 缓存目录不可写（被文件占位等） | 只 warning，检索照常返回结果 |
| 缓存载荷里的 chunk 已不存在 | 视为未命中，重新检索 |
| 缓存维度与当前后端不一致 | 忽略缓存、全量重新编码 |

### 4.5 已知未实现

- **无 TTL、无容量上限、无 LRU 淘汰**：缓存文件只增不减（删掉 `.cache/` 或 `reset` 即可重建）；
- **无跨进程文件锁**：多进程并发写同一 JSONL 可能丢行（丢行只影响命中率，不影响正确性）；
- **无压缩**；缓存目录随知识库走，暂不支持全局共享缓存目录；
- **查询缓存键尚未包含 `top_k` / `rerank` 开关**（`top_n` 已进入键）。也就是说，同一个 `(query, filter, top_n)` 在 `rerank=True` 与 `rerank=False` 之间、或不同 `top_k` 之间仍会互相命中。当前实现按"同问题同过滤同 top_n 的结果稳定"来假设；若调用方会频繁切换重排或 `top_k`，请先关查询缓存，或把这几个参数也加进键（见 `PipelineCache.query_cache_key`）。

---

## 5. 三、审计日志

### 5.1 落盘与配置

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_AUDIT_ENABLED` | `true` | 总开关 |
| `RAG_AUDIT_DIR` | 空 | 空 = 跟随数据根，即 `rag_data/.audit/`；可指到独立审计盘 |

文件：`<audit_dir>/<kb>.jsonl`，**追加写**（一条一行，便于 `tail` 与 `grep`）。

> 为什么默认跟随数据根而不是写死 `rag_data/.audit/`：测试与 selftest 用的是临时数据目录，跟随走就不会往真实 `rag_data/` 写东西；而默认数据根本身就是 `rag_data`，所以默认行为仍是任务要求的 `rag_data/.audit/`（`tests/test_rag_audit.py::test_audit_default_dir_follows_data_dir` 锁住了这条）。

### 5.2 字段说明（真实记录）

```json
{
  "ts": "2026-10-11T10:00:09.511+00:00",
  "kb": "meta_demo",
  "kind": "query",                       // search | query(一次调用只写一条)
  "query": "授信额度上限是多少",
  "filter": {},                           // 归一化前的原始过滤条件
  "rewrite_mode": "local",
  "variants": ["授信额度上限是多少"],       // 实际用于检索的变体(含原始查询)
  "variant_hits": [{"variant": "...", "dense_hits": 10, "sparse_hits": 6}],
  "llm_rewrite": {"used": false, "error": ""},
  "retry": {"enabled": true, "triggered": false, "gate_before": 0.3935, "gate_after": null, "improved": null},
  "gate_score": 0.3935,                   // 判定分(向量余弦)
  "fused_count": 10,
  "filtered_out": 0,                      // 被元数据过滤剔除的候选数
  "cache": {"used": false, "type": "generation", "hit": false, "key": "fba96750...", "data_version": 1},
  "results": [                            // 结果摘要:**不含原文**
    {"rank": 1, "chunk_id": "5276a610...", "source": "风控额度制度.md", "chunk_index": 0,
     "score": 0.8629, "dense": 0.3935, "sparse": 24.014, "channels": ["dense", "sparse"]}
  ],
  "result_count": 4,
  "llm": {"called": true, "elapsed": 6.929, "refusal_reason": "llm_error"},   // 拒答原因/是否进大模型
  "answer_len": 178,
  "grounded": false,
  "citations": [{"index": 1, "source": "风控额度制度.md", "chunk_index": 0, "score": 0.8629}],
  "embedding_fingerprint": "hash:512",
  "audit_dir": "G:\\deepsearch-agents-main\\rag_data\\.audit"
}
```

**刻意不记原文**：只记 `chunk_id/source/chunk_index/分数/通道`。审计日志一旦带正文，既会迅速膨胀，也会把敏感正文复制到另一个文件里（反而扩大泄露面）。
测试 `test_audit_does_not_store_full_chunk_text` 用一段 640 字的正文证明日志里不含它，且单条记录远小于正文长度。

### 5.3 只读查看入口

```powershell
python -m app.rag.cli audit -k meta_demo --tail 20            # 人类可读
python -m app.rag.cli audit -k meta_demo --tail 5 --kind query # 只看问答类
python -m app.rag.cli audit -k meta_demo --tail 5 --json      # JSONL 原样输出(便于 jq/脚本)
```

真实输出（节选）：

```
审计目录:G:\deepsearch-agents-main\rag_data\.audit
知识库:meta_demo(最近 2 条)

2026-10-11T10:00:01.201+00:00 [search] kb=meta_demo query='授信额度上限'
    filter={"部门": "风控"} 改写=local 变体=1 重试=False 判定分=0.4824
    #1 风控额度制度.md#0 score=0.9725 dense=0.4824 通道=['dense', 'sparse']
    #2 风控额度制度.md#1 score=0.4994 dense=0.1722 通道=['dense', 'sparse']
    指纹=hash:512
2026-10-11T10:00:09.511+00:00 [query] kb=meta_demo query='授信额度上限是多少'
    filter={} 改写=local 变体=1 重试=False 判定分=0.3935
    #1 风控额度制度.md#0 score=0.8629 dense=0.3935 通道=['dense', 'sparse']
    生成:called=True 耗时=6.929s 拒答原因=llm_error
    指纹=hash:512
```

### 5.4 降级

| 情况 | 行为 |
| --- | --- |
| 审计目录不可写（权限/被文件占位） | `AuditLogger.record()` 捕获异常 → `logger.warning`，返回 `False`；查询结果**不受影响**（`failures` 计数可在 `stats()` 里看到） |
| 关闭审计 | 不创建目录、不写文件 |
| 日志文件被截断/半行 | `read_audit()` 跳过无法解析的行，不影响其余记录 |

---

## 6. 向后兼容（硬指标）

| 位置 | 策略 |
| --- | --- |
| `search()` | 新增 `filter=None`（默认不过滤）与内部参数 `_record_audit`；不传时检索路径与引入前一致 |
| `query()` | 新增 `filter=None`；不传时与引入前一致（只多写一条审计） |
| 默认值 | `RagConfig.cache_enabled=False`、`audit_enabled=True`、`audit_dir=""`（跟随数据根）；`Settings` 同名默认一致 |
| `RetrievedChunk` | 只新增 `cached: bool = False` 字段（有默认值 → 既有构造方式不变；`to_dict()` 多一个键） |
| 缓存关闭时 | 不创建 `.cache/` 目录、不读不写；检索结果逐值不变 |
| 元数据 | 只影响**带 frontmatter** 的文档（本仓库既有语料都没有 frontmatter，切块/向量不变）；没有 frontmatter 的文档行为完全不变 |
| 锁住兼容性的用例 | `test_rag_metadata.py::test_search_without_filter_matches_legacy_path`（按历史实现手工算一遍再逐值比对）、`test_empty_filter_dict_is_same_as_no_filter`、`test_rag_cache.py::test_search_with_cache_disabled_matches_uncached_content` |

实测数字（本次，最终 516 passed）：

```
$ python -m pytest tests -q
516 passed          # 374(既有) + 50(metadata) + 19(cache) + 19(audit) + 63(eval_offline+eval_ragas) + 1(缓存 top_n 回归) 等

$ python -m app.rag.selftest
──── 自测结果:18 通过 / 0 失败 ────
```

---

## 7. 配置清单（`.env.example` 已同步）

| 变量 | 取值 | 默认 | 作用 |
| --- | --- | --- | --- |
| `RAG_CACHE_ENABLED` | true/false | `false` | 三级缓存总开关；缓存文件在 `rag_data/<kb>/.cache/` |
| `RAG_AUDIT_ENABLED` | true/false | `true` | 审计日志开关 |
| `RAG_AUDIT_DIR` | 路径 | 空（跟随 `RAG_DATA_DIR` 下的 `.audit/`） | 审计文件目录 |

CLI 开关：`--filter '<JSON>'`、`--cache` / `--no-cache`（`search`/`ask`/`eval`/`ingest`）、`audit` 子命令（`--tail` / `--kind` / `--json`）。

`.gitignore` 已排除运行时产物（`rag_data/*/.cache/`、`rag_data/.audit/`）——`rag_data/` 本身是纳入版本控制的，所以必须单独排除这两类生成物。

---

## 8. 未实现 / 预留的部分（如实列出）

1. **过滤只做元数据白名单**：不支持范围比较（`版本 > 2`）、模糊匹配、字段间逻辑（AND/OR 嵌套）、正则；
2. **过滤不参与索引**：不做"过滤后重建向量索引"，因此过滤只在召回候选（默认 top_k=10 的并集）内生效——如果符合过滤条件的片段一条都没进候选，结果仍为空（这是设计取舍：不改向量检索本身）；
3. **缓存无 TTL/容量上限/LRU**（见 §4.5），也没有跨进程文件锁；
4. **查询缓存键未包含 `top_k/rerank`**（见 §4.5 的显式警告，`top_n` 已包含）；如需严格多参数缓存需扩展键；
5. **审计日志无轮转（rotation）与归档**：长期运行会持续增长，建议外部按大小切割/归档；
6. **审计不含原文也不含"用户身份"**：多租户场景应把 `identity.user_id` 一并写入（当前 Agent 侧拿得到，但 RAG 层没有透传，避免越权记录 PII）；
7. **frontmatter 只支持 Markdown 头部**：PDF/DOCX 的文档属性（标题/作者/页数）未抽取；
8. **元数据不参与 embedding**：`heading` 之外没有把元数据拼进向量文本（例如把"部门:风控"拼进 chunk 一起编码），因此按元数据做**语义**检索不可用；
9. **`list_knowledge_bases` 的字段清单来自 `meta.json`**：若手工修改过 `chunks.jsonl` 而未重新入库，清单可能与实际 chunk 不一致（首次查询会按实际 chunk 重算并落盘）。

---

## 9. 复现命令（最小集）

```powershell
# 1) 建带元数据的演示库(含 frontmatter / 无 / 非法 三类文档)
python -m app.rag.cli ingest -k meta_demo -p examples/test_docs_metadata --embedding-provider hash
python -m app.rag.cli list                                     # 看"可过滤字段"

# 2) 元数据过滤
python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "准确率" -n 3
python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "准确率" -n 3 --filter '{"部门": "风控"}'
python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "制度要求" -n 5 --filter '{"类型": ["制度", "规范"]}'
python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "准确率" -n 3 --filter '{"部门": "法务"}'

# 3) 缓存(miss → hit → 失效)
python -m app.rag.cli ingest -k meta_demo -p examples/test_docs_metadata --embedding-provider hash --cache   # 0/12
python -m app.rag.cli ingest -k meta_demo -p examples/test_docs_metadata --embedding-provider hash --cache   # 12/12
python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "授信额度上限" -n 2 --cache           # 未命中
python -m app.rag.cli search -k meta_demo --embedding-provider hash -q "授信额度上限" -n 2 --cache           # 命中([缓存])

# 4) 审计
python -m app.rag.cli audit -k meta_demo --tail 5
python -m app.rag.cli audit -k meta_demo --tail 5 --json

# 5) 测试
python -m pytest tests/test_rag_metadata.py tests/test_rag_cache.py tests/test_rag_audit.py -q   # 86 passed
python -m pytest tests -q                                                                        # 513 passed(另有 1 个队友在改的 OCR 用例)
python -m app.rag.selftest                                                                        # 18 通过 / 0 失败
```

---

## 10. 改动清单

| 路径 | 类型 | 说明 |
| --- | --- | --- |
| `app/rag/metadata.py` | 新增 | frontmatter/file 元数据抽取、filter 归一化与匹配、字段清单、提示文案 |
| `app/rag/cache.py` | 新增 | `JsonlCache` / `PipelineCache`（embedding / query / generation 三级缓存 + 统计） |
| `app/rag/audit.py` | 新增 | `AuditLogger`（JSONL 追加写、失败降级）、记录构造、`read_audit` / `format_audit_records` |
| `app/rag/pipeline.py` | 修改 | 入库元数据 + frontmatter 剥离 + 可过滤字段落盘；`search(filter=..., _record_audit=...)`；查询/生成缓存；`_encode_documents`（embedding 缓存）；`reset` 清缓存；`stats`/`format_search_audit` 增字段 |
| `app/rag/vectorstore.py` | 修改 | `data_version`（增删改递增、落盘）、`list_knowledge_bases` 暴露 `data_version/filter_fields`、跳过点目录 |
| `app/rag/types.py` | 修改 | `RetrievedChunk.cached`（默认 False） |
| `app/rag/config.py` | 修改 | `cache_enabled` / `audit_enabled` / `audit_dir` + `cache_dir()` / `audit_path()` + Settings 映射 |
| `app/core/config.py` | 修改 | 同名 Settings 字段（`rag_cache_enabled` / `rag_audit_enabled` / `rag_audit_dir`） |
| `app/rag/cli.py` | 修改 | `--filter` / `--cache` / `--no-cache`（search/ask/eval/ingest）、`audit` 子命令、`list` 展示可过滤字段、过滤空结果提示、非法 filter 退出码 2 |
| `app/rag/eval.py` | 修改 | `evaluate(..., filter=None)` 并写入 config（便于"加过滤前后"对比评测） |
| `app/tools/rag_tools.py` | 修改 | 两个查询工具新增 `filter_json`（docstring 写明格式与可用字段）、过滤空结果专用提示、`list` 工具展示可过滤字段 |
| `.env.example` | 修改 | 缓存/审计配置段与用法示例 |
| `.gitignore` | 修改 | 排除 `rag_data/*/.cache/`、`rag_data/.audit/` |
| `tests/test_rag_metadata.py` | 新增 | 50 用例 |
| `tests/test_rag_cache.py` | 新增 | 18 用例 |
| `tests/test_rag_audit.py` | 新增 | 18 用例 |
| `examples/test_docs_metadata/*.md` | 新增 | 6 份演示语料（3 份合法 frontmatter + 1 份无 + 2 份非法/未闭合） |
| `rag_data/meta_demo/` | 运行时数据 | 演示库（12 片段，hash:512），可由 §9 第 1 步一条命令重建；可删 |
