# t15 独立验证报告：元数据过滤、缓存、审计日志与 RAGAS 评测方案

> 任务：t15「独立验证元数据过滤、缓存、审计日志与 RAGAS 评测方案」（验证对象：t13、t14）
> 方法：**不接受自述** —— 自造语料（不复用对方 demo）、给真实实现套计数器、与 `git HEAD`（引入本功能前的实现）
> 逐值对拍、回退变异（改坏缓存键看测试是否报红）、保留同款功能但注入 stub 大模型、真实跑 CLI 与脚本。
> 范围：只读核查 + 报告。**未修改任何业务代码与既有文档**；临时脚本与产物在 `tmpdir/verif_t15/`（已 gitignore）。

---

## 0. 结论摘要

| 判定项 | 结论 | 关键证据 |
| --- | --- | --- |
| 元数据抽取 | ✅ 可用 | 自造 4 份（2 合法 frontmatter / 1 无 / 1 非法）→ **4/4 入库**，非法 frontmatter 只给 1 条 warning；frontmatter 已从正文剥掉；`meta.json` 落 `filter_fields` |
| 检索期过滤 | ✅ 可用且语义正确 | 等值 / 列表命中其一 / 多字段 AND / 缺字段=不匹配 / 空列表不可满足 / 忽略大小写与空白；`filtered_out` 计数正确；**空结果给的是可操作的提示，不是空** |
| 向后兼容 | ✅ **成立（逐值一致）** | 与 `git HEAD` 版 pipeline 在 5 条查询上 `(chunk_id, score, rrf)` **5/5 完全相同**；`RetrievedChunk.cached` 默认 False；缓存关闭时不建 `.cache/` |
| 缓存（三级） | ✅ 可用 | 计数 embedder：第二次入库 `encode()` **0 次调用**（命中 2/2）；查询缓存 miss→hit 且分数/顺序逐值相同、`cached=True`；改库后版本 4→5 **整体失效**；生成缓存让 stub 大模型只被调用 **1 次** |
| 审计日志 | ✅ 可用 | 46 条真实记录字段齐全；**不含正文**（自造 240 字正文不在日志里）；`audit` CLI 可读回；**审计目录被文件占位时查询照常返回**（failures=1/2） |
| RAGAS 方案 | ✅ 方案可用（本机不可跑，已如实标注） | `--check` 真实给出中文指引 + **exit=2**；组装逻辑用 stub rows 走通（contexts=`list[list[str]]`、跳过记原因、`ground_truth` 列整体省略）；无硬编码密钥、报告/控制台只输出脱敏 key |
| 离线近似指标 | ✅ 可用且不夸大 | 真实跑 `eval_offline` exit=0；**我自己造的编造数字（2.1%、1.5 亿元）被列入 `suspected_hallucinations`**；无可核验内容时给 `None` 而非 1.0；报告自带 `is_ragas:false`+disclaimer |
| 有无夸大 | ✅ 未见夸大 | 文档把"extractive 天然 ~100% 只证明链路通"、"RAGAS 本机装不上"、"过滤不参与索引"、"缓存无 TTL/LRU/锁"都写在正文；我发现的差异都是**文档陈旧或偏保守** |

---

## 1. 时间线（并发编辑对我的测量的影响）

本任务执行期间 t13/t14 的作者仍在改这一片代码并最终提交，我的测量跨越了三个状态，如实记录：

| 时刻 | 工作区状态 | 我的测量 |
| --- | --- | --- |
| ~18:05–18:08 | 未提交的工作树（t13/t14 交付态） | `pytest tests -q` → **514 passed ×2**（21.57s / 20.33s，exit=0）；selftest 18/0 |
| ~18:11 | 作者新增 `test_query_cache_is_isolated_by_top_n` 但未写完（`tests/test_rag_cache.py` mtime 18:10:25） | **1 failed, 514 passed**：`NameError: name 'DOC3' is not defined`（该文件 134 行） |
| ~18:13 起 | 作者补完该用例并提交 `d1c2f41`（同时清理 `rag_data/meta_demo/`、更新 `.gitignore`） | `pytest tests -q` → **515 passed in 19.54s / 19.00s ×2**（exit=0）；五个新测试文件 150 passed；`git status` 干净 |

> 也就是说：**F1（下节）是验证期间的瞬时失败，作者已在 18:11:10 修掉**；我的最终判定基于提交 `d1c2f41` 之后的状态。

---

## 2. 全量回归与卫生

```powershell
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
python -m pytest tests -q      # 提交后：515 passed in 19.54s / 19.00s  exit=0（跑两次）
python -m app.rag.selftest     # ──── 自测结果:18 通过 / 0 失败 ────  exit=0
python -m pytest tests/test_rag_metadata.py tests/test_rag_cache.py tests/test_rag_audit.py tests/test_rag_eval_offline.py tests/test_rag_eval_ragas.py -q
                               # 150 passed in 6.17s
```

逐文件用例数（实测 collect）：metadata **50**、cache **19**、audit **18**、eval_offline **30**、eval_ragas **33**
（t13 文档写 86 = 50+18+18，现为 87；t14 文档写 63 = 30+33 ✅ 准确；另有 OCR 24、rewrite 49）。

`rag_data/` 终态：`fin`（10-10 17:07:35，未被改动）、`README.md`、`.audit/`（`fin.jsonl` 24KB、`meta_demo.jsonl` 19KB，运行时产物；
`git check-ignore` 命中 `.gitignore:32:.audit/` ✅）；演示库 `meta_demo/` 已由作者清理，`git status` 干净。

---

## 3. 元数据与过滤（t13）

### 3.1 我自造的语料（与 `examples/test_docs_metadata` 无关）

`tmpdir/verif_t15/docs_meta/`：`风控指标.md`（部门:风控/类型:规范/密级:内部/版本:2）、`法务合同.md`（部门:法务/类型:制度）、
`无元数据.md`（无 frontmatter）、`非法.md`（`部门: [未闭合` 的坏 YAML）。

```
files=4/4 skipped=0 chunks=5
warnings=['非法.md:frontmatter 解析失败(ParserError: while parsing a flow sequence …),已按普通正文处理']
meta.json filter_fields=['密级', '版本', '类型', '部门']  data_version=1
chunk 风控指标.md#0 metadata={"file":"风控指标.md","suffix":".md","path":"…","size":643,"mtime":"…","部门":"风控","类型":"规范","密级":"内部","版本":2}
chunk 无元数据.md#0 metadata={"file":"无元数据.md","suffix":".md",…}          ← 无 frontmatter 也有系统字段
非法.md 是否被入库: True ；frontmatter 是否从正文剥掉: True
```

→ 与文档 §3.1/§3.2「非法只 warning、不跳过文件」「frontmatter 从正文剥掉」一致。

### 3.2 过滤语义（自造语料的实测）

| 过滤条件 | 结果 | filtered_out |
| --- | --- | --- |
| 不过滤 `准确率` | 5 条：风控指标#0 / 法务合同#0 / 无元数据#0 / 非法#0 / 非法#1 | 0 |
| `{"部门":"风控"}` | 1 条：风控指标#0（**全部命中项都是风控**） | 4 |
| `{"部门":"法务"}` | 1 条：法务合同#0 | 4 |
| `{"类型":["制度","规范"]}` | 2 条：风控指标#0 + 法务合同#0 | 3 |
| `{"部门":"法务部"}`（不存在） | **0 条** → 白名单语义成立 | 5 |
| `{"部门":"风控","类型":"规范"}` / `{...,"类型":"制度"}` | 1 条 / **0 条** → 多字段 AND 成立 | 4 / 5 |
| `{"部门":""," 风控 "}`（带空白） | 命中 → 忽略大小写与首尾空白 | — |
| `filter={}` | 与不传 filter **逐值相同** | — |
| `{"部门":[]}` | 0 条（不可满足） | 5 |

空结果提示（`metadata.format_filter_hint` 实测输出，与文档 §3.4④ 逐字一致）：

```
没有符合过滤条件的片段(过滤条件:部门=['法务部'])。当前可用于过滤的字段:密级、版本、类型、部门。
请检查字段名与取值,或先执行 python -m app.rag.cli list 查看各库的过滤字段。
```

### 3.3 CLI / 工具侧（真实命令）

```
$ python -m app.rag.cli list
  - cliverif: 5 chunks | 向量:hash:512 | 更新:2026-10-11 18:09:39 | 数据版本:6 | 可过滤字段:密级、版本、类型、部门
$ python -m app.rag.cli search -k cliverif … -q "准确率" -n 3 --filter '{\"部门\": \"风控\"}'
命中 1 条 … · 元数据过滤:{'部门': '风控'}(过滤掉 4 条候选)
$ python -m app.rag.cli search -k cliverif … -q "准确率" -n 3 --filter '{\"部门\": \"法务部\"}'
命中 0 条 … · 元数据过滤:{'部门': '法务部'}(过滤掉 5 条候选)
没有符合过滤条件的片段(过滤条件:部门=['法务部'])。当前可用于过滤的字段:密级、版本、类型、部门。…
$ python -m app.rag.cli search -k cliverif -q "x" --filter '{不是 JSON}'
过滤条件错误:filter 不是合法 JSON:…   →  echo $LASTEXITCODE = 2
```

工具侧（`app/tools/rag_tools.py`）：

```
[list_knowledge_bases] - cliverif: 5 个片段 | 向量模型 hash:512 | … | 可过滤字段 密级、版本、类型、部门
                       提示:…调用检索/问答工具时可用 filter_json 传 JSON(如 {"部门": "风控"})
[search filter_json 正确]   知识库 cliverif 中没有与该问题足够相关的资料(最高匹配度 0.180,低于阈值 0.2)。   ← 过滤命中但门槛不足
[search filter_json 无匹配] 知识库 cliverif 中没有符合过滤条件的片段(过滤条件:部门=['法务部'])。当前可用于过滤的字段:…
[search filter_json 非法]   过滤条件不合法:filter 不是合法 JSON:…            ← 未抛异常（工具层铁律成立）
[工具 docstring 写明 filter_json] True
```

### 3.4 向后兼容：与 `git HEAD`（引入本功能前的 pipeline）逐值对拍

```
HEAD 版 pipeline.py：701 行；含 'filter=' 调用: False
  q='准确率'          一致=True  新/旧均 [('4ac74b4fa13dc20392ae', 0.773974, 0.028571)]
  q='制度要求'         一致=True
  q='授信额度'         一致=True
  q='合同审阅'         一致=True  新/旧均 [('3c44f5a107ecfa24e849', 0.975, 0.029508)]
  q='不存在的问题xyz'    一致=True
逐值一致：5/5
RetrievedChunk 默认 cached=False ；to_dict 含 cached: True
缓存关闭时 .cache 目录是否存在: False
```

→ 文档 §6「不传 filter 时与引入前一致」**成立**（而且是直接对拍旧实现，比其自带快照用例更强）。

---

## 4. 缓存（t13）

### 4.1 embedding 缓存（我自己的计数 embedder）

```
[冷缓存] chunks=2 encode_calls=1 texts=2 stats={embedding_hits:0, embedding_misses:2}
[热缓存] 第二次入库：本次 encode_calls=0 texts=0；累计 hits=2
[判定] 第二次入库没有再调用 encode: True
[dim=64 新库] 键 hash:512 vs hash:64 不同 → 全部未命中（encode_calls=1, misses=3）
```
与文档 §4.2 的「0/12 → 12/12」同性质；我用**独立计数 embedder**证明，不依赖对方用例。

### 4.2 查询缓存与失效

```
[第一次] cached=[False,False] cache={'used':True,'type':'query','hit':False,'key':'f6c7e955…','data_version':4} scores=[0.975, 0.0]
[第二次] cached=[True,True]  cache={…,'hit':True,'key':'f6c7e955…','data_version':4}        scores=[0.975, 0.0]  ← 逐值相同
[版本]   4 → 5（入库新文档）；同查询 hit=False, data_version=5                                 ← 整体失效
CLI 对照：· 缓存:query 未命中(数据版本 6) → · 缓存:query 命中(数据版本 6)，结果行末尾标 [缓存]
```

**关于 `top_n` 的核对（重要）**：我实测 `query_cache_key(q,"",2) != query_cache_key(q,"",10)`，且用 `top_n=2` 写入缓存后
用 `top_n=10` 查询 **miss** → 代码 `cache.py:209 = compound_key(query, filter, top_n, fingerprint, data_version)` 里 **top_n 确实进键**。
（我在 18:05 读到的文档旧版还写着"键里没有 top_n"；作者已在 18:08:59 改为「`top_n` 已进入键，尚未包含 `top_k`/`rerank`」，
**当前文档与代码一致** —— 见 F7。）

### 4.3 生成缓存 / reset / 降级

```
[生成缓存] stub 大模型被调用 1 次；a1.cache_hit=False  a2.cache_hit=True  a1.answer == a2.answer
[reset] 前：.cache 内 3 个文件（embeddings/query_cache/generation_cache.jsonl）
[reset] 后：.cache 目录**仍然存在**（内容被清空，data_version 归零）      ← 与文档 §4.1 措辞不符，见 F3
[降级] 往 query_cache.jsonl 追加半截 JSON → 查询仍返回 2 条，未抛异常
[降级] 注入"chunk 已不存在"的假载荷 → 返回真实检索结果（首位 chunk 仍在库中）
[关闭] cache_enabled=False：不建 .cache/，cached 全为 False
```

**回退变异（看测试是否真的锁住）**：

| 变异 | 结果 |
| --- | --- |
| A：`query_cache_key` 去掉 `data_version` | `tests/test_rag_cache.py` → **2 failed**，含 `test_query_cache_invalidated_when_store_changes` ✅ |
| B：`content_key` 换成常量（键不含文本） | → **2 failed**，含 `test_pipeline_cache_embedding_roundtrip` ✅ |

→ 「库变更 → 缓存整体失效」「embedding 按内容寻址」两条语义**确实被测试锁住**（不是空断言）。

---

## 5. 审计日志（t13）

### 5.1 字段与"不含原文"

```
审计目录(默认跟随数据根)=…\rag_data\.audit  文件=metaverif.jsonl 存在=True
记录数=46 kinds=['query','search']
最后一条字段: [audit_dir, cache, embedding_fingerprint, filter, filtered_out, fused_count, gate_score, kb, kind,
              llm, llm_rewrite, query, result_count, results, retry, rewrite_mode, ts, variant_hits, variants]
results[0] 字段: [channels, chunk_id, chunk_index, dense, rank, score, source, sparse]
审计文件是否含正文长句（应 False）: False ；是否含 240 字正文片段: False
审计文件 37206 字节 / 46 条 → 平均 ~800 字节/条
```

`query()` 记录（stub 大模型，两组对照）：

```
[高分问题] stub 被调用 1 次；kind=query filter={"部门":"风控"} gate_score=0.3154 filtered_out=4
           cache={used:false,type:generation,key:d33d5ca5…,data_version:11} result_count=1
           llm={called:true,elapsed:0.0,refusal_reason:""} answer_len=21 grounded=true
           citations=[{index:1,source:风控指标.md,chunk_index:0,score:0.775}] embedding_fingerprint=hash:512
[低分问题] stub 被调用 0 次；answer='知识库中没有找到与该问题相关的资料…'
           llm={called:true,elapsed:0.002,refusal_reason:"score_below_threshold"} grounded=false
```

CLI 读回（`python -m app.rag.cli audit -k cliverif --tail 2`）输出格式与文档 §5.3 一致：
`[search] kb=… query=… / filter={} 改写=local 变体=1 重试=False 判定分=… / 缓存:query hit=False key=… / #1 … 指纹=hash:512`。

### 5.2 审计失败不影响查询（真实复现）

把 `audit_dir` 指到一个**普通文件**（`blocker_file.txt`），使 `<dir>/<kb>.jsonl` 无法创建：

```
[5c] 查询仍返回 3 条（未抛异常）；audit failures=1 describe={'enabled': True, 'dir': '…blocker_file.txt', 'writes': 0, 'failures': 1}
[5c] query 也照常返回: answer_len=33；audit failures=2
```

→ 文档 §5.4「写入失败只 warning、查询不受影响、failures 可见」**成立**。

> ⚠️ 一处字段语义问题（F2）：`llm.called` 实际取自 `pipeline.py:880  llm_called = self._llm_client() is not None`，
> 即"客户端可用"，而不是"本次真的调用了大模型"。低分拒答路径里 stub 调用次数为 **0**，日志却记 `called: true`。

---

## 6. RAGAS / 离线评测（t14）

### 6.1 缺依赖提示（真实命令，exit=2）

```
$ python -m app.rag.eval_ragas --check
缺少依赖:datasets、ragas。
RAGAS 属于「用户侧可选依赖」，本项目 pyproject.toml 不包含它。
请在你要跑评测的机器上安装: pip install ragas datasets
若 RAGAS 需要本地向量模型 / 中文分词，可能还要: pip install sentence-transformers
并在 .env 里配好 judge 大模型（OPENAI_BASE_URL / OPENAI_API_KEY / LLM_DEEPSEEK 或 LLM_QWEN_MAX）。
装好后重跑: python -m app.rag.eval_ragas -e … -k fin -o output/ragas.json
相关文档:docs/rag-eval-guide.md
暂时装不上的话，可先用零依赖近似指标…: python -m app.rag.eval_offline -e … --answer-mode extractive
$ echo $LASTEXITCODE → 2
$ python -m app.rag.eval_ragas            # 不带 -e 且非 --check → argparse 提示「-e/--eval-set 是必需的（只有 --check 可以省略）」exit=2
```
`pyproject.toml` 确实**未被修改**（`git status --porcelain pyproject.toml` 为空），与文档「不进入核心依赖」一致。

### 6.2 离线替代真实跑通（我自己跑，exit=0）

```
$ python -m app.rag.eval_offline -e examples/ragas_eval_set.example.json -k fin --answer-mode extractive --top-n 4 -o tmpdir/verif_t15/offline_report.json
── 生成质量离线近似评测（非 RAGAS）──  知识库: fin   答案来源: extractive   样本数: 4   耗时: 1.537s
context_faithfulness_approx : 100.00%      context_coverage_approx : 100.00%
疑似幻觉答案比例            : 0.00%        拒答准确率 : N/A（本模式不适用）
⚠️ 离线近似，非 RAGAS：… 正式生成质量评测请使用 RAGAS（见 docs/rag-eval-guide.md）。
注意: answer_mode=extractive：答案由检索到的原文直接拼接，因此两个指标天然接近 100%，只能证明评测链路可用…
报告已写入: tmpdir\verif_t15\offline_report.json
```

报告结构核对：`is_ragas=false`、`not_ragas=true`、`disclaimer`、`ragas_metrics_not_included` 均在；
**平均值只统计"可答复且可核验"的 3 条**（`counts={answerable:3,no_answer:1,checkable_answers:3}`），
拒答题**不混入** faithfulness/coverage 均值（不会把 N/A 伪装成 100%）—— 与文档 §8.2/§8.3 的说明一致。

### 6.3 幻觉识别（我自己造编造数字）

```
context = ["模型准确率为 91.5%,依据 3.2.1 标准,A13 设备用于采集。"]
[1a] answer="模型准确率为 2.1%,成本 1.5 亿元,A13 设备正常。"
     → score=0.3333  checkable=True  suspected=['2.1%', '1.5 亿元']        ← 两个编造数字都被标记
[1b] answer="这套方案整体不错,值得推广。"
     → score=None    checkable=False                                     ← 无可核验内容给 None，不是 1.0
[1c] answer 全部可核验 → score=1.0
[1e] normalize_contexts([None,'x','']) → ['x']                            ← None 不会变成字符串 "None"
```

### 6.4 RAGAS 组装逻辑（stub rows，不装 ragas 也能验）

```
records.dataset 行数=2 ；字段=['answer','contexts','question']
contexts 类型: list / 元素类型: str                    ← 强制 list[list[str]]，与文档一致
包含 ground_truth 列: False
跳过原因: [{"q2","答案为空"}, {"q3","没有检索到上下文，无法评测生成质量"}, {"(评测集整体)","有 1 条缺 ground_truth，已整体省略该列"}]
[全部条目都有 ground_truth 时] 含 ground_truth 列: True ；各列长度一致 {question:2, answer:2, contexts:2, ground_truth:2}
```

### 6.5 密钥：不硬编码、只脱敏

```
resolve_judge_config() 键: ['api_key','base_url','model','temperature']        ← 从 .env/settings 读
public_judge_config(): {"api_key": "sk-b******a7a2", "api_key_configured": true}
mask_secret('sk-abcdefghijklmnop') = sk-a******mnop ；mask_secret('')/None = '(未配置)'
[密钥扫描] eval_ragas.py: 无 ；eval_offline.py: 无
result_to_report() 里 report["judge"] = public_judge_config(...)               ← 报告与控制台都只拿脱敏值
```
`examples/ragas_eval_set.example.json` 实测 4 条：3 条带 `ground_truth`+`expected_sources`、1 条 `expect_no_answer` —— 与文档 §5 一致。

---

## 7. 文档声称 vs 实测（差异表）

| 文档所说的 | 我的实测 | 判定 |
| --- | --- | --- |
| §3.1–3.4 元数据/过滤/提示/退出码（含 `列表值`、`AND`、空结果提示、非法 filter 退出码 2） | 逐条复现（自造语料） | ✅ 一致 |
| §3.5 工具侧 `filter_json` + list 展示可过滤字段 + 非法 JSON 不抛异常 | 逐条复现 | ✅ 一致 |
| §4.2 embedding 冷/热（0/12 → 12/12） | 我的语料 0/5 → 5/5，且第二次 `encode()` 调用数为 0 | ✅ 一致 |
| §4.3 miss → hit → 失效；`reset()` 清理 | 逐条复现；但 `reset()` 后 **`.cache/` 目录仍在**（内容清空） | ⚠️ §4.1 措辞不符（F3） |
| §4.5「查询缓存键**没有** `top_n`」 | 旧版文档如此；现版本已改为「`top_n` 已进键，`top_k`/`rerank` 未进」——与 `cache.py:209` 一致 | ✅ 已由作者修正（F7） |
| §5.2 审计字段（含 `filter/variants/retry/cache/results/llm/citations/指纹`） | 46 条真实记录字段齐全；`llm.called` 语义偏松 | ⚠️ 字段齐全，`called` 语义见 F2 |
| §5.2「刻意不记原文」 | 自造 240 字正文在日志里查不到 | ✅ 一致 |
| §5.4 审计失败降级（failures 可见、查询不受影响） | 目录被文件占位 → 查询照常返回，failures=1/2 | ✅ 一致 |
| §6 向后兼容（不传 filter 逐值一致、`cached` 默认 False、缓存关闭不建目录） | 与 HEAD 对拍 5/5；`cached=False`；关闭时无 `.cache/` | ✅ 一致 |
| §6 实测数字「460 passed」「513 passed, 1 failed」 | 我实测：提交前 514 passed×2；提交后 **515 passed×2**（`d1c2f41`） | ⚠️ 陈旧快照（F5） |
| §10 用例数「metadata 50 / cache 18 / audit 18」 | 实测 50 / **19** / 18（cache 多了新加的 top_n 用例） | ⚠️ 数字差 1（F5） |
| 《eval 指南》§4.3 `--check` 退出码 2、§7.2 报告含脱敏 key、§8.1 本机装不上、§11 交付清单 | 逐条复现；`pyproject.toml` 确实未改 | ✅ 一致 |
| 《eval 指南》§3 的经验阈值（≥0.7 等） | 属建议值，非可验证断言 | — 不判 |

---

## 8. 遗留问题（findings）

| ID | 严重度 | 问题 | 证据 | requiredFix |
| --- | --- | --- | --- | --- |
| **F1** | low（**已在验证期间修复**） | 18:10:25 作者新增 `tests/test_rag_cache.py::test_query_cache_is_isolated_by_top_n` 时引用了未定义的 `DOC3`，导致该时刻工作树 **1 failed, 514 passed** | 该文件 mtime 18:10:25；失败详情 `NameError: name 'DOC3' is not defined`（134 行）；18:11:10 补完后我实测 515 passed×2 | 无需再修（已由作者在 `d1c2f41` 修掉）；建议后续新增用例时先本地跑一遍该文件再提交 |
| **F2** | low | 审计字段 `llm.called` 语义不实：取自"客户端是否可用"（`pipeline.py:880`），低分拒答路径**根本没调大模型**也记 `called: true` | stub 调用次数=0，而记录 `llm={called:true,elapsed:0.002,refusal_reason:"score_below_threshold"}` | 二选一：①把 `llm_called` 改为按"是否真的进入生成调用"计算（如由 `AnswerGenerator` 返回是否调用）；②保留实现但把字段更名为 `llm_client_available`，并在文档 §5.2 注明 |
| **F3** | low | 文档 §4.1 写「`reset()` 会把版本归零并**删除整个 `.cache/` 目录**」，实现只删三个缓存文件（`cache.py:285-301`），目录保留 | 我的探针：`reset()` 前 3 个文件、后 `.cache` 目录 `exists=True` | 改文档措辞为「清空 `.cache/` 内的缓存文件（目录保留）」，或让 `clear()` 追加一次 `rmdir` |
| **F4** | low | 文档 §3.4/§9 的过滤命令在 **Windows PowerShell 5.1**（本仓库 shell）下不可照抄：内层双引号被剥离，实参变成 `{部门: 风控}` → `过滤条件错误…` exit=2 | 我按文档原样执行 → exit=2；改成 `--filter '{\"部门\": \"风控\"}'` 即正常（1 条命中、`元数据过滤:{'部门': '风控'}(过滤掉 4 条候选)`） | 在 §3.4/§9 增加一行 PS 5.1 提示（转义引号或改用 `pwsh 7`/bash），或提供 `--filter-file <path>` 用法示例 |
| **F5** | low | 文档测试数字为陈旧快照：§6「460 passed」「513 passed, 1 failed」、§10「cache 18 例」；实际提交后 **515 passed**、cache **19 例**（t14 的 63 例 ✅ 准确） | 我实测 515 passed×2（19.54s/19.00s）、`pytest --collect-only` 逐文件计数 | 更新 §6/§9/§10 的数字为 515/87/19，并删掉"队友在改的 OCR 用例"这句（该失败已不存在） |
| **F6** | low | `app/rag/_write_probe.py` 仍存在于磁盘（内容已被清空为一行注释，且已被 `.gitignore` 忽略，不会进版本库） | `Get-Content app/rag/_write_probe.py` → 仅一行注释；`git ls-files` 为空；`.gitignore` 末尾新增 `app/rag/_write_probe.py` | 由有 `app/` 写权限的人物理删除（我的任务范围是"只读核查 + 报告"，未执行删除）；删除后可一并移除 `.gitignore` 里那条沙箱专用忽略 |
| **F7** | info（**验证期间已由作者处理**） | 我 18:05 读到的文档旧版称"缓存键没有 `top_n`"，与 `cache.py:209` 不符；`rag_data/meta_demo/` 空目录残留；`.audit/` 未被忽略 | 18:08:59 文档已改为「`top_n` 已进入键」；18:12 后 `meta_demo/` 已删除、`.gitignore` 命中 `.audit/`；`git status` 干净 | 无需动作 |

**证据等级说明（不拔高）**：
- RAGAS **真实评测未运行**（本机无 `ragas`/`datasets`，且 judge 需付费）：其"装好后 `--check` 返回 0"与"真实指标数值"**未验证**；已验证的是缺依赖提示、退出码 2、组装逻辑（stub）、脱敏与报告结构，文档对"本机跑不了"已如实标注。
- 生成缓存、审计 `llm` 字段、Agent 工具侧过滤均用 **stub 大模型**验证（零 API 花费）；**未**用真实大模型复核 `llm.elapsed` 的量级（文档示例 6.929s 未复核）。
- 缓存"半截 JSON/维度不一致"的降级：我验证了半截 JSON 与"载荷 chunk 已删"两条；"维度不一致"未单独造（代码路径已读，未实测）。

---

## 9. 最终判定

* **三项企业级能力真实可用**：元数据抽取与过滤（语义完整、空结果提示可操作）、三级缓存（命中/失效/降级/关闭全部实测）、
  审计日志（字段齐全、不含原文、写入失败不拖垮查询）。
* **向后兼容成立**：不传 `filter` 时与 `git HEAD` 实现 **5/5 逐值一致**；`cached` 有默认值；缓存关闭零副作用。
* **RAGAS 方案可用且诚实**：缺依赖给中文指引 + exit=2；组装逻辑与脱敏经 stub 验证；离线替代指标真跑通过，
  幻觉识别与"无可核验→None"都符合文档，且报告显式声明"非 RAGAS、extractive 天然 ~100% 只证明链路通"。
* **无夸大**：我发现的 7 条差异全部是"文档陈旧/措辞/字段命名"级别，其中两条（F1/F7）在本轮由作者自行修掉。
  **没有发现"声称已实现但代码没有"**，也没有把未实现部分说成已完成。

---

## 10. 证据落点（`tmpdir/verif_t15/`）

| 文件 | 内容 |
| --- | --- |
| `pytest_run1..5.txt` | 全量回归 5 次（514×2 → 1 failed → 515×2）与 selftest |
| `probe_ent.py` / `probe_ent_out.txt` | 自造语料、过滤语义、HEAD 对拍、审计字段与失败降级 |
| `probe_cache.py` / `probe_cache_out.txt` | 三级缓存、top_n 键、版本失效、reset、降级、关闭 |
| `mutate_cache_key.py` / `mutate_emb_key.py` / `pytest_mut_*.txt` | 两处回退变异（去掉版本号 / 键不含文本）→ 测试报红 |
| `probe_eval.py` / `probe_eval_out.txt` | 离线近似指标、RAGAS 组装/脱敏、工具侧 filter_json |
| `offline_stdout.txt` / `offline_report.json` | 离线评测真实输出与 JSON 报告 |
| `ragas_check.txt` | `--check` 真实输出（exit=2） |
| `probe_eval_stdout.txt`、`pytest_new3.txt`、`pytest_run3.txt` | 其余原始输出 |

> 本次唯一新增的仓库文件是本报告；业务代码、既有文档、`rag_data/` 均未被改动（`git status` 干净）。
