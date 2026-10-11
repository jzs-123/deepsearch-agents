# 自研 RAG 查询改写（Query Rewrite）：两级改写 + 多查询 RRF 融合 + 低分重试

> 版本：v1（t11 / 新功能 2/2）
> 代码：`app/rag/rewrite.py`（改写器）、`app/rag/pipeline.py::RagPipeline.search()`（接入）、`app/rag/cli.py`（可见性）
> 测试：`tests/test_rag_rewrite.py`（49 用例）｜ 评测集：`examples/rag_eval_set_rewrite.json` + 语料 `examples/test_docs_rewrite/`
> 前置文档：`docs/rag-self-built.md`（5.4 分词歧义、5.10 排序分/判定分不得混用）、`docs/rag-testing-guide.md`（命令）

---

## 1. 结论摘要

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | 解决的问题是**问法与文档写法对不上**：短查询（「成交比例」「达标率」「限额」）、口语化（「请问一下夜间要是没人回咋办」）、异词（问「成交比例」而文档写「转化率」） | §4、§5 |
| 2 | 两级改写：`local`（零成本，口语归一化 + 同义词/领域词扩展 + 专有名词数字强化）与 `llm`（一次调用生成 N 个变体 + 一句 HyDE 式"可能答案"），失败**一律降级为本地变体** | §2、§3.2、§6.4 |
| 3 | 接入方式是「**首轮不动 + 低分才重试**」：改写关闭时 `search()` 与引入前**逐值一致**（顺序/分数/通道都比对过），既有关闭开关也保证既有 292+ 用例全绿 | §3.1、§7.1 |
| 4 | 收益实测（改写专用评测集，10 题）：Hit@5 **87.50% → 100%**、MRR **0.7812 → 1.0000**、拒答准确率 **100% → 100%**；判定分层面 **4 条查询从"会拒答"变成"可答复"** | §5.1、§5.2 |
| 5 | 既有评测集**不回归**：`rag_eval_set.json` 100%/MRR 0.8333 → 不变；`rag_eval_set_finance.json` 100%/1.0000 → 不变（基线本来就饱和，改写无处提升） | §5.3 |
| 6 | 可审计性是默认输出：CLI `search`/`eval`、Agent 工具、`pipeline.last_search_debug` 都能看到**用了哪些变体、各自命中几条、是否重试、重试前后判定分** | §4 |
| 7 | 大模型改写实测可用但有成本：真实调用新增 4 个变体、耗时 **19.08s**；模型不可达时降级为本地改写且检索照常返回 | §6 |
| 8 | 诚实记录的局限：同义词表覆盖有限（「请问一下夜间要是没人回咋办」改写后判定分 0.0723→**0.1991**，仍**差 0.0009 没过阈值**）；hash 兜底向量下语义改写收益有限；拒答守门问题重试后分数也会被抬高（本项目实测仍低于阈值，但余量变小） | §8 |

---

## 2. 设计

### 2.1 两级改写器

```
question
  ├─ 第 1 级 local（零成本，默认开启）
  │   ① 口语/疑问句归一化:去掉「请问」「一下」「咋办」「是多少呢」这类对检索无贡献的成分
  │   ② 关键词 + 同义词/领域词扩展:jieba 分词(缺 jieba 时退化为单字+二元组,与 bm25.py 同一套兜底)
  │      再并入命中的同义词扩展词,词表内置在 rewrite.py,可用 RAG_REWRITE_SYNONYMS 追加
  │   ③ 专有名词/数字强化:型号(A13)、编号(3.2.1)、比例(91.5%)、金额(5000 元)、日期(2026 年 3 月)
  │      单独构成一个变体——这些词关键词路最敏感,混在长句里会被稀释
  └─ 第 2 级 llm（默认关闭，要花钱）
      一次调用生成 N 个检索变体 + 1 句"可能的答案片段"(轻量 HyDE:用答案的用词去检索)
      复用 app/rag/generator.py 的 ChatClient 与项目 .env,不新增大模型依赖
```

### 2.2 变体组装规则（决定"只增不减召回"）

| 规则 | 说明 |
| --- | --- |
| **原始查询永远第一项** | `rewrite(question)[0] == question`。变体只往后追加，所以改写开启后**不可能**因为改写而召回不到原来能召回的内容（融合层面是原始查询结果集的超集，见 `test_rewrite_does_not_lose_original_recall`） |
| 去重 | 变体之间、与原始查询之间按清洗后的字符串去重 |
| 数量上限 | `max_variants`（配置 `RAG_QUERY_VARIANTS`，默认 3，**含**原始查询） |
| 模式语义 | `off` 只返回原始查询；`local` 原始+本地；`llm` 原始+大模型（**失败则退回本地变体**）；`local+llm` 原始+本地+大模型 |
| 空/异常输入 | 空查询返回 `[""]`；模式非法抛 `ValueError`（配置早失败）；`rewrite_query()` 便捷函数任何异常都退化为 `[原始查询]` |

### 2.3 内置词表（节选）

```
成交比例 → 转化率/成交转化率      达标率 → 目标完成率/完成率/达成率
限额     → 授信额度/额度上限/单笔上限   没人回 → 无人应答/未响应
夜间     → 非工作时间/夜间时段      超时   → 超出时限/升级处理
复购 → 复购率/回购   客单价 → 平均订单金额   坏账 → 不良贷款/逾期
```

设计原则：**只放确定成立的业务等价词，不放"可能相关"的词**——错扩比不扩更伤精度。脏配置（缺 `=`、key 为空）只记录 warning 并忽略该条，不让整条检索链路起不来。

---

## 3. 接入检索链路

### 3.1 三种路径（向后兼容是硬指标）

```
search(question, rewrite=None)   # None=按配置(应用默认 local) / False=强制关闭 / True=首轮即改写
  │
  ├─ rewrite=False 或 配置 off :  单查询 → hybrid_search → 重排      ← 与引入改写前逐值一致
  │
  ├─ rewrite=True                : 变体 → 多变体检索 → RRF 融合 → 重排（不做低分重试）
  │
  └─ rewrite=None + 配置 local/llm:
       ① 首轮:只用原始查询(同上单查询路径)
       ② gate_before = grounded_score(结果)   # 向量余弦,不是重排分
       ③ gate_before < min_score 且 retry_on_low_score → 用变体重试一次
       ④ gate_after = grounded_score(重试结果);只有 gate_after >= gate_before 才采用重试结果
```

「首轮不动」有两个好处：**改写关闭时零成本零行为变化**；改写开启时绝大多数正常查询（判定分本来就够）也不会多花一次检索。

### 3.2 多变体融合（复用现成 RRF）

- 每个变体各跑一路稠密 + 一路稀疏，通道名带变体后缀：`dense@v0` / `sparse@v1` …
- 用 `app/rag/hybrid.py::reciprocal_rank_fusion` 融合（**复用，不另写一套**），权重沿用「稠密 1.0 / 稀疏 0.8」，原始查询权重不打折、变体乘 `VARIANT_DISCOUNT=0.85`（避免变体把原始查询挤掉）
- **去重靠 chunk_id**（`reciprocal_rank_fusion` 内部就用 chunk_id 建表）
- 融合后把 `dense_score`（各变体最高向量余弦）与 `sparse_score`（各变体最高 BM25 分）补回候选：RRF 只认通道名 `dense`/`sparse`，带后缀就不会自动填，**不补回来判定分就会退化成排序分**（正是 `docs/rag-self-built.md` 5.10 那类坑）
- 重排：多变体时用「在所有变体里的最高词面分」排序（只按原始短查询打分会让变体召回的片段重新掉下去，等于白改）；LLM 重排按原问题做列表式重排

### 3.3 判定分口径（重试与拒答必须同一把尺子）

重试前后的比较、最终拒答判定，**统一用 `RagPipeline.grounded_score()`（向量余弦）**，而不是 `results[0].score`（重排分，尺度完全不同）。落地时踩到过一个具体坑并已修：

> 早期版本用「重排前候选」算重试判定分，审计里出现 `判定分 0.1541 → 0.1541，改善=否`，而实际返回结果的判定分是 **0.2181**（重排把正确片段提到了第一位）。修法：每轮尝试都跑完「检索→重排」再算分（`_search_once`），审计与返回结果同源。

---

## 4. 可审计性（默认可见，不是 debug 开关）

### 4.1 `pipeline.last_search_debug`

```python
{
  "question": "成交比例",
  "rewrite_mode": "local",
  "variants": ["成交比例", "成交 交比 比例 转化率 成交转化率"],
  "variant_count": 2,
  "variant_hits": [{"variant": "成交比例", "dense_hits": 10, "sparse_hits": 10}, ...],
  "fused_count": 10,
  "gate_score": 0.2181,                 # 最终结果的向量余弦(与返回结果同源)
  "retry": {"enabled": True, "triggered": True, "threshold": 0.2,
            "gate_before": 0.1541, "gate_after": 0.2181, "improved": True,
            "variants": [...]},
  "rewrite": {"mode": "local", "llm_used": False, "llm_error": "", "degraded": False,
              "local_variants": [...], "elapsed": 0.0004},
  "rerank": True,
  "results": [{"rank": 1, "source": "交易指标.md", "chunk_index": 0,
               "channels": ["dense@v0","sparse@v0","dense@v1","sparse@v1"],
               "score": 0.6258, "dense": 0.2181, "sparse": 23.2068, "rrf": 0.0523}]
}
```

### 4.2 CLI 默认输出（`search`，实测）

```
$ .\.venv\Scripts\python.exe -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 1
问题:成交比例
命中 1 条(向量后端:hash:512,重排:lexical)
查询改写:local(变体 2 个 | 融合候选 10 条 | 判定分(向量余弦) 0.2181)
  · 变体「成交比例」→ 向量命中 10 条 / 关键词命中 10 条
  · 变体「成交 交比 比例 转化率 成交转化率」→ 向量命中 10 条 / 关键词命中 10 条
  · 低分重试:已触发(判定分 0.1541 → 0.2181,阈值 0.2,改善=是)

[1] score=0.6258 rrf=0.0523 dense=0.2181 sparse=23.2068 通道=['dense@v0', 'sparse@v0', 'dense@v1', 'sparse@v1']
    来源:交易指标.md#0 平台交易指标 > 一、转化情况
```

关闭改写时输出的是状态而不是变体清单（同样是默认输出）：

```
查询改写:off(变体 1 个 | 融合候选 10 条 | 判定分(向量余弦) 0.1541)
  · 变体「成交比例」→ 向量命中 10 条 / 关键词命中 10 条
  · 低分重试:已关闭
```

`--json` 额外给出结构化字段：`{"rewrite": {...}, "retry": {...}, "variants": [...], "variant_hits": [...], "results": [...]}`。

### 4.3 `eval` 的改写统计（实测）

```
查询改写: local(多变体查询 8/10 | 触发重试 8 | 重试后改善 6 | 大模型改写 0 次 / 降级 0 次)
逐条结果里还会标:  [HIT rank=1 score=0.2344 变体=2 已重试] 成交比例 -> ...
```

### 4.4 Agent 工具

`search_knowledge_base` / `ask_knowledge_base` 在**真的发生改写或重试**时追加：

```
【检索过程】
查询改写:local(变体 2 个 | 融合候选 10 条 | 判定分(向量余弦) 0.2181)
  · 变体「成交比例」→ 向量命中 10 条 / 关键词命中 10 条
  · 变体「成交 交比 比例 转化率 成交转化率」→ 向量命中 10 条 / 关键词命中 10 条
  · 低分重试:已触发(判定分 0.1541 → 0.2181,阈值 0.2,改善=是)
```

没有改写/重试时不追加，避免每轮都往模型上下文塞无关状态（既有工具输出格式与断言不受影响）。

---

## 5. 收益实测

### 5.1 改写专用评测集（`examples/rag_eval_set_rewrite.json`）

语料：`examples/test_docs_rewrite/`（10 份制度/指标文档 → 22 个片段，`hash:512` 兜底向量）。语料里**故意不出现查询原词**（例如只写「目标完成率」「授信额度上限」，不写「达标率」「限额」），否则测不出改写的作用。

| 配置 | 解释器 | Hit@5 | MRR | Recall@5 | 拒答准确率 | 多变体查询 | 触发重试 | 重试改善 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 关闭（基线） | 系统 3.10（有 jieba） | 87.50% | 0.7812 | 87.50% | 100% | 0/10 | 0 | 0 |
| B 默认（local + 低分重试） | 系统 3.10（有 jieba） | **100.00%** | **1.0000** | **100.00%** | 100% | 8/10 | 8 | 6 |
| C 始终改写（`--rewrite-always`） | 系统 3.10（有 jieba） | **100.00%** | **1.0000** | **100.00%** | 100% | 10/10 | 0 | 0 |
| A 关闭（基线） | `.venv` 3.12（**无 jieba**，BM25 退化） | 87.50% | 0.7812 | 87.50% | 100% | 0/10 | 0 | 0 |
| B 默认 | `.venv` 3.12（无 jieba） | **100.00%** | **1.0000** | **100.00%** | 100% | 8/10 | 8 | 5 |
| C 始终改写 | `.venv` 3.12（无 jieba） | **100.00%** | **1.0000** | **100.00%** | 100% | 10/10 | 0 | 0 |

命令（照抄即可复现）：

```powershell
python -m app.rag.cli ingest -k rewrite_demo -p examples/test_docs_rewrite --embedding-provider hash
python -m app.rag.cli eval   -k rewrite_demo --embedding-provider hash -e examples/rag_eval_set_rewrite.json -n 5 --no-rewrite
python -m app.rag.cli eval   -k rewrite_demo --embedding-provider hash -e examples/rag_eval_set_rewrite.json -n 5
python -m app.rag.cli eval   -k rewrite_demo --embedding-provider hash -e examples/rag_eval_set_rewrite.json -n 5 --rewrite-always
```

### 5.2 判定分前后对比（比 Hit@5 更能说明问题）

`Hit@k` 只看"正确文档在不在前 k 条"；但**到底会不会被拒答**取决于判定分（向量余弦 ≥ `RAG_MIN_SCORE=0.2`）。下表是同一 KB、同一查询的实测（系统 3.10，确定性）：

| 查询 | 判定分(关) | 低于阈值? | 判定分(开) | 触发重试 | 重试改善 | 首位来源(关) | 首位来源(开) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 成交比例 | 0.1541 | 是 | **0.2181** | 是 | 是 | 绩效考核办法.md#0 | **交易指标.md#0** |
| 达标率 | 0.1246 | 是 | **0.2418** | 是 | 是 | 风控指标口径.md#0 | 风控指标口径.md#0 |
| 限额 | 0.1595 | 是 | **0.4497** | 是 | 是 | 风控指标口径.md#1 | 风控指标口径.md#1 |
| 请问一下夜间要是没人回咋办 | 0.0723 | 是 | 0.1991 | 是 | 是 | 客服响应规范.md#1 | 客服响应规范.md#1 |
| KPI 达标情况 | 0.0415 | 是 | 0.1680 | 是 | 是 | 差旅费报销制度.md#0 | **风控指标口径.md#0** |
| 夜间没人应答算不算超时 | 0.0550 | 是 | **0.2273** | 是 | 是 | 客服响应规范.md#0 | 客服响应规范.md#1 |
| 客服响应时限是多少（对照） | 0.3782 | 否 | 0.3782 | 否 | — | 客服响应规范.md#0 | 客服响应规范.md#0 |
| A13 模型的准确率（对照） | 0.3655 | 否 | 0.3655 | 否 | — | 风控指标口径.md#0 | 风控指标口径.md#0 |
| 公司食堂本周的菜单是什么（应拒答） | 0.1469 | 是 | 0.1739 | 是 | 是 | 办公设备领用规定.md#0 | 办公设备领用规定.md#0 |

**读法**：6 条短/口语/异词查询在关闭改写时判定分全部低于 0.2（`ask` 会直接拒答），开启后 **4 条越过阈值变成可答复**；2 条对照查询分数与来源完全不变（改写没有乱动正常路径）；应拒答问题的分数被抬高了 0.027 但**仍低于阈值，拒答结论不变**。

### 5.3 单查询证据：改写前召不回、改写后召回并置顶

```
$ .\.venv\Scripts\python.exe -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 3 --no-rewrite
命中 3 条
[1] 来源:绩效考核办法.md#0 ...   [2] 来源:供应商管理办法.md#0 ...   [3] 来源:风控指标口径.md#0 ...
（正确文档 交易指标.md 不在前 3；判定分 0.1541 < 0.2 → ask 会拒答）

$ .\.venv\Scripts\python.exe -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 3
[1] score=0.6258 dense=0.2181 通道=['dense@v0','sparse@v0','dense@v1','sparse@v1']
    来源:交易指标.md#0 平台交易指标 > 一、转化情况
    （"平台整体的转化率为 3.2%" —— 正是问题想问的内容；判定分 0.2181 > 0.2 → 可答复）
```

### 5.4 既有评测集不回归

| 评测集 / 语料 | 关闭改写 | 默认（改写+重试） | 结论 |
| --- | --- | --- | --- |
| `examples/rag_eval_set.json` + `examples/test_docs` | Hit@5 100% / MRR 0.8333 / 拒答 100% | Hit@5 100% / MRR 0.8333 / 拒答 100% | **无变化**（重试触发 2 次并改善，但指标已饱和） |
| `examples/rag_eval_set_finance.json` + `rag_data/fin` | Hit@5 100% / MRR 1.0000 / 拒答 100% | Hit@5 100% / MRR 1.0000 / 拒答 100% | **无变化** |

> ⚠️ **必须说明"为什么没提升"**：这两份评测集的基线本来就是 100%/MRR 1.0（语料与查询用词高度重合，见 `docs/rag-self-built.md` 7.2 的局限说明），**天花板已到，改写无处发挥**。真正能体现收益的是 §5.1 里刻意构造的短查询/异词查询集。

---

## 6. 大模型级（llm / local+llm）

### 6.1 实测：真实调用成功

```
$ .\.venv\Scripts\python.exe -m app.rag.cli search -k rewrite_demo --embedding-provider hash `
    -q "请问一下夜间要是没人回咋办" -n 1 --rewrite-mode local+llm --rewrite-always --variants 4
查询改写:local+llm(变体 4 个 | 融合候选 10 条 | 判定分(向量余弦) 0.1991)
  · 变体「请问一下夜间要是没人回咋办」→ 向量命中 10 条 / 关键词命中 10 条
  · 变体「夜间要是没人回」→ 向量命中 10 条 / 关键词命中 4 条
  · 变体「夜间 间要 要是 是没 没人 人回 无人应答 未响应 非工作时间 夜间时段」→ ...
  · 变体「夜间无人回复 夜间值班 空岗 处理流程 升级联系人」→ ...          ← 大模型产出
  · 大模型改写:成功(新增 4 个变体,耗时 19.0835s)
    来源:客服响应规范.md#1 客服响应规范 > 二、非工作时间的安排
```

### 6.2 实测：模型不可达时降级（关键安全属性）

```
$ $env:OPENAI_BASE_URL='http://127.0.0.1:9/v1'    # 指向不可达地址
$ .\.venv\Scripts\python.exe -m app.rag.cli search -k rewrite_demo --embedding-provider hash `
    -q "成交比例" -n 1 --rewrite-mode local+llm --rewrite-always --variants 3
WARNING app.rag.rewrite: 大模型查询改写失败,降级为本地改写:Error code: 502
命中 1 条
查询改写:local+llm(变体 2 个 | 融合候选 10 条 | 判定分(向量余弦) 0.2181)
  · 变体「成交比例」→ ...   · 变体「成交 交比 比例 转化率 成交转化率」→ ...
  · 大模型改写:失败→已降级为本地规则(InternalServerError: Error code: 502)
[1] 来源:交易指标.md#0      ← 检索照常可用
exit=0
```

客户端不可用（没有 api_key/model）时同样只降级、不报错，`rewrite.degraded=True`（`tests/test_rag_rewrite.py` 用 stub 覆盖了这个分支，测试里绝不联网）。

### 6.3 成本与延迟

| 模式 | 每次查询的额外开销 | 实测 |
| --- | --- | --- |
| `local` | 0 次大模型调用、毫秒级字符串处理；多查询时向量/BM25 各多跑 (N-1) 次 | `rewrite.elapsed` 通常 < 1ms；检索本体是毫秒级 |
| `llm` / `local+llm` | **1 次大模型调用**（只在首轮判定分低或 `--rewrite-always` 时发生） | 本项目配置的推理型模型实测 **19.08s / 次**（`deepseek-v4-pro`） |
| 低分重试 | 命中重试时才多一轮「检索 → 重排」 | 例：`search()` 单次从 ~10ms 量级增到 ~20ms 量级（词面重排，无大模型） |

### 6.4 踩坑：推理模型的 token 预算（已修）

`.env` 配的是推理型模型时，**推理过程先消耗 completion tokens**。实测：

```
max_tokens=400 : finish_reason=length, content='' , reasoning_tokens=400
max_tokens=1200: finish_reason=length, content='成交比例 转化率 成交' , reasoning_tokens=1192
```

结论：`max_tokens=400` 时 content 恒为空字符串 → 每次都"看起来像大模型坏了"，其实是预算不够。代码已改为 `LLM_MAX_TOKENS=1500`（`app/rag/rewrite.py`），并保留"内容无法解析 → 降级本地"的兜底。**换模型/换供应商后要重新确认这个预算。**

---

## 7. 配置与兼容性

### 7.1 新增配置（`app/core/config.py` + `.env.example`）

| 变量 | 取值 | 应用默认 | 说明 |
| --- | --- | --- | --- |
| `RAG_QUERY_REWRITE` | `off` / `local` / `llm` / `local+llm` | **`local`** | 应用级（CLI / Agent 工具经 `load_rag_config()` 读取）默认开启本地改写 |
| `RAG_QUERY_VARIANTS` | ≥1 | `3` | 变体总数上限（含原始查询） |
| `RAG_RETRY_ON_LOW_SCORE` | true/false | `true` | 首轮判定分 < `RAG_MIN_SCORE` 时用变体重试一次 |
| `RAG_REWRITE_SYNONYMS` | `key=alt1\|alt2;key2=alt` | 空 | 追加同义词（内置表在 `app/rag/rewrite.py`） |

> **一个刻意的默认值差异**：`RagConfig`（数据类）的 `query_rewrite` 默认是 **`off`**，而 `Settings` 默认是 **`local`**。
> 原因：现有 292+ 用例与其它调用方直接构造 `RagConfig`，数据类默认 `off` 保证**引入改写前后行为逐值一致**（兼容硬指标）；
> 应用侧（CLI、Agent 工具）走 `load_rag_config()`，因此**默认开启**改写。切换点只有一处：`RagConfig.query_rewrite` → `Settings.rag_query_rewrite`。

### 7.2 CLI 开关（`search` / `ask` / `eval` 通用）

```
--rewrite-mode {off,local,llm,local+llm}   临时覆盖模式
--no-rewrite                               等于 --rewrite-mode off（对照基线用）
--variants N                               变体上限
--no-retry                                 低分不自动重试
--rewrite-always（search/eval）             首轮就用全部变体（不做低分重试）
```

### 7.3 测试

```
tests/test_rag_rewrite.py            49 个用例
  ├─ 改写器契约(原始查询第一/上限/空输入/非法模式/去重)          8
  ├─ 本地规则(口语归一化/同义词扩展/专有名词数字/配置词表/脏配置)  8
  ├─ 大模型级(stub:解析、HyDE、异常与不可用降级、乱码降级、两级合并) 7
  ├─ 检索接入(关闭时逐值一致、低分重试与判定分、关闭重试、多变体去重与通道、只增不减召回、审计) 15
  ├─ Agent 工具可见性(发生改写/重试才追加)                        3
  └─ CLI 可见性(文本/JSON/eval 统计)                             5
```

命令：`$env:PYTHONPATH='.uv-cache\pytest-deps'; python -m pytest tests -q`（或 `python -m pytest tests -q`，`pyproject.toml` 已配 `-p no:tmpdir -p no:cacheprovider`）。

---

## 8. 已知局限（不许只报好数据）

| # | 局限 | 证据 / 影响 |
| --- | --- | --- |
| 1 | **同义词表覆盖有限**，覆盖率上不去 | 「请问一下夜间要是没人回咋办」改写后判定分 0.0723 → **0.1991**，差 **0.0009** 没过阈值（`ask` 仍会拒答）。这类查询需要更全的领域词表或真实语义向量，不能靠规则穷举 |
| 2 | **hash 兜底向量下，"语义改写"帮助有限** | 当前环境没有 `sentence-transformers`，向量是字符 n-gram 哈希（只匹配词面）。改写之所以有效，主要靠**把文档里的词面加进查询**；换成真实 BGE 后同义改写（无字面重叠）才会真正发挥语义作用 |
| 3 | **判定分被整体抬高会挤压拒答余量** | 应拒答问题「公司食堂本周的菜单是什么」重试后判定分 0.1469 → 0.1739（+0.027），本例仍低于 0.2，但**换语料可能越过阈值**变成误答。启用重试后必须重新标定 `RAG_MIN_SCORE`（用 `eval` 看可答/不可答的分数分布） |
| 4 | 「达标率→目标完成率」这类同义扩展**可能扩错方向** | 例：「KPI 达标情况」首位来源从「差旅费报销制度」变成「风控指标口径」（改对了），但判定分 0.1680 仍低于阈值；词表里 `达标 → 目标完成率/完成率/达成率` 全部命中的前提下仍不足以越过阈值，说明**单靠规则改写的上限就在这里** |
| 5 | 多变体让**重排成本翻倍**（仅重试路径） | 低分重试会跑两轮「检索→重排」。词面重排是毫秒级，可忽略；若把 `RAG_RERANK_MODE=llm` 与重试同时开启，则会变成 **2 次大模型重排调用**，成本翻倍——高成本组合请谨慎 |
| 6 | LLM 改写延迟与不确定性 | 实测 **19.08s/次**（推理型模型）；同一次查询两次调用可能给出不同变体（非确定性），因此**评测数字只在 `local` 模式下具有可复现性**，`llm` 模式下请用多次运行或固定模型 |
| 7 | 词的切分质量受 jieba 是否安装影响 | `.venv` 缺 jieba → BM25 退化、关键词变体也会退化为字级（`成交 交比 比例 转化率`）；系统 3.10 有 jieba 时变体更干净（`成交 比例 转化率`）。两套环境的「多变体查询数/重试改善数」都列在 §5.1，数字略有差异属正常 |
| 8 | 变体上限压到 2 时收益明显下降 | `RAG_QUERY_VARIANTS=2` 只能容纳「原始 + 一个变体」，关键词扩展与专有名词强化不能兼得；默认 3 是收益/成本的折中 |

---

## 9. 复现命令（最小集）

```powershell
# 0) 建演示库（22 个片段，hash 兜底向量；rag_data/rewrite_demo 已在仓库工作区）
python -m app.rag.cli ingest -k rewrite_demo -p examples/test_docs_rewrite --embedding-provider hash

# 1) 单查询:改写关闭 vs 默认(local+低分重试)
python -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 3 --no-rewrite
python -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 3

# 2) 评测对比(表格数据来源)
python -m app.rag.cli eval -k rewrite_demo --embedding-provider hash -e examples/rag_eval_set_rewrite.json -n 5 --no-rewrite
python -m app.rag.cli eval -k rewrite_demo --embedding-provider hash -e examples/rag_eval_set_rewrite.json -n 5
python -m app.rag.cli eval -k rewrite_demo --embedding-provider hash -e examples/rag_eval_set_rewrite.json -n 5 --rewrite-always

# 3) 大模型级(要花钱;约 20s/次)
python -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 1 --rewrite-mode local+llm --rewrite-always --variants 4

# 4) 大模型不可用时的降级(不花钱:指向不可达地址)
$env:OPENAI_BASE_URL='http://127.0.0.1:9/v1'
python -m app.rag.cli search -k rewrite_demo --embedding-provider hash -q "成交比例" -n 1 --rewrite-mode local+llm --rewrite-always
Remove-Item Env:OPENAI_BASE_URL

# 5) 测试
python -m pytest tests/test_rag_rewrite.py -q
python -m pytest tests -q                     # 基线 292 + 本功能 49 = 341（不含队友在改的 OCR 用例）
python -m app.rag.selftest                    # 期望 18 通过 / 0 失败
```

---

## 10. 改动清单

| 路径 | 类型 | 说明 |
| --- | --- | --- |
| `app/rag/rewrite.py` | 新增 | 两级改写器（`QueryRewriter` / `rewrite_query` / `build_rewriter` / `parse_synonyms` / `RewriteResult`）+ 内置词表 + 大模型提示词 |
| `app/rag/pipeline.py` | 修改 | `search(rewrite=...)`、多变体检索与 RRF 融合（`_retrieve_variants`）、低分重试（`_search_once`）、`last_search_debug`、`format_search_audit()`、`stats()` 增字段、`query()` debug 增字段 |
| `app/rag/config.py` | 修改 | `RagConfig` 增 `query_rewrite` / `query_variants` / `retry_on_low_score` / `rewrite_synonyms` + 校验 + `load_rag_config()` 映射 |
| `app/core/config.py` | 修改 | `Settings` 增 4 个 `rag_*` 字段（应用默认 `local`） |
| `app/rag/cli.py` | 修改 | `search` / `ask` / `eval` 增改写开关；`search` 默认打印变体与重试审计（文本 + JSON）；`eval` 打印改写统计 |
| `app/rag/eval.py` | 修改 | `evaluate(..., rewrite=None)`、`EvalResult.rewrite_mode/rewrite_stats`、逐条结果带 `variants`/`retry_triggered`、`format_report` 增一行 |
| `app/tools/rag_tools.py` | 修改 | 检索/问答工具在发生改写或重试时附带「【检索过程】」审计块 |
| `.env.example` | 修改 | 新增查询改写配置段与对比实验命令 |
| `tests/test_rag_rewrite.py` | 新增 | 49 个用例（契约 / 本地规则 / 大模型降级 / 链路接入 / 工具与 CLI 可见性） |
| `examples/test_docs_rewrite/*.md` | 新增 | 10 份演示语料（3 份目标 + 7 份干扰，共 22 片段） |
| `examples/rag_eval_set_rewrite.json` | 新增 | 10 条「短查询/异词/口语」评测集（含 2 条拒答守门） |
| `rag_data/rewrite_demo/` | 运行时数据 | 演示库（hash:512，22 片段），可由 §9 第 0 步一条命令重建；可删 |
