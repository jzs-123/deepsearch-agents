# t12 独立验证报告：OCR 与查询改写两个新功能

> 任务：t12「独立验证 OCR 与查询改写两个新功能（含回退变异与收益独立复算）」
> 验证人：verif ｜ 验证对象：t10（OCR）、t11（查询改写）
> 方法：**不接受自述** —— 自造与对方不同的样本、给真实实现套计数器、把修复回退做变异测试、
> 拿 `git HEAD`（引入改写前的实现）逐值对拍、自己从原始数据重算指标、离线复跑。
> 范围：只读核查 + 报告。**未修改任何业务代码与既有文档**，临时脚本与产物在 `tmpdir/verif_t12/`（已被 `.gitignore` 忽略）。

---

## 0. 结论摘要

| 判定项 | 结论 | 我的关键证据 |
| --- | --- | --- |
| OCR 真的可用 | ✅ 可用 | 自造**无文本层**扫描件 PDF → 入库 1 chunk → 检索命中 `score=1.0000`；图片 → `score=0.9771` |
| 「文本层优先」真的成立 | ✅ 成立（计数取证） | 我自造的混合 PDF（第 1 页图 + 第 2 页文本）**只 OCR 了 1 页**；整批 4 个样本 `OCR_CALLS=3`（扫描件 1 + 混合第 1 页 1 + 图片 1）；文本层 PDF `OCR_CALLS=0`、真实 17 页 PDF 整份 `text_layer / 21272 字符 / OCR_CALLS=0` |
| 降级真的不炸 | ✅ 成立 | load 级 5 种缺层 + ingest 级 3 种配置，**全部不抛异常**，均有 warning，链路仍可用 |
| OCR 质量数据是否如实 | ✅ 如实，含差案例 | 文档 §6.3 那张表（113/55、1389/1523、1344/1311、1169/1267）与封面乱码原文**逐项复现**；我自己还能造出第三种错法（`3.75`→`3 ． 75`），与其「错字随排版变」的声明一致 |
| 改写真的可行且有收益 | ✅ 有收益（我自己算的） | 关闭 `Hit@5 87.50% / MRR 0.7812 / 拒答 100%` → 默认与始终改写 `100% / 1.0000 / 100%`；我**从 per_query 原始数据重算**，与报告值一致 |
| 向后兼容真的成立 | ✅ 更强于其自证 | 与 `git HEAD`（引入改写前）的实现**10/10 逐值一致**（chunk_id + score + rrf，顺序也一致） |
| 原始查询不变量 | ✅ 成立 | 4 种模式 × 6 条查询（含空串）全部 `variants[0] == 原查询` 且 ≤ `max_variants` |
| 大模型降级真的不阻断 | ✅ 成立 | 端点不可达 → `llm_error=InternalServerError: Error code: 502`、返回结果正常，且与 local 模式 **5/5 结果等价** |
| 低分重试真的用向量余弦 | ✅ 成立（变异证明） | 把 `grounded_score` 强制成 0.99 → 不重试；强制 0.01 → 必重试；把 `grounded_score` 换成 `results[0].score` → **5 条测试转红** |
| 测试是否联网 | ✅ 不联网 | `OPENAI_API_KEY=''` + `OPENAI_BASE_URL=http://127.0.0.1:1/v1` 下 **364 passed**；测试代码无 requests/httpx/openai/tavily 引用 |
| 有无夸大 | ✅ 未见夸大 | 两文档都把差案例、未实测分支（rapidocr）、未提升项写在正文；我发现的差异方向是**文档比实测更保守**（见 §6/F4） |
| 交付完整度 | ⚠️ 2 处缺口 | **F1**：OCR 用例硬依赖**未提交**的二进制样本 PDF（缺文件时 9 条用例 FileNotFoundError）；**F2**：第三方 OCR 引擎自身抛异常时调用侧不兜底（内建三个后端不触发） |

---

## 1. 全量回归（命令 + 真实输出）

```powershell
cd G:\deepsearch-agents-main
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
$env:RAG_TEST_TMP = '...\tmpdir\verif_t12\pt1'      # 会话独有临时根
python -m pytest tests -q        # 第 1 次：364 passed in 16.59s   exit=0
                                 # 第 2 次（换 RAG_TEST_TMP）：364 passed in 16.38s  exit=0
python -m pytest tests -q        # 离线复跑（OPENAI_API_KEY=''、OPENAI_BASE_URL='http://127.0.0.1:1/v1'）
                                 # → 364 passed in 17.85s   exit=0
python -m app.rag.selftest       # → ──── 自测结果:18 通过 / 0 失败 ────   exit=0
python -m pytest tests/test_rag_ocr.py   --collect-only -q   # 23 tests collected
python -m pytest tests/test_rag_rewrite.py --collect-only -q # 49 tests collected
```

* 用例数自洽：292（既有）+ 23（OCR）+ 49（改写）= **364**，且 **0 skipped** ——
  OCR 的 4 条 `requires_windows_ocr` 真实用例在本机**实际执行**（不是静默跳过）。
* 工作区卫生：`rag_data/` 仍为 `fin`（10-10 17:07:35）、`rewrite_demo`（10-11 17:05:01）、`README.md`；
  我全部实验都写到 `tmpdir/verif_t12/**`，**没有触碰真实 `rag_data/`**。
  > 说明：`rag_data/rewrite_demo/` 是 t11 为演示写入真实数据目录的知识库（22 片段），
  > 其文档 §10 已声明「可由一条命令重建、可删」——属**已披露**的运行时数据，非测试副产物（见 F5）。

---

## 2. OCR（t10）逐项独立复现

### 2.1 我自造的样本（与 `tests/fixtures` 内容不同）

`tmpdir/verif_t12/make_samples.py`（PIL + reportlab，与对方生成器无关的独立实现）：

```
wrote scan_verif.pdf 39621 bytes     # 整页位图，无文本层
wrote text_verif.pdf 37955 bytes     # reportlab 直写文本（有文本层）
wrote mixed_verif.pdf 76670 bytes    # 第 1 页位图 + 第 2 页文本层
wrote img_verif.png  20882 bytes     # 单张图片
```

计数器方式：给**真实实现**套计数（`WindowsOcrEngine.recognize` / `WindowsPdfRenderer.render_page` 包一层）。

```
[engine] name=windows available=True  powershell=...\WindowsPowerShell\v1.0\powershell.exe
[1a 自造扫描件 3.1s]        extraction='ocr'        chars=52  OCR_CALLS=1 RENDER_CALLS=1
[1b 自造文本层 0.0s]        extraction='text_layer' chars=379 OCR_CALLS=0 RENDER_CALLS=0   ← 文本层优先
[1c 自造混合(图+文本) 2.2s] extraction='mixed'      chars=433 OCR_CALLS=1 RENDER_CALLS=1   ← 逐页判定
[1d 自造图片 0.8s]          extraction='ocr'        chars=30  OCR_CALLS=1 RENDER_CALLS=0
[2 fixture text_sample.pdf] extraction='text_layer' chars=351 OCR_CALLS=0
[2 fixture scan_sample.pdf] extraction='ocr'        chars=51  OCR_CALLS=1
```

`1c` 是「逐页判定」最直接的证据：**同一份 PDF 里只有无文本层的那一页被渲染+识别**。

### 2.2 端到端：扫描件 → 入库 → 检索（隔离 `RAG_DATA_DIR`）

```
[入库] files=4/4 skipped=0 chunks_added=4 total=4 OCR_CALLS=3
   detail: img_verif.png   chars=30  chunks=1 extraction=ocr
   detail: mixed_verif.pdf chars=433 chunks=1 extraction=mixed
   detail: scan_verif.pdf  chars=52  chunks=1 extraction=ocr
   detail: text_verif.pdf  chars=379 chunks=1 extraction=text_layer
[检索] q='库存周转率'            -> mixed_verif.pdf#0 score=1.0000
[检索] q='缺货率与平均周转天数'   -> mixed_verif.pdf#0 score=0.9995
[检索] q='仓储管理部 运营报告'    -> mixed_verif.pdf#0 score=0.9963
[检索] q='风控模型的准确率'       -> img_verif.png#0  score=0.9771
```

* 整批 `OCR_CALLS=3` = 扫描件 PDF(1) + 混合 PDF 第 1 页(1) + 图片(1)，与预期完全一致；
* `IngestReport.details[].extraction` 确实落到了报告里（文档 §5.2 的说法成立）。

### 2.3 降级矩阵（任何一层缺失都不应抛异常）

load 级（注入/配置 5 种缺层，全部 **未抛异常**，`extraction='none'` + 明确 warning）：

| 场景 | 结果 | warning |
| --- | --- | --- |
| `OcrOptions(backend='none')` | 不抛，chars=0 | 「OCR 后端不可用(backend=none),扫描件页将取不到内容…」 |
| `OcrOptions(enabled=False)` | 不抛，chars=0 | 「共 1 页未提取到文本(可能是扫描件/图片页)」 |
| `OcrOptions(min_text_chars=0)` | 不抛，chars=0 | 同上（显式关闭页级 OCR） |
| 注入 `NoneOcrEngine()` | 不抛，chars=0 | 「OCR 后端不可用(backend=windows)…」 |
| 注入 `NonePdfRenderer()` | 不抛，chars=0 | 「缺少可用的 PDF 渲染器,扫描件页将取不到内容」 |

ingest 级（整批入库，全部 **exit=0 未抛异常**，链路仍可用）：

```
[OCR 关闭]  未抛异常 files=2/4 skipped=2 total=2
[后端 none] 未抛异常 files=2/4 skipped=2 total=2
[阈值 0]    未抛异常 files=3/4 skipped=1 total=3   ← 混合 PDF 第 2 页文本层仍入库
```

### 2.4 识别质量：文档声称 vs 我的实测

| 文档声称 | 我的实测 | 判定 |
| --- | --- | --- |
| §5.3 贝莱德整份：`extraction=text_layer chars=21272 OCR_CALLS=0` | `extraction='text_layer' chars=21272 OCR_CALLS=0 warnings=[]` | **逐字符一致** |
| §5.3 `text_sample.pdf` 351 字符 / 0 次 OCR | 351 / 0 | 一致 |
| §5.3 `scan_sample.pdf` 51 字符 / 1 次 OCR | 51 / 1，且 OCR 文本＝文档所载 `风控模型的准确率为 91 丐 ％ …` | 一致（逐字） |
| §6.3 表格 1/5/7/9 页 = 113/55、1389/1523、1344/1311、1169/1267 | 113/55、1389/1523、1344/1311、1169/1267 | **全部一致** |
| §6.3 封面 OCR 原文（`6 年全球…不伺 00 添发`） | `6 年全球 / 投资展望 / 突破传统界限 / BlackRock / 贝莱德 / 贝莱德智库 / 业投资者使用 ， 不伺 00 添发` | 一致（含错字） |
| §6.4 第 7 页 OCR「额外读到图表标题 `债务水平分化`」 | 含 `债务水平分化`、含 `1950` | 一致 |
| §6.4 第 9 页 OCR 重复小标题 `…面临 / 面临限制` | `人工智能建设面临\n面临限制\n2024 一 2030 年…` | 一致 |
| §6.1 `91.5%` → `91 · 5 ％` | 我自造图片 OCR：`风控模型的准确率为 91 · 5 ％` | 一致 |
| §6.2 `91.5%` → `91 丐 ％` | 同一句在扫描件 PDF 里复现 | 一致 |

**我自己的独立质量数字**（自造扫描件，40px 微软雅黑，scale=2.0）：

```
原文    : 库存周转率为3.75次仓储管理部2026年第三季度运营报告缺货率与平均周转天数是核心指标（44 字符）
OCR 归一: 库存周转率为 3 ． 75 次 / 仓储管理部 2026 年第三季度运营报告 / 缺货率与平均周转天数是核心指标
按位一致率 = 43/44 = 97.73%；唯一错误：小数点的「.」→ 全角「．」，且两侧多出空格
```

→ 与文档结论一致：**汉字质量可用、数字与标点不可信**。我遇到的错法与文档记录的两种都不同
（`．` vs `·` vs `丐`），恰好印证其「**错的具体字符随渲染倍率/排版而变，结论稳定**」这一说明；
文档**没有只报好数据**：封面差案例、页脚乱码、丢 `202`、`rapidocr` 分支未实测都写在正文（§8.2、§9）。

### 2.5 回退变异（看测试是否真的锁住行为）

| 变异 | 做法 | 结果 |
| --- | --- | --- |
| B：默认阈值抬到 1e9（强制所有页都 OCR） | pytest 插件改 `OcrOptions.from_settings` | **364 passed** → 「文本层优先」的锁只在**用例显式注入 `OcrOptions`** 的路径上生效（注入是合法测试手法；这是覆盖边界，不是缺陷） |
| C：样本 PDF 指向不存在路径 | pytest 插件改 `test_rag_ocr.SCAN_PDF/TEXT_PDF` | **9 failed / 14 passed**（全部 `FileNotFoundError`）→ 见 F1 |

---

## 3. 查询改写（t11）逐项独立复现

### 3.1 不变量：原始查询永远第一

4 种模式 × 6 条查询（含空串、口语、专有名词、数字、百分比）全部满足 `variants[0] == 原始查询` 且 `len ≤ 3`：

```
[local  ] '成交比例'            -> ['成交比例', '成交 比例 转化率 成交转化率']
[local  ] '请问一下夜间要是没人回咋办' -> ['请问一下夜间要是没人回咋办', '夜间要是没人回', '夜间 要是 没人 无人应答 未响应 非工作时间 夜间时段']
[local  ] 'A13 型号的限额是多少'    -> ['A13 型号的限额是多少', 'A13 型号的限额', 'a13 型号 限额 授信额度 额度上限 单笔上限']
[local  ] '2026 年第三季度库存周转率 1.8%' -> [原查询, '2026 第三季度 库存 周转率 1.8', '1.8 1.8% 2026 年 2026']
[off/llm/local+llm] 同上（llm 不可用时也返回本地变体，未抛异常）
[边界] max_variants=1 -> ['成交比例']；mode='bogus' -> ValueError（早失败）
```

### 3.2 向后兼容：与 `git HEAD`（引入改写前）逐值对拍

把 `git show HEAD:app/rag/pipeline.py`（403 行，**不含任何 rewrite 关键字**）落到 tmpdir 里 import，
对同一份知识库、10 条查询比较 `(chunk_id, score, rrf_score)` 序列：

```
逐值一致检查：10/10 完全一致（不一致 0 条）
 例 q='成交比例' 新=[('6ab9a9fb',0.304167),('40a35bce',0.197727)] 旧=同
    q='客服响应时限是多少' 新=[('92652528',0.804116),('080043d6',0.182068)] 旧=同
```

→ 文档 §3.1 的「改写关闭时与引入前逐值一致」**成立**，而且这个证法比其自带用例更强（直接对拍旧实现）。

### 3.3 收益独立复算（我自己从 `per_query` 重算，不看报告值）

语料 `examples/test_docs_rewrite`（**11 个 .md**，实测入库 `files=11/11 → 22 chunks`），评测集 10 题、k=5：

| 配置 | 报告值 Hit@5 / MRR / Recall / 拒答 | **我重算** Hit@5 / MRR / Recall | 多变体 | 触发重试 | 重试改善 |
| --- | --- | --- | --- | --- | --- |
| 关闭（基线） | 87.50% / 0.7812 / 87.50% / 100% | 87.50% / 0.7812 / 87.50% | 0/10 | 0 | 0 |
| 默认（local + 低分重试） | 100% / 1.0000 / 100% / 100% | 100% / 1.0000 / 100% | 8/10 | 8 | **7** |
| 始终改写（`rewrite=True`） | 100% / 1.0000 / 100% / 100% | 100% / 1.0000 / 100% | 10/10 | 0 | 0 |

* 系统 3.10（有 jieba）与 `.venv` 3.12（无 jieba）**两套解释器**我都跑了，Hit@5/MRR/Recall/拒答四项完全一致；
* 「重试改善」实测两套解释器都是 **7**，而文档 §5.1 写 6（3.10 行）/5（.venv 行）、§4.3 写 6 → 见 F4；
* 拒绝答守门未被破坏：应拒答两题仍判为拒答（`0.1739` / `0.1147`，均 < 0.2）。

### 3.4 单查询与判定分（文档 §5.2/§5.3 的复算）

```
# .venv 3.12（无 jieba）—— 与文档 §5.2/§5.3 数字逐值一致
[--no-rewrite] gate=0.1541 首位=绩效考核办法.md#0  retry=False
[默认]         gate=0.2181 首位=交易指标.md#0     retry=True  变体=['成交比例','成交 交比 比例 转化率 成交转化率']
[gate 表] 0.1541→0.2181 | 0.1246→0.2418 | 0.1595→0.4497 | 0.0723→0.1991 | 0.0415→0.1680 | 0.0550→0.2273
          | 0.1469→0.1739（应拒答） | 0.1147→0.1147（应拒答）
# 系统 3.10（有 jieba）—— 同一批查询因为关键词变体不同，数值更高
[gate 表] 0.1541→0.2344 | 0.1246→0.2458 | 0.1595→0.4497 | 0.0723→0.2485 | 0.0415→0.1792 | 0.0550→0.3436
```

→ 文档 §5.2/§8.1 的数字**是 `.venv` 那一套**（可逐值复现），但文档没有标注解释器；
在 3.10 上口语例的判定分是 **0.2485（>0.2）**，即 §8.1「0.0723→0.1991 仍差 0.0009 未过阈值」这条局限
**在 3.10 上不成立**（不是夸大，而是漏标解释器 + 数字偏保守，见 F4）。

### 3.5 低分重试确实用 `grounded_score`（回退变异）

```
[5a 强制判定分 0.99] retry.triggered=False
[5b 强制判定分 0.01] retry.triggered=True  gate_before=0.01 gate_after=0.01 variants=[原查询, 关键词变体]
=> 重试门槛读的确实是 grounded_score（向量余弦）

# 更强的变异：把 grounded_score 换成 results[0].score 后跑整套测试
python -m pytest tests -q -p mutate_gate   →  5 failed, 359 passed
  FAILED test_agent_rag_tools.py::test_search_tool_gate_uses_vector_cosine
  FAILED test_rag_eval.py::test_refusal_uses_vector_cosine_not_rank_score
  FAILED test_rag_pipeline.py::test_grounded_score_prefers_dense_cosine_over_rank_score
  FAILED test_rag_pipeline.py::test_grounded_score_picks_first_available_dense_score
  FAILED test_rag_pipeline.py::test_grounded_score_ignores_rerank_score_scale
=> 「排序分/判定分不得混用」这条语义被测试真正锁住（不是空断言）
```

### 3.6 大模型降级（不花钱、不联网）

```
[6a local+llm 但地址不可达(http://127.0.0.1:9)] 6.8s 返回 3 条
     rewrite_mode=local+llm  llm_used=False  llm_error='InternalServerError: Error code: 502'
     variants=[原查询, 关键词变体]（本地变体仍在）
[等价性] local+llm(不可达) 与 local 模式在同 5 条查询上：结果与判定分 **5/5 完全等价**
[6c] 空 api_key 客户端 available=False
[WARNING] 大模型查询改写失败,降级为本地改写:Error code: 502
```

### 3.7 测试是否真的不联网

```
$env:OPENAI_API_KEY=''; $env:OPENAI_BASE_URL='http://127.0.0.1:1/v1'
python -m pytest tests -q   →  364 passed in 17.85s  exit=0
grep tests/*.py  (requests|httpx|openai|tavily|urllib|socket)  →  0 命中
```

---

## 4. 文档一致性核对

| 项 | 文档 | 实测 | 判定 |
| --- | --- | --- | --- |
| `docs/rag-ocr.md` 行数 | 未自报 | 352 行 | — |
| `docs/rag-query-rewrite.md` | 自报 331 行 | **398 行** | 行数偏低（F4 同类） |
| OCR 文档 §10.3「23 例」 | 23 | `23 tests collected` | 一致 |
| 改写文档 §7.3「49 例」 | 49 | `49 tests collected` | 一致 |
| OCR 文档 §10.2「`pytest tests -q` → 315 passed（292+23）」 | 315 | 当前 **364 passed**；且发布当时裸跑实为 11 failed（t10 自己的任务报告已披露） | 命令/数字陈旧（F3） |
| 改写文档 §5.1「10 份制度/指标文档」 | 10 | 语料目录 **11 个 .md**，入库 11/11 → 22 chunks | 文档偏少 1（F4） |
| 改写文档 §5.1/§4.3「重试改善 6 / 5」 | 6、5 | 两套解释器均 **7** | 数字陈旧（F4） |
| 改写文档 §7.1「Settings 默认 local / RagConfig 默认 off」 | 刻意差异 | `query_rewrite='local'`（load_rag_config）与 `RagConfig().query_rewrite='off'` | **一致** |
| 改写文档 §4.4「只在发生改写/重试时追加【检索过程】」 | — | `rag_tools.py:43` 判据 `variant_count>1 or retry or llm_error` | **一致** |
| 配置同步 | 4+4 个 `rag_*` 字段进 `Settings` / `RagConfig` / `.env.example` | 实测到位（`config.py:81-98`、`.env.example` diff） | 一致 |
| 「声称已实现但代码没有」 | — | 未发现（逐条比对 §2/§3 的功能点都落到代码） | 通过 |
| 「代码实现了但文档没写」 | — | 未发现实质性遗漏（`extraction` 进 `IngestReport.details`、`last_search_debug`、CLI JSON 字段都有写） | 通过 |

---

## 5. 未验证 / 证据等级说明

| 项 | 等级 | 说明 |
| --- | --- | --- |
| 真实大模型改写成功（文档 §6.1：新增 4 个变体、19.08s/次） | **未验证** | 需要真实付费调用，本次只验证了**降级路径**（不可达地址 → 502 → 本地兜底）。文档对该路径的描述与我的实测一致 |
| `rapidocr` 后端真实识别 | **未验证（文档已声明）** | 本机未安装，`build_ocr_engine('rapidocr')` 实测降级为 `NoneOcrEngine` + warning；文档 §8.2/§9.10 自己就写了「未在本机实测」 |
| 非 Windows 平台行为 | **按构造推断** | `available` 判据是 `sys.platform != 'win32'`（代码可读），未在非 Windows 跑过 |
| `llm` 重排 + 重试的成本翻倍（文档 §8.5） | **未验证** | 未开 `RAG_RERANK_MODE=llm` |
| 「文本层优先」在默认配置路径上的锁 | **覆盖边界** | 变异 B（默认阈值 1e9）套件仍全绿：锁只覆盖显式注入 `OcrOptions` 的用例；注入是合法手法，故不判缺陷 |

---

## 6. 遗留问题（findings）

| ID | 严重度 | 问题（具体到文件与行为） | 证据 | requiredFix |
| --- | --- | --- | --- | --- |
| **F1** | **medium** | `tests/test_rag_ocr.py:31-32` 硬依赖 `tests/fixtures/scan_sample.pdf`、`text_sample.pdf`，但这两个二进制样本**未纳入版本控制**（`git status` 显示整个 `tests/fixtures/` 为 `??`），且测试模块**没有**「缺失则生成」逻辑 → clean checkout 上 `pytest tests` 会出现 9 条 `FileNotFoundError` | 变异 C（把样本路径指向不存在文件）：`9 failed, 14 passed`，全部 `FileNotFoundError`；`rag-ocr.md` §10.1 只把生成脚本列为可选步骤，§10.2 的 `pytest tests -q` 未含该步 | 二选一：①把两个 PDF（各约 40KB）一起提交；②在 `tests/test_rag_ocr.py` 加 session 级 autouse 夹具，`if not SCAN_PDF.exists(): importlib 加载 make_scan_sample.py 并生成`（生成器已在文件内被 import，改动量约 5 行）。推荐 ②，避免二进制入仓 |
| **F2** | low | 第三方/自定义 `OcrEngine.recognize()` 抛异常时调用侧不兜底：`loaders._ocr_pdf_page`（`loaders.py:283`）与 `loaders.load_image`（`loaders.py:330`）直接调用 `engine.recognize(...)`，异常会穿透 `load_pdf`/`load_image` 并中断整批入库；与 `ocr.py:146` 的接口约定（「失败返回空串并 warning，不抛业务异常」）以及 `rag-ocr.md` §4.2「单页渲染/识别失败 → 该页留 warning + 空文本，不影响其它页」不一致。内建三个后端内部都 try/except，故当前链路不触发 | 注入 `RaisingEngine`（available=True、recognize 抛 RuntimeError）后：`[3h] RAISED RuntimeError ← 调用侧未兜底`、`[3h' 图片分支] RAISED` | 在 `_ocr_pdf_page` 与 `load_image` 里把 `engine.recognize(...)` 包进 `try/except Exception`，失败时 `warnings.append(f"第 N 页 OCR 失败:{exc}")` 并返回空串（与渲染失败同款处理），同时补 1 条用例 |
| **F3** | low | `docs/rag-ocr.md` §10.2 的命令与数字陈旧：写 `python -m pytest tests -q` → 「315 passed」，既未标注需要先跑 `make_scan_sample.py`，也未标注当时 bare run 因并发的改写用例有 11 failed；当前该命令实测 **364 passed** | `pytest tests -q` = 364 passed；t10 自己的任务报告写「裸跑 11 failed」 | 把 §10.2 更新为「先 `python tests/fixtures/make_scan_sample.py`（或依赖 F1 的自动生成）→ `pytest tests -q` → **364 passed**」，并注明用例数会随其它任务增长 |
| **F4** | low | `docs/rag-query-rewrite.md` 三处数字与当前实现不符：① §5.1 写「10 份制度/指标文档」，语料实际 **11 个 .md**；② §5.1/§4.3 的「重试后改善 6（3.10）/5（.venv）/6」实测**两套解释器均为 7**（`improved = gate_after > gate_before`，含 1 条应拒答对照题也计为"改善"）；③ §5.2/§8.1 的判定分表未标注解释器——数字与 `.venv`（无 jieba）逐值一致，但在系统 3.10（有 jieba）上更高（口语例 0.2485 而非 0.1991），导致 §8.1「仍差 0.0009 未过阈值」在 3.10 上不成立 | 我的两套解释器实测：`retry_improved=7`（两者）；`.venv` gate 表与文档逐值一致；3.10 gate 表更高 | 把 §5.1 的「10 份」改 11；把「重试改善」由 6/5 更新为 7（并说明该计数含应拒答对照题）；在 §5.2/§8.1 表头标注解释器（`.venv` 3.12 无 jieba），并补一行 3.10 的数值或注明 3.10 下口语例已过阈值 |
| **F5** | low | t11 把演示知识库写进了**真实数据目录** `rag_data/rewrite_demo/`（22 片段），使 `rag_data/` 不再只含 `fin + README.md`。文档 §10 已声明「运行时数据、可由一条命令重建、可删」，属已披露 | `rag_data/` 实际内容；`docs/rag-query-rewrite.md:397` | 若希望真实数据目录保持干净：删除 `rag_data/rewrite_demo/`，并在 §9 第 0 步保留重建命令即可；或在文档里明确「演示库允许留在 `rag_data/`」 |
| **F6** | low | 两个新功能的**全部代码/文档/测试目前都处于未提交状态**（`app/rag/ocr.py`、`render.py`、`rewrite.py`、`_powershell.py`、两份文档、`tests/test_rag_ocr.py`、`tests/test_rag_rewrite.py`、`tests/fixtures/`、`examples/test_docs_rewrite/` 均为 `??`；`.env.example`、`config.py`、`cli.py`、`config.py`、`eval.py`、`loaders.py`、`pipeline.py`、`rag_tools.py` 为 `M`）。仓库当前 HEAD 里这两个功能都不存在 | `git status --porcelain` 全量输出 | 交付前做一次提交（建议与 F1 一起：提交时补上样本生成夹具，避免二进制或缺失文件进仓） |

> 补充说明（不构成 finding）：变异 B 显示「有文本层不触发 OCR」的锁只覆盖显式注入 `OcrOptions` 的路径；
> 由于注入是该套件一贯且合法的写法，且判定逻辑本身同源，这只是**覆盖面边界**，已在 §5 记录。

---

## 7. 最终判定

* **OCR**：功能真实可用（自造样本端到端检索命中）、**文本层优先**有计数证据（混合 PDF 只 OCR 1 页）、
  降级矩阵全部不炸、质量数据（含差案例）**逐项可复现且未夸大**。
* **查询改写**：不变量成立、**与引入前实现 10/10 逐值一致**、收益经我独立复算为
  `Hit@5 87.50%→100%`、`MRR 0.7812→1.0000`、拒答保持 100%、大模型不可达时与 local 模式 5/5 等价、
  低分重试口径经变异测试证明是向量余弦、测试全程离线。
* **不足**：F1（medium，OCR 用例依赖未提交的样本文件）与 F2（low，引擎异常未兜底）需要修；
  F3–F6 是文档数字/卫生与提交状态问题。**没有发现夸大表述**，两处差异方向都是文档比实测更保守。

---

## 8. 证据落点（全部在 `tmpdir/verif_t12/`）

| 文件 | 内容 |
| --- | --- |
| `pytest_run1.txt` / `pytest_run2.txt` / `pytest_offline.txt` | 全量回归 364 passed ×2 + 离线复跑 |
| `selftest.txt` | `18 通过 / 0 失败` |
| `make_samples.py` + `samples/` | 我自造的扫描件/文本层/混合 PDF 与图片 |
| `probe_ocr.py` / `probe_ocr_out2.txt` | 计数器、降级矩阵、质量复算、贝莱德 1/5/7/9 页、抛异常引擎 |
| `probe_ocr_e2e.py` / `ocr_e2e.txt` | 自造样本 → 入库 → 检索（OCR_CALLS=3） |
| `probe_ocr_degrade_ingest.py` / `ocr_degrade.txt` | 入库级降级（关闭 / none / 阈值 0） |
| `probe_ocr_doc_claims.py` / `ocr_doc_claims.txt` | 贝莱德整份 21272/0 与第 7/9 页图表标题核对 |
| `probe_rewrite.py` + `probe_rewrite_out_310.txt` / `_venv312.txt` | 不变量、HEAD 对拍、eval 复算、重试变异、大模型降级（两套解释器） |
| `probe_rewrite_demo.py` / `rwdemo_stdout.txt` | 用交付方自己的 `rag_data/rewrite_demo` 复算（排除语料差异） |
| `mutate_gate.py` / `pytest_mutate_gate.txt` | 变异 A：`grounded_score`→排序分 → 5 failed |
| `mutate_ocr_threshold.py` / `pytest_mutate_ocr.txt` | 变异 B：默认阈值 1e9 → 364 passed（覆盖边界） |
| `patch_missing_fixture.py` / `pytest_missing_fixture.txt` | 变异 C：样本 PDF 缺失 → 9 failed（F1 证据） |
| `rw310_stdout.txt` / `rwvenv_stdout.txt` | 探针控制台输出（含 warning） |

> 本报告是本次唯一新增的仓库文件；业务代码、既有文档、`rag_data/` 均未被改动。
