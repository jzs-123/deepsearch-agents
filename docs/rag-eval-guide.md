# RAG 性能检测指南：检索指标（eval.py）与生成指标（RAGAS）

> 适用代码：`app/rag/eval.py`（已有，检索评测）、`app/rag/eval_ragas.py`（本指南新增，生成评测）、
> `app/rag/eval_offline.py`（本指南新增，零依赖离线近似）
> 相关设计文档：`docs/rag-self-built.md`（自研 RAG 的取舍与踩坑）

---

## 0. 一句话结论

**检索好不好，用 `eval.py` 测；答案好不好，用 RAGAS 测。两者互补，不能互相替代。**
本机装不上 RAGAS 时，用 `eval_offline.py` 拿一个**明确标注为非 RAGAS** 的近似抓手，
它只证明"链路通"，不证明"生成好"。

---

## 1. 为什么需要两套评测：检索指标 vs 生成指标

RAG 分两段：**检索**（把资料找出来）和**生成**（模型拿着资料写答案）。
两段会各自出错，而且错法完全不同：

| | 检索层 | 生成层 |
| --- | --- | --- |
| 回答的问题 | 该找的资料找到了吗？排序对吗？该拒答时拒了吗？ | 答案忠于资料吗？答到点子上了吗？ |
| 典型故障 | 召回漏了关键片段；无关片段排在最前；该拒答却硬答 | **幻觉**：数字/结论超出资料；答非所问；把资料说反 |
| 现有工具 | `app/rag/eval.py` → Hit@k / MRR / Recall@k / 拒答准确率 | **RAGAS** → faithfulness / answer_relevancy / context_precision … |
| 是否需要大模型 | 不需要（纯计算） | 需要（LLM 当评委 judge） |

**最容易被混淆的一点**：`eval.py` 里的"拒答准确率"是**检索层**的指标（比较的是向量余弦与阈值），
它**不检查答案内容**。所以哪怕 Hit@5 = 100%、拒答 100%，答案照样可能满嘴幻觉 ——
那正是 RAGAS 要抓的东西。

> 现状参考（`docs/rag-self-built.md` §7）：示例语料与真实研报语料上检索指标都是
> Hit@5 100% / MRR 1.0000 / 拒答 100%。**这些数字完全没有回答"答案是否忠于资料"**，
> 这就是本指南存在的理由。

---

## 2. RAGAS 是什么

RAGAS（Retrieval Augmented Generation Assessment）是一个**用大模型当评委**的 RAG 评测框架。
它把一次问答拆成 `question / answer / contexts / ground_truth` 四元组，然后用 LLM 逐项打分，
输出 0~1 之间的指标。

它测的是**生成质量**，与我们自研的检索评测互补。

**成本与性质**：每个指标都要调用 LLM（`answer_relevancy` 还需要 embedding 模型），
所以它有**费用、延迟、以及一定的随机性**。建议：
固定 judge 模型与版本；`temperature=0`；把 judge 配置写进报告（我们的脚本已自动记录，且**对 key 脱敏**）。

---

## 3. 三个常用指标：含义、范围、什么算好

范围统一是 **0 ~ 1，越高越好**。下面给的是**经验参考值，不是硬标准** ——
阈值随语料难度、judge 模型、甚至提示词而漂移，请用同一 judge 做纵向对比（改前 vs 改后）而不是横向攀比。

### 3.1 faithfulness（忠实度）—— 最重要

* **问的是**：答案里的每一条论断，是否都能由**检索到的上下文**支撑？
* **怎么算**：把答案拆成一堆论断，逐条问 LLM"这条能从上下文推出来吗"，比例即得分。
* **读法**：
  * `1.0`：答案完全由资料支撑（没有编造）。
  * `0.6`：约 4 成论断在资料里找不到依据 —— **有幻觉风险**，要看是哪几条。
  * 低分常见原因：模型用先验知识"补全"了资料里没有的数字/结论。
* **什么算好**：生产环境一般要求 **≥ 0.85**；涉及金额/合规的场景建议 **≥ 0.95**。
* **注意**：它**不**评价"答案是否切题"。答案只要全部来自上下文，哪怕答非所问也能拿高分。

### 3.2 answer_relevancy（答案相关性）

* **问的是**：这个答案是否真的在回答**用户的问题**？
* **怎么算**：拿生成的答案反推"它可能在回答什么问题"，再与原问题算相似度（所以**需要 embedding 模型**）。
* **读法**：`1.0` 高度切题；`< 0.6` 通常说明答非所问、或者答案里塞了大量与问题无关的上下文。
* **什么算好**：一般 **≥ 0.8**。
* **重要陷阱**：**拒答类答案是低分的**（"知识库中没有找到相关信息"与任何问题都不相似）。
  所以评测集里 `expect_no_answer: true` 的题目**必须排除在生成质量打分之外** ——
  我们的脚本会自动跳过它们并记录原因。

### 3.3 context_precision（上下文精确率）

* **问的是**：检索到的上下文里，**相关的那些是否排在前面**？（衡量检索噪声）
* **需要** `ground_truth`。
* **读法**：`1.0` 相关的都在最前面；低分说明前面塞了一堆无关片段，会稀释生成质量。
* **什么算好**：**≥ 0.7** 比较健康。它低的时候，先回去调 `RAG_TOP_K`/重排，而不是调提示词。

### 3.4 附加：context_recall（上下文召回率）

* **问的是**：`ground_truth` 里的事实，有多少能在检索到的上下文里找到？（衡量漏检）
* **需要** `ground_truth`；**≥ 0.8** 比较健康。它低说明该找的没找到，属于检索问题。

> 指标选择建议：**起步只用 `faithfulness` + `answer_relevancy`**（不需要参考答案，最省事）；
> 等积累了带 `ground_truth` 的评测集，再加 `context_precision` / `context_recall` 定位是检索还是生成的问题。

---

## 4. 在“你自己的机器上”安装与配置

> ⚠️ **RAGAS 不是本项目的核心依赖**，`pyproject.toml` **不会**包含它。
> 它是**用户侧可选安装**：装不上不影响项目运行，只是没法跑生成质量评测。

### 4.1 安装

```bash
pip install ragas datasets
```

* `ragas` 依赖 `datasets`（HuggingFace），后者常会牵出 `sentence-transformers` / torch 系依赖 ——
  这是**几百 MB 到数 GB** 的下载量，也是本机（Windows 沙箱）装不上的原因。
* 如果 `answer_relevancy` 报缺 embedding 模型，再补：

```bash
pip install sentence-transformers
```

* 或者：把 `answer_relevancy` 从指标里去掉（`--metrics faithfulness`），
  只跑纯 LLM judge 的指标，就能避开 embedding 依赖。

### 4.2 配置 judge 大模型（复用项目 `.env`）

RAGAS 的评委就是一个 OpenAI 兼容模型，**直接复用项目现有配置**，无需另外申请：

```dotenv
# .env（项目根目录）
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_DEEPSEEK=deepseek-v3            # 或 LLM_QWEN_MAX=qwen-max
```

也支持用环境变量**临时覆盖**（优先级高于 `.env`）：

| 环境变量 | 作用 |
| --- | --- |
| `RAGAS_LLM_MODEL` | judge 模型名 |
| `RAGAS_JUDGE_BASE_URL` | judge 接口地址 |
| `RAGAS_JUDGE_API_KEY` | judge 密钥 |
| `RAGAS_EMBEDDING_MODEL` | `answer_relevancy` 用的向量模型（默认 `text-embedding-v3`） |

**安全**：脚本**不硬编码任何密钥**，一律读 `.env` / 环境变量；
报告与控制台里只输出**脱敏形式**（如 `sk-1******cdef`），**绝不打印原始 key**。

### 4.3 先自检依赖

```bash
python -m app.rag.eval_ragas --check
```

装好了 → 打印"ragas 与 datasets 已安装"，退出码 `0`；
没装 → 打印中文安装指引，退出码 `2`（**不会抛裸 ImportError**）。

---

## 5. 准备评测集

格式：JSON 数组、`{"items": [...]}` 或 JSONL 均可。示例文件：
**`examples/ragas_eval_set.example.json`**

```json
{
  "items": [
    {
      "question": "2025年第四季度中国人民银行实施了什么类型的货币政策",
      "ground_truth": "实施适度宽松的货币政策，并推出一揽子货币金融政策组合，强化逆周期调节。",
      "expected_sources": ["中国人民银行2025年第四季度中国货币政策执行报告63页.pdf"]
    },
    { "question": "公司食堂本周的菜单是什么", "expect_no_answer": true }
  ]
}
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `question` | ✅ | 用户问题 |
| `ground_truth` | ➖ | 参考答案。**只有 all 条目都提供时才会带上**（`context_precision` / `context_recall` 需要它） |
| `expected_sources` | ➖ | 期望来源文件名（与 `eval.py` 的检索评测集**同一套写法**，可共用一份） |
| `expect_no_answer` | ➖ | `true` 表示该题应拒答；**不参与生成质量打分**，自动跳过 |

> **为什么要显式跳过拒答题**：`answer_relevancy` 会给拒答文案打低分（见 §3.2），
> 不排除的话会把整体指标拉低，制造"生成质量差"的假象。这条在 `docs/rag-self-built.md` §5.11
> 有同类教训 —— **评测脚本是尺子，尺子必须先校准**。

---

## 6. 怎么跑

```bash
# 1)（可选）先自检依赖
python -m app.rag.eval_ragas --check

# 2) 跑评测（默认指标：faithfulness + answer_relevancy）
python -m app.rag.eval_ragas -e examples/ragas_eval_set.example.json -k fin -o output/ragas_report.json

# 3) 只跑某几个指标（避开 embedding 依赖的最省事方式）
python -m app.rag.eval_ragas -e examples/ragas_eval_set.example.json -k fin --metrics faithfulness

# 4) 要 context_precision / context_recall，评测集必须带 ground_truth
python -m app.rag.eval_ragas -e examples/ragas_eval_set.example.json -k fin \
    --metrics faithfulness,answer_relevancy,context_precision,context_recall
```

参数：`-e` 评测集、`-k` 知识库名、`-o` 报告输出、`--top-n` 送入评测的片段数、`--metrics` 指标列表。

流程：对每条 `question` 调 `RagPipeline.query()` → 拿到 `answer` + 检索到的 `contexts` + `citations`
→ 组装成 RAGAS 记录 → 跑指标 → 输出 JSON 报告 + 控制台摘要。

---

## 7. 怎么读报告

### 7.1 控制台摘要

```
── RAGAS 生成质量评测 ──
评测条数: 3   跳过: 1
judge: deepseek-v3 @ https://dashscope.aliyuncs.com/compatible-mode/v1  key=sk-1******cdef

指标:
  faithfulness         0.8700
  answer_relevancy     0.8200

跳过的条目:
  - 公司食堂本周的菜单是什么: expect_no_answer：应拒答的题不参与生成质量评测
```

### 7.2 JSON 报告结构

```json
{
  "report_type": "ragas",
  "is_ragas": true,
  "evaluated_count": 3,
  "metrics": { "faithfulness": 0.87, "answer_relevancy": 0.82 },
  "skipped": [ { "question": "...", "reason": "..." } ],
  "has_ground_truth": true,
  "unavailable_metrics": [],
  "notes": ["已用 .env / 环境变量构造 judge LLM 与 embeddings"],
  "judge": { "model": "deepseek-v3", "base_url": "...", "api_key": "sk-1******cdef", "api_key_configured": true }
}
```

### 7.3 读法三条

1. **先看 `faithfulness`**：它是有没有幻觉的直接信号。低于 0.85 就去看是哪些答案。
2. **`answer_relevancy` 低 ≠ 检索差**：先确认拒答题已被跳过（看 `skipped`），再判断是不是提示词/生成的问题。
3. **`unavailable_metrics` 与 `skipped` 不是装饰**：它们说明"这次评测少算了什么"，写结论前必须看一眼。

---

## 8. 本机装不上 RAGAS：原因与离线替代

### 8.1 为什么本机跑不了

| 依赖 | 状态 | 后果 |
| --- | --- | --- |
| `ragas` | **未装** | 无法评测生成质量 |
| `datasets` | **未装** | RAGAS 的数据容器 |
| `sentence-transformers` | **未装** | `answer_relevancy` 的 embedding 依赖 |

根因：`datasets`/`sentence-transformers` 会牵出 **torch 系依赖**，而本项目开发环境的 `pip`
在装这类包时会**进程空转**（CPU 持续上升但无网络连接）。因此**本沙箱不装 RAGAS**，
只交付"能在用户机器上直接跑的脚本"。

真实演示（退出码 `2`）：

```
$ python -m app.rag.eval_ragas --check
缺少依赖:datasets、ragas。
RAGAS 属于「用户侧可选依赖」，本项目 pyproject.toml 不包含它。
请在你要跑评测的机器上安装:
    pip install ragas datasets
...
暂时装不上的话，可先用零依赖近似指标（明确标注为非 RAGAS）:
    python -m app.rag.eval_offline -e examples/ragas_eval_set.example.json -k fin --answer-mode extractive
```

### 8.2 离线替代：`app/rag/eval_offline.py`

**它是近似，不是 RAGAS。** 报告里 `is_ragas: false`、带 `disclaimer`，
且明确列出"本报告不含哪些 RAGAS 指标"。

它做两件事，只用标准库：

| 指标 | 含义 | 做法 |
| --- | --- | --- |
| `context_faithfulness_approx` | 答案里的**可核验事实**（数字/百分比/金额/型号/编号）能否在上下文中找到 | 抽事实 → 字面包含判断 → 找不到的记为**疑似幻觉** |
| `context_coverage_approx` | 上下文中包含答案**关键术语**的比例 | 术语集合更宽（事实 + 英文词 + 中文切词） |

真实运行（零依赖、不联网、不调用大模型）：

```bash
python -m app.rag.eval_offline -e examples/ragas_eval_set.example.json -k fin \
    --answer-mode extractive --top-n 4 -o output/ragas_offline.json
```

```
── 生成质量离线近似评测（非 RAGAS）──
知识库: fin   答案来源: extractive
样本数: 4   耗时: 1.741s

context_faithfulness_approx : 100.00%
context_coverage_approx     : 100.00%
疑似幻觉答案比例            : 0.00%
拒答准确率                  : N/A（本模式不适用）

⚠️  离线近似，非 RAGAS：仅做「可核验事实的字面包含」判断，无法识别同义改写/语义蕴含，
    存在假阳性与假阴性；正式生成质量评测请使用 RAGAS（见 docs/rag-eval-guide.md）。

注意: answer_mode=extractive：答案由检索到的原文直接拼接，因此两个指标天然接近 100%，
      只能证明评测链路可用，不能说明生成质量好。
注意: answer_mode=extractive 不会拒答，故拒答准确率记 N/A。
```

**为什么会是 100%？** 因为 `--answer-mode extractive` 的"答案"就是把检索到的原文拼起来，
它当然能在上下文里找到 —— 这个模式的作用是**验证评测链路可用**、并提供
"检索足够好时生成指标的上界"，而**不是**衡量模型。要评估真实生成质量，必须用：
`--answer-mode pipeline`（需要配好大模型）或 RAGAS。

### 8.3 离线近似的三条诚实边界（别当成 RAGAS 用）

1. **只做字面包含**：答案正确但换了说法（"适度宽松"写成"偏宽松"）会被误报为疑似幻觉；
   反过来，一字不差照抄上下文却逻辑错误，它查不出来。
2. **无可核验内容时给 `None`，不给 1.0**：答案通篇没有数字/专有名词时，判为"不可核验"并排除出均值，
   避免把"空话"判成满分。
3. **它不是 faithfulness**：RAGAS 的 faithfulness 判断的是**论断能否被上下文蕴含**（语义级），
   本近似判断的是**数值字面是否出现**（字符级）。**两者的分数不可比、不可互换、不得混在同一张表里。**

---

## 9. 推荐工作流

```
① 准备评测集（examples/ragas_eval_set*.json；检索与生成可共用同一份）
        │
② python -m app.rag.eval.py 口径的检索评测：Hit@k / MRR / Recall@k / 拒答率
        │   不达标 → 调 chunk_size / top_k / 重排 / 阈值，别碰提示词
        ▼   达标
③ python -m app.rag.eval_offline（可选，本机就能跑）：确认链路通、拿到上界
        │
④ python -m app.rag.eval_ragas：faithfulness / answer_relevancy（+ context_precision/recall）
        │   不达标 → 看是"检索噪声"（context_precision 低）还是"生成跑偏"（faithfulness 低）
        ▼
⑤ 把 judge 配置、指标、跳过的条目一并写进评测报告，便于复现
```

**定位小结**：`eval.py` 是**回归防线**（每次改检索参数都跑，快且免费）；
RAGAS 是**验收尺子**（改动大时跑，慢且有成本）；`eval_offline.py` 是**应急抓手**（装不上 RAGAS 时用）。

---

## 10. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `ModuleNotFoundError: ragas` | 没装。`pip install ragas datasets`（用户侧可选，不进 `pyproject.toml`） |
| `answer_relevancy` 报缺 embedding | 装 `sentence-transformers`，或改用 `--metrics faithfulness` 避开 |
| 指标全是 `N/A` / 报错说没有可评测记录 | 检索没返回上下文（空库 / 库名写错 / 阈值太高）。先看 `skipped` 里的原因 |
| `faithfulness` 异常高但答案明显不对 | 检查 `contexts` 是不是恰好包含了答案里的数字（字面包含的局限）；RAGAS 也可能被强提示词"骗"过 |
| 拒答题拉低了 `answer_relevancy` | 评测集里给它加 `expect_no_answer: true`，脚本会自动跳过 |
| 报告里 `unavailable_metrics` 非空 | 你的 ragas 版本没提供该指标，换版本或改用其它指标 |

---

## 11. 交付文件清单（t14）

| 文件 | 作用 |
| --- | --- |
| `app/rag/eval_ragas.py` | RAGAS 评测脚本（可选依赖；缺依赖给中文指引；key 脱敏） |
| `app/rag/eval_offline.py` | 零依赖离线近似（明确标注非 RAGAS） |
| `examples/ragas_eval_set.example.json` | 评测集示例（含 `ground_truth` 与 `expect_no_answer` 两种条目） |
| `tests/test_rag_eval_ragas.py` | 组装逻辑 / 缺依赖提示 / 脱敏，**全部 stub，不装不调 ragas** |
| `tests/test_rag_eval_offline.py` | 事实抽取 / **幻觉数字识别** / 报告诚实性 |
| `docs/rag-eval-guide.md` | 本文档 |

**未改动 `pyproject.toml`**：RAGAS 属于用户侧可选安装，不进入项目核心依赖。
