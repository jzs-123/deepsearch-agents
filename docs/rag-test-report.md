# 自研 RAG + Agent 工具层：pytest 测试套件报告（t2 交付）

> 任务：t2「为 app/rag 与 Agent 工具层建立 pytest 测试套件」
> 依据：`docs/rag-agent-test-plan.md`（t1 审计）、`docs/rag-self-built.md`、captain 环境情报
> 结论：**14 个测试文件、292 个用例、292 passed、exit code 0**，全离线、不加载真实模型、不调用大模型、不写 `rag_data/`。

---

## 1. 结论摘要

| 项 | 结果 |
| --- | --- |
| 测试文件数 | **14**（要求 ≥4） |
| 用例总数（collect） | **292**（要求 ≥25） |
| 运行结果 | **292 passed, 0 failed**（exit=0） |
| 离线 | 是：向量后端固定注入 `HashEmbedder`，`sentence-transformers` 未安装也可跑通 |
| 大模型 | 不调用。conftest 强制清空 `OPENAI_API_KEY`（见 §9.3） |
| 数据隔离 | 所有知识库落在 `tmp_path`，**全程未写 `rag_data/`** |
| 覆盖 t2 的 7 类测试点 | 全部覆盖（§4） |
| 历史回归 bug 专门用例 | 全部覆盖（§5） |
| 发现但未修的业务 bug | **1 个：D1**（见 §7）；除它之外为空 |

### D1 生命周期说明（诚实记录）

t2 交付时 D1 **尚未修复**，因此当时有 2 个用例以 `xfail(strict=False)` 锁它的**期望行为**，
运行结果是 `290 passed, 2 xfailed`。captain 随后在 `c811a96` 修复了 D1，这两条用例按设计
自动翻转为 `XPASS`（**未改测试**，证明防线生效）。

但 `xfail(strict=False)` 在修复后会变成"哑火的锁"：将来 D1 若回归，失败会被报成 xfail
而**静默吞掉**。因此修复落地后，我已把这两条改回**硬断言**（去掉 xfail 标记），
它们现在断言的是**修复后的正确行为**，回归会立刻报红。当前运行结果：**292 passed**。

即：D1 已从「未修缺陷」转为「已修 + 已armed 的回归锁」，见 §7。

---

## 2. 确切运行命令

### 2.1 标准命令（当前环境唯一可跑通的方式）

```powershell
cd G:\deepsearch-agents-main
$env:PYTHONPATH      = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests -q
```

因为 `pyproject.toml` 里配了 `testpaths = ["tests"]`，**不带路径**也可以：

```powershell
python -m pytest -q          # 实测：292 passed
```

### 2.2 为什么需要 `PYTHONPATH`

本沙箱下系统 site-packages **不可写**，`python -m pip install pytest` 会输出一行
`Defaulting to user installation ...` 后**空转 3 分钟以上无任何输出**（已按规则终止、未重试）。
因此 pytest 采用工作区内的**离线副本**：`.uv-cache/pytest-deps/`（纯 Python 包，从 conda 环境
`langchain1.2` 里已有的 pytest 9.0.3 复制而来，无需联网；`.uv-cache/` 已被 `.gitignore` 忽略）。
运行时借 `PYTHONPATH` 引入，**不修改任何用户环境**。

> 若以后 `.venv` 里装了 pytest（t3 负责），可直接用：
> `.\.venv\Scripts\python.exe -m pytest tests -q`（截至本报告，`.venv` 尚无 pytest）。

### 2.3 关键点：**不需要 `--basetemp`，也不需要设 `TEMP`**

`pyproject.toml` 的 `addopts` 里带了 `-p no:tmpdir -p no:cacheprovider`，因此：

* pytest **根本不创建 basetemp** —— 之前那整类「`0o700` 目录自毁 / 跨会话不可删 / 复用了被毁目录就全崩」的问题
  被从设计上绕开了，不再需要在命令里传 `--basetemp` 或 `TEMP`；
* `tmp_path` 由 `tests/conftest.py` **自供**，一律用**默认 mode** 建目录；
* 实测证据：`%TEMP%` 下没有 `pytest-of-*` 目录；仓库根的 `.pytest_cache` 时间戳停在
  早先他人的运行（16:49:26），**未被本套件触碰**。

这也意味着：**「temp 必须在同一条命令里设置、不能依赖外部状态」这个坑在本套件里不存在** ——
没有任何环境变量参与临时目录的定位。

### 2.4 `pyproject.toml` 新增配置（仅新增 pytest 配置，未改业务配置）

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-p no:tmpdir -p no:cacheprovider"
markers = ["e2e: 需要真实大模型/网络/已入库知识库的端到端用例（默认跳过，RAG_E2E=1 开启）"]
```

---

## 3. 测试清单（每个文件覆盖什么）

| 文件 | 用例数 | 覆盖内容 |
| --- | --- | --- |
| `tests/conftest.py` | — | 沙箱安全 `tmp_path`、仓库根入 `sys.path`、`RAG_DATA_DIR` 隔离、强制离线（清空 `OPENAI_API_KEY`）、`HashEmbedder`/`RagPipeline`/假大模型客户端等夹具 |
| `tests/test_rag_types.py` | 9 | `Chunk`/`RetrievedChunk`/`Answer` 序列化往返、未知字段兼容、引用格式、`content_fingerprint`、`make_chunk_id` 稳定性与敏感性 |
| `tests/test_rag_config.py` | 13 | 非法 provider/rerank_mode/chunk_size/overlap/top_n 校验、不可变性、`with_kb`/`kb_dir`、`ensure_within` 越界防护 |
| `tests/test_rag_loaders.py` | 27 | `normalize_text`（换行/零宽/\xa0/CJK 空格）、多编码 txt（utf-8/BOM/GBK）、Markdown 标题与 title、CSV 行转句、PDF 空页与缺依赖提示、不支持后缀、`iter_supported_files`、**缺陷 D1 的观测与期望用例** |
| `tests/test_rag_chunking.py` | 20 | 标题识别与 `re.MULTILINE` 回归、标题路径栈、中文编号标题 vs 正文列表、重叠窗口（含句边界回退）、无标点硬切、`max_chunks`、代码块围栏、chunk 长度上界 |
| `tests/test_rag_embedding.py` | 22 | hash 后端形状/归一化/确定性/零向量不产生 NaN、`build_embedder` 分派与 `local→hash` 降级、`allow_fallback=False` 必须报错、openai 缺 key、指纹格式、懒加载 |
| `tests/test_rag_vectorstore.py` | 22 | 持久化往返、`chunk_id` 幂等去重、`doc_id` 整篇替换、指纹校验三条路径（打开/追加/维度）、落盘撕裂截断、`clear`、`search` 排序与 top_k 夹取、`list_knowledge_bases` 容错 |
| `tests/test_rag_bm25.py` | 18 | 型号 `A13`/编号 `3.2.1` 整体保留、停用词、jieba 缺失退化、专有名词命中、TF/IDF 单调性、标题参与索引、`explain` |
| `tests/test_rag_hybrid.py` | 16 | RRF 三态（仅稠密/仅稀疏/两路）、名次贡献与加权算术、双路共识胜过单路第一、rank 重编号、top_k、`overlap_count`、r 权重覆盖 |
| `tests/test_rag_rerank.py` | 21 | **中文分词致重排失效的回归**、二元组与字符覆盖率两级兜底、rerank 真的换序、同分稳定、`off/lexical/llm` 分派与降级、`_parse_order` 鲁棒性、LLM 重排失败 fail-open |
| `tests/test_rag_generator.py` | 18 | `build_context` 编号/标题/按整条截断、**低于门槛拒答且不调用大模型**、排序分与判定分分离、无客户端/抛异常降级、refusal marker 出口识别、长回答不误判 |
| `tests/test_rag_pipeline.py` | 27 | `grounded_score` 取向量余弦（**5.10 回归**）、`add_texts` 走切块路径、幂等、`search` 排名与通道、`query` 拒答不调模型、`ingest` 报告与整篇替换、跳过原因、`reset`/`stats`、`get_pipeline` 缓存、chunk_size 变更告警 |
| `tests/test_rag_eval.py` | 21 | 来源归一化（**假失败回归**）、`load_eval_set` 三种格式、Hit@k/MRR/Recall@k、**拒答判定与 `grounded_score` 一致（假通过回归）**、`format_report`、评测不调用生成模型 |
| `tests/test_agent_rag_tools.py` | 26 | 三个 `@tool` 的形态与参数 schema、必须 `.invoke({...})`（直接调用 TypeError）、空库/不存在/指纹不匹配/异常等**所有分支都返回中文提示不抛异常**、**A3 门槛口径回归**、**A4 幽灵目录回归**、离线性守门 |
| `tests/test_agent_prompts_contract.py` | 10 | `prompts.yml` 与 `sub_agents_content` 键集合一致、`rag` 已取代 `ragflow`、三个工具名出现在子智能体提示词、`knowledge_base_agent` 接线、全 `app/` 无 `ragflow_tools` 导入残留 |

> 用例数按 `def test_` 统计；`@pytest.mark.parametrize` 展开后 collect 总数为 **292**。

---

## 4. t2 要求的 7 类测试点 → 用例映射（验收对照）

| # | 要求 | 覆盖文件 | 代表用例 |
| --- | --- | --- | --- |
| 1 | 切块（MULTILINE、中文编号粘连、重叠、硬切） | `test_rag_chunking.py` | `test_md_heading_finditer_finds_every_heading`、`test_cn_numbered_body_line_is_not_a_heading`、`test_with_overlap_prefers_sentence_boundary`、`test_hard_split_for_long_text_without_punctuation` |
| 2 | 向量库指纹校验与幂等 | `test_rag_vectorstore.py` | `test_fingerprint_mismatch_on_open_raises`、`test_same_chunk_id_is_deduplicated`、`test_delete_document_replaces_whole_document` |
| 3 | 混合检索 RRF 三态 + BM25 专有名词 | `test_rag_hybrid.py`、`test_rag_bm25.py` | `test_dense_only_hit_records_dense_channel`、`test_two_channel_hit_accumulates_both_scores`、`test_proper_noun_query_hits_the_right_chunk` |
| 4 | 重排不得因分词差异整体失效 | `test_rag_rerank.py` | `test_score_for_split_words_is_not_zero`、`test_rerank_actually_reorders_candidates` |
| 5 | 拒答门槛用向量余弦 + 低于阈值不调大模型 | `test_rag_pipeline.py`、`test_rag_generator.py` | `test_grounded_score_prefers_dense_cosine_over_rank_score`、`test_query_refuses_irrelevant_question_without_calling_llm`、`test_gate_below_threshold_refuses_without_calling_llm` |
| 6 | 评测来源比对忽略目录大小写 + 拒答与线上一致 | `test_rag_eval.py` | `test_matches_ignores_directory_prefix`、`test_evaluate_hit_when_expected_source_only_written_as_filename`、`test_refusal_uses_vector_cosine_not_rank_score` |
| 7 | 工具层三工具 `.invoke()` 全分支中文提示 | `test_agent_rag_tools.py` | `test_tools_are_not_directly_callable`、`test_search_never_raises_on_embedding_mismatch`、`test_search_on_missing_kb_reports_and_creates_nothing` |

---

## 5. 历史回归 bug → 专门用例（逐条锁死）

| 历史 bug（`docs/rag-self-built.md`） | 锁它的用例 |
| --- | --- |
| 5.3 `re.MULTILINE` 缺失 → 标题全丢、整篇只切 1 块 | `test_md_heading_pattern_has_multiline_flag`、`test_md_heading_finditer_finds_every_heading`、`test_chunk_document_splits_on_every_markdown_heading` |
| 5.2 中文编号标题粘连正文（畸形标题） | `test_cn_numbered_body_line_is_not_a_heading`、`test_is_cn_title_decision_table` |
| 5.4 中文分词差异致重排整体失效（分数全 0） | `test_score_for_split_words_is_not_zero`、`test_feature_extraction_adds_adjacent_bigrams`、`test_char_coverage_ignores_low_signal_chars` |
| 5.5 换向量模型静默产生垃圾结果 | `test_fingerprint_mismatch_on_open_raises`、`test_fingerprint_mismatch_on_add_raises` |
| 5.7 重复入库导致结果全是同一段 / 过期版本残留 | `test_same_chunk_id_is_deduplicated`、`test_ingest_same_file_twice_replaces_instead_of_appending`、`test_add_texts_goes_through_chunk_document` |
| 5.10 排序分与判定分混用 → 拒答率 100%→50% | `test_grounded_score_prefers_dense_cosine_over_rank_score`、`test_gate_uses_gate_score_not_rank_score`、`test_search_tool_gate_uses_vector_cosine`、`test_refusal_uses_vector_cosine_not_rank_score` |
| 5.11 评测来源比对假失败（Hit@5 报 0%） | `test_matches_ignores_directory_prefix`、`test_evaluate_hit_when_expected_source_only_written_as_filename` |
| 5.11 评测拒答判定假通过（尺子未校准） | `test_refusal_uses_vector_cosine_not_rank_score`、`test_refusal_miss_when_cosine_above_threshold` |
| 5.8 依赖缺失被静默跳过 | `test_ingest_unsupported_file_is_skipped_with_reason`、`test_local_embedder_raises_actionable_error_without_dependency` |

---

## 6. 实际运行结果（原始输出）

命令：

```powershell
cd G:\deepsearch-agents-main
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests -q
```

原始 stdout（**当前**，D1 修复后）：

```
........................................................................ [ 24%]
........................................................................ [ 49%]
........................................................................ [ 73%]
........................................................................ [ 98%]
....                                                                     [100%]
292 passed in 6.73s
```

D1 修复**前**（t2 交付时刻）的原始 stdout，保留作为对照：

```
........................................................................ [ 24%]
........................................................................ [ 49%]
.............................................xx......................... [ 73%]
........................................................................ [ 98%]
....                                                                     [100%]
290 passed, 2 xfailed in 8.88s
```

修复落地时那两条 xfail 自动翻转（**未改测试**即证明防线生效）：

```
XPASS tests/test_rag_loaders.py::test_normalize_should_preserve_blank_line_between_cjk_paragraphs - D1 未修：normalize_text 用 r'\s+' 而非 r'[ \t]+'，CJK 字符之间的换行/空行被整体吃掉
XPASS tests/test_rag_loaders.py::test_markdown_structure_should_survive_normalisation - D1 未修：H1 被粘上正文后整篇退化成 1 个 chunk、heading 全空（症状同历史 bug 5.3，根因不同）
290 passed, 2 xpassed in 6.73s
```

随后去掉 xfail 标记（转为硬断言）→ `292 passed`。

收集数：

```
292 tests collected in 1.15s
```

可重入性：同一条命令**连续执行两次**均为 `292 passed`、`exit=0`
（不存在 basetemp 被复用/被毁的问题，因为压根没有 basetemp）。

---

## 7. 发现但未修的业务 bug 清单

> 按 t2 约定：发现业务代码缺陷**记录并交回 captain，不顺手改**。除下列 1 条外，
> 本轮**没有发现其他未修缺陷**（明确声明：清单非空，但只有 1 条）。
>
> **状态更新**：D1 在 t2 交付后由 captain 于 `c811a96` 修复（`\s+` → `[ \t]+`），
> 现为**已修 + 已 armed 的回归锁**，不再是未修缺陷。以下保留原始发现记录（当时的形态），
> 并补充修复后的验证结果。

### D1 — `normalize_text` 会吃掉 CJK 字符之间的换行，静默破坏结构感知切块

**状态：已修（captain `c811a96`，t2 交付后修复）；t2 全程未改业务代码。**

* **位置**：`app/rag/loaders.py:61`
  ```python
  text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
  ```
* **根因**：注释写的是"去掉中文字符之间**被逐字抽取产生的空格**"，但用的是 `\s+`，
  而 `\s` **包含换行符**。于是只要空行两侧都是中文字符（不是标点），整个换行就被删掉。
  按意图应写 `[ \t]+`。
* **实测证据**：
  | 输入 | 实际输出 | 期望 |
  | --- | --- | --- |
  | `"甲\n\n乙"` | `"甲乙"` | `"甲\n\n乙"` |
  | `"甲\n乙"` | `"甲乙"` | 视场景 |
  | `"第一段内容。\n\n第二段内容。"` | 保持不变 | 保持不变 |
  | `"第一段：\n\n第二段"` | 保持不变 | 保持不变 |

  （后两行以**标点**结尾，因此不触发替换 —— 这正是该缺陷长期不易被发现的原因。）

* **下游后果（真正的危害）**：Markdown 一级标题后面紧跟中文正文时，标题行会把正文
  **吞进自己**，导致 `chunking._split_headings()` 一个 section 都切不出来，走
  `if not sections` 兜底分支 —— **整篇退化成 1 个 chunk、`heading` 全空**。
  实测：
  ```
  # 风控报告\n\n风控模型的准确率为 91.5%。\n\n## 授信\n\n授信要求复核。
  → title: "风控报告风控模型的准确率为 91.5%."   （标题被粘上正文）
  → chunk_document(...) → 1 个 chunk，heading == ""
  ```
  这与历史 bug 5.3（漏 `re.MULTILINE`）的**症状一模一样，但根因完全不同** ——
  排查时极易被误判成"正则又漏了 MULTILINE"。
* **严重度**：**high**。理由：
  1. 不抛任何异常，只是**静默降低检索质量**——正是本项目反复强调"最难发现的一类 bug"；
  2. 触发条件极普遍：中文文档里"标题行以汉字结尾 + 下一段以汉字开头"几乎必然命中；
  3. 直接打掉 `docs/rag-self-built.md` 2.1/5.2 主打的"结构感知切块"能力，
     并让 chunk 的 `heading`（用于检索匹配的结构信息）全部为空；
  4. 与 `LoadedDocument` 自己的契约冲突：其 docstring 明确写"正文(**段落之间用空行分隔**)"。
* **影响范围**：`.md` / `.txt` / `.csv` 以及 PDF 逐页文本（都经过 `normalize_text`）。
  `app/rag/selftest.py` 与示例评测之所以没报警，是因为它们直接构造 `LoadedDocument`、
  或在语料上恰好未触发（见上表"以标点结尾"的行）。
* **建议修复**（未执行，越界）：把该行改为
  ```python
  text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", text)
  ```
  保留其后的 `re.sub(r"\n{3,}", "\n\n", text)` 做空行压缩；
  修完请把 `app/rag/selftest.py` 与 `examples/` 评测集一并重跑，确认 Hit/MRR 未回退。
* **回归防线**：埋了 2 条用例
  （`test_normalize_preserves_blank_line_between_cjk_paragraphs`、
  `test_markdown_structure_survives_normalisation`）。
  修复前它们以 `xfail(strict=False)` 锁期望行为；修复后自动 `XPASS`（证明防线生效），
  随后已去掉 xfail 标记转为**硬断言**，避免将来回归被 xpass 静默掩盖。
  其中 `test_markdown_structure_survives_normalisation` 直接断言
  `doc.title == "风控报告"`、`len(chunks) >= 2`、`all(chunk.heading)`，
  把 D1 回归的三个可观察症状一次性钉死。
* **修复后的独立验证（captain 提供）**：`python -m app.rag.selftest` → 18 通过 / 0 失败；
  用修复后代码重建 `fin` 知识库，chunk 数 **238 → 248**（恢复 10 处结构边界），
  评测无回退：Hit@5 100% / MRR 1.0000 / Recall@5 100% / 拒答 100%。
* **本次自检的局限（诚实说明）**：我原打算用"把 `[ \t]+` 改回 `\s+`、看断言是否报红"来**直接**
  验证防线，但**写入 `app/rag/loaders.py` 被沙箱拒绝**（`app/` 下不可写），该实验未执行、
  结论未取得；`loaders.py` 保持 captain 的修复版、未被改动（`git diff` 为空）。
  防线的有效性由此前 `xfail → XPASS` 的实测翻转证明，而不是由这次失败的实验证明。

---

## 8. 本轮已验证「已修并被回归锁锁死」的集成缺陷

这两条由 captain 在 `39b4b91` 修复，本套件按**修复后**的行为写断言（不锁旧行为）：

| 编号 | 缺陷 | 锁它的用例 | 断言要点 |
| --- | --- | --- | --- |
| A3 | `search_knowledge_base` 用排序/重排分与按余弦标定的 `min_score` 比较（5.10 同类问题发生在工具层） | `test_search_tool_gate_uses_vector_cosine`、`test_search_tool_returns_fragments_when_cosine_is_high` | 造"排序分 0.95 越阈值、余弦 0.01 远低于阈值"的候选：必须报"资料不足"，且提示里的分数是 **0.010（余弦）**、不得出现 `0.950`；反向用例保证不是恒拒答 |
| A4 | 传不存在的 `kb_name` 会凭空创建空库目录 | `test_search_on_missing_kb_reports_and_creates_nothing`、`test_ask_on_missing_kb_reports_and_creates_nothing`、`test_missing_kb_hint_lists_available_knowledge_bases`、`test_get_pipeline_create_false_raises_instead_of_creating` | 返回含"不存在 / 未创建也不会自动创建 / 可用库清单"的中文提示；调用前后**数据目录下目录集合必须完全相同**，且目标目录不存在 |

---

## 9. 环境偏差点与说明（复现必读）

### 9.1 落点偏差：`tests/` 而非 `app/rag/tests/`

t2 默认约定写的是 `app/rag/tests/**`，但本沙箱 **`app/` 下完全不可写**（目录和文件都被拒绝），实测：

| 目标 | 结果 |
| --- | --- |
| 新建 `app/rag/tests/` | 拒绝 |
| 新建 `app/rag/_tests_probe/` | 拒绝 |
| 新建 `app/rag/_probe_file.py` | 拒绝 |
| 新建 `app/_probe_dir/` | 拒绝 |
| 新建仓库根 `tests/` | **成功** |

所以备选方案里的"单文件 `app/rag/tests_rag.py`"同样不可行。经 captain 批准改用仓库根 `tests/`，
并相应把 `testpaths` 指向 `tests`。**这是本交付唯一的落点偏差。**

### 9.2 pytest 来自工作区内的离线副本

见 §2.2。`uv` 亦不可用（其缓存在工作区外，实测 `os error 5 拒绝访问`）。
若确需 uv，必须先设 `$env:UV_CACHE_DIR='G:\deepsearch-agents-main\.uv-cache\uv'`。

### 9.3 强制离线：本机 `.env` 里配了**真实**大模型 key

`app/core/config.py` 的 `settings` 是 `lru_cache` 模块级单例，而本机 `.env` 里
`openai_api_key` / `llm_model` / `openai_base_url` **均为非空**。工具层
`ask_knowledge_base → pipeline.query() → _llm_client()` 因此会拿到**真实客户端**并真的联网调用大模型
（开发期实测确实发生了，返回了一段像答案的文本）。

这不满足"离线、可复现"的验收要求，且要花钱、结果不确定。因此 `tests/conftest.py` 的 autouse 夹具
在重建 settings 之前 `setenv("OPENAI_API_KEY", "")`，使 `ChatClient.available is False`。
并埋了一条**守门用例**：

* `test_suite_runs_offline_without_llm_credentials` —— 断言 `load_rag_config().llm_api_key == ""`
  且 `ChatClient.available is False`。若将来有人放开这个开关，套件会立刻报红。

### 9.4 数据目录隔离

`RAG_DATA_DIR` 指向 `tmp_path`，且 autouse 夹具重建 `settings` 并把 `app.rag.config.settings`
一并替换（import 之后只改环境变量是无效的，因为 `settings` 已缓存）。
另有 `test_suite_runs_offline...` 同族的目录集合断言证明不会产生幽灵目录。
**全程未创建/修改/删除 `rag_data/`（用户 `fin` 库原样）。**

### 9.5 已知环境限制（已从设计上规避，不是阻塞项）

本环境有**两类**临时目录风险，团队已定论（措辞按 captain 终审口径）：

1. **`0o700` 目录自毁**：3.12/3.13 下用 `os.mkdir(p, 0o700)` 建出的目录
   （pytest 自带 tmpdir、`tempfile.mkdtemp` 都走它），**同一进程随即就 listdir/write/自删失败**；
   3.10.9 免疫。captain 用三解释器同进程对照稳定复现。
   *（如实记录分歧：在我的会话里 3.12.11 下 `0o700` 与 `mkdtemp` 均成功，说明该现象随会话/ACL
   上下文而异；但它在 captain 的会话里稳定复现，故作为真实风险保留，不因单点观察而淡化。）*
2. **跨会话所有权 / ACL**：删除**他人会话**创建的目录在 3.10 / 3.12 / 3.13 下**都会失败**
   （`rmtree/rmdir → WinError 5`）。`tmpdir/` 下遗留了一批这样的目录
   （`pt`、`ptbase`、`pdir1`、`ptfresh`、`pytest-of-lenovo` 等），连 `icacls`/`Get-Acl` 读取都是
   `Access is denied`，`Remove-Item -Recurse -Force` 也删不掉。

**t2 的处置（从设计上规避，不依赖绕过技巧）**：

* `pyproject.toml` 的 `addopts` 带 `-p no:tmpdir` → **pytest 根本不创建 basetemp**，
  第 1 类问题被整体消除；实测 `%TEMP%` 下没有 `pytest-of-*`，仓库根 `.pytest_cache` 未被触碰；
* `tmp_path` 由 `conftest.py` 用**默认 mode** 自供；
* 临时根使用**本会话独有**的 `tmpdir/pytest-tmp-tester`（可用 `RAG_TEST_TMP` 覆盖），
  不复用任何已毁路径，规避第 2 类；
* ACL 修复**未执行**：修复脚本必须以非受限（unconfined）方式运行，而本会话审批已禁用，
  按 skill 规定记为 undiagnosed 并停止（未手改任何权限）。

**这不是阻塞项**：t2 已在设计层面规避整类问题，后续任务无需把它当阻塞项处理。

### 9.6 仓库卫生

* `tests/` 下测试文件是要入库的正式交付物；
* 临时根 `tmpdir/pytest-tmp-tester`、pytest 离线副本 `.uv-cache/pytest-deps` 均已被 `.gitignore` 忽略；
* 运行过程不产生 `.pytest_cache`（`-p no:cacheprovider`）。

---

## 10. 复现清单（照抄即可）

```powershell
# 0) 一次性：确认 pytest 离线副本仍在（拷贝自 conda langchain1.2 的 pytest 9.0.3）
Test-Path G:\deepsearch-agents-main\.uv-cache\pytest-deps\pytest

# 1) 跑全套（从仓库根执行）
cd G:\deepsearch-agents-main
$env:PYTHONPATH       = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest tests -q
# 期望：292 passed（exit=0）

# 2) 只看 xfail 原因
python -m pytest tests -q -rx

# 3) 只跑某一条历史回归
python -m pytest tests/test_rag_chunking.py -q -k multiline
python -m pytest tests/test_rag_rerank.py   -q -k split_words
python -m pytest tests/test_rag_pipeline.py -q -k grounded_score
python -m pytest tests/test_agent_rag_tools.py -q -k "cosine or missing_kb"
```
