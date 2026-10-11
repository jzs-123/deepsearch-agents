# 更新日志

> 本项目在自研 RAG 基础上持续迭代。按时间倒序记录每次可交付的功能变更。

---

## 2026-10-10 · 企业级能力批次（OCR / 查询改写 / 元数据 / 缓存 / 审计 / RAGAS）

### 新增功能

**1. OCR 机制（零依赖，Windows 内置引擎）**

- 可插拔 OCR 后端：`WindowsOcrEngine`（默认）/ `RapidOcrEngine` / `NoneOcrEngine`，后端不可用时自动降级为 none + warning，不抛异常。
- `Windows.Data.Pdf` 页渲染（`app/rag/render.py`），与 OCR 解耦——**扫描件 PDF 可页转图再识别**。
- 接入 `loaders.py`：
  - 图片文件（`.png/.jpg/.jpeg/.bmp/.webp/.tif/.tiff`）直接走 OCR；
  - PDF **逐页判定**：文本层不足才对该页 OCR，充足则不触发（用计数器取证，避免用更差的识别结果替换干净原文）；
  - `LoadedDocument.extraction` 标注来源（`text_layer` / `ocr` / `mixed` / `none`）。
- 质量如实记录：正文页识别好（散文近乎逐字），设计类封面页差（丢数字、错字随倍率变）——定位为**兜底**而非常规路径。

**2. 查询改写（本地规则 + 大模型两级）**

- 本地级：口语归一化 + jieba 关键词/同义词扩展 + 专有名词/数字强化（零成本）。
- 大模型级：生成多个检索变体（HyDE 思路），不可用/超时自动降级为本地改写。
- `pipeline.search()` 多查询 RRF 融合 + 低分自动重试；判定统一用向量余弦（`grounded_score()`），绝不用重排分。
- 收益实测：短/口语/异词查询评测集 **关闭 87.5% → 开启 100%（MRR 1.0000）**；4 条查询从"低于阈值拒答"变"可答复"；既有评测集数字完全不变（基线已饱和）。

**3. 元数据过滤**

- 自动抽取：文件级（path/suffix/size/mtime）+ Markdown YAML frontmatter（部门/类型/密级/版本等）。
- `search(filter=...)` 在召回后、重排前做白名单过滤（等值 / 列表值 / 多字段 AND）。
- CLI `--filter '<JSON>'`、Agent 工具 `filter_json`；过滤空结果给明确提示（含可用字段清单），非静默空。

**4. 缓存机制（默认关闭）**

- 三级缓存：embedding / 查询结果 / 生成结果；键含向量指纹 + 数据版本号。
- 库内任何 chunk 增删改 → 数据版本号递增 → 整体失效；`reset()` 清空缓存文件。
- 命中结果标注 `cached=True`，可审计。默认关闭（避免过期答案风险）。

**5. 审计日志（默认开启）**

- JSONL 追加写，每次 `search`/`query` 一条；含时间/query/filter/改写变体/缓存命中/重试/结果摘要（**不含原文**）/是否调大模型与耗时/拒答原因/向量指纹。
- `python -m app.rag.cli audit -k <kb> --tail N [--json]` 可读回。
- 写失败只 warning，**审计绝不拖垮查询**。

**6. RAGAS 性能评测（脚本 + 文档 + 离线替代）**

- `app/rag/eval_ragas.py`：RAGAS 组装层（跳无上下文/空答案/应拒答，`contexts` 强制 `list[list[str]]`），judge 复用 `.env` 配置。
- `app/rag/eval_offline.py`：**零依赖离线近似**——数字/专有名词幻觉识别，明确标注"非 RAGAS"，无可核验内容给 `None` 而非满分。
- 详细使用见 `docs/rag-eval-guide.md`（本机装 RAGAS 需 `pip install ragas datasets`）。

### 修复

- **审计 `llm.called` 语义**：原来用"客户端是否可用"判断，拒答时误记 true；改为记录"是否真的发起生成调用"。
- **查询缓存键补入 `top_n`**：避免同 query 不同 `top_n` 互相命中导致少拿结果。
- **OCR 识别异常兜底**：第三方引擎抛异常时降级为 warning，不中断整批入库。

### 文档

- `docs/rag-ocr.md`、`docs/rag-query-rewrite.md`、`docs/rag-enterprise-features.md`、`docs/rag-eval-guide.md`
- `docs/rag-new-features-verification.md`（OCR + 改写独立验证）、`docs/rag-enterprise-verification.md`（企业级能力独立验证）
- `docs/rag-testing-guide.md`、`docs/rag-upload-and-ask-guide.md`

### 测试

- `python -m pytest tests -q` → **516 passed**（含元数据 50 / 缓存 19 / 审计 19 / OCR 24 / 改写 49 / 评测 63 等）
- `python -m app.rag.selftest` → 18 通过 / 0 失败

---

## 2026-10-10 · 自研 RAG MVP（替代 RAGFlow）

- 完整 RAG 链路：多格式解析 → 结构感知切块 → 向量化（local/openai/hash）→ numpy 向量库 → BM25 稀疏检索 → RRF 混合融合 → 重排 → 带引用生成。
- 知识库向量指纹校验、拒答门槛、CLI（list/ingest/search/ask/eval/stats/reset/audit）、检索评测（Hit@k/MRR/Recall@k/拒答率）。
- 修复 `normalize_text` 吞中文换行（标题吞正文）、工具层拒答门槛用排序分、技能文档残留 RAGFlow 引用、幽灵知识库等缺陷。
- 详见 `docs/rag-self-built.md`。
