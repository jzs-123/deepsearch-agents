# OCR（图片 / 扫描件兜底）

> 任务：t10 ｜ 作者：envi ｜ 日期：2026-10-10 ｜ 关键代码：`app/rag/ocr.py`、`app/rag/render.py`、`app/rag/loaders.py`
> 方案：**方案 A —— Windows 内置 OCR 引擎（`Windows.Media.Ocr`），零 pip 依赖**

---

## 1. 一句话结论

给自研 RAG 加了 OCR 能力：**图片文件**与**PDF 里没有文本层的页**（扫描件）现在能取出文字并正常入库检索。
它是**兜底**，不是主路径 —— 有文本层的页面**绝不会**被 OCR 覆盖（第 5 节有实测证据）。

---

## 2. 为什么是「兜底」而不是「默认」

本机实测的两组极端数据决定了这个定位（完整数据见第 6 节）：

| 场景 | 文本层 | OCR | 谁更好 |
| --- | --- | --- | --- |
| 真实 PDF **封面页**（设计类、低对比、大字排版） | 113 字符，内容正确 | **55 字符**，丢字 + 页脚乱码 | **文本层完胜** |
| 合成**扫描件**（无文本层） | **0 字符** | 51 字符，中文可读可检索 | OCR 是唯一出路 |

所以策略是**逐页判定**：文本层够用就不碰 OCR；文本层不够才补 OCR。
反过来做（整份优先 OCR）会把干净的原文换成带错字的识别结果，是净损失。

---

## 3. 架构与调用链

```
app/rag/pipeline.py  ingest()
  └─ loaders.load_document(path, ocr=RagConfig.ocr_options())
       ├─ .png/.jpg/.jpeg/.bmp/.webp/.tif/.tiff ──► load_image()
       │        └─ ocr.build_ocr_engine().recognize(图片字节)         ← 图片没有文本层,只能 OCR
       │
       └─ .pdf ─────────────────────────────────► load_pdf()
                ├─ pypdf.page.extract_text()  ──► 逐页文本层
                ├─ len(该页文本) < RAG_OCR_MIN_TEXT_CHARS ?
                │     ├─ 否 ──► 用文本层        (extraction = text_layer)
                │     └─ 是 ──► render.build_pdf_renderer().render_page()   ← Windows.Data.Pdf(零 pip 依赖)
                │                   └─ ocr.build_ocr_engine().recognize(png)← Windows.Media.Ocr
                │                   └─ 与文本层合并/取优
                └─ 汇总 ──► LoadedDocument(
                                text, headings,
                                extraction = text_layer | ocr | mixed | none )

【解耦点】渲染器与 OCR 引擎是两组独立接口,互不知道对方存在:
    render.PdfRenderer        : available / render_page(pdf_bytes, page_index, scale)
    ocr.OcrEngine             : available / recognize(image_bytes) -> str
换渲染器(如 PyMuPDF)只动 render.py;换 OCR(如 rapidocr)只动 ocr.py。
```

**子进程调用基础设施**在 `app/rag/_powershell.py`（渲染与 OCR 共用），它解决两个实测坑：

1. **用 `-EncodedCommand`(UTF-16LE base64) 而不是 .ps1 文件** —— PowerShell 5.1 会把无 BOM 的
   `.ps1` 当 ANSI 读，脚本里的中文注释直接把解析搞崩（`Unexpected token '}'`）；执行策略也会拦 `.ps1`。
2. **管道被拒时自动回落到临时文件重定向** —— 沙箱禁止程序打开命名管道，而 `capture_output=True`
   依赖它。先走管道，一旦 `OSError` 就改用「stdin/stdout/stderr 全部重定向到临时文件」。
   （本会话实测管道可用，但兜底路径有专门用例覆盖，见第 10 节。）

---

## 4. 引擎选择与降级行为

### 4.1 后端

| `RAG_OCR_BACKEND` | 实现 | 可用条件 | 不可用时 |
| --- | --- | --- | --- |
| `windows`（默认） | `Windows.Media.Ocr`，经 PowerShell 5.1 + WinRT | `sys.platform == "win32"` 且能找到 `powershell.exe` | 降级 `none` + **warning** |
| `rapidocr` | `rapidocr_onnxruntime` | 已安装该包 | 降级 `none` + **warning** |
| `none` | 空实现 | 总是"可用"（即明确不做 OCR） | — |

未知后端名同样降级 `none` + warning。

### 4.2 降级是**多级、逐层**的，任何一层缺失都不影响入库

| 层 | 缺失时的行为 |
| --- | --- |
| 总开关 `RAG_OCR_ENABLED=false` | 图片 → 空文本 + warning；扫描件页 → 只保留文本层 |
| `RAG_OCR_MIN_TEXT_CHARS=0` | 关闭页级 OCR（扫描件页取不到内容，这是显式选择） |
| OCR 后端不可用 | 降级 `none`，PDF 仍走文本层，warning 提示可能的原因 |
| PDF 渲染器不可用 | 同上（非 Windows 平台会遇到） |
| 单页渲染/识别失败 | 该页留 warning + 空文本，**不影响其它页** |
| 图片读取失败 | 空文本 + warning |

**设计约定**：`OcrEngine.recognize()` 失败时返回空串并 `logger.warning`，**不抛业务异常**；
`pipeline.ingest()` 原有的「未解析出文本 → 跳过该文件并记 warning」逻辑继续生效。

---

## 5. 与文本层的关系

### 5.1 逐页判定，不是整份判定

真实文档大量是**混合体**：正文有文本层，封面/图表/插页是图。整份判定要么丢掉图里的内容，
要么对整份做无谓且昂贵的 OCR。所以阈值判定放在**页**这一级。

### 5.2 `LoadedDocument.extraction` 的语义

| 取值 | 含义 |
| --- | --- |
| `text_layer` | 全部来自文本层，**没有调用过 OCR** |
| `ocr` | 全部来自 OCR（图片文件，或整页扫描件） |
| `mixed` | 部分页文本层、部分页 OCR |
| `none` | 没取到任何文本（OCR 关闭 / 后端不可用 / 真的没有字） |

该字段同时进入 `IngestReport.details[]`（键名 `extraction`），排障时可据此判断
「这个文件的正文是原文还是识别结果」。**注意**：chunk 层目前还没有携带该标记
（见第 9 节「已知限制」）。

### 5.3 文本层优先的实测证据

```
$ .\.venv\Scripts\python.exe <注入计数器的脚本>
实跑（贝莱德 17 页报告，21272 字符）：
  extraction=text_layer   chars=21272   OCR_CALLS=0   warnings=[]
实跑（tests/fixtures/text_sample.pdf，351 字符）：
  extraction=text_layer   chars=351     OCR_CALLS=0
实跑（tests/fixtures/scan_sample.pdf，无文本层）：
  extraction=ocr          chars=51      OCR_CALLS=1
```

计数为 0 才算通过 —— 只断言文本内容无法证明"没调用 OCR"。

### 5.4 合并策略

某页文本层不足时：渲染该页 → OCR → 与文本层合并：

- 文本层为空 → 用 OCR；
- 两者都有 → **都保留**（OCR 结果在前，文本层在后），宁可多留一点让切块与去重处理，
  也不因为"择一"而丢字。

---

## 6. 质量实测（含差案例）

全部数据在 2026-10-10 于本机实测，渲染倍率 `scale=2.0`。

### 6.1 合成图（白底黑字、40px 微软雅黑）—— 质量好

```
输入文本 : 风控模型的准确率为 91.5%。
OCR 原始 : 风 控 模 型 的 准 确 率 为 91 · 5 ％ 。
归一化后 : 风控模型的准确率为 91 · 5 ％ 。      ← normalize_text 清掉了汉字间空格
```

- ✅ 汉字全部正确，**汉字之间会插空格**（`normalize_text` 已自动清除）；
- ❌ **字符混淆不会被修正**：小数点 `.` → `·`，百分号 `%` → `％`。
  这是识别误差而不是排版噪声，`normalize_text` 无法也不应该"猜"回去。

### 6.2 合成扫描件 PDF（`tests/fixtures/scan_sample.pdf`）

```
OCR 归一化后:
  风控模型的准确率为 91 丐 ％
  风险管理部编制 2026 年模型监控报告
  信用评分与逾期率是核心指标
```

- ✅ 三行中文全部可读，检索「风控模型的准确率」能命中；
- ❌ `91.5%` 这次变成 `91 丐 ％`（`.5` 被识别成汉字 `丐`）—— **同一句话在不同尺寸/排版下错法不同**，
  说明这类百分比数字**不能信任 OCR 结果**。

### 6.3 真实 PDF：封面页 vs 正文页（`贝莱德BlackRock2026年全球投资展望报告17页.pdf`）

| 页 | 文本层字符 | OCR 字符 | 观察 |
| --- | --- | --- | --- |
| 第 1 页（**封面**，设计类大字低对比） | 113 | **55** | **差案例**：OCR 丢字 + 页脚乱码 |
| 第 5 页（正文，投资环境） | 1389 | 1523 | 正文散文识别准确，额外读到图表标题 |
| 第 7 页（正文，金融杠杆） | 1344 | 1311 | OCR 额外读到图表标题 `债务水平分化` |
| 第 9 页（正文，人工智能） | 1169 | 1267 | OCR **重复/串行了一个小标题** |

**差案例原文（封面页）：**

```
文本层 : BlackRock / Investment / Institute / 2026年全球 / 投资展望 / 突破传统界限
         文件仅供派发予合格境內机构投资者、专业投资者使用，不得公幵派发
OCR    : 6 年全球 / 投资展望 / 突破传统界限 / BlackRock / 贝莱德 / 贝莱德智库
         业投资者使用 ， 不伺 00 添发
```

- OCR 把 `2026年全球` 读成 `6 年全球`（**丢了 "202"**）；
- 页脚 `文件仅供派发予合格境內机构投资者……` 被读成 `业投资者使用 ， 不伺 00 添发`（**乱码**）；
- 封面 OCR 只有 55 字符，不到文本层的一半。

> **关于错字的一个诚实说明**：任务情报里 captain 在封面页观察到 `贝 菜 德`（正确为"贝莱德"）。
> 我在 `scale=2.0` 下复现同一页时 `贝莱德` **读对了**，但同样丢了 `202`、同样把页脚读成
> `不伺 00 添发`。也就是说 **错的具体字符随渲染倍率/排版而变，但"封面页质量不可用"这个结论稳定**。
> 不要依赖任何单次观测的具体错字。

> **另一个诚实说明**：文本层也不是永远完美。上面封面页的文本层里 `不得公幵派发` 的 `幵`
> 就是 PDF 自带的错字（原字应为"开"）。所以正确说法是"**有文本层时优先用文本层**"，
> 而不是"文本层一定正确"。

### 6.4 正文页的真实表现（比封面更有代表性）

OCR 在**正文页**上的表现明显好于封面：

```
第 5 页 OCR 片段:
  投资环境
  美国经济增长有望
  突破长期趋势
  我们认为 2026 年人工智能资本支出仍将推动经济增长 ， 今年这类投资对美
  国经济增长的贡献已达到其历史平均值的三倍 。 这种资本密集型增长动力
  或将持续至明年 ， 即便劳动力市场持续降温 ， 仍能支撑经济增长保持韧性 。
  劳动力市场处于 " 不招不裁 " 的停滞状态
```

散文段落基本逐字正确（含引号与破折号），并且**额外读到了文本层里没有的图表标签**
（第 7 页的 `债务水平分化`、`1950 一 2025 年美国政府及企业债务状况`）——
这是 OCR 相对文本层的**意外收益**：文字层的抽取顺序可能漏掉浮动文本框，OCR 反而看到。

但正文页也暴露了两类稳定缺陷：

1. **数字与标点易错**：范围号 `–` 被读成汉字 `一`（`1950 一 2025`、`2024 一 2030`），
   数字与汉字之间会插空格（`2026 年`、`2030 年`）—— 后者 `normalize_text` **不会**清理
   （它只清汉字之间的空格，数字/ASCII 周围的空格不动）；
2. **串行与重复**：第 9 页 OCR 把一个小标题读成 `人工智能建设面临 / 面临限制`（重复），
   多栏/图文混排的行序不保证。

### 6.5 结论

- **有文本层 → 一律用文本层**（本实现已如此，并有计数用例锁死）；
- **无文本层 → OCR 可用**，中文散文质量够检索用；
- **不要相信 OCR 的数字、百分号、范围号与小数**：`91.5%` 实测出现过 `91 · 5 ％` 与 `91 丐 ％` 两种错法；
- **不要拿封面/海报类页面做验收样本**，它们本来就是 OCR 最差的场景。

---

## 7. 配置项

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_OCR_ENABLED` | `true` | 总开关。关闭后图片不解析、扫描件页只保留文本层 |
| `RAG_OCR_BACKEND` | `windows` | `windows` / `rapidocr` / `none` |
| `RAG_OCR_MIN_TEXT_CHARS` | `50` | PDF 单页文本层字符数低于该值时对该页做 OCR；`0` = 关闭页级 OCR |
| `RAG_OCR_LANGUAGE` | `zh-Hans-CN` | OCR 语言（BCP-47）；Windows 需已安装对应语言包 |

对应的 `RagConfig` 字段为 `ocr_enabled` / `ocr_backend` / `ocr_min_text_chars` / `ocr_language`，
可用 `RagConfig.ocr_options()` 取得 `OcrOptions`（`loaders` 只认 `OcrOptions`，不反向依赖 `RagConfig`）。
`.env.example` 已同步。

---

## 8. Windows 独占性，以及怎么换到别的平台

### 8.1 为什么现在是 Windows 独占

默认后端用的是两条 **WinRT API**：

- OCR：`Windows.Media.Ocr.OcrEngine`；
- **PDF 页渲染**：`Windows.Data.Pdf.PdfDocument`（把指定页渲染成 PNG）。

两者都不需要 pip 安装，但**只在 Windows 上存在**。非 Windows 上：

| 能力 | 非 Windows 行为 |
| --- | --- |
| 图片 OCR | 若把后端设为 `rapidocr` 并安装依赖 → **可用** |
| 扫描件 PDF OCR | **不可用**（缺 PDF 渲染器）→ PDF 仍能解析文本层，只是扫描件页取不到内容 |

### 8.2 换到 rapidocr（跨平台）

```powershell
pip install rapidocr-onnxruntime
$env:RAG_OCR_BACKEND = 'rapidocr'      # 或写进 .env
```

代码路径已就绪（`ocr.RapidOcrEngine`），未安装时**自动降级并 warning**。
⚠️ **本机未安装该包**（方案 A 要求零依赖，且任务明令不得装包），因此
`RapidOcrEngine.recognize()` 的解析逻辑**未在本机实测**；已测的是「未安装 → 降级为 none」这条路径。

### 8.3 换掉 PDF 渲染器（非 Windows 想支持扫描件时）

实现 `app/rag/render.py::PdfRenderer` 接口并用它替换 `build_pdf_renderer()` 的返回即可：

```python
class MyRenderer(PdfRenderer):
    name = "mupdf"
    @property
    def available(self) -> bool: ...
    def render_page(self, pdf_bytes: bytes, page_index: int, *, scale: float = 2.0) -> bytes:
        ...  # 返回 PNG 字节
```

`ocr.py` 不需要任何改动 —— 它只要求「给我图片字节」。

---

## 9. 已知限制

1. **设计类/海报类页面质量差**（第 6.3 节实测：文本层 113 → OCR 55 字符，丢字 + 乱码）。
2. **数字与标点不可信**：`.`→`·`、`%`→`％`、`–`→`一`；百分比小数会丢位（`91.5%` → `91 丐 ％`）。
   需要精确数字的场景请以文本层或原始文档为准。
3. **多栏 / 图文混排的行序不保证**：OCR 按识别行输出，可能把图表标签与正文交错，甚至重复小标题（实测 `人工智能建设面临 / 面临限制`）。
4. **表格结构完全丢失**：只会得到零散文本，没有行列关系。表格类内容应走 `docx` 表格解析或数据库，不要指望 OCR。
5. **数字/ASCII 周围的空格不会被清理**：`normalize_text` 只清汉字之间的空格（这是有意的，见 `loaders.normalize_text` 注释里的历史 bug）。
6. **依赖系统语言包**：Windows 未安装 `zh-Hans-CN` 语言包时引擎创建失败 → 降级 `none`。
7. **速度**：每页都会起一个 PowerShell 子进程（约 1~3 秒/页），整本扫描件入库会明显变慢；
   当前**没有页级并发**。
8. **chunk 不携带 `extraction` 标记**：目前只有 `LoadedDocument` 与 `IngestReport.details` 有。
   也就是说引用出处里写的是文件名与页码，读者无法从引用本身判断"这句是原文还是 OCR 猜的"。
   若要彻底解决，需要把 `extraction` 透传到 `Chunk`（属后续工作）。
9. **无 bbox/坐标**：只取纯文本，不保留字的位置，因此无法做"原图高亮"或按区域二次校对。
10. **`rapidocr` 分支未实测**（见 8.2）。

---

## 10. 复现与验证

### 10.1 生成样本（可复现，无需联网）

```powershell
python tests/fixtures/make_scan_sample.py
# → tests/fixtures/scan_sample.pdf  (无文本层,40 KB)
# → tests/fixtures/text_sample.pdf  (有文本层,38 KB,351 字符)
```

生成脚本用 `PIL` 把文字渲染成位图、`reportlab` 嵌入 PDF（两个包都已在依赖里）；
仓库自带的 5 份 PDF 全都有文本层，**没有扫描件样本**，所以必须合成。

### 10.2 跑测试

```powershell
$env:PYTHONPATH = 'G:\deepsearch-agents-main\.uv-cache\pytest-deps'
python -m pytest tests -q
# 实测：315 passed（原 292 + 新增 23），exit=0
python -m pytest tests\test_rag_ocr.py -q
# 实测：23 passed in 11.04s
python -m app.rag.selftest
# 实测：18 通过 / 0 失败
```

### 10.3 测试覆盖（`tests/test_rag_ocr.py`，23 例）

| 关注点 | 用例 |
| --- | --- |
| 汉字间空格归一化 + 不吞换行回归 | `test_normalize_text_removes_spaces_between_han_chars` / `..._keeps_newlines_and_ocr_confusions` |
| 后端降级（未知/未装/显式 none） | `test_build_ocr_engine_*` |
| 配置校验与映射 | `test_rag_config_validates_ocr_fields` / `..._maps_to_ocr_options` |
| 图片 OCR 入库 | `test_load_document_dispatches_image_to_ocr` / `test_real_ocr_image_ingest_and_search` |
| 后端不可用不抛异常 | `test_load_image_with_unavailable_engine_warns_instead_of_raising` |
| **文本层充足不触发 OCR（计数取证）** | `test_text_layer_pdf_does_not_trigger_ocr` |
| 关闭开关 / 阈值 0 | `test_load_image_with_ocr_disabled_warns` / `test_ocr_threshold_zero_disables_page_ocr` |
| 扫描件端到端（真实 OCR） | `test_real_ocr_reads_scanned_pdf` / `test_real_ocr_scanned_pdf_ingest_and_search` |
| 渲染失败降级 | `test_render_page_failure_is_reported_as_warning_not_exception` |
| 管道被拒的文件重定向兜底 | `test_run_powershell_falls_back_to_files_when_pipes_denied` |

真实 OCR 用例带 `requires_windows_ocr` 标记：引擎不可用时 **skip 而不是 fail**，
便于在非 Windows / 无语言包的环境下跑其余用例。
