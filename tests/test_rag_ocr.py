"""OCR 功能测试(t10)。

覆盖范围:
* 汉字间空格归一化(OCR 输出的典型副作用),以及 `normalize_text` **不吞换行**的回归;
* 后端不可用 / 未知后端 / 关闭开关时的**降级**行为:只 warning,**不抛异常**;
* **有文本层的 PDF 不触发 OCR**(最关键的一条 —— 否则等于用更差的识别结果替换干净原文);
* 图片文件走 OCR 入库;
* 合成扫描件 PDF(无文本层)端到端入库,并能检索到其中的内容;
* 沙箱拒绝命名管道时,PowerShell 调用的**文件重定向兜底**路径。

样本:`tests/fixtures/scan_sample.pdf`(无文本层)/ `text_sample.pdf`(有文本层),
由 `tests/fixtures/make_scan_sample.py` 用 PIL + reportlab 合成,可复现(仓库自带的
5 份 PDF 全都有文本层,没有扫描件样本)。

真实 OCR 每次要起一个 PowerShell 子进程(约 1~3 秒),因此依赖真实引擎的用例统一加
`requires_windows_ocr`:引擎不可用时 **skip 而不是 fail**,便于在非 Windows 上跑其余用例。
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest

from app.rag.loaders import IMAGE_SUFFIXES, SUPPORTED_SUFFIXES, load_document, load_image, normalize_text
from app.rag.ocr import NoneOcrEngine, OcrEngine, OcrOptions, RapidOcrEngine, build_ocr_engine

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCAN_PDF = FIXTURES / "scan_sample.pdf"
TEXT_PDF = FIXTURES / "text_sample.pdf"

# OCR 阈值(与生产默认值一致)
MIN_CHARS = 50


def _windows_ocr_available() -> bool:
    from app.rag.ocr import WindowsOcrEngine

    return WindowsOcrEngine().available


requires_windows_ocr = pytest.mark.skipif(
    not _windows_ocr_available(),
    reason="需要 Windows 内置 OCR 引擎(Windows.Media.Ocr),本机不可用",
)


class _StubEngine(OcrEngine):
    """可控的假 OCR 后端:记录调用次数,可指定是否可用与返回文本。"""

    def __init__(self, text: str = "识 别 结 果", available: bool = True, name: str = "stub") -> None:
        super().__init__()
        self._text = text
        self._available = available
        self.name = name

    @property
    def available(self) -> bool:
        return self._available

    def recognize(self, image_bytes: bytes) -> str:
        self.calls += 1
        return self._text


class _RaisingEngine(OcrEngine):
    """会抛异常的第三方引擎:验证 loaders 必须把识别异常兜住,而不是让它穿透。"""

    name = "raising"

    @property
    def available(self) -> bool:
        return True

    def recognize(self, image_bytes: bytes) -> str:
        raise RuntimeError("engine exploded")



@lru_cache(maxsize=1)
def _fixture_generator():
    """加载样本生成脚本,复用其中的「渲染中文文字为位图」能力(避免重复字体逻辑)。"""
    spec = importlib.util.spec_from_file_location("_make_scan_sample", FIXTURES / "make_scan_sample.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_sample_png(path: Path) -> Path:
    generator = _fixture_generator()
    image = generator.render_text_image(generator.SCAN_LINES)
    image.save(path)
    return path


# ────────────────────────── 归一化 ──────────────────────────


def test_normalize_text_removes_spaces_between_han_chars():
    """OCR 会在汉字之间插空格,必须被清掉,否则 BM25 分词与向量都会被污染。"""
    ocr_like = "风 控 模 型 的 准 确 率 为 91 · 5 ％"
    out = normalize_text(ocr_like)
    assert "风控模型的准确率为" in out


def test_normalize_text_keeps_newlines_and_ocr_confusions():
    """两个反向约束:

    1. **不能吞换行**(历史 bug:用 `\\s+` 会把「标题\\n\\n正文」压成「标题正文」,
       导致整篇退化成 1 个 chunk)。这里锁死 `[ \\t]+` 的行为;
    2. OCR 的**字符混淆不是排版噪声**,`normalize_text` 不该也无法修正
       (`。`→`·`、`%`→`％`、小数点丢失)——这是识别误差,只能靠文档如实告知使用者。
    """
    out = normalize_text("风 控\n\n# 标题\n\n正文段落")
    assert "\n\n" in out
    assert "# 标题" in out
    assert "正文段落" in out

    confused = normalize_text("准 确 率 为 91 · 5 ％")
    assert "91 · 5 ％" in confused  # 未被"修好",如实保留


# ────────────────────────── 后端构造与降级 ──────────────────────────


def test_build_ocr_engine_unknown_backend_degrades_to_none():
    engine = build_ocr_engine("totally-unknown")
    assert isinstance(engine, NoneOcrEngine)
    assert engine.available is False
    assert engine.recognize(b"whatever") == ""  # 不抛异常


def test_build_ocr_engine_none_backend():
    engine = build_ocr_engine("none")
    assert isinstance(engine, NoneOcrEngine)
    assert engine.available is False


def test_build_ocr_engine_unavailable_rapidocr_degrades_to_none():
    if RapidOcrEngine().available:
        pytest.skip("本机已安装 rapidocr,无法验证「未安装 → 降级」这条路径")
    engine = build_ocr_engine("rapidocr")
    assert isinstance(engine, NoneOcrEngine)
    assert engine.recognize(b"x") == ""


def test_rag_config_validates_ocr_fields(tmp_path):
    from app.rag.config import RagConfig

    with pytest.raises(ValueError, match="ocr_backend"):
        RagConfig(data_dir=tmp_path, ocr_backend="bogus")
    with pytest.raises(ValueError, match="ocr_min_text_chars"):
        RagConfig(data_dir=tmp_path, ocr_min_text_chars=-1)


def test_rag_config_maps_to_ocr_options(tmp_path):
    from app.rag.config import RagConfig

    cfg = RagConfig(
        data_dir=tmp_path,
        ocr_enabled=True,
        ocr_backend="none",
        ocr_min_text_chars=7,
        ocr_language="en-US",
    )
    opts = cfg.ocr_options()
    assert (opts.enabled, opts.backend, opts.min_text_chars, opts.language) == (True, "none", 7, "en-US")


def test_image_suffixes_are_supported():
    assert {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"} <= SUPPORTED_SUFFIXES
    assert IMAGE_SUFFIXES <= SUPPORTED_SUFFIXES


# ────────────────────────── 图片入库 ──────────────────────────


def test_load_document_dispatches_image_to_ocr(tmp_path):
    """图片 → OCR 变文本,且标注 extraction='ocr'(并顺手归一化汉字间空格)。"""
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")  # 内容不重要:注入了假引擎
    spy = _StubEngine(text="图 片 里 的 字")
    doc = load_document(image, ocr=OcrOptions(enabled=True, backend="windows"), ocr_engine=spy)
    assert spy.calls == 1
    assert doc.extraction == "ocr"
    assert doc.text == "图片里的字"


def test_load_image_with_unavailable_engine_warns_instead_of_raising(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    engine = _StubEngine(available=False)
    doc = load_image(image, "shot.png", ocr=OcrOptions(enabled=True, backend="none"), ocr_engine=engine)
    assert doc.text == ""
    assert doc.extraction == "none"
    assert doc.warnings  # 有 warning,而不是抛异常
    assert engine.calls == 0


def test_load_image_with_ocr_disabled_warns(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    doc = load_image(image, "shot.png", ocr=OcrOptions(enabled=False))
    assert doc.text == ""
    assert doc.extraction == "none"
    assert any("OCR 已关闭" in w for w in doc.warnings)


def test_load_image_swallows_engine_exception(tmp_path):
    """第三方 OCR 引擎抛异常时必须被兜住(降级为 warning),不能中断整批入库。

    对应 t12 评审发现的 F2:loaders 早期直接调用 engine.recognize() 未包 try/except。
    """
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    doc = load_image(image, "shot.png", ocr=OcrOptions(enabled=True, backend="windows"), ocr_engine=_RaisingEngine())
    assert doc.text == ""
    assert doc.extraction == "none"
    assert any("识别失败" in w for w in doc.warnings)


# ────────────────────────── 文本层优先(最关键) ──────────────────────────


def test_text_layer_pdf_does_not_trigger_ocr():
    """**最关键的一条**:文本层充足时绝不能调用 OCR。

    否则就是把干净的原文替换成带错字的识别结果(见 docs/rag-ocr.md 的质量实测),
    对收录正式文档是净损失。用调用计数取证,而不是只看文本内容。
    """
    spy = _StubEngine(text="我 是 O C R 结 果")
    doc = load_document(
        TEXT_PDF,
        ocr=OcrOptions(enabled=True, backend="windows", min_text_chars=MIN_CHARS),
        ocr_engine=spy,
    )
    assert spy.calls == 0, "文本层充足却调用了 OCR"
    assert doc.extraction == "text_layer"
    assert doc.char_count > MIN_CHARS
    assert "我是OCR结果" not in doc.text.replace(" ", "")


def test_pdf_degrades_to_text_layer_when_ocr_backend_unavailable():
    """OCR 后端不可用 → 有文本层的 PDF 仍然正常,只留 warning,不抛异常。"""
    engine = _StubEngine(available=False)
    doc = load_document(
        TEXT_PDF,
        ocr=OcrOptions(enabled=True, backend="none", min_text_chars=MIN_CHARS),
        ocr_engine=engine,
    )
    assert doc.extraction == "text_layer"
    assert doc.char_count > MIN_CHARS
    assert engine.calls == 0
    assert any("OCR 后端不可用" in w for w in doc.warnings)


def test_ocr_threshold_zero_disables_page_ocr():
    """阈值设为 0 = 关闭页级 OCR:扫描件就取不到内容,这是显式选择而非 bug。"""
    spy = _StubEngine(text="不该被调用")
    doc = load_document(
        SCAN_PDF,
        ocr=OcrOptions(enabled=True, backend="windows", min_text_chars=0),
        ocr_engine=spy,
    )
    assert spy.calls == 0
    assert doc.text == ""
    assert doc.extraction == "none"


def test_ocr_disabled_on_scanned_pdf_yields_text_layer_none():
    doc = load_document(SCAN_PDF, ocr=OcrOptions(enabled=False))
    assert doc.text == ""
    assert doc.extraction == "none"
    assert doc.warnings


# ────────────────────────── 扫描件 PDF ──────────────────────────


def test_scanned_pdf_uses_injected_engine_and_marks_ocr():
    """合成扫描件没有文本层 → 必须走 OCR,且标注 extraction='ocr'。"""
    spy = _StubEngine(text="风 控 模 型 的 准 确 率 为 91.5%")
    doc = load_document(
        SCAN_PDF,
        ocr=OcrOptions(enabled=True, backend="windows", min_text_chars=MIN_CHARS),
        ocr_engine=spy,
    )
    assert spy.calls == 1
    assert doc.extraction == "ocr"
    assert "风控模型的准确率" in doc.text


def test_render_page_failure_is_reported_as_warning_not_exception():
    """渲染失败(例如渲染器不可用)只能降级成 warning + 空文本。"""

    class _BrokenRenderer:
        name = "broken"

        @property
        def available(self) -> bool:
            return True

        def render_page(self, pdf_bytes: bytes, page_index: int, *, scale: float = 2.0) -> bytes:
            raise RuntimeError("模拟渲染失败")

    doc = load_document(
        SCAN_PDF,
        ocr=OcrOptions(enabled=True, backend="windows", min_text_chars=MIN_CHARS),
        ocr_engine=_StubEngine(text="不该被调用"),
        renderer=_BrokenRenderer(),
    )
    assert doc.text == ""
    assert doc.extraction == "none"
    assert any("渲染失败" in w for w in doc.warnings)


# ────────────────────────── 端到端(真实 OCR) ──────────────────────────


@requires_windows_ocr
def test_real_ocr_reads_scanned_pdf():
    """真实 Windows OCR:无文本层的扫描件必须识别出中文内容。"""
    doc = load_document(
        SCAN_PDF,
        ocr=OcrOptions(enabled=True, backend="windows", min_text_chars=MIN_CHARS),
    )
    assert doc.extraction == "ocr"
    compact = doc.text.replace(" ", "")
    assert "风控模型" in compact
    assert "风险管理部" in compact


@requires_windows_ocr
def test_real_ocr_scanned_pdf_ingest_and_search(pipeline):
    """扫描件 PDF 端到端:入库成功 + 能检索到其中的内容(本功能的核心价值)。"""
    report = pipeline.ingest(SCAN_PDF)
    assert report.files_indexed == 1, report.warnings
    assert report.chunks_added >= 1

    hits = pipeline.search("风控模型的准确率", rewrite=False)
    assert hits, "扫描件里的内容没有被检索到"
    assert any("风控模型" in hit.chunk.text.replace(" ", "") for hit in hits)


@requires_windows_ocr
def test_real_ocr_image_ingest_and_search(pipeline, tmp_path):
    """图片文件端到端:走 OCR 入库,并能被检索到。"""
    image = _make_sample_png(tmp_path / "scan.png")
    report = pipeline.ingest(image)
    assert report.files_indexed == 1, report.warnings
    assert report.chunks_added >= 1

    hits = pipeline.search("风控模型的准确率", rewrite=False)
    assert hits, "图片里的内容没有被检索到"
    assert any("风控模型" in hit.chunk.text.replace(" ", "") for hit in hits)


@requires_windows_ocr
def test_real_renderer_produces_png():
    from app.rag.render import build_pdf_renderer

    renderer = build_pdf_renderer("windows")
    assert renderer.available
    png = renderer.render_page(SCAN_PDF.read_bytes(), 0, scale=2.0)
    assert png.startswith(b"\x89PNG")


def test_build_pdf_renderer_unknown_backend_is_none():
    from app.rag.render import NonePdfRenderer, build_pdf_renderer

    renderer = build_pdf_renderer("bogus")
    assert isinstance(renderer, NonePdfRenderer)
    assert renderer.available is False


# ────────────────────────── 管道被拒时的兜底 ──────────────────────────


def test_run_powershell_falls_back_to_files_when_pipes_denied(monkeypatch):
    """沙箱拒绝命名管道时,必须自动改用临时文件重定向,而不是直接失败。

    本沙箱禁止程序打开命名管道(`capture_output=True` 依赖它)。实测本会话里
    powershell 的管道捕获可用,但这里强制让第一次调用抛 `OSError`,
    验证兜底路径真的能拿到结果。
    """
    import app.rag._powershell as ps

    real_run = ps.subprocess.run
    calls = {"n": 0}

    def flaky_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(5, "拒绝访问。(模拟命名管道被沙箱拒绝)")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(ps.subprocess, "run", flaky_run)
    returncode, stdout, _stderr = ps.run_powershell_script(
        "[Console]::Out.Write('FALLBACK_OK')", "", timeout=120
    )
    assert calls["n"] == 2, "第一次应抛 OSError 并触发文件重定向兜底"
    assert returncode == 0
    assert "FALLBACK_OK" in stdout
