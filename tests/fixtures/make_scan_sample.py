"""生成 OCR 用例所需的两个 PDF 样本(可复现)。

仓库自带的 5 份 PDF 全部有文本层(2 万~4.6 万字符),**没有扫描件样本**,
所以这里用 ``PIL`` + ``reportlab``(都已在依赖里)合成两个互补的样本:

- ``scan_sample.pdf``:文字先渲染成位图再嵌进 PDF → **没有文本层**,只能靠 OCR;
- ``text_sample.pdf``:reportlab 直接写文本 → **有文本层**,用于验证「有文本层不触发 OCR」。

用法(在仓库根执行):

    python tests/fixtures/make_scan_sample.py

依赖:``Pillow``、``reportlab``(无需联网、无需额外安装)。
注意:**不要**把这里的文本换成非中文,样本的用途就是验证中文 OCR。
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# 样本正文:刻意混入「专有名词 + 数字 + 百分号」,便于观察 OCR 对数字/标点的失真
SCAN_LINES = [
    "风控模型的准确率为 91.5%",
    "风险管理部编制 2026 年模型监控报告",
    "信用评分与逾期率是核心指标",
]

# 文本层样本:内容与规模都足够(> 阈值 50 字符),用于验证「不触发 OCR」
TEXT_LINES = [
    "风控模型监控季度报告",
    "本报告由风险管理部编制,涵盖信用评分、逾期率与模型稳定性三类指标。",
    "2026 年第二季度风控模型的准确率为 91.5%,较上季度提升 1.2 个百分点。",
] * 4  # 重复几段,确保文本层字符数明显高于 OCR 阈值

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
    r"C:\Windows\Fonts\simsun.ttc",  # 宋体
    r"C:\Windows\Fonts\simhei.ttf",  # 黑体
    r"C:\Windows\Fonts\Deng.ttf",  # 等线
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError(
        "找不到可用的中文字体(尝试过:%s)。本脚本需要系统装有中文字体才能生成样本。"
        % ", ".join(FONT_CANDIDATES)
    )


def render_text_image(lines: list[str], *, font_size: int = 40, width: int = 1240) -> Image.Image:
    """把文字渲染成白底黑字位图(模拟「扫描/拍照得到的图」)。"""
    font = _load_font(font_size)
    line_height = int(font_size * 1.8)
    height = line_height * len(lines) + line_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((40, line_height // 2 + index * line_height), line, font=font, fill="black")
    return image


def build_scanned_pdf(out_path: Path) -> Path:
    """合成「扫描件 PDF」:整页是一张位图,**不含任何文本层**。"""
    image = render_text_image(SCAN_LINES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    pdf = canvas.Canvas(str(out_path), pagesize=A4)
    pdf.drawImage(
        ImageReader(image),
        0,
        page_height - page_width * image.height / image.width,
        width=page_width,
        height=page_width * image.height / image.width,
    )
    pdf.showPage()
    pdf.save()
    return out_path


def build_text_pdf(out_path: Path) -> Path:
    """合成「有文本层的 PDF」:正常写文本,可被 pypdf 直接抽取。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    pdf = canvas.Canvas(str(out_path), pagesize=A4)
    # reportlab 内置字体不含中文;用系统字体注册一个,保证文本层里连中文也是真字符
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    if font_path is None:
        raise RuntimeError("找不到可用的中文字体,无法生成文本层样本")
    try:
        pdfmetrics.registerFont(TTFont("SampleCJK", font_path))
        font_name = "SampleCJK"
    except Exception:  # .ttc 集合字体可能注册失败 → 退回内置字体(文本层仍是真文本)
        font_name = "Helvetica"

    text = pdf.beginText(40, page_height - 60)
    text.setFont(font_name, 13)
    text.setLeading(20)
    for line in TEXT_LINES:
        text.textLine(line)
    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    return out_path


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    scan = build_scanned_pdf(out_dir / "scan_sample.pdf")
    text = build_text_pdf(out_dir / "text_sample.pdf")
    print("wrote", scan, scan.stat().st_size, "bytes")
    print("wrote", text, text.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
