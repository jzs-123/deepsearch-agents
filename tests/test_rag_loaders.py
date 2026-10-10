"""app/rag/loaders.py：多格式解析与文本归一化。

重点在两件历史踩坑：
* PDF 逐字抽取会在中文字符之间留下空格（docs/rag-self-built.md 5.1），不清理会污染分词与 chunk 边界；
* 依赖缺失（如 pypdf）必须给出**可执行**的提示，而不是静默跳过（5.8）。
"""

from __future__ import annotations

import pytest

from app.rag.chunking import chunk_document
from app.rag.loaders import (
    SUPPORTED_SUFFIXES,
    extract_markdown_headings,
    iter_supported_files,
    load_csv,
    load_document,
    load_markdown,
    load_plain_text,
    normalize_text,
    read_text_file,
)


# ── normalize_text ──────────────────────────────────────────────────


def test_normalize_unifies_newlines():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_normalize_strips_zero_width_and_nbsp():
    # \xa0 先变成普通空格，随后作为「CJK 之间的多余空格」被合并掉
    assert normalize_text("风\u200b控\ufeff策略\xa0要求") == "风控策略要求"
    assert "\u200b" not in normalize_text("甲\u200b乙")
    assert "\ufeff" not in normalize_text("甲\ufeff乙")


def test_normalize_removes_spaces_between_cjk_chars():
    """PDF 逐字抽取的典型副作用：中文字符之间出现空格。"""
    assert normalize_text("风 控 策 略") == "风控策略"


def test_normalize_keeps_spaces_around_latin():
    assert normalize_text("A13 型号") == "A13 型号"


def test_normalize_keeps_blank_line_when_line_ends_with_punctuation():
    """以标点结尾时不受影响 —— 这也解释了缺陷 D1 为什么不容易被发现。"""
    assert normalize_text("第一段内容。\n\n第二段内容。") == "第一段内容。\n\n第二段内容。"
    assert normalize_text("第一段：\n\n第二段") == "第一段：\n\n第二段"


def test_normalize_preserves_blank_line_between_cjk_paragraphs():
    """D1 回归锁（D1 已修：`\\s+` → `[ \\t]+`）。

    `LoadedDocument` 承诺「段落之间用空行分隔」，归一化不得把 CJK 之间的空行删掉。
    这里断言的是**修复后的正确行为**（不再用 xfail），所以一旦回归会立刻报红，
    而不是被 xfail 静默吞掉。
    """
    assert normalize_text("第一段\n\n第二段") == "第一段\n\n第二段"
    assert normalize_text("甲\n\n乙") == "甲\n\n乙"


def test_markdown_structure_survives_normalisation(tmp_path):
    """D1 回归锁（下游后果）——D1 已修。

    失效时的可观察症状：H1 吞掉正文 → `_split_headings` 切不出 section →
    走 `if not sections` 兜底 → 整篇退化成 1 个 chunk、heading 全空
    （与历史 bug 5.3 症状完全相同、根因不同）。
    """
    path = tmp_path / "报告.md"
    path.write_text("# 风控报告\n\n风控模型的准确率为 91.5%。\n\n## 授信\n\n授信要求复核。", encoding="utf-8")
    doc = load_markdown(path, "报告.md")
    assert doc.title == "风控报告", "标题被正文粘住了（D1 回归）"
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    assert len(chunks) >= 2, "整篇退化成 1 个 chunk（D1 回归）"
    assert all(chunk.heading for chunk in chunks), "heading 全空（D1 回归）"


def test_normalize_trims_edges():
    assert normalize_text("  \n 内容 \n  ") == "内容"


# ── read_text_file ──────────────────────────────────────────────────


def test_read_text_file_utf8(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("中文内容", encoding="utf-8")
    assert read_text_file(path) == "中文内容"


def test_read_text_file_utf8_bom(tmp_path):
    path = tmp_path / "b.txt"
    path.write_bytes("中文内容".encode("utf-8-sig"))
    assert read_text_file(path) == "中文内容"


def test_read_text_file_gbk_fallback(tmp_path):
    """中文 Windows 上不少 txt 是 GBK，直接 utf-8 读会抛 UnicodeDecodeError。"""
    path = tmp_path / "c.txt"
    path.write_bytes("风控策略".encode("gbk"))
    assert read_text_file(path) == "风控策略"


# ── 各格式 loader ───────────────────────────────────────────────────


def test_load_plain_text_uses_filename_as_title(tmp_path):
    path = tmp_path / "报告.txt"
    path.write_text("正文内容", encoding="utf-8")
    doc = load_plain_text(path, "报告.txt")
    assert doc.title == "报告"
    assert doc.text == "正文内容"
    assert doc.headings == []


def test_load_markdown_extracts_headings_and_title(tmp_path):
    # 每段以非 CJK 字符开头，避开缺陷 D1（CJK 之间的空行会被吃掉，见文件后半部分）
    path = tmp_path / "白皮书.md"
    path.write_text("# 电商白皮书\n\nRisk: 第一段正文。\n\n## 第一节\n\nA: 内容", encoding="utf-8")
    doc = load_markdown(path, "白皮书.md")
    assert doc.title == "电商白皮书"
    assert [h[1] for h in doc.headings] == ["电商白皮书", "第一节"]
    assert doc.char_count == len(doc.text)


def test_load_markdown_without_headings_falls_back_to_stem(tmp_path):
    path = tmp_path / "无标题.md"
    path.write_text("只有正文", encoding="utf-8")
    assert load_markdown(path, "无标题.md").title == "无标题"


def test_load_markdown_tags_empty_file_with_warning(tmp_path):
    path = tmp_path / "空.md"
    path.write_text("", encoding="utf-8")
    doc = load_markdown(path, "空.md")
    assert doc.warnings and "空" in doc.warnings[0]


def test_extract_markdown_headings_offsets():
    headings = extract_markdown_headings("# 甲\n\nBody\n\n## 乙")
    assert [text for _, text in headings] == ["甲", "乙"]
    assert headings[0][0] == 0
    assert headings[1][0] > headings[0][0]


def test_load_csv_turns_rows_into_sentences(tmp_path):
    path = tmp_path / "渠道.csv"
    path.write_text("渠道,分成比例\n线上,30%\n线下,15%\n", encoding="utf-8")
    doc = load_csv(path, "渠道.csv")
    assert "渠道: 线上；分成比例: 30%。" in doc.text
    assert doc.title == "渠道"


def test_load_csv_empty_file_warns(tmp_path):
    path = tmp_path / "空.csv"
    path.write_text("", encoding="utf-8")
    doc = load_csv(path, "空.csv")
    assert doc.text == ""
    assert doc.warnings


# ── load_document 分派 ──────────────────────────────────────────────


def test_load_document_dispatches_by_suffix(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# 标题\n\nBody: 正文", encoding="utf-8")
    assert load_document(path).title == "标题"


def test_load_document_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="文件不存在"):
        load_document(tmp_path / "nope.md")


def test_load_document_unsupported_suffix_lists_supported(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="暂不支持"):
        load_document(path)


def test_load_document_uses_relative_path_when_base_dir_given(tmp_path):
    sub = tmp_path / "子目录"
    sub.mkdir()
    path = sub / "a.md"
    path.write_text("# 标题", encoding="utf-8")
    doc = load_document(path, base_dir=tmp_path)
    assert doc.source_path == "子目录/a.md"  # 统一正斜杠，便于引用与比对


def test_load_document_falls_back_to_basename_when_outside_base(tmp_path):
    other = tmp_path / "outside"
    other.mkdir()
    path = other / "a.md"
    path.write_text("# 标题", encoding="utf-8")
    assert load_document(path, base_dir=tmp_path / "elsewhere").source_path == "a.md"


# ── iter_supported_files ────────────────────────────────────────────


def test_iter_supported_files_accepts_single_supported_file(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("x", encoding="utf-8")
    assert iter_supported_files(path) == [path]


def test_iter_supported_files_skips_unsupported_single_file(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("x", encoding="utf-8")
    assert iter_supported_files(path) == []


def test_iter_supported_files_recurses_and_filters(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "c.xyz").write_text("x", encoding="utf-8")

    found = {p.name for p in iter_supported_files(tmp_path)}
    assert found == {"a.md", "b.txt"}


def test_supported_suffixes_cover_documented_formats():
    assert {".md", ".txt", ".pdf", ".docx", ".csv"} <= SUPPORTED_SUFFIXES
