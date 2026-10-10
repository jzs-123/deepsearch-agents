"""app/rag/chunking.py：结构感知切块。

本文件锁死两条历史 bug：

* 历史 bug 1（docs/rag-self-built.md 5.3）：`_MD_HEADING` 漏了 `re.MULTILINE`。
  没有它时 `^`/`$` 只匹配整串首尾，逐行 `match()` 看起来正常，但全文 `finditer`
  一条标题都匹配不到 —— 整篇只切出 1 个 chunk、heading 全空。**不抛异常，只让效果变差**，
  是最难发现的一类 bug。见 `test_md_heading_finditer_finds_every_heading`。

* 历史 bug 2（5.2 踩坑细节）：中文编号标题没有 `#` 标记，正文里的编号列表
  （「一、核心结论金融与电商……」）也匹配编号正则，最初被粘成了畸形标题。
  见 `test_cn_numbered_body_line_is_not_a_heading`。
"""

from __future__ import annotations

import re

import pytest

from app.rag.chunking import (
    _CN_HEADING,
    _MD_HEADING,
    _is_cn_title,
    _with_overlap,
    chunk_document,
)
from app.rag.loaders import LoadedDocument


def _doc(text: str, source_path: str = "a.md") -> LoadedDocument:
    return LoadedDocument(text=text, source_path=source_path, title=source_path)


# ── 历史 bug 1：re.MULTILINE ──────────────────────────────────────────


def test_md_heading_pattern_has_multiline_flag():
    """判据直接读 flags（8 == re.MULTILINE），防止有人"简化"正则时把它去掉。"""
    assert _MD_HEADING.flags & re.MULTILINE


def test_md_heading_finditer_finds_every_heading():
    """真正的回归锁：全文 finditer 必须找到每一行标题。

    这正是漏 MULTILINE 时静默失效的地方：逐行 match 全部通过，finditer 却一条都找不到。
    """
    text = "# 第一章\n\n内容A\n\n## 1.1 小节\n\n内容B\n\n# 第二章\n\n内容C"
    headings = [m.group(2).strip() for m in _MD_HEADING.finditer(text)]
    assert headings == ["第一章", "1.1 小节", "第二章"]
    assert len(headings) > 1, "整篇只识别出一个标题即说明 MULTILINE 失效"


def test_chunk_document_splits_on_every_markdown_heading():
    """失效时的可观察后果：整篇只切出 1 个 chunk、heading 为空。"""
    doc = _doc("# 第一章 总则\n\n内容一。\n\n## 1.1 细则\n\n内容二。")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    assert len(chunks) >= 2
    assert all(chunk.heading for chunk in chunks)
    assert "第一章 总则 > 1.1 细则" in {chunk.heading for chunk in chunks}


def test_heading_path_uses_three_level_stack():
    doc = _doc("# 甲\n\nA\n\n## 乙\n\nB\n\n### 丙\n\nC")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    assert "甲 > 乙 > 丙" in {chunk.heading for chunk in chunks}


# ── 历史 bug 2：中文编号标题 vs 正文编号列表 ────────────────────────


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("总则", True),
        ("", True),
        ("核心结论金融与电商业务正在加速融合。", False),  # 带句号 → 正文
        ("先做 A，再做 B", False),                        # 带逗号 → 正文
        ("标" * 40, False),                               # 过长 → 正文
    ],
)
def test_is_cn_title_decision_table(tail, expected):
    assert _is_cn_title(tail) is expected


def test_cn_numbered_heading_is_recognised():
    assert _CN_HEADING.match("一、总则")
    assert _CN_HEADING.match("第三章 风控")
    assert _CN_HEADING.match("3.2.1 授信规则")


def test_cn_numbered_body_line_is_not_a_heading():
    """历史 bug 2 的回归锁：畸形标题「一、核心结论金融与电商……」必须不再出现。"""
    doc = _doc("# 第一章\n\n一、核心结论金融与电商业务正在加速融合。\n\n后续正文。")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    headings = {chunk.heading for chunk in chunks}
    assert headings == {"第一章"}
    assert not any("核心结论" in heading for heading in headings)
    # 该行必须作为正文被保留下来，而不是被吃掉
    assert any("核心结论金融与电商" in chunk.text for chunk in chunks)


def test_cn_numbered_heading_still_works_when_genuine():
    doc = _doc("一、总则\n\n第一条内容。\n\n二、细则\n\n第二条内容。")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    headings = {chunk.heading for chunk in chunks}
    assert {"一、总则", "二、细则"} <= headings


# ── 重叠窗口 ─────────────────────────────────────────────────────────


def test_with_overlap_prefers_sentence_boundary():
    """重叠从上一块的句边界回退，不从句子中间截断。"""
    prev = "第一句。第二句。第三句。"
    assert _with_overlap(prev, "新内容。", 8) == "第三句。\n新内容。"


def test_with_overlap_falls_back_to_raw_tail_without_boundary():
    assert _with_overlap("abcdef", "新。", 3) == "def\n新。"


@pytest.mark.parametrize("overlap", [0, -5])
def test_with_overlap_disabled_returns_text_unchanged(overlap):
    assert _with_overlap("上一块内容", "新内容", overlap) == "新内容"


def test_with_overlap_no_previous_chunk():
    assert _with_overlap("", "新内容", 10) == "新内容"


def test_chunk_overlap_actually_carries_text_between_chunks():
    """端到端：开启重叠后，后一块的头部应带上上一块的尾部文字。"""
    doc = _doc("第一句话。第二句话。第三句话。第四句话。第五句话。第六句话。")
    chunks = chunk_document(doc, chunk_size=12, chunk_overlap=6)
    assert len(chunks) >= 2
    head_of_second = chunks[1].text.split("\n")[0]
    assert head_of_second in chunks[0].text, (
        f"第二块开头 {head_of_second!r} 未出现在第一块中，重叠未生效"
    )


# ── 兜底硬切 / 上限 / 代码块 / 边界 ──────────────────────────────────


def test_hard_split_for_long_text_without_punctuation():
    """无标点的长表格行只能硬切，且每块不超过 chunk_size。"""
    doc = _doc("甲" * 250)
    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 3
    assert [len(chunk.text) for chunk in chunks] == [100, 100, 50]


def test_max_chunks_caps_output():
    """超大文件不能撑爆内存。"""
    doc = _doc("\n\n".join(f"# 标题{i}\n\n内容{i}。" for i in range(10)))
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=0, max_chunks=3)
    assert len(chunks) == 3


def test_headings_inside_code_fence_are_ignored():
    doc = _doc("# 真标题\n\n正文A。\n\n```\n# 假标题\ncode\n```\n\n正文B。")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    headings = {chunk.heading for chunk in chunks}
    assert not any("假标题" in heading for heading in headings)
    assert "# 假标题" in "\n".join(chunk.text for chunk in chunks)


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc("   \n  "), chunk_size=100, chunk_overlap=0) == []


def test_chunk_ids_are_unique_and_indexed_in_order():
    doc = _doc("# 甲\n\nA\n\n# 乙\n\nB\n\n# 丙\n\nC")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_all_chunks_share_document_fingerprint():
    from app.rag.types import content_fingerprint

    doc = _doc("# 甲\n\nA\n\n# 乙\n\nB")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=0)
    assert {chunk.doc_id for chunk in chunks} == {content_fingerprint(doc.text)}


def test_chunk_length_never_exceeds_size_plus_overlap():
    doc = _doc("句子一。句子二。句子三。句子四。句子五。句子六。句子七。句子八。")
    size, overlap = 20, 8
    chunks = chunk_document(doc, chunk_size=size, chunk_overlap=overlap)
    assert all(len(chunk.text) <= size + overlap for chunk in chunks)
