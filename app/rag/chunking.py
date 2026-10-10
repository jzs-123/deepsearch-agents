"""
文本切块层

切块是 RAG 里最容易被低估、也最容易毁掉效果的一环:
- chunk 太大 -> 检索命中里混入无关内容,噪声拉低生成质量;
- chunk 太小 -> 一句话被拦腰截断,答案碎片化;
- 硬按固定字数切 -> 表格被切散、标题和正文分家。

所以这里采用「结构优先 + 长度兜底」的递归切分:
1. 先按 Markdown / 中文编号标题切大段(文档结构是天然的语义边界);
2. 大段内按空行切段落;
3. 段落还超长就按中文句末标点切句;
4. 单句仍然超长(如无标点的长表格行)才硬切。

每个 chunk 都带 heading(所属标题路径),让「第三章 > 3.2 风控」这类结构信息一起参与
检索匹配,同时正文保持可读。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.loaders import LoadedDocument
from app.rag.types import Chunk, content_fingerprint, make_chunk_id

# 标题行:Markdown 标题或「第X章 / 一、/ 1.1」这类中文编号标题
# 必须带 re.MULTILINE:不带的话 ^ 只匹配整串开头、$ 只匹配整串末尾,
# 逐行 match 时看似正常,但对全文 finditer 会一条标题都匹配不到(踩过的坑)。
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_CN_HEADING = re.compile(
    r"^(第[一二三四五六七八九十百零\d]+[章节篇部分]|[一二三四五六七八九十]+、|\d+(?:\.\d+){1,3}\s+)\s*(.{0,60})$",
    re.MULTILINE,
)
# 句末标点作为切点
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])")
# 代码块围栏:代码块内部不做标题识别与切分
_FENCE = re.compile(r"^\s*```")
# 标题里不应该出现的标点:出现说明这一行是正文而非标题(中文编号标题没有 #,只能靠这个区分)
_TITLE_PUNCT = "。！？；!?;，,、"


@dataclass
class _Unit:
    """切块中间态:一段带有结构上下文与原文偏移的文本。"""

    text: str
    heading_path: str
    start: int


def _is_cn_title(tail: str) -> bool:
    """判断中文编号行后面跟的内容到底是不是标题的一部分。

    中文文档常用「一、核心结论」这类没有 # 的编号标题,但正文里也常见「1. 先做 A,再做 B。」
    这类编号列表。两者都匹配编号正则,只能靠后面内容的特征区分:
    - 标题:短、不含句末标点、不含逗号;
    - 正文:较长,或带标点。
    """
    tail = (tail or "").strip()
    if not tail:
        return True
    if len(tail) > 30:
        return False
    return not any(ch in tail for ch in _TITLE_PUNCT)


def _split_headings(text: str) -> list[tuple[str, int, str]]:
    """按标题把正文切成 (标题路径, 起始偏移, 正文) 列表。

    标题路径形如 "第二章 风控 > 2.1 授信规则",用最近三级拼接。
    """
    lines = text.split("\n")
    sections: list[tuple[str, int, str]] = []
    stack: list[str] = []
    buffer: list[str] = []
    current_path = ""
    cursor = 0
    body_start = 0
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer)
        if body.strip():
            # 记录该段正文在原文中的绝对起始位置(去掉前导空行)
            leading = len(body) - len(body.lstrip("\n"))
            sections.append((current_path, body_start + leading, body.strip()))

    for line in lines:
        line_len = len(line) + 1  # 换行符也计入偏移
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            cursor += line_len
            continue

        if not in_fence:
            md_match = _MD_HEADING.match(line)
            cn_match = None if md_match else _CN_HEADING.match(line.strip())
            if cn_match and not _is_cn_title(cn_match.group(2)):
                # 形似编号但其实是正文(如「一、核心结论金融与电商正在加速融合。」),
                # 放弃标题判定,当作普通段落处理,否则标题会被粘上正文句子
                cn_match = None
            if md_match or cn_match:
                flush()
                buffer = []
                body_start = cursor + line_len
                if md_match:
                    level = len(md_match.group(1))
                    title = md_match.group(2).strip()
                else:
                    title = line.strip()
                    numbering = cn_match.group(1).strip()
                    # 中文编号标题没有显式层级,按编号里点数推断深度
                    level = numbering.count(".") + 1
                stack = stack[: level - 1]
                while len(stack) < level - 1:
                    stack.append("")
                stack.append(title)
                current_path = " > ".join([item for item in stack if item])
                cursor += line_len
                continue

        if not buffer:
            body_start = cursor
        buffer.append(line)
        cursor += line_len

    flush()
    if not sections:
        sections.append(("", 0, text.strip()))
    return sections


def _split_sentences(text: str) -> list[str]:
    """按句末标点切句,标点保留在句尾。"""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hard_split(text: str, size: int) -> list[str]:
    """无标点长文本(如表格行)的兜底硬切。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _pack_units(
    units: list[_Unit],
    size: int,
    joiner: str,
) -> list[tuple[str, str, int]]:
    """按长度贪心装箱,返回 (文本, 标题路径, 起始偏移) 列表。

    装箱时保留第一个消费到的 unit 的标题与偏移,这样倾斜的标题信息不会丢。
    """
    packed: list[tuple[str, str, int]] = []
    current = ""
    current_heading = ""
    current_start = 0

    def push() -> None:
        nonlocal current, current_heading, current_start
        if current:
            packed.append((current, current_heading, current_start))
        current = ""
        current_heading = ""
        current_start = 0

    for unit in units:
        text = unit.text.strip()
        if not text:
            continue
        # 单个单元本身超长:先落盘当前箱,再把单元硬切成多块
        if len(text) > size:
            push()
            for piece in _hard_split(text, size):
                packed.append((piece, unit.heading_path, unit.start))
            continue

        candidate = f"{current}{joiner}{text}" if current else text
        if len(candidate) <= size:
            if not current:
                current_heading = unit.heading_path
                current_start = unit.start
            current = candidate
        else:
            push()
            current = text
            current_heading = unit.heading_path
            current_start = unit.start

    push()
    return packed


def _with_overlap(prev_text: str, text: str, overlap: int) -> str:
    """给 chunk 头部拼上上一块的尾部内容,避免答案恰好被切在边界。

    重叠部分从上一块尾部按句边界回退,尽量不从句子中间截断。
    """
    if overlap <= 0 or not prev_text:
        return text
    tail = prev_text[-overlap:]
    # 尽量从句号/换行之后开始,让拼接后的开头是完整句子
    for match in re.finditer(r"[。！？；!?;\n]", tail):
        candidate = tail[match.end() :].strip()
        if candidate:
            return f"{candidate}\n{text}"
    return f"{tail}\n{text}"


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    max_chunks: int = 2000,
) -> list[Chunk]:
    """把解析后的文档切成带结构信息的 chunk 列表。

    :param doc: loaders 产出的文档
    :param chunk_size: 单个 chunk 目标字符数
    :param chunk_overlap: 相邻 chunk 重叠字符数
    :param max_chunks: 单文档 chunk 上限,防止超大文件撑爆内存
    """
    if not doc.text.strip():
        return []

    chunks: list[Chunk] = []
    index = 0

    for heading_path, section_start, body in _split_headings(doc.text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        units: list[_Unit] = []
        search_from = section_start

        for para in paragraphs:
            # 逐段在原文中推进定位,拿到该段落的绝对偏移
            found = doc.text.find(para[:30], search_from)
            if found < 0:
                found = search_from
            search_from = found

            if len(para) <= chunk_size:
                units.append(_Unit(text=para, heading_path=heading_path, start=found))
            else:
                # 段落过长:先按句装箱,句内超长再由 _pack_units 硬切
                offset = found
                for piece in _pack_units(
                    [_Unit(t, heading_path, offset) for t in _split_sentences(para)],
                    chunk_size,
                    joiner="",
                ):
                    units.append(_Unit(piece[0], piece[1], piece[2]))

        prev_text = ""
        for text, heading, start in _pack_units(units, chunk_size, joiner="\n"):
            final_text = _with_overlap(prev_text, text, chunk_overlap)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.source_path, index, final_text),
                    text=final_text.strip(),
                    source=doc.source_path.split("/")[-1],
                    source_path=doc.source_path,
                    chunk_index=index,
                    heading=heading,
                    doc_id=content_fingerprint(doc.text),
                )
            )
            prev_text = text
            index += 1
            if len(chunks) >= max_chunks:
                return chunks

    return chunks


__all__ = ["chunk_document"]
