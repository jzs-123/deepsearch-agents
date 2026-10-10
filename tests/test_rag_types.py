"""app/rag/types.py：数据模型与指纹。

覆盖：序列化往返、未知字段兼容、引用格式、chunk_id 稳定性与敏感性。
chunk_id 的稳定性是「重复入库幂等」的地基，所以这里单独锁死。
"""

from __future__ import annotations

import pytest

from app.rag.types import Answer, Chunk, RetrievedChunk, content_fingerprint, make_chunk_id


def test_chunk_to_dict_from_dict_roundtrip(make_chunk):
    chunk = make_chunk(text="风控准确率为 91.5%。", index=3, heading="第一章 > 1.1")
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk


def test_chunk_from_dict_ignores_unknown_fields():
    """旧数据里多出来的字段不能炸掉加载（前向兼容）。"""
    data = {
        "chunk_id": "abc",
        "text": "内容",
        "source": "a.md",
        "source_path": "a.md",
        "chunk_index": 0,
        "future_field": "未来才有的字段",
    }
    chunk = Chunk.from_dict(data)
    assert chunk.chunk_id == "abc"
    assert not hasattr(chunk, "future_field")


def test_chunk_citation_format(make_chunk):
    assert make_chunk(source="报告.pdf", index=7).citation == "[报告.pdf#7]"


def test_retrieved_chunk_to_dict_embeds_chunk(make_chunk):
    item = RetrievedChunk(chunk=make_chunk(text="内容"), score=0.5, dense_score=0.4, rank=1)
    data = item.to_dict()
    assert data["chunk"]["text"] == "内容"
    assert data["dense_score"] == 0.4
    assert data["rank"] == 1


def test_answer_defaults_are_grounded():
    answer = Answer(question="q", answer="a")
    assert answer.grounded is True
    assert answer.citations == []
    assert answer.debug == {}
    assert answer.to_dict()["answer"] == "a"


def test_content_fingerprint_is_stable_and_sensitive():
    assert content_fingerprint("同样内容") == content_fingerprint("同样内容")
    assert content_fingerprint("内容A") != content_fingerprint("内容B")


def test_make_chunk_id_stable_for_same_input():
    """同一份文件同一位置同一内容必须得到同一个 chunk_id —— 幂等去重靠它。"""
    first = make_chunk_id("docs/a.md", 2, "内容")
    second = make_chunk_id("docs/a.md", 2, "内容")
    assert first == second


@pytest.mark.parametrize(
    "args",
    [
        ("docs/b.md", 2, "内容"),   # 换路径
        ("docs/a.md", 3, "内容"),   # 换序号
        ("docs/a.md", 2, "别的内容"),  # 换内容
    ],
)
def test_make_chunk_id_changes_when_input_changes(args):
    assert make_chunk_id(*args) != make_chunk_id("docs/a.md", 2, "内容")


def test_make_chunk_id_only_reads_first_120_chars():
    """chunk_id 只取内容前 120 字，超出部分不参与（与实现约定一致）。"""
    head = "甲" * 120
    assert make_chunk_id("a.md", 0, head + "尾部A") == make_chunk_id("a.md", 0, head + "尾部B")
