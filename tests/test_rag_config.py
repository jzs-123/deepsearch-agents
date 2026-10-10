"""app/rag/config.py：配置装配与校验。

设计意图是「脏配置要在构造阶段就炸，而不是跑到检索阶段才出问题」，所以这里逐条锁死校验分支。
"""

from __future__ import annotations

import dataclasses

import pytest

from app.rag.config import EMBEDDING_PROVIDERS, RERANK_MODES, RagConfig, ensure_within


def _cfg(tmp_path, **overrides):
    base = dict(data_dir=tmp_path / "rag_data")
    base.update(overrides)
    return RagConfig(**base)


def test_defaults_are_sane(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.default_kb == "default"
    assert cfg.embedding_provider in EMBEDDING_PROVIDERS
    assert cfg.rerank_mode in RERANK_MODES
    assert 0 <= cfg.chunk_overlap < cfg.chunk_size
    assert cfg.top_n <= cfg.top_k


def test_unknown_embedding_provider_rejected(tmp_path):
    with pytest.raises(ValueError, match="embedding_provider"):
        _cfg(tmp_path, embedding_provider="magic")


def test_unknown_rerank_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="rerank_mode"):
        _cfg(tmp_path, rerank_mode="bert")


@pytest.mark.parametrize("size", [0, -1])
def test_non_positive_chunk_size_rejected(tmp_path, size):
    with pytest.raises(ValueError, match="chunk_size"):
        _cfg(tmp_path, chunk_size=size)


@pytest.mark.parametrize("overlap", [-1, 500])
def test_overlap_out_of_range_rejected(tmp_path, overlap):
    """overlap >= chunk_size 会让切块窗口无法前进（退化成死循环），必须拒绝。"""
    with pytest.raises(ValueError, match="chunk_overlap"):
        _cfg(tmp_path, chunk_size=500, chunk_overlap=overlap)


def test_top_n_greater_than_top_k_rejected(tmp_path):
    with pytest.raises(ValueError, match="top_n"):
        _cfg(tmp_path, top_k=3, top_n=4)


def test_config_is_immutable(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.chunk_size = 999  # type: ignore[misc]


def test_with_kb_returns_copy_and_keeps_other_params(tmp_path):
    cfg = _cfg(tmp_path, default_kb="default", chunk_size=333)
    other = cfg.with_kb("fin")
    assert other.default_kb == "fin"
    assert other.chunk_size == 333
    assert cfg.default_kb == "default"  # 原对象不受影响


def test_with_kb_none_falls_back_to_default(tmp_path):
    cfg = _cfg(tmp_path, default_kb="default")
    assert cfg.with_kb(None).default_kb == "default"


def test_kb_dir_is_data_dir_slash_kb(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.kb_dir("fin") == cfg.data_path / "fin"
    assert cfg.kb_dir() == cfg.data_path / "default"


def test_ensure_within_accepts_child(tmp_path):
    base = tmp_path / "rag_data"
    base.mkdir()
    target = base / "fin"
    assert ensure_within(base, target) == target.resolve()


def test_ensure_within_rejects_escape(tmp_path):
    """知识库名里带 ../ 不能越界读写。"""
    base = tmp_path / "rag_data"
    base.mkdir()
    with pytest.raises(ValueError, match="非法路径"):
        ensure_within(base, tmp_path / "outside")


def test_ensure_within_accepts_base_itself(tmp_path):
    base = tmp_path / "rag_data"
    base.mkdir()
    assert ensure_within(base, base) == base.resolve()
