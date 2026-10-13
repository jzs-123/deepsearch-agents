import json
from types import SimpleNamespace

import pytest

from app.rag.config import RagConfig
from app.rag.pipeline import get_pipeline
from app.security import knowledge_bases as access
from app.security.auth import AuthError, Identity, set_identity_context, reset_identity_context
from app.tools import rag_tools


@pytest.fixture
def scoped_config(tmp_path, monkeypatch):
    config = RagConfig(data_dir=tmp_path / "bases", embedding_provider="hash")
    monkeypatch.setattr(access, "load_rag_config", lambda: config)
    monkeypatch.setattr(access, "settings", SimpleNamespace(auth_enabled=True, rag_shared_kbs=["public"]))
    return config


def _make_kb(root, name):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(json.dumps({"chunk_count": 1}), encoding="utf-8")


def test_tenant_lists_and_same_name_are_isolated(scoped_config):
    alice = Identity("alice", "a", authenticated=True)
    bob = Identity("bob", "b", authenticated=True)
    a_config = access.knowledge_config(alice)
    b_config = access.knowledge_config(bob)
    assert a_config.data_path != b_config.data_path
    _make_kb(a_config.data_path, "finance")
    _make_kb(b_config.data_path, "secret")
    _make_kb(scoped_config.data_path, "public")
    _make_kb(scoped_config.data_path, "legacy-private")
    assert {item["name"] for item in access.visible_knowledge_bases(alice)} == {"finance", "public"}
    assert access.query_knowledge_config("secret", alice).data_path == a_config.data_path
    assert access.query_knowledge_config("public", alice).data_path == scoped_config.data_path


def test_scoped_access_requires_identity(scoped_config):
    with pytest.raises(AuthError):
        access.knowledge_config()


def test_agent_cannot_bypass_scope_using_path(scoped_config):
    identity = Identity("alice", "a", authenticated=True)
    token = set_identity_context(identity)
    try:
        with pytest.raises(ValueError):
            access.query_knowledge_config("../other")
        reply = rag_tools.search_knowledge_base.invoke({"question": "secret", "kb_name": "../other"})
        assert "secret text" not in reply
        assert "检索失败" in reply
    finally:
        reset_identity_context(token)


def test_pipeline_cache_separates_data_roots(tmp_path):
    a = RagConfig(data_dir=tmp_path / "a", embedding_provider="hash")
    b = RagConfig(data_dir=tmp_path / "b", embedding_provider="hash")
    first = get_pipeline("finance", config=a)
    second = get_pipeline("finance", config=b)
    assert first is not second
    assert first.kb_dir != second.kb_dir
