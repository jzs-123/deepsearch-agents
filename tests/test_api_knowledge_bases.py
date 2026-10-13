import pytest

from app.api.knowledge_bases import (
    normalize_kb_name,
    normalize_thread_id,
    public_kb_summary,
    safe_session_upload_name,
    safe_upload_name,
)


def test_normalize_kb_name_accepts_user_facing_names():
    assert normalize_kb_name("财务制度-2026") == "财务制度-2026"


@pytest.mark.parametrize("name", ["", "../secret", "a/b", "a\\b", ".", ".."])
def test_normalize_kb_name_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        normalize_kb_name(name)


def test_safe_upload_name_rejects_paths_and_unknown_formats():
    with pytest.raises(ValueError):
        safe_upload_name("../policy.pdf")
    with pytest.raises(ValueError):
        safe_upload_name("policy.exe")
    assert safe_upload_name("policy.pdf") == "policy.pdf"


@pytest.mark.parametrize("thread_id", ["", "../other", "a/b", "a\\b", "a:b", "has space"])
def test_normalize_thread_id_rejects_unsafe_values(thread_id):
    with pytest.raises(ValueError):
        normalize_thread_id(thread_id)


def test_normalize_thread_id_accepts_frontend_uuid_and_legacy_name():
    assert normalize_thread_id("550e8400-e29b-41d4-a716-446655440000")
    assert normalize_thread_id("test_session_001") == "test_session_001"


@pytest.mark.parametrize(
    "name",
    ["../report.pdf", "report.exe", "CON.txt", "report.pdf.", "a/b.txt", "a\\b.txt"],
)
def test_safe_session_upload_name_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        safe_session_upload_name(name)


@pytest.mark.parametrize("name", ["notes.txt", "report.pdf", "table.xlsx", "data.csv"])
def test_safe_session_upload_name_accepts_readable_formats(name):
    assert safe_session_upload_name(name) == name


def test_public_kb_summary_recommends_qdrant_for_large_local_index():
    summary = public_kb_summary({"name": "large", "chunk_count": 100_000}, "numpy")
    assert "建议切换 Qdrant" in summary["recommendation"]
    assert "path" not in summary
