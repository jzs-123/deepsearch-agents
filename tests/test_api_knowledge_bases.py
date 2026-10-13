import pytest

from app.api.knowledge_bases import normalize_kb_name, public_kb_summary, safe_upload_name


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


def test_public_kb_summary_recommends_qdrant_for_large_local_index():
    summary = public_kb_summary({"name": "large", "chunk_count": 100_000}, "numpy")
    assert "建议切换 Qdrant" in summary["recommendation"]
    assert "path" not in summary
