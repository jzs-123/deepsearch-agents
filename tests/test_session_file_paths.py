import pytest

from app.utils.path_utils import resolve_path


def test_session_relative_paths_and_virtual_prefixes(tmp_path):
    session = tmp_path / "session_a"
    assert resolve_path("sub/report.md", str(session)) == str((session / "sub/report.md").resolve())
    assert resolve_path("/workspace/report.md", str(session)) == str((session / "report.md").resolve())
    assert resolve_path("output/session_a/report.md", str(session)) == str((session / "report.md").resolve())


@pytest.mark.parametrize("path", ["../secret.txt", "output/session_b/secret.txt", "updated/session_b/../../../../secret.txt"])
def test_cross_session_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        resolve_path(path, str(tmp_path / "session_a"))


def test_absolute_path_outside_session_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_path(str(tmp_path / "session_b/secret.txt"), str(tmp_path / "session_a"))


def test_missing_context_rejected():
    with pytest.raises(ValueError):
        resolve_path("secret.txt")
