from io import BytesIO

import pytest
from fastapi import HTTPException

from app.api import uploads


class _Upload:
    def __init__(self, name: str, content: bytes):
        self.filename = name
        self._file = BytesIO(content)
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        return self._file.read(size)

    async def close(self) -> None:
        self._file.close()
        self.closed = True


def _upload(name: str, content: bytes) -> _Upload:
    return _Upload(name, content)


def _run(coroutine):
    """驱动本测试的纯内存协程，避开沙箱内不可用的 Windows socketpair。"""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("测试替身不应产生异步挂起点")


def test_session_upload_streams_file_into_isolated_directory(tmp_path, monkeypatch):
    result = _run(uploads.save_session_uploads([_upload("notes.txt", b"hello")], "thread_1", tmp_path))

    assert result == ["notes.txt"]
    assert (tmp_path / "session_thread_1" / "notes.txt").read_bytes() == b"hello"


def test_session_upload_rejects_unsafe_thread_before_writing(tmp_path, monkeypatch):
    with pytest.raises(HTTPException) as error:
        _run(uploads.save_session_uploads([_upload("notes.txt", b"hello")], "../escape", tmp_path))

    assert error.value.status_code == 400
    assert not any(path.name.startswith("session_") for path in tmp_path.iterdir())


def test_session_upload_rejects_oversize_and_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_SESSION_FILE_BYTES", 4)

    with pytest.raises(HTTPException) as error:
        _run(uploads.save_session_uploads([_upload("notes.txt", b"12345")], "thread_1", tmp_path))

    assert error.value.status_code == 413
    assert list((tmp_path / "session_thread_1").iterdir()) == []


def test_session_upload_rolls_back_batch_when_later_file_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_SESSION_FILE_BYTES", 4)

    files = [_upload("ok.txt", b"1234"), _upload("large.txt", b"12345")]
    with pytest.raises(HTTPException) as error:
        _run(uploads.save_session_uploads(files, "thread_1", tmp_path))

    assert error.value.status_code == 413
    assert list((tmp_path / "session_thread_1").iterdir()) == []


def test_session_upload_refuses_overwrite(tmp_path, monkeypatch):
    target_dir = tmp_path / "session_thread_1"
    target_dir.mkdir()
    (target_dir / "notes.txt").write_text("old", encoding="utf-8")

    with pytest.raises(HTTPException) as error:
        _run(uploads.save_session_uploads([_upload("notes.txt", b"new")], "thread_1", tmp_path))

    assert error.value.status_code == 409
    assert (target_dir / "notes.txt").read_text(encoding="utf-8") == "old"


def test_session_upload_enforces_total_capacity_including_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_SESSION_TOTAL_BYTES", 6)
    target = tmp_path / "session_thread_1"
    target.mkdir()
    (target / "old.txt").write_bytes(b"1234")
    with pytest.raises(HTTPException) as error:
        _run(uploads.save_session_uploads([_upload("new.txt", b"123")], "thread_1", tmp_path))
    assert error.value.status_code == 413
    assert [path.name for path in target.iterdir()] == ["old.txt"]


def test_session_upload_rejects_too_many_files_and_closes_them(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_SESSION_FILES_PER_UPLOAD", 1)
    files = [_upload("a.txt", b"a"), _upload("b.txt", b"b")]
    with pytest.raises(HTTPException) as error:
        _run(uploads.save_session_uploads(files, "thread_1", tmp_path))
    assert error.value.status_code == 400
    assert all(file.closed for file in files)


def test_cancellation_cleans_partial_upload(tmp_path):
    import asyncio

    class CancelledUpload(_Upload):
        async def read(self, size=-1):
            raise asyncio.CancelledError

    upload = CancelledUpload("notes.txt", b"hello")
    with pytest.raises(asyncio.CancelledError):
        _run(uploads.save_session_uploads([upload], "thread_1", tmp_path))
    assert list((tmp_path / "session_thread_1").iterdir()) == []
    assert upload.closed
