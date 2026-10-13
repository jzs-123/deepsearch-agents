"""会话附件上传的校验与原子写入。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Sequence

from fastapi import HTTPException, UploadFile

from app.api.knowledge_bases import normalize_thread_id, safe_session_upload_name

MAX_SESSION_FILE_BYTES = 50 * 1024 * 1024
MAX_SESSION_FILES_PER_UPLOAD = 10
MAX_SESSION_TOTAL_BYTES = 200 * 1024 * 1024


async def save_session_uploads(
    files: Sequence[UploadFile],
    thread_id: str,
    root: Path,
) -> list[str]:
    """校验并原子保存一批会话附件；任一文件失败时回滚本批次。"""
    try:
        safe_thread_id = normalize_thread_id(thread_id)
        if not files:
            raise ValueError("请至少选择一个文件")
        if len(files) > MAX_SESSION_FILES_PER_UPLOAD:
            raise ValueError(f"一次最多上传 {MAX_SESSION_FILES_PER_UPLOAD} 个文件")
        safe_names = [safe_session_upload_name(file.filename or "") for file in files]
        if len(set(safe_names)) != len(safe_names):
            raise ValueError("一次上传中不能包含同名文件")
    except ValueError as exc:
        for upload in files:
            await upload.close()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_dir = root / f"session_{safe_thread_id}"
    saved_paths: list[Path] = []
    temp_paths: list[Path] = []
    saved_files: list[str] = []

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        existing_total = sum(path.stat().st_size for path in target_dir.iterdir() if path.is_file())
        for upload, safe_name in zip(files, safe_names):
            target = target_dir / safe_name
            if target.exists():
                raise HTTPException(status_code=409, detail=f"文件 {safe_name} 已存在，请重命名后上传")

            temp = target_dir / f".{uuid.uuid4().hex}.uploading"
            temp_paths.append(temp)
            written = 0
            with temp.open("xb") as buffer:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_SESSION_FILE_BYTES:
                        raise HTTPException(status_code=413, detail=f"{safe_name} 超过 50MB 限制")
                    if existing_total + written > MAX_SESSION_TOTAL_BYTES:
                        raise HTTPException(status_code=413, detail="当前会话附件总量超过 200MB 限制")
                    buffer.write(chunk)

            temp.replace(target)
            temp_paths.remove(temp)
            saved_paths.append(target)
            saved_files.append(safe_name)
            existing_total += written

        return saved_files
    except BaseException:
        for path in saved_paths + temp_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for upload in files:
            await upload.close()


__all__ = [
    "MAX_SESSION_FILE_BYTES",
    "MAX_SESSION_FILES_PER_UPLOAD",
    "MAX_SESSION_TOTAL_BYTES",
    "save_session_uploads",
]
