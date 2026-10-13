"""
FastAPI 接口层与项目闭环入口(企业级重构版)

负责承接前端的任务提交、任务取消、文件上传/下载、输出文件列表查询、历史会话
管理与 WebSocket 长连接。相比教学版,本模块新增:

1. 请求 ID 中间件(全链路关联)
2. 进程内限流(RATE_LIMIT_PER_MINUTE)
3. 任务并发信号量(MAX_CONCURRENT_TASKS)
4. 身份解析与多租户隔离(默认关闭,兼容旧前端)
5. 健康检查 /api/health 与指标 /api/metrics
"""

import asyncio
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.agent.main_agent import run_deep_agent
from app.api.knowledge_bases import normalize_kb_name, public_kb_summary, safe_upload_name
from app.api.monitor import manager, monitor
from app.core.config import settings
from app.core.logging import (
    configure_logging,
    get_logger,
    reset_request_context,
    set_request_context,
)
from app.observability.metrics import metrics
from app.security.auth import AuthError, Identity, resolve_identity
from app.utils.history_manager import (
    delete_session,
    get_session,
    get_session_messages,
    list_sessions,
)

logger = get_logger("api.server")

_started_at = time.time()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """服务生命周期入口:绑定事件循环、初始化日志与并发信号量。"""
    configure_logging()
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)
    logger.info("服务启动,绑定事件循环: %s", id(loop))
    yield
    logger.info("服务关闭")


# 当前文件位于 app/api/server.py,运行时目录统一收敛到 app 目录
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

app = FastAPI(title="DeepAgents API", lifespan=lifespan)

# output 保存每个会话最终工作区,前端只允许从这里浏览和下载生成文件
output_dir = project_root / "output"
output_dir.mkdir(exist_ok=True)

# updated 暂存用户上传文件,run_deep_agent 启动时会复制到对应 output/session_xxx
updated_dir = project_root / "updated"
updated_dir.mkdir(exist_ok=True)

kb_upload_dir = updated_dir / "knowledge_bases"
kb_upload_dir.mkdir(exist_ok=True)

MAX_KB_FILE_BYTES = 50 * 1024 * 1024
_kb_ingest_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


# ─── 中间件 ──────────────────────────────────────────────────


class RequestIDMiddleware:
    """为每个 HTTP 请求生成 X-Request-ID,并绑定到日志上下文。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        rid = str(uuid.uuid4())

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", rid.encode("utf-8")))
                message["headers"] = headers
            await send(message)

        tokens = set_request_context(request_id=rid)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_request_context(*tokens)


class RateLimitMiddleware:
    """进程内令牌桶限流(按客户端 IP 的滑动窗口)。"""

    def __init__(self, app, per_minute: int):
        self.app = app
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client") or ("unknown", 0)
        ip = client[0]
        now = time.monotonic()
        window = self._hits[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            await JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁,请稍后重试"},
            )(scope, receive, send)
            return
        window.append(now)
        await self.app(scope, receive, send)


# 教学项目通常前后端分别本地启动,这里放开跨域以便 Vite 页面直接调用 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)


# ─── 任务并发治理 ────────────────────────────────────────────

# 保存 thread_id -> 后台 Agent 任务,用于同一会话任务替换和主动取消
active_tasks: dict[str, asyncio.Task] = {}
_task_semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)


class TaskRequest(BaseModel):
    """前端启动任务时提交的请求体。"""

    query: str
    thread_id: str = None


def _forget_task(thread_id: str, task: asyncio.Task) -> None:
    """清理已结束任务的登记关系,避免误删同 thread_id 下刚启动的新任务。"""
    if active_tasks.get(thread_id) is task:
        active_tasks.pop(thread_id, None)


def _resolve_identity(headers) -> Identity:
    """解析身份,鉴权失败统一转 401。"""
    try:
        return resolve_identity(headers)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def _run_task_safely(query: str, thread_id: str, identity: Identity) -> None:
    """包装 run_deep_agent,确保任务结束(含异常/取消)后释放并发信号量。"""
    try:
        await run_deep_agent(query, thread_id, identity=identity)
    finally:
        _task_semaphore.release()


@app.post("/api/task")
async def run_task(request: TaskRequest, http_request: Request):
    """
    启动一次 DeepAgents 后台任务。

    HTTP 请求只负责创建后台协程并立即返回,后续执行轨迹、子智能体调用和最终
    答案都会由 monitor 通过 `/ws/{thread_id}` 推送给同一会话的前端。
    """
    identity = _resolve_identity(http_request.headers)
    thread_id = request.thread_id or str(uuid.uuid4())

    # 全局并发上限
    if _task_semaphore.locked():
        raise HTTPException(status_code=429, detail="并发任务数已达上限,请稍后重试")

    # 同一个 thread_id 只保留一个活跃任务,新任务会先取消旧任务
    old_task = active_tasks.get(thread_id)
    if old_task and not old_task.done():
        old_task.cancel()

    await _task_semaphore.acquire()
    task = asyncio.create_task(_run_task_safely(request.query, thread_id, identity))
    active_tasks[thread_id] = task
    task.add_done_callback(lambda finished_task: _forget_task(thread_id, finished_task))
    logger.info("任务已启动 thread_id=%s user=%s", thread_id, identity.user_id)

    return {"status": "started", "thread_id": thread_id}


@app.post("/api/task/{thread_id}/cancel")
async def cancel_task(thread_id: str):
    """取消指定 thread_id 对应的后台 Agent 任务(立即通知前端,再注入取消信号)。"""
    task = active_tasks.get(thread_id)
    if not task or task.done():
        active_tasks.pop(thread_id, None)
        raise HTTPException(status_code=404, detail="任务不存在或已结束")

    monitor.report_task_cancelled()
    _forget_task(thread_id, task)
    task.cancel()

    return {"status": "cancelled", "thread_id": thread_id}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), thread_id: str = Form(...)):
    """文件上传接口:按会话隔离保存到 updated/session_{thread_id}。"""
    target_dir = updated_dir / f"session_{thread_id}"
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        file_path = target_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    return {"status": "uploaded", "files": saved_files}


@app.get("/api/knowledge-bases")
async def get_knowledge_bases():
    """列出前端可选择的知识库及规模信息。"""
    from app.rag.config import load_rag_config
    from app.rag.vectorstore import list_knowledge_bases

    config = load_rag_config()
    bases = list_knowledge_bases(config.data_path)
    return {
        "default_kb": config.default_kb,
        "knowledge_bases": [public_kb_summary(item, config.vectorstore) for item in bases],
    }


@app.post("/api/knowledge-bases/ingest")
async def ingest_knowledge_base(
    files: List[UploadFile] = File(...),
    kb_name: str = Form(...),
    document_key: str = Form(""),
    revision_id: str = Form(""),
    effective_from: str = Form(""),
):
    """上传文档并直接入库；同一 document_key 的新版本会替换旧版本。"""
    try:
        safe_kb = normalize_kb_name(kb_name)
        if not files:
            raise ValueError("请至少选择一个文件")
        if len(files) > 1 and any((document_key, revision_id, effective_from)):
            raise ValueError("文档身份和版本字段只适用于单文件上传")
        safe_names = [safe_upload_name(file.filename or "") for file in files]
        if len(set(safe_names)) != len(safe_names):
            raise ValueError("一次上传中不能包含同名文件")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    batch_dir = kb_upload_dir / f"{safe_kb}_{uuid.uuid4().hex}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    try:
        for upload, safe_name in zip(files, safe_names):
            target = batch_dir / safe_name
            written = 0
            with target.open("wb") as buffer:
                while chunk := await upload.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_KB_FILE_BYTES:
                        raise HTTPException(status_code=413, detail=f"{safe_name} 超过 50MB 限制")
                    buffer.write(chunk)

        async with _kb_ingest_locks[safe_kb]:
            def run_ingest():
                from app.rag.pipeline import clear_pipeline_cache, get_pipeline

                pipeline = get_pipeline(kb=safe_kb, create=True)
                source = batch_dir / safe_names[0] if len(safe_names) == 1 else batch_dir
                report = pipeline.ingest(
                    source,
                    document_key=document_key.strip() or None,
                    revision_id=revision_id.strip() or None,
                    effective_from=effective_from.strip() or None,
                )
                clear_pipeline_cache()
                return report.to_dict()

            report = await asyncio.to_thread(run_ingest)
        return {"status": "indexed", "report": report}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("知识库入库失败 kb=%s", safe_kb)
        raise HTTPException(status_code=500, detail=f"知识库入库失败：{exc}") from exc
    finally:
        for upload in files:
            await upload.close()
        shutil.rmtree(batch_dir, ignore_errors=True)


@app.get("/api/download")
async def download_file(path: str):
    """文件下载接口:严格限制在 output 目录内,防路径穿越。"""
    try:
        abs_path = Path(path).resolve()
        output_abs = output_dir.resolve()
        if not abs_path.is_relative_to(output_abs):
            return {"error": "拒绝访问: 只能下载输出目录下的文件"}
    except Exception:
        return {"error": "无效的路径参数"}

    if not abs_path.exists():
        return {"error": "文件不存在"}

    return FileResponse(abs_path, filename=abs_path.name)


@app.get("/api/files")
async def list_files(path: str):
    """文件列表查询接口:返回 output 目录下的文件元数据。"""
    try:
        abs_path = Path(path).resolve()
        output_abs = output_dir.resolve()
        if not abs_path.is_relative_to(output_abs):
            return {"error": "拒绝访问: 只能访问输出目录下的文件"}
    except Exception as e:
        return {"error": f"路径无效: {e}"}

    if not abs_path.exists():
        return {"error": "目录不存在"}

    files = []
    try:
        for file_path in abs_path.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append(
                    {
                        "name": file_path.name,
                        "type": "file",
                        "path": str(file_path),
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    }
                )
    except Exception as e:
        return {"error": str(e)}

    files.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return {"files": files}


# ─── 历史会话 API ──────────────────────────────────────────────


@app.get("/api/sessions")
async def get_sessions():
    """历史会话列表接口。"""
    try:
        sessions = list_sessions(limit=50)
        return {"sessions": sessions}
    except Exception as e:
        logger.error("获取会话列表失败: %s", e)
        return {"sessions": [], "error": str(e)}


@app.get("/api/sessions/{thread_id}")
async def get_session_detail(thread_id: str):
    """会话详情接口:返回元数据与所有历史消息。"""
    try:
        session = get_session(thread_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")
        messages = get_session_messages(thread_id)
        session_dir = output_dir / f"session_{thread_id}"
        return {
            "session": session,
            "messages": messages,
            "session_path": str(session_dir) if session_dir.exists() else "",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取会话详情失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{thread_id}")
async def remove_session(thread_id: str):
    """删除会话接口:若在运行则先取消,再删除会话及其消息。"""
    try:
        task = active_tasks.get(thread_id)
        if task and not task.done():
            monitor.report_task_cancelled()
            task.cancel()
            _forget_task(thread_id, task)

        deleted = delete_session(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="会话不存在或已删除")
        return {"status": "deleted", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("删除会话失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── 健康检查与指标 ──────────────────────────────────────────


@app.get("/api/health")
async def health():
    """健康检查:返回服务状态、运行时长与指标快照。"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "uptime_seconds": round(time.time() - _started_at, 2),
        "metrics": metrics.snapshot(),
    }


@app.get("/api/metrics")
async def get_metrics():
    """Prometheus 文本格式指标。"""
    from fastapi.responses import Response

    return Response(content=metrics.render_prometheus(), media_type="text/plain; charset=utf-8")


# ─── WebSocket ────────────────────────────────────────────────


@app.websocket("/ws/{thread_id}")
async def websocket_endpoint(websocket: WebSocket, thread_id: str):
    """WebSocket 实时通讯核心接口,按 thread_id 定向推送 monitor 事件。"""
    logger.info("建立 WebSocket 连接: %s", thread_id)
    await manager.connect(websocket, thread_id)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "pong", "message": f"服务端已收到: {data}"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, thread_id)
        logger.info("WebSocket 已断开: %s", thread_id)
    except Exception as e:
        logger.warning("WebSocket 连接异常: %s", e)
        manager.disconnect(websocket, thread_id)


if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
