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
from math import ceil
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
from app.agent.result import AgentResult, StructuredResultError
from langgraph.errors import GraphRecursionError
from app.api.access import claim_session, require_session, resolve_owned_output_path, task_key
from app.api.knowledge_bases import (
    normalize_kb_name,
    public_kb_summary,
    safe_upload_name,
)
from app.api.monitor import manager, monitor
from app.api.uploads import save_session_uploads
from app.api import task_store
from app.security.knowledge_bases import knowledge_config, visible_knowledge_bases
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
    get_session_messages,
    list_sessions,
)

logger = get_logger("api.server")

_started_at = time.time()
_shutting_down = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """服务生命周期入口:绑定事件循环、初始化日志与并发信号量。"""
    configure_logging()
    global _shutting_down
    _shutting_down = False
    loop = asyncio.get_running_loop()
    manager.set_loop(loop)
    task_store.interrupt_unfinished()
    logger.info("服务启动,绑定事件循环: %s", id(loop))
    try:
        yield
    finally:
        _shutting_down = True
        running = list(active_tasks.values())
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        task_store.interrupt_unfinished()
        manager.loop = None
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
_session_upload_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_task_submit_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


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
                headers={"Retry-After": str(max(1, ceil(60 - (now - window[0]))))},
            )(scope, receive, send)
            return
        window.append(now)
        await self.app(scope, receive, send)


# CORS 包住限流响应并先处理预检，Request-ID 同时覆盖被限流的请求。
app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After", "X-Request-ID"],
)


# ─── 任务并发治理 ────────────────────────────────────────────

# 保存 thread_id -> 后台 Agent 任务,用于同一会话任务替换和主动取消
active_tasks: dict[tuple[str, str, str], asyncio.Task] = {}
_task_semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)


class TaskRequest(BaseModel):
    """前端启动任务时提交的请求体。"""

    query: str
    thread_id: str = None


@app.get("/api/auth/status")
async def auth_status(http_request: Request):
    try:
        identity = resolve_identity(http_request.headers)
    except AuthError:
        return {"enabled": settings.auth_enabled, "authenticated": False}
    return {"enabled": settings.auth_enabled, "authenticated": True, "user_id": identity.user_id}


class LoginRequest(BaseModel):
    api_key: str


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    identity = _resolve_identity({"authorization": f"Bearer {request.api_key}"})
    response = JSONResponse({"user_id": identity.user_id, "tenant_id": identity.tenant_id})
    response.set_cookie(
        "deepsearch_auth", request.api_key, httponly=True,
        secure=settings.app_env == "prod", samesite="strict", max_age=28800,
    )
    return response


@app.post("/api/auth/logout")
async def logout():
    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie("deepsearch_auth")
    return response


def _forget_task(key: tuple[str, str, str], task: asyncio.Task) -> None:
    """清理已结束任务的登记关系,避免误删同 thread_id 下刚启动的新任务。"""
    if active_tasks.get(key) is task:
        active_tasks.pop(key, None)


def _resolve_identity(headers) -> Identity:
    """解析身份,鉴权失败统一转 401。"""
    try:
        return resolve_identity(headers)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


def _report_status(task_id, thread_id, status, **fields):
    task_store.transition(task_id, status, **fields)
    monitor._emit("task_status", fields.get("error_message") or f"任务状态: {status}",
                  {"status": status, **fields}, thread_id=thread_id)


async def _run_task_safely(query: str, thread_id: str, identity: Identity, task_id: str) -> None:
    """落盘后才报告终态；信号量由完成回调释放，覆盖启动前取消。"""
    token = task_store.current_task_id.set(task_id)
    try:
        _report_status(task_id, thread_id, "running")
        result = await run_deep_agent(query, thread_id, identity=identity)
        snapshot = task_store.get_task(task_id, identity)
        if snapshot and snapshot["status"] == "cancelling":
            _report_status(task_id, thread_id, "cancelled")
            monitor.report_task_cancelled(thread_id)
        else:
            output = result if isinstance(result, AgentResult) else AgentResult(status="success", answer=result)
            if output.status == "failed":
                _report_status(task_id, thread_id, "failed", result=output.answer,
                               output=output.model_dump(), error_code="AGENT_REPORTED_FAILURE",
                               error_message=output.answer)
            else:
                _report_status(task_id, thread_id, "succeeded", result=output.answer, output=output.model_dump())
    except asyncio.CancelledError:
        if _shutting_down:
            _report_status(task_id, thread_id, "interrupted", error_code="SERVICE_SHUTDOWN",
                           error_message="服务停止，任务已中断，请重新提交")
        else:
            _report_status(task_id, thread_id, "cancelled")
            monitor.report_task_cancelled(thread_id)
        raise
    except Exception as exc:
        logger.exception("任务执行失败 task_id=%s", task_id)
        timed_out = isinstance(exc, TimeoutError)
        code = "EXECUTION_TIMEOUT" if timed_out else "AGENT_EXECUTION_FAILED"
        message = "任务执行超时，请稍后重试" if timed_out else "任务执行失败，请稍后重试"
        if isinstance(exc, StructuredResultError):
            code, message = "INVALID_AGENT_RESULT", "回答格式校验失败，请重新提交任务"
        elif isinstance(exc, GraphRecursionError):
            code, message = "AGENT_STEP_LIMIT", "任务达到执行步数上限，请缩小问题范围"
        _report_status(task_id, thread_id, "failed",
                       error_code=code, error_message=message)
    finally:
        task_store.current_task_id.reset(token)


def _task_done(key: tuple[str, str, str], task: asyncio.Task, task_id: str, identity: Identity) -> None:
    _task_semaphore.release()
    _forget_task(key, task)
    if not task.cancelled():
        task.exception()
    snapshot = task_store.get_task(task_id, identity)
    if snapshot and snapshot["status"] not in task_store.TERMINAL:
        # 协程启动前被取消时，包装层的 finally 不会执行。
        token = task_store.current_task_id.set(task_id)
        try:
            _report_status(task_id, key[2], "interrupted" if _shutting_down else "cancelled")
        finally:
            task_store.current_task_id.reset(token)


@app.post("/api/task")
async def run_task(request: TaskRequest, http_request: Request):
    """
    启动一次 DeepAgents 后台任务。

    HTTP 请求只负责创建后台协程并立即返回,后续执行轨迹、子智能体调用和最终
    答案都会由 monitor 通过 `/ws/{thread_id}` 推送给同一会话的前端。
    """
    identity = _resolve_identity(http_request.headers)
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="请输入研搜任务")
    thread_id = claim_session(request.thread_id or str(uuid.uuid4()), identity)
    key = task_key(thread_id, identity)

    async with _task_submit_locks[thread_id]:
        old_task = active_tasks.get(key)
        if old_task and not old_task.done():
            old_task.cancel()
            await asyncio.gather(old_task, return_exceptions=True)
        if _task_semaphore.locked():
            raise HTTPException(status_code=429, detail="并发任务数已达上限,请稍后重试")
        await _task_semaphore.acquire()
        try:
            task_id = task_store.create_task(thread_id, identity, request.query)
            task = asyncio.create_task(_run_task_safely(request.query, thread_id, identity, task_id))
            active_tasks[key] = task
            task.add_done_callback(lambda finished_task: _task_done(key, finished_task, task_id, identity))
        except BaseException:
            _task_semaphore.release()
            raise
    logger.info("任务已启动 thread_id=%s user=%s", thread_id, identity.user_id)

    return {"status": "started", "thread_id": thread_id, "task_id": task_id}


@app.post("/api/task/{thread_id}/cancel")
async def cancel_task(thread_id: str, http_request: Request):
    """请求取消；协程真正退出后由包装层报告 cancelled。"""
    identity = _resolve_identity(http_request.headers)
    require_session(thread_id, identity)
    key = task_key(thread_id, identity)
    task = active_tasks.get(key)
    if not task or task.done():
        active_tasks.pop(key, None)
        raise HTTPException(status_code=404, detail="任务不存在或已结束")

    snapshot = task_store.latest_task(thread_id, identity)
    if not snapshot or snapshot["status"] in task_store.TERMINAL:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    token = task_store.current_task_id.set(snapshot["task_id"])
    try:
        if snapshot["status"] != "cancelling":
            _report_status(snapshot["task_id"], thread_id, "cancelling")
    finally:
        task_store.current_task_id.reset(token)
    task.cancel()

    return {"status": "cancelling", "thread_id": thread_id, "task_id": snapshot["task_id"]}


@app.get("/api/tasks/{task_id}")
async def task_snapshot(task_id: str, http_request: Request):
    identity = _resolve_identity(http_request.headers)
    snapshot = task_store.get_task(task_id, identity)
    if not snapshot:
        raise HTTPException(status_code=404, detail="任务不存在或无权访问")
    return snapshot


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, http_request: Request, after: int = 0):
    identity = _resolve_identity(http_request.headers)
    if not task_store.get_task(task_id, identity):
        raise HTTPException(status_code=404, detail="任务不存在或无权访问")
    return {"events": task_store.get_events(task_id, max(0, after))}


@app.get("/api/sessions/{thread_id}/task")
async def session_task(thread_id: str, http_request: Request):
    identity = _resolve_identity(http_request.headers)
    require_session(thread_id, identity)
    return {"task": task_store.latest_task(thread_id, identity)}


@app.post("/api/upload")
async def upload_files(
    http_request: Request,
    files: List[UploadFile] = File(...),
    thread_id: str = Form(...),
):
    """文件上传接口:校验后按会话隔离、限量保存到 updated 目录。"""
    identity = _resolve_identity(http_request.headers)
    thread_id = claim_session(thread_id, identity)
    try:
        async with _session_upload_locks[thread_id]:
            saved_files = await save_session_uploads(files, thread_id, updated_dir)
        return {"status": "uploaded", "files": saved_files}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("会话附件上传失败 thread_id=%s", thread_id)
        raise HTTPException(status_code=500, detail="文件上传失败") from exc


@app.get("/api/knowledge-bases")
async def get_knowledge_bases(http_request: Request):
    """列出前端可选择的知识库及规模信息。"""
    identity = _resolve_identity(http_request.headers)
    config = knowledge_config(identity)
    bases = visible_knowledge_bases(identity)
    return {
        "default_kb": config.default_kb,
        "knowledge_bases": [public_kb_summary(item, config.vectorstore) for item in bases],
    }


@app.post("/api/knowledge-bases/ingest")
async def ingest_knowledge_base(
    http_request: Request,
    files: List[UploadFile] = File(...),
    kb_name: str = Form(...),
    document_key: str = Form(""),
    revision_id: str = Form(""),
    effective_from: str = Form(""),
):
    """上传文档并直接入库；同一 document_key 的新版本会替换旧版本。"""
    identity = _resolve_identity(http_request.headers)
    config = knowledge_config(identity)
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

        async with _kb_ingest_locks[str(config.kb_dir(safe_kb))]:
            def run_ingest():
                from app.rag.pipeline import clear_pipeline_cache, get_pipeline

                pipeline = get_pipeline(kb=safe_kb, config=config, create=True)
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
async def download_file(path: str, http_request: Request):
    """文件下载接口:严格限制在 output 目录内,防路径穿越。"""
    identity = _resolve_identity(http_request.headers)
    abs_path = resolve_owned_output_path(path, output_dir, identity)

    if not abs_path.exists():
        return {"error": "文件不存在"}

    return FileResponse(abs_path, filename=abs_path.name)


@app.get("/api/files")
async def list_files(path: str, http_request: Request):
    """文件列表查询接口:返回 output 目录下的文件元数据。"""
    identity = _resolve_identity(http_request.headers)
    abs_path = resolve_owned_output_path(path, output_dir, identity)

    if not abs_path.exists():
        return {"error": "目录不存在"}

    files = []
    try:
        for file_path in abs_path.rglob("*"):
            if file_path.relative_to(output_dir).parts[1:2] == ("skills",):
                continue
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
async def get_sessions(http_request: Request):
    """历史会话列表接口。"""
    identity = _resolve_identity(http_request.headers)
    try:
        sessions = list_sessions(50, identity.user_id, identity.tenant_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error("获取会话列表失败: %s", e)
        return {"sessions": [], "error": str(e)}


@app.get("/api/sessions/{thread_id}")
async def get_session_detail(thread_id: str, http_request: Request):
    """会话详情接口:返回元数据与所有历史消息。"""
    identity = _resolve_identity(http_request.headers)
    try:
        session = require_session(thread_id, identity)
        messages = get_session_messages(thread_id, identity.user_id, identity.tenant_id)
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
async def remove_session(thread_id: str, http_request: Request):
    """删除会话接口:若在运行则先取消,再删除会话及其消息。"""
    identity = _resolve_identity(http_request.headers)
    require_session(thread_id, identity)
    key = task_key(thread_id, identity)
    try:
        task = active_tasks.get(key)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            _forget_task(key, task)

        deleted = delete_session(thread_id, identity.user_id, identity.tenant_id)
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
    try:
        identity = resolve_identity(websocket.headers)
        thread_id = claim_session(thread_id, identity)
    except (AuthError, HTTPException):
        await websocket.close(code=4401)
        return
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
