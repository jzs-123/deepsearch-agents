import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  cancelTask,
  deleteSessionApi,
  fetchSessionDetail,
  fetchSessionList,
  fetchLatestTask,
  fetchTaskEvents,
  listSessionFiles,
  startTask,
  uploadSessionFiles
} from "../lib/api";
import { WS_BASE_URL } from "../lib/config";
import { createThreadId, getStoredThreadId, storeThreadId } from "../lib/thread";
import { mergeTaskEvents, taskDisplay, resultAnswer } from "../lib/taskState";
import type {
  ConnectionState,
  MonitorMessage,
  OutputFile,
  SessionMessage,
  SessionSummary,
  SocketMessage,
  UploadedItem
} from "../types";

const MAX_EVENTS = 120;

function extractString(data: Record<string, unknown>, key: string): string | null {
  const value = data[key];
  return typeof value === "string" ? value : null;
}

export function useDeepAgentSession() {
  const initialThreadIdRef = useRef(getStoredThreadId());
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const heartbeatTimerRef = useRef<number | undefined>(undefined);
  const uploadedNameSetRef = useRef<Set<string>>(new Set());
  const taskIdRef = useRef<string | null>(null);
  const terminalTaskIdRef = useRef<string | null>(null);
  const replayCursorRef = useRef(0);
  const requestEpochRef = useRef(0);
  const [threadId, setThreadId] = useState(getStoredThreadId);
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const [events, setEvents] = useState<MonitorMessage[]>([]);
  const [files, setFiles] = useState<OutputFile[]>([]);
  const [sessionPath, setSessionPath] = useState("");
  const [result, setResult] = useState("");
  const [lastError, setLastError] = useState("");
  const [lastPongAt, setLastPongAt] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedItems, setUploadedItems] = useState<UploadedItem[]>([]);
  const [sessionList, setSessionList] = useState<SessionSummary[]>([]);
  const [historyMessages, setHistoryMessages] = useState<SessionMessage[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const clearSocketTimers = useCallback(() => {
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = undefined;
    }
    if (heartbeatTimerRef.current) {
      window.clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = undefined;
    }
  }, []);

  const resetSession = useCallback(() => {
    taskIdRef.current = null;
    replayCursorRef.current = 0;
    requestEpochRef.current += 1;
    const nextThreadId = createThreadId();
    storeThreadId(nextThreadId);
    setThreadId(nextThreadId);
    setEvents([]);
    setFiles([]);
    setSessionPath("");
    setResult("");
    setLastError("");
    setUploadedItems([]);
    setHistoryMessages([]);
    uploadedNameSetRef.current.clear();
    setIsRunning(false);
    setIsCancelling(false);
  }, []);

  const refreshFiles = useCallback(async () => {
    if (!sessionPath) {
      return;
    }

    const response = await listSessionFiles(sessionPath);
    if (response.error) {
      throw new Error(response.error);
    }
    setFiles(response.files || []);
  }, [sessionPath]);

  useEffect(() => {
    let disposed = false;

    function connect() {
      clearSocketTimers();
      const hadSocket = Boolean(socketRef.current);
      socketRef.current?.close();
      setConnectionState(hadSocket ? "reconnecting" : "connecting");

      const socket = new WebSocket(`${WS_BASE_URL}/ws/${encodeURIComponent(threadId)}`);
      socketRef.current = socket;

      socket.onopen = () => {
        if (disposed) {
          return;
        }
        setConnectionState("connected");
        setLastError("");
        heartbeatTimerRef.current = window.setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send("ping");
          }
        }, 25000);
      };

      socket.onmessage = (event) => {
        if (socketRef.current !== socket) {
          return;
        }
        try {
          const payload = JSON.parse(event.data) as SocketMessage;
          if (payload.type === "pong") {
            setLastPongAt(new Date().toISOString());
            return;
          }

          if (payload.type !== "monitor_event") {
            return;
          }

          if (payload.task_id && payload.task_id !== taskIdRef.current) {
            return; // 状态查询确认当前任务后，再补发其事件，忽略旧任务的迟到消息。
          }
          if (payload.task_id) {
            setEvents((previous) => mergeTaskEvents(previous, [payload], payload.task_id!));
          } else {
            setEvents((previous) => [...previous, payload].slice(-MAX_EVENTS));
          }

          if (payload.event === "session_created") {
            const path = extractString(payload.data, "path");
            if (path) {
              setSessionPath(path);
            }
          }

          if (payload.event === "task_result") {
            const finalResult = resultAnswer(payload.data);
            setResult(finalResult || payload.message);
            void fetchSessionList().then((response) => {
              if (!response.error) {
                setSessionList(response.sessions || []);
              }
            }).catch(() => undefined);
          }

          if (payload.event === "task_cancelled") {
            setResult((previous) => previous || payload.message);
            setIsRunning(false);
            setIsCancelling(false);
          }

          if (payload.event === "error") {
            setLastError(payload.message);
          }

          if (payload.event === "task_status") {
            const status = extractString(payload.data, "status");
            const running = ["queued", "running", "cancelling"].includes(status || "");
            if (!running && payload.task_id) terminalTaskIdRef.current = payload.task_id;
            if (running && terminalTaskIdRef.current === payload.task_id) return;
            setIsRunning(running);
            setIsCancelling(status === "cancelling");
          }
        } catch (error) {
          setLastError(error instanceof Error ? error.message : "WebSocket 消息解析失败");
        }
      };

      socket.onerror = () => {
        if (!disposed && socketRef.current === socket) {
          setLastError("WebSocket 连接异常，请确认后端服务已启动");
        }
      };

      socket.onclose = () => {
        if (socketRef.current !== socket) {
          return;
        }
        clearSocketTimers();
        if (disposed) {
          setConnectionState("closed");
          return;
        }
        setConnectionState("reconnecting");
        reconnectTimerRef.current = window.setTimeout(connect, 2000);
      };
    }

    connect();

    return () => {
      disposed = true;
      clearSocketTimers();
      socketRef.current?.close();
    };
  }, [clearSocketTimers, threadId]);

  // 快照决定状态，持久事件补齐断线窗口；游标只由有序 HTTP 补发推进。
  useEffect(() => {
    let disposed = false;
    let timer: number;
    async function recover() {
      const epoch = requestEpochRef.current;
      let delay = connectionState === "connected" ? 15000 : 5000;
      try {
        const { task } = await fetchLatestTask(threadId);
        if (disposed || epoch !== requestEpochRef.current || !task) return;
        if (taskIdRef.current !== task.task_id) {
          taskIdRef.current = task.task_id;
          replayCursorRef.current = 0;
          setEvents([]);
        }
        const display = taskDisplay(task);
        if (!display.isRunning) delay = 30000;
        if (!display.isRunning) terminalTaskIdRef.current = task.task_id;
        if (!display.isRunning || terminalTaskIdRef.current !== task.task_id) {
          setIsRunning(display.isRunning);
          setIsCancelling(display.isCancelling);
        }
        if (display.result) setResult(display.result);
        if (display.error) setLastError(display.error);
        let page: { events: MonitorMessage[] };
        do {
          page = await fetchTaskEvents(task.task_id, replayCursorRef.current);
          if (disposed || epoch !== requestEpochRef.current || taskIdRef.current !== task.task_id) return;
          const incoming = page.events;
          setEvents((previous) => mergeTaskEvents(previous, incoming, task.task_id));
          for (const event of page.events) {
            replayCursorRef.current = Math.max(replayCursorRef.current, event.seq || 0);
            if (event.event === "session_created") {
              const path = extractString(event.data, "path");
              if (path) setSessionPath(path);
            }
            if (event.event === "task_result") {
              const answer = resultAnswer(event.data);
              if (answer) setResult(answer);
            }
            if (event.event === "task_status" && terminalTaskIdRef.current !== task.task_id) {
              const status = extractString(event.data, "status") || "";
              const running = ["queued", "running", "cancelling"].includes(status);
              if (!running) terminalTaskIdRef.current = task.task_id;
              setIsRunning(running);
              setIsCancelling(status === "cancelling");
              const error = extractString(event.data, "error_message");
              if (error) setLastError(error);
              if (status === "cancelled") setResult("任务已取消");
            }
          }
        } while (page.events.length === 500);
      } catch (error) {
        if (error instanceof ApiError && error.status === 429) delay = Math.max(delay, error.retryAfterMs);
        // 新会话尚未创建或连接暂时不可用，下次补发继续。
      } finally {
        if (!disposed) timer = window.setTimeout(recover, delay);
      }
    }
    void recover();
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [threadId, connectionState]);

  useEffect(() => {
    if (!sessionPath) {
      return;
    }

    let disposed = false;
    let timer: number;
    let fileError = "";
    async function pollFiles() {
      let delay = 5000;
      let limited = false;
      try {
        await refreshFiles();
        if (!disposed && fileError) {
          const recoveredError = fileError;
          setLastError((previous) => previous === recoveredError ? "" : previous);
          fileError = "";
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 429) {
          limited = true;
          delay = Math.max(delay, error.retryAfterMs);
        }
        fileError = error instanceof Error ? error.message : "文件列表刷新失败";
        if (!disposed) setLastError(fileError);
      } finally {
        if (!disposed && (isRunning || limited)) timer = window.setTimeout(pollFiles, delay);
      }
    }
    void pollFiles();
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [isRunning, refreshFiles, sessionPath]);

  const submitTask = useCallback(
    async (query: string) => {
      const cleanQuery = query.trim();
      if (!cleanQuery) {
        throw new Error("请输入研搜任务");
      }

      setIsRunning(true);
      requestEpochRef.current += 1;
      taskIdRef.current = null;
      replayCursorRef.current = 0;
      setIsCancelling(false);
      setEvents([]);
      setResult("");
      setLastError("");
      try {
        const response = await startTask(cleanQuery, threadId);
        taskIdRef.current = response.task_id;
        if (response.thread_id && response.thread_id !== threadId) {
          storeThreadId(response.thread_id);
          setThreadId(response.thread_id);
        }
        return response;
      } catch (error) {
        setIsRunning(false);
        setIsCancelling(false);
        throw error;
      }
    },
    [threadId]
  );

  const cancelCurrentTask = useCallback(async () => {
    if (!isRunning) {
      throw new Error("当前没有正在执行的任务");
    }

    setIsCancelling(true);
    setLastError("");
    try {
      const response = await cancelTask(threadId);
      if (response.status === "cancelled") {
        setIsRunning(false);
        setIsCancelling(false);
        setResult((previous) => previous || "任务已取消");
      }
      return response;
    } catch (error) {
      setIsCancelling(false);
      throw error;
    }
  }, [isRunning, threadId]);

  const uploadFiles = useCallback(
    async (items: UploadedItem[]) => {
      if (items.length === 0) {
        throw new Error("请选择要上传的文件");
      }

      const nextItems = items.filter((item) => !uploadedNameSetRef.current.has(item.name));

      if (nextItems.length === 0) {
        return {
          status: "uploaded",
          files: Array.from(uploadedNameSetRef.current)
        };
      }

      setIsUploading(true);
      setLastError("");
      try {
        const response = await uploadSessionFiles(
          nextItems.map((item) => item.raw),
          threadId
        );
        setUploadedItems((previous) => {
          const names = new Set(previous.map((item) => item.name));
          const next = [...previous];
          nextItems.forEach((item) => {
            if (!names.has(item.name)) {
              names.add(item.name);
              uploadedNameSetRef.current.add(item.name);
              next.push(item);
            }
          });
          return next;
        });
        return response;
      } finally {
        setIsUploading(false);
      }
    },
    [threadId]
  );

  // ─── 历史会话管理 ──────────────────────────────────────

  const loadSessionList = useCallback(async (): Promise<SessionSummary[]> => {
    try {
      const response = await fetchSessionList();
      if (!response.error) {
        const sessions = response.sessions || [];
        setSessionList(sessions);
        return sessions;
      }
    } catch {
      // 静默处理，历史列表加载失败不影响主流程
    }
    return [];
  }, []);

  const loadSessionHistory = useCallback(async (targetThreadId: string) => {
    requestEpochRef.current += 1;
    taskIdRef.current = null;
    replayCursorRef.current = 0;
    setIsLoadingHistory(true);
    try {
      const response = await fetchSessionDetail(targetThreadId);
      storeThreadId(targetThreadId);
      setThreadId(targetThreadId);
      setHistoryMessages(response.messages);
      setEvents([]);
      setFiles([]);
      setSessionPath(response.session_path || "");
      setResult("");
      setLastError("");
      setUploadedItems([]);
      uploadedNameSetRef.current.clear();
      setIsRunning(false);
      setIsCancelling(false);
      return response;
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  const deleteSession = useCallback(async (targetThreadId: string) => {
    await deleteSessionApi(targetThreadId);
    // 如果删除的是当前会话，新建一个空会话
    if (targetThreadId === threadId) {
      const nextThreadId = createThreadId();
      storeThreadId(nextThreadId);
      setThreadId(nextThreadId);
      setEvents([]);
      setFiles([]);
      setSessionPath("");
      setResult("");
      setLastError("");
      setHistoryMessages([]);
    }
    // 刷新列表
    await loadSessionList();
  }, [loadSessionList, threadId]);

  // 首次打开或刷新页面时，恢复 localStorage 中当前会话的完整消息。
  useEffect(() => {
    let disposed = false;
    const initialThreadId = initialThreadIdRef.current;
    void loadSessionList().then((sessions) => {
      if (!disposed && sessions.some((item) => item.thread_id === initialThreadId)) {
        void loadSessionHistory(initialThreadId);
      }
    });
    return () => {
      disposed = true;
    };
  }, [loadSessionHistory, loadSessionList]);

  const stats = useMemo(() => {
    const toolEvents = events.filter((event) => event.event === "tool_start").length;
    const assistantEvents = events.filter((event) => event.event === "assistant_call").length;
    const errorEvents = events.filter((event) => event.event === "error").length;

    return {
      toolEvents,
      assistantEvents,
      errorEvents,
      fileCount: files.length
    };
  }, [events, files.length]);

  return {
    connectionState,
    deleteSession,
    events,
    files,
    historyMessages,
    isCancelling,
    isLoadingHistory,
    isRunning,
    isUploading,
    lastError,
    lastPongAt,
    loadSessionHistory,
    loadSessionList,
    refreshFiles,
    resetSession,
    result,
    sessionList,
    sessionPath,
    stats,
    cancelCurrentTask,
    submitTask,
    threadId,
    uploadFiles,
    uploadedItems
  };
}
