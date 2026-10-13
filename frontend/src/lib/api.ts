import { API_BASE_URL } from "./config";
import type {
  CancelTaskResponse,
  FileListResponse,
  KnowledgeBaseIngestResponse,
  KnowledgeBaseListResponse,
  SessionDetail,
  SessionListResponse,
  TaskResponse,
  TaskSnapshot,
  MonitorMessage,
  UploadResponse,
} from "../types";

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, { ...init, credentials: "include" });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "detail" in payload
        ? String(payload.detail)
        : `HTTP ${response.status}`;
    throw new Error(message);
  }

  return payload as T;
}

export function fetchAuthStatus(): Promise<{ enabled: boolean; authenticated: boolean; user_id?: string }> {
  return requestJson(apiUrl("/api/auth/status"));
}

export function loginWithApiKey(apiKey: string): Promise<{ user_id: string; tenant_id: string }> {
  return requestJson(apiUrl("/api/auth/login"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
}

export async function startTask(query: string, threadId: string): Promise<TaskResponse> {
  return requestJson<TaskResponse>(apiUrl("/api/task"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query,
      thread_id: threadId
    })
  });
}

export async function cancelTask(threadId: string): Promise<CancelTaskResponse> {
  return requestJson<CancelTaskResponse>(apiUrl(`/api/task/${encodeURIComponent(threadId)}/cancel`), {
    method: "POST"
  });
}

export function fetchLatestTask(threadId: string): Promise<{ task: TaskSnapshot | null }> {
  return requestJson(apiUrl(`/api/sessions/${encodeURIComponent(threadId)}/task`));
}

export function fetchTaskEvents(taskId: string, after: number): Promise<{ events: MonitorMessage[] }> {
  return requestJson(apiUrl(`/api/tasks/${encodeURIComponent(taskId)}/events?after=${after}`));
}

export async function uploadSessionFiles(
  files: File[],
  threadId: string
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("thread_id", threadId);
  files.forEach((file) => formData.append("files", file));

  return requestJson<UploadResponse>(apiUrl("/api/upload"), {
    method: "POST",
    body: formData
  });
}

export async function listKnowledgeBases(): Promise<KnowledgeBaseListResponse> {
  return requestJson<KnowledgeBaseListResponse>(apiUrl("/api/knowledge-bases"));
}

export async function ingestKnowledgeBase(input: {
  kbName: string;
  files: File[];
  documentKey?: string;
  revisionId?: string;
  effectiveFrom?: string;
}): Promise<KnowledgeBaseIngestResponse> {
  const formData = new FormData();
  formData.append("kb_name", input.kbName);
  input.files.forEach((file) => formData.append("files", file));
  formData.append("document_key", input.documentKey || "");
  formData.append("revision_id", input.revisionId || "");
  formData.append("effective_from", input.effectiveFrom || "");
  return requestJson<KnowledgeBaseIngestResponse>(apiUrl("/api/knowledge-bases/ingest"), {
    method: "POST",
    body: formData,
  });
}

export async function listSessionFiles(path: string): Promise<FileListResponse> {
  const url = new URL(apiUrl("/api/files"));
  url.searchParams.set("path", path);
  return requestJson<FileListResponse>(url);
}

export function getDownloadUrl(path: string): string {
  const url = new URL(apiUrl("/api/download"));
  url.searchParams.set("path", path);
  return url.toString();
}

// ─── 历史会话 API ──────────────────────────────────────────

export async function fetchSessionList(): Promise<SessionListResponse> {
  return requestJson<SessionListResponse>(apiUrl("/api/sessions"));
}

export async function fetchSessionDetail(threadId: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(apiUrl(`/api/sessions/${encodeURIComponent(threadId)}`));
}

export async function deleteSessionApi(threadId: string): Promise<{ status: string; thread_id: string }> {
  return requestJson<{ status: string; thread_id: string }>(apiUrl(`/api/sessions/${encodeURIComponent(threadId)}`), {
    method: "DELETE"
  });
}
