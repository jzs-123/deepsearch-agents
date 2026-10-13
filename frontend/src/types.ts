export type ConnectionState = "connecting" | "connected" | "reconnecting" | "closed";

export type MonitorEventName =
  | "session_created"
  | "tool_start"
  | "assistant_call"
  | "task_result"
  | "task_cancelled"
  | "error"
  | string;

export interface MonitorMessage {
  type: "monitor_event";
  event: MonitorEventName;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface RagSource {
  rank: number;
  source: string;
  source_path: string;
  chunk_index: number;
  heading: string;
  score: number;
  dense_score: number | null;
  text: string;
}

export interface RagSourceTrace {
  kb: string;
  question: string;
  grounded: boolean;
  gate_score: number;
  threshold: number;
  sources: RagSource[];
}

export interface PongMessage {
  type: "pong";
  message: string;
}

export type SocketMessage = MonitorMessage | PongMessage;

export interface TaskResponse {
  status: "started" | string;
  thread_id: string;
}

export interface CancelTaskResponse {
  status: "cancelled" | "cancelling" | string;
  thread_id: string;
  message?: string;
}

export interface UploadResponse {
  status: "uploaded" | string;
  files: string[];
}

export interface KnowledgeBaseSummary {
  name: string;
  chunk_count: number;
  embedding: string;
  updated_at: string;
  data_version: number;
  filter_fields: string[];
  backend: string;
  recommendation: string;
}

export interface KnowledgeBaseListResponse {
  default_kb: string;
  knowledge_bases: KnowledgeBaseSummary[];
}

export interface IngestReport {
  kb: string;
  files_total: number;
  files_indexed: number;
  files_skipped: number;
  chunks_added: number;
  chunks_replaced: number;
  total_chunks: number;
  elapsed: number;
  warnings: string[];
}

export interface KnowledgeBaseIngestResponse {
  status: "indexed" | string;
  report: IngestReport;
}

export interface OutputFile {
  name: string;
  type: "file" | string;
  path: string;
  size: number;
  mtime: number;
}

export interface FileListResponse {
  files?: OutputFile[];
  error?: string;
}

export interface UploadedItem {
  uid: string;
  name: string;
  size: number;
  raw: File;
}

// ─── 历史会话类型 ──────────────────────────────────────────

export interface SessionSummary {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface SessionMessage {
  role: string;
  content: string;
  events_json: string;
  created_at: string;
}

export interface SessionDetail {
  session: SessionSummary;
  messages: SessionMessage[];
  session_path?: string;
}

export interface SessionListResponse {
  sessions: SessionSummary[];
  error?: string;
}

export interface ChatTurn {
  id: string;
  content: string;
  events: MonitorMessage[];
  files: OutputFile[];
  isRunning: boolean;
  result: string;
  timestamp: string;
}
