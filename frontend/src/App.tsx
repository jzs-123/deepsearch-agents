import {
  ApiOutlined,
  BranchesOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  FileSearchOutlined,
  HistoryOutlined,
  PlusOutlined,
  ToolOutlined
} from "@ant-design/icons";
import { Alert, App as AntApp, Button } from "antd";
import { useEffect, useRef, useState } from "react";
import { ChatComposer } from "./components/ChatComposer";
import { ConversationThread } from "./components/ConversationThread";
import { KnowledgeBaseManager } from "./components/KnowledgeBaseManager";
import { API_BASE_URL, WS_BASE_URL } from "./lib/config";
import { restoreChatTurns } from "./lib/history";
import { useDeepAgentSession } from "./hooks/useDeepAgentSession";
import type { ChatTurn, ConnectionState, UploadedItem } from "./types";

function connectionLabel(state: ConnectionState): string {
  const labels: Record<ConnectionState, string> = {
    connecting: "连接中",
    connected: "已连接",
    reconnecting: "重连中",
    closed: "已关闭"
  };
  return labels[state];
}

function createTurn(content: string): ChatTurn {
  return {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
    content,
    events: [],
    files: [],
    isRunning: true,
    result: "",
    timestamp: new Date().toISOString()
  };
}

export default function App() {
  const { message } = AntApp.useApp();
  const [query, setQuery] = useState("");
  const [stagedItems, setStagedItems] = useState<UploadedItem[]>([]);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [knowledgeBaseOpen, setKnowledgeBaseOpen] = useState(false);
  const activeTurnIdRef = useRef<string | null>(null);
  const streamRef = useRef<HTMLElement | null>(null);
  const session = useDeepAgentSession();

  useEffect(() => {
    const activeTurnId = activeTurnIdRef.current;
    if (!activeTurnId) {
      if (session.historyMessages.length === 0) {
        return;
      }
      setTurns((previous) => previous.map((turn, index) =>
        index === previous.length - 1 ? { ...turn, files: session.files } : turn));
      return;
    }
    setTurns((previous) => {
      return previous.map((turn) => turn.id === activeTurnId
        ? {
            ...turn,
            events: session.events,
            files: session.files,
            isRunning: session.isRunning,
            result: session.result
          }
        : turn);
    });
  }, [session.events, session.files, session.historyMessages, session.isRunning, session.result]);

  useEffect(() => {
    activeTurnIdRef.current = null;
    setTurns(restoreChatTurns(session.historyMessages));
  }, [session.historyMessages]);

  // 移除自动滚动到底部行为，保留用户当前滚动位置

  async function handleSubmit() {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      message.warning("请输入研搜任务");
      return;
    }

    const nextTurn = createTurn(cleanQuery);
    activeTurnIdRef.current = nextTurn.id;
    setTurns((previous) => [...previous, nextTurn]);
    setQuery("");

    try {
      await session.submitTask(cleanQuery);
      session.loadSessionList(); // 刷新列表，让新会话立即出现
      message.success("任务已启动，执行过程会显示在对话中");
    } catch (error) {
      setTurns((previous) =>
        previous.map((turn) =>
          turn.id === nextTurn.id
            ? {
                ...turn,
                isRunning: false,
                result: error instanceof Error ? error.message : "任务启动失败"
              }
            : turn
        )
      );
      message.error(error instanceof Error ? error.message : "任务启动失败");
    }
  }

  async function handleCancel() {
    try {
      const response = await session.cancelCurrentTask();
      message.info(response.status === "cancelling" ? "取消请求已发送，正在等待当前调用结束" : "任务已取消");
    } catch (error) {
      message.error(error instanceof Error ? error.message : "取消任务失败");
    }
  }

  async function handleUpload(items: UploadedItem[]) {
    try {
      const response = await session.uploadFiles(items);
      setStagedItems([]);
      message.success(`已上传 ${response.files.length} 个文件`);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "上传失败");
    }
  }

  function handleNewSession() {
    activeTurnIdRef.current = null;
    session.resetSession();
    setTurns([]);
    setQuery("");
    setStagedItems([]);
  }

  async function handleLoadSession(threadId: string) {
    activeTurnIdRef.current = null;
    try {
      await session.loadSessionHistory(threadId);
    } catch (error) {
      message.error(error instanceof Error ? error.message : "历史会话加载失败");
    }
  }

  const online = session.connectionState === "connected";

  return (
    <div className="chat-app-shell min-h-dvh">
      <aside className="chat-sidebar" aria-label="会话信息">
        <div className="sidebar-brand">
          <span className="brand-mark" aria-hidden>
            <i />
            <i />
            <i />
          </span>
          <div>
            <span className="panel-kicker">DEEPSEARCH OS</span>
            <h1>深度研搜</h1>
            <p>多智能体研究工作台</p>
          </div>
        </div>

        <Button className="new-chat-button" block icon={<PlusOutlined />} onClick={handleNewSession}>
          新建研搜
        </Button>
        <Button
          block
          className="kb-manage-button"
          icon={<DatabaseOutlined />}
          onClick={() => setKnowledgeBaseOpen(true)}
        >
          上传知识库
        </Button>

        <div className="sidebar-thread">
          <span className="sidebar-label">ACTIVE THREAD</span>
          <strong className="thread-id" title={session.threadId}>
            {session.threadId.slice(0, 8)}
          </strong>
          <span className="thread-signal">实时上下文已连接</span>
        </div>

        <span className="sidebar-label">SYSTEM PULSE</span>
        <div className="sidebar-status-list">
          <div className={`sidebar-status ${online ? "sidebar-status--online" : "sidebar-status--warn"}`}>
            <ApiOutlined aria-hidden />
            <span>WebSocket</span>
            <strong>{connectionLabel(session.connectionState)}</strong>
          </div>
          <div className="sidebar-status">
            <BranchesOutlined aria-hidden />
            <span>助手调度</span>
            <strong>{session.stats.assistantEvents}</strong>
          </div>
          <div className="sidebar-status">
            <ToolOutlined aria-hidden />
            <span>工具调用</span>
            <strong>{session.stats.toolEvents}</strong>
          </div>
          <div className={session.stats.errorEvents > 0 ? "sidebar-status sidebar-status--error" : "sidebar-status"}>
            <CloseCircleOutlined aria-hidden />
            <span>异常</span>
            <strong>{session.stats.errorEvents}</strong>
          </div>
        </div>

        <div className="sidebar-section">
          <span className="sidebar-label">
            <HistoryOutlined aria-hidden /> RECENT THREADS
          </span>
          {session.sessionList.length === 0 ? (
            <p className="history-empty">
              暂无历史会话
            </p>
          ) : (
            <ul className="history-list">
              {session.sessionList.map((s) => (
                <li
                  key={s.thread_id}
                  className={s.thread_id === session.threadId ? "history-item history-item--active" : "history-item"}
                >
                  <button
                    className="history-entry"
                    onClick={() => {
                      if (s.thread_id !== session.threadId) {
                        void handleLoadSession(s.thread_id);
                      }
                    }}
                    type="button"
                  >
                    <span className="history-index" aria-hidden />
                    <span className="history-title">
                      {s.title || s.thread_id.slice(0, 8)}
                    </span>
                  </button>
                  <button
                    aria-label={`删除会话 ${s.title || s.thread_id.slice(0, 8)}`}
                    className="history-delete"
                    onClick={() => {
                      void session.deleteSession(s.thread_id);
                    }}
                    type="button"
                  >
                    <DeleteOutlined aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="sidebar-section">
          <span className="sidebar-label">AGENT NETWORK</span>
          <ul className="agent-mini-list">
            <li>
              <CloudServerOutlined aria-hidden />
              网络搜索助手
            </li>
            <li>
              <DatabaseOutlined aria-hidden />
              数据库查询助手
            </li>
            <li>
              <FileSearchOutlined aria-hidden />
              知识库助手（自研 RAG）
            </li>
          </ul>
        </div>

        <div className="sidebar-section sidebar-endpoints">
          <span className="sidebar-label">ENDPOINTS</span>
          <code>{API_BASE_URL}</code>
          <code>{WS_BASE_URL}</code>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-topbar">
          <div>
            <span className="panel-kicker">RESEARCH CANVAS</span>
            <h2>把问题变成证据链</h2>
            <p className="topbar-subtitle">检索、推理与交付，在同一条对话里完成</p>
          </div>
          <div className={`run-indicator ${session.isRunning ? "run-indicator--live" : ""}`}>
            {session.isRunning ? <BranchesOutlined aria-hidden /> : <CheckCircleOutlined aria-hidden />}
            {session.isRunning ? "研搜中" : "待命"}
          </div>
        </header>

        {session.lastError ? (
          <Alert
            className="chat-alert"
            message={session.lastError}
            showIcon
            type="error"
          />
        ) : null}

        <section className="chat-stream-panel" ref={streamRef}>
          <ConversationThread
            onUseExample={setQuery}
            turns={turns}
          />
        </section>

        <ChatComposer
          isCancelling={session.isCancelling}
          isRunning={session.isRunning}
          isUploading={session.isUploading}
          onCancel={handleCancel}
          onNewSession={handleNewSession}
          onQueryChange={setQuery}
          onStagedItemsChange={setStagedItems}
          onSubmit={handleSubmit}
          onUpload={handleUpload}
          query={query}
          stagedItems={stagedItems}
          uploadedItems={session.uploadedItems}
        />
      </main>
      <KnowledgeBaseManager
        onClose={() => setKnowledgeBaseOpen(false)}
        open={knowledgeBaseOpen}
      />
    </div>
  );
}
