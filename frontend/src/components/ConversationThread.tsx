import {
  BranchesOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  DownloadOutlined,
  FileMarkdownOutlined,
  FilePdfOutlined,
  FileSearchOutlined,
  FileTextOutlined,
  StopOutlined,
  ToolOutlined,
} from "@ant-design/icons";
import { Button, Tooltip } from "antd";
import { useEffect, useRef, useState } from "react";
import { getDownloadUrl } from "../lib/api";
import { MarkdownRenderer } from "./MarkdownRenderer";
import type { ChatTurn, MonitorMessage, OutputFile, RagSourceTrace } from "../types";

interface ConversationThreadProps {
  onUseExample: (prompt: string) => void;
  turns: ChatTurn[];
}

const TASK_EXAMPLES = [
  {
    tool: "网络搜索工具",
    title: "联网趋势研判",
    prompt:
      "请使用网络搜索工具，检索 2026 年跨境电商 AI 客服趋势，列出 5 条关键变化，并附上来源链接。",
    icon: <CloudServerOutlined aria-hidden />,
  },
  {
    tool: "数据库查询工具",
    title: "药品库存排查",
    prompt:
      "请使用数据库查询工具，查询库存大于 100 的药品，按库存量升序列出药品名称、批次号、仓库位置和过期日期。",
    icon: <DatabaseOutlined aria-hidden />,
  },
  {
    tool: "内部知识库",
    title: "内部文档问答",
    prompt:
      "请使用知识库助手，查询公司内部白皮书中关于品类策略的内容，并整理成三条可执行建议。",
    icon: <FileSearchOutlined aria-hidden />,
  },
  {
    tool: "文件读取工具",
    title: "上传文件分析",
    prompt:
      "请使用文件读取工具，读取我上传的文件，提炼核心观点、风险点和待补充信息，并给出下一步分析计划。",
    icon: <FileTextOutlined aria-hidden />,
  },
  {
    tool: "Markdown/PDF 工具",
    title: "生成交付报告",
    prompt:
      "请使用 Markdown 文档生成工具和 Markdown 转 PDF 工具，基于本次调研结果生成一份 Markdown 报告，并转换成 PDF 保存到当前工作目录。",
    icon: <FileMarkdownOutlined aria-hidden />,
  },
];

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "--:--";
  }
  return date.toLocaleTimeString("zh-CN", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function parseTime(value: string): number | null {
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? null : time;
}

function formatDuration(value: number): string {
  const totalSeconds = Math.max(0, Math.floor(value / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const paddedMinutes = String(minutes).padStart(2, "0");
  const paddedSeconds = String(seconds).padStart(2, "0");

  if (hours > 0) {
    return `${hours}:${paddedMinutes}:${paddedSeconds}`;
  }
  return `${paddedMinutes}:${paddedSeconds}`;
}

function getLastEventTime(
  events: MonitorMessage[],
  eventName?: string,
): number | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!eventName || event.event === eventName) {
      return parseTime(event.timestamp);
    }
  }
  return null;
}

function getThinkingDuration(
  events: MonitorMessage[],
  fallbackStart: string,
  isRunning: boolean,
  now: number,
): string {
  const startedAt =
    (events[0] ? parseTime(events[0].timestamp) : null) ??
    parseTime(fallbackStart) ??
    now;
  const finishedAt =
    getLastEventTime(events, "task_result") ??
    (!isRunning ? getLastEventTime(events) : null) ??
    now;
  return formatDuration(finishedAt - startedAt);
}

function EventIcon({ event }: { event: string }) {
  if (event === "assistant_call") {
    return <BranchesOutlined aria-hidden />;
  }
  if (event === "tool_start") {
    return <ToolOutlined aria-hidden />;
  }
  if (event === "session_created") {
    return <FileSearchOutlined aria-hidden />;
  }
  if (event === "task_result") {
    return <CheckCircleOutlined aria-hidden />;
  }
  if (event === "task_cancelled") {
    return <StopOutlined aria-hidden />;
  }
  if (event === "error") {
    return <CloseCircleOutlined aria-hidden />;
  }
  return <ClockCircleOutlined aria-hidden />;
}

function FileIcon({ name }: { name: string }) {
  if (name.endsWith(".pdf")) {
    return <FilePdfOutlined aria-hidden />;
  }
  if (name.endsWith(".md")) {
    return <FileMarkdownOutlined aria-hidden />;
  }
  return <FileTextOutlined aria-hidden />;
}

function ThinkingTimeline({ events }: { events: MonitorMessage[] }) {
  const timelineRef = useRef<HTMLOListElement | null>(null);

  // 移除自动滚动到底部行为，保留用户当前滚动位置

  if (events.length === 0) {
    return (
      <div className="thinking-empty">
        <ClockCircleOutlined aria-hidden />
        等待后端推送执行事件
      </div>
    );
  }

  return (
    <ol className="thinking-timeline" ref={timelineRef}>
      {events.map((event, index) => (
        <li
          className={`thinking-event thinking-event--${event.event}`}
          key={`${event.timestamp}-${index}`}
        >
          <span className="thinking-event-icon">
            <EventIcon event={event.event} />
          </span>
          <div>
            <div className="thinking-event-meta">
              <span>{event.event}</span>
              <time dateTime={event.timestamp}>
                {formatTime(event.timestamp)}
              </time>
            </div>
            <p>{event.message}</p>
            {event.event === "assistant_call" ||
            event.event === "tool_start" ? (
              <code>{JSON.stringify(event.data)}</code>
            ) : null}
          </div>
        </li>
      ))}
    </ol>
  );
}

function ArtifactShelf({ files }: { files: OutputFile[] }) {
  if (files.length === 0) {
    return (
      <div className="artifact-empty">
        <FileSearchOutlined aria-hidden />
        暂无输出文件
      </div>
    );
  }

  return (
    <div className="artifact-shelf">
      {files.map((file) => (
        <div className="artifact-card" key={file.path}>
          <span className="artifact-icon">
            <FileIcon name={file.name} />
          </span>
          <div className="artifact-copy">
            <strong title={file.name}>{file.name}</strong>
            <span>{formatBytes(file.size)}</span>
          </div>
          <Tooltip title="下载">
            <Button
              aria-label={`下载 ${file.name}`}
              className="artifact-download"
              href={getDownloadUrl(file.path)}
              icon={<DownloadOutlined />}
              shape="circle"
            />
          </Tooltip>
        </div>
      ))}
    </div>
  );
}

function isRagSourceTrace(value: unknown): value is RagSourceTrace {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Partial<RagSourceTrace>;
  return typeof candidate.kb === "string" && Array.isArray(candidate.sources);
}

function SourceTrace({ events }: { events: MonitorMessage[] }) {
  const traces: RagSourceTrace[] = [];
  events.forEach((event) => {
    if (event.event === "rag_sources" && isRagSourceTrace(event.data)) {
      traces.push(event.data);
    }
  });

  if (traces.length === 0) {
    return null;
  }

  const sourceCount = traces.reduce((total, trace) => total + trace.sources.length, 0);
  return (
    <details className="source-trace" open>
      <summary>
        <span>
          <FileSearchOutlined aria-hidden />
          检索依据
        </span>
        <strong>{sourceCount}</strong>
      </summary>
      <div className="source-trace-runs">
        {traces.map((trace, traceIndex) => (
          <section className="source-trace-run" key={`${trace.kb}-${trace.question}-${traceIndex}`}>
            <div className="source-trace-meta">
              <span>{trace.kb}</span>
              <span className={trace.grounded ? "source-gate source-gate--pass" : "source-gate"}>
                {trace.grounded ? "依据充分" : "低于拒答阈值"} · {trace.gate_score.toFixed(3)} / {trace.threshold.toFixed(3)}
              </span>
            </div>
            <p className="source-question">{trace.question}</p>
            <ol className="source-list">
              {trace.sources.map((source) => (
                <li key={`${source.source}-${source.chunk_index}-${source.rank}`}>
                  <div className="source-heading">
                    <strong>{source.source} #{source.chunk_index}</strong>
                    <span>{source.heading || "无标题"}</span>
                    <code>{source.score.toFixed(3)}</code>
                  </div>
                  <p>{source.text}</p>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </details>
  );
}

function ThinkingLoader({ durationLabel }: { durationLabel: string }) {
  return (
    <div
      className="thinking-loader"
      aria-live="polite"
      aria-label="正在生成回复"
    >
      <div className="loader-status">
        <span className="loader-pulse" aria-hidden />
        <strong>正在研搜</strong>
        <span className="loader-duration">已思考 {durationLabel}</span>
        <span className="loader-dots" aria-hidden>
          <i />
          <i />
          <i />
        </span>
      </div>
      <div className="loader-track" aria-hidden />
      <ul className="loader-steps" aria-hidden>
        <li>理解问题</li>
        <li>调度工具</li>
        <li>汇总答案</li>
      </ul>
    </div>
  );
}

function AssistantMessage({
  events,
  files,
  isRunning,
  result,
  timestamp,
}: Pick<ChatTurn, "events" | "files" | "isRunning" | "result" | "timestamp">) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!isRunning) {
      return;
    }

    const timer = window.setInterval(() => {
      setNow(Date.now());
    }, 1000);

    return () => window.clearInterval(timer);
  }, [isRunning]);

  const durationLabel = getThinkingDuration(events, timestamp, isRunning, now);
  const isCancelled = events.some((event) => event.event === "task_cancelled");
  const syncLabel = isRunning
    ? `生成中 · 思考 ${durationLabel}`
    : `${isCancelled ? "已取消" : "已同步"} · 用时 ${durationLabel}`;

  return (
    <article className="chat-message chat-message--assistant">
      <div className="message-avatar">AI</div>
      <div className="message-bubble">
        <div className="message-meta">
          <span>DeepSearch Agents</span>
          <time>{syncLabel}</time>
        </div>

        <details
          className="thinking-block"
          open={isRunning || events.length > 0}
        >
          <summary>
            <span>
              <BranchesOutlined aria-hidden />
              深度研搜过程
            </span>
            <strong>{events.length}</strong>
          </summary>
          <ThinkingTimeline events={events} />
        </details>

        {result ? (
          <div className="assistant-answer">
            <MarkdownRenderer content={result} />
          </div>
        ) : (
          <div className="assistant-answer assistant-answer--pending">
            {isRunning ? (
              <ThinkingLoader durationLabel={durationLabel} />
            ) : (
              "任务完成后会在这里显示最终回复。"
            )}
          </div>
        )}

        <SourceTrace events={events} />

        <details
          className="thinking-block artifact-block"
          open={files.length > 0}
        >
          <summary>
            <span>
              <FileSearchOutlined aria-hidden />
              输出文件
            </span>
            <strong>{files.length}</strong>
          </summary>
          <ArtifactShelf files={files} />
        </details>
      </div>
    </article>
  );
}

export function ConversationThread({
  onUseExample,
  turns,
}: ConversationThreadProps) {
  if (turns.length === 0) {
    return (
      <div className="conversation-empty">
        <div className="empty-examples">
          <div className="empty-orbit" aria-hidden>
            <span className="orbit-core">DS</span>
            <span className="orbit-ring orbit-ring--one">
              <i />
            </span>
            <span className="orbit-ring orbit-ring--two">
              <i />
              <i />
            </span>
            <span className="orbit-label orbit-label--web">WEB</span>
            <span className="orbit-label orbit-label--data">DATA</span>
            <span className="orbit-label orbit-label--rag">RAG</span>
          </div>
          <div className="empty-examples-copy">
            <span className="panel-kicker">START A RESEARCH RUN</span>
            <h3>一个问题，调动整支研究团队</h3>
            <p>
              描述目标即可。主智能体会选择检索路径、调用工具，并把证据、过程与交付物放回这里。
            </p>
          </div>

          <div className="example-grid" aria-label="研搜任务示例">
            {TASK_EXAMPLES.map((example) => (
              <button
                className="example-card"
                key={example.tool}
                onClick={() => onUseExample(example.prompt)}
                type="button"
              >
                <span className="example-icon">{example.icon}</span>
                <span className="example-copy">
                  <span>{example.tool}</span>
                  <strong>{example.title}</strong>
                  <small>{example.prompt}</small>
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="conversation-thread" aria-label="聊天消息流">
      {turns.map((turn) => (
        <div className="conversation-turn" key={turn.id}>
          <article className="chat-message chat-message--user">
            <div className="message-bubble">
              <div className="message-meta">
                <span>你</span>
                <time dateTime={turn.timestamp}>
                  {formatTime(turn.timestamp)}
                </time>
              </div>
              <p>{turn.content}</p>
            </div>
          </article>
          <AssistantMessage
            events={turn.events}
            files={turn.files}
            isRunning={turn.isRunning}
            result={turn.result}
            timestamp={turn.timestamp}
          />
        </div>
      ))}
    </div>
  );
}
