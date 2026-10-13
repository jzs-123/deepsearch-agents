import type { MonitorMessage, TaskSnapshot } from "../types";

export function mergeTaskEvents(previous: MonitorMessage[], incoming: MonitorMessage[], taskId: string, limit = 120): MonitorMessage[] {
  const events = new Map<number, MonitorMessage>();
  for (const event of [...previous, ...incoming]) {
    if (event.task_id === taskId && typeof event.seq === "number") {
      events.set(event.seq, event);
    }
  }
  return [...events.values()].sort((a, b) => a.seq! - b.seq!).slice(-limit);
}

export function taskDisplay(snapshot: TaskSnapshot) {
  return {
    isRunning: ["queued", "running", "cancelling"].includes(snapshot.status),
    isCancelling: snapshot.status === "cancelling",
    result: snapshot.status === "cancelled" ? "任务已取消" : resultAnswer({ output: snapshot.output, result: snapshot.result }) || "",
    error: snapshot.error_message,
  };
}

export function resultAnswer(data: Record<string, unknown>): string | null {
  const output = data.output;
  if (output && typeof output === "object") {
    const value = output as Record<string, unknown>;
    if (value.schema_version !== "1.0" && value.schema_version !== "2.0") return null;
    return typeof value.answer === "string" ? value.answer : null;
  }
  return typeof data.result === "string" ? data.result : null;
}
