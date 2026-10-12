import type { ChatTurn, MonitorMessage, SessionMessage } from "../types";

function parseEvents(message: SessionMessage): MonitorMessage[] {
  try {
    const value = JSON.parse(message.events_json || "[]") as unknown;
    if (!Array.isArray(value)) {
      return [];
    }

    return value.flatMap((item) => {
      if (!item || typeof item !== "object") {
        return [];
      }
      const event = item as Record<string, unknown>;
      if (typeof event.event !== "string") {
        return [];
      }
      return [{
        type: "monitor_event" as const,
        event: event.event,
        message: typeof event.message === "string" ? event.message : event.event,
        data: event.data && typeof event.data === "object"
          ? event.data as Record<string, unknown>
          : {},
        timestamp: typeof event.timestamp === "string"
          ? event.timestamp
          : message.created_at,
      }];
    });
  } catch {
    return [];
  }
}

export function restoreChatTurns(messages: SessionMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];

  messages.forEach((message, index) => {
    if (message.role === "user") {
      turns.push({
        id: `${message.created_at}-${index}`,
        content: message.content,
        events: [],
        files: [],
        isRunning: false,
        result: "",
        timestamp: message.created_at,
      });
      return;
    }

    if (message.role !== "assistant" && message.role !== "system") {
      return;
    }

    let turn = turns[turns.length - 1];
    if (!turn || turn.result) {
      turn = {
        id: `${message.created_at}-${index}`,
        content: "",
        events: [],
        files: [],
        isRunning: false,
        result: "",
        timestamp: message.created_at,
      };
      turns.push(turn);
    }

    turn.events.push(...parseEvents(message));
    turn.result = message.content;
  });

  return turns;
}
