import test from "node:test";
import assert from "node:assert/strict";
import { mergeTaskEvents, taskDisplay, resultAnswer } from "../.test-build/lib/taskState.js";

const event = (seq, task_id = "current") => ({ task_id, seq, event: "tool_start" });

test("replay and live messages merge in sequence without duplicates or previous task messages", () => {
  const result = mergeTaskEvents([event(3), event(1)], [event(2), event(3), event(4, "old")], "current");
  assert.deepEqual(result.map(item => item.seq), [1, 2, 3]);
});

test("versioned answers consume v2 and legacy v1; unknown schemas are not silently accepted", () => {
  assert.equal(resultAnswer({ output: { schema_version: "2.0", answer: "new" }, result: "old" }), "new");
  assert.equal(resultAnswer({ output: { schema_version: "1.0", answer: "legacy" } }), "legacy");
  assert.equal(resultAnswer({ output: { schema_version: "3.0", answer: "unknown" }, result: "old" }), null);
  assert.equal(resultAnswer({ output: { schema_version: "2.0", answer: 123 } }), null);
  assert.equal(resultAnswer({ result: "old task" }), "old task");
});

test("bounded display keeps latest events while replay cursor can track all persisted events", () => {
  assert.deepEqual(mergeTaskEvents([], [event(3), event(1), event(2)], "current", 2).map(item => item.seq), [2, 3]);
});

test("all terminal snapshots stop running; cancelling remains active until confirmed", () => {
  for (const status of ["succeeded", "failed", "cancelled", "interrupted"]) {
    const view = taskDisplay({ status, result: "answer", error_message: "failure" });
    assert.equal(view.isRunning, false);
    assert.equal(view.isCancelling, false);
  }
  assert.equal(taskDisplay({ status: "cancelling" }).isRunning, true);
  assert.equal(taskDisplay({ status: "cancelling" }).isCancelling, true);
});
