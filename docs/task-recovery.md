# 任务状态与断线恢复

每次提交返回 `thread_id` 和独立的 `task_id`。同一会话的下一轮拥有新的任务 ID，替换运行中的任务前会等待旧协程退出。

## 状态及接口

```text
queued -> running -> succeeded / failed
                  -> cancelling -> cancelled
queued / running / cancelling -> interrupted
```

- `GET /api/tasks/{task_id}`：持久快照，含输入、结果、错误码、起止时间和真实状态。
- `GET /api/sessions/{thread_id}/task`：当前会话最新任务；无任务时 `task=null`。
- `GET /api/tasks/{task_id}/events?after=<seq>`：按序补发最多 500 条。收到满页时继续用最后的序号请求。
- `POST /api/task/{thread_id}/cancel`：返回 `cancelling`；协程退出后快照和事件才变为 `cancelled`。

这些接口都校验 owner。删除会话后，其任务及事件也无法通过 API 访问。

每条事件先写入 SQLite，再实时推送，包含 `task_id` 和递增 `seq`。同一会话支持多个 WebSocket 连接。前端按任务 ID、序号去重并排序，最多展示最近 120 条；完整事件仍在数据库中。重连和刷新后通过快照与补发接口恢复，补发游标只由有序 HTTP 数据推进。

## 错误与服务恢复

Agent 异常会向包装层重新抛出，失败不会再保存空助手回答。只有完整执行及历史保存成功后，才报告最终回答。

- `AGENT_EXECUTION_FAILED`：执行异常，详细堆栈留在服务端日志。
- `EXECUTION_TIMEOUT`：执行抛出超时异常。
- `SERVICE_SHUTDOWN`：正常停服中断仍在执行的任务。
- `SERVICE_RESTART`：启动时发现数据库里遗留的未完成任务。

中断任务不会自动重跑，用户可以检查已生成文件后重新提交，避免重复执行带有副作用的工具。

## 部署边界

当前方案用于单个 API Worker。任务执行和并发锁仍在进程内；启动时会把未完成任务标为中断。**不要用多个 Worker 共享这套执行逻辑**，否则一个 Worker 启动可能中断另一个 Worker 的记录。多进程或多机部署需要先完成清单中的持久队列及数据库升级。

前端保留文件轮询；此项完成的是状态恢复和事件补发，产物主动推送仍可在后续引用统一任务中增加。

## 验证记录（2026-10-13）

完整后端回归 `643 passed, 4 skipped`，前端事件去重、结果版本兼容及终态测试 `4 passed`，生产构建通过。跳过项需要真实外部服务或指定运行环境。

隔离数据库和假模型的浏览器验证：执行中刷新仍显示研搜中，恢复工具事件；任务完成后显示最终回答并回到待命。测试未调用真实 LLM、网络搜索或数据库查询。

追加真实接口验证：DeepSeek 的思考模式会拒绝结构化输出使用的 `tool_choice=required`，造成所有任务在首个模型请求返回 400。主 Agent 在官方 DeepSeek 接口的结构化调用中关闭思考模式，仍保留严格 Schema 与纠错预算。限制见 [DeepSeek 接口说明](https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/)。

修复后，使用当前真实模型完成最小计算任务；通过运行中的前端代理提交，任务状态为 `succeeded`，结果为 `schema_version=2.0`、`2 + 2 = 4`，浏览器中历史与回答正常显示。Agent/API/结果专项回归 `30 passed`，新增 MockTransport 用例覆盖服务端 400 限制。此次没有运行真实网络搜索、业务数据库或私人知识库问题。
