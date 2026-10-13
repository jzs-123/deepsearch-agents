# 会话与知识库隔离配置

默认 `AUTH_ENABLED=false`，保留本地匿名开发模式。多人使用时，在 `.env` 配置：

```dotenv
AUTH_ENABLED=true
AUTH_USERS={"replace-with-alice-key":{"user_id":"alice","tenant_id":"school-a"},"replace-with-bob-key":{"user_id":"bob","tenant_id":"school-b"}}
CORS_ORIGINS=["http://localhost:5173","http://127.0.0.1:5173"]
```

把示例凭证换为独立、随机的真实凭证，重启后端。网页会显示登录页；脚本在请求头中发送 `Authorization: Bearer <key>`。用户和租户由服务端绑定，伪造 `X-User-Id`、`X-Tenant-Id` 不会改变身份。

网页凭证放在 HttpOnly Cookie 中，WebSocket 和下载同样校验。前后端地址请统一使用 `localhost`，或统一使用 `127.0.0.1`。生产部署设置 `APP_ENV=prod` 并使用 HTTPS；页面域名需加入 `CORS_ORIGINS`。

## 隔离范围

- 会话列表、历史、任务启动/取消、文件上传/列表/下载、WebSocket 均校验会话 owner。
- 同一 `thread_id` 不能被另一用户接管；删除会话后保留身份占用记录，防止残留文件被新用户继承。
- Agent 的内置文件工具与自定义文件工具只能访问当前会话目录。技能复制到会话目录后加载。
- 用户长期记忆命名空间包含租户 ID 和用户 ID。
- 新知识库位于 `rag_data/.tenants/<租户标识哈希>/`，同租户成员共享该租户的知识库；不同租户的同名知识库互不影响。
- 开启鉴权后，原 `rag_data/<kb>` 不会自动公开。若某个旧库确实允许所有已登录用户读取，设置 `RAG_SHARED_KBS=["fin"]`。公共库通过网页只读，上传写入租户自己的库，同名时优先使用租户库。

## 旧数据兼容

会话表首次加载时自动增加身份字段，旧会话保留 `anonymous/default` 归属。`AUTH_API_KEY` 是旧单用户兼容选项，固定映射到这一身份。不要给不同用户使用同一个凭证。

旧长期记忆不会自动迁入新的租户命名空间，避免把历史记忆错误分配给其他用户。迁移前应人工确认归属并备份。

远程 Qdrant 的新库使用独立 collection；旧远程库如果仍使用共用的 `rag_chunks`，程序会拒绝打开非空旧库，提示备份后重新入库。程序不会自动删除或重建旧知识库。本地 Qdrant 保留原有目录兼容。

## 验证

权限、路径、上传、租户知识库和 Agent 文件沙箱都有离线回归测试。真实模型与真实远程 Qdrant 服务未在本次验证中调用。
