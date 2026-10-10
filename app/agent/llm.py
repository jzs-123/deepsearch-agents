"""
大模型初始化模块

负责从 .env 中读取模型配置，并创建项目统一复用的模型对象
后续主智能体和子智能体都从这里导入 model，避免在多个文件里重复加载环境变量
"""

import httpx
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

from app.core.config import settings

# find_dotenv 会从当前目录向上查找 .env，适合脚本和 Web 服务从不同入口启动的场景
# 集中配置已由 app.core.config 收敛,这里保留 load_dotenv 以兼容脚本直接运行
load_dotenv(find_dotenv())

# 配置带重试的 HTTP 客户端，解决 DeepSeek API 偶发连接重置（10054）的问题
# connect=10s 连接超时，总超时 120s（DeepSeek 推理较慢时可能超过 60s）
# max_retries=3：总共 3 次尝试（1 次原始 + 2 次重试）
_http_timeout = httpx.Timeout(120.0, connect=10.0)

# 同步客户端
http_client = httpx.Client(
    timeout=_http_timeout,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
)

# 异步客户端
http_async_client = httpx.AsyncClient(
    timeout=_http_timeout,
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
)

# 使用 OpenAI 兼容协议接入模型；具体模型名由集中配置 Settings.llm_model 控制
model = init_chat_model(
    model=settings.llm_model,
    model_provider="openai",
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    max_retries=3,
    timeout=120.0,
    http_client=http_client,
    http_async_client=http_async_client,
)
