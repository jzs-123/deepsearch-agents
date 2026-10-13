"""
集中配置模块

用 pydantic-settings 把散落在各模块的 os.getenv 收敛到一处,提供类型化、
带默认值的配置访问。所有新增配置均有安全默认值,保证旧流程开箱可用。

环境变量自动绑定规则(大小写不敏感):
    llm_deepseek  <->  LLM_DEEPSEEK
    mysql_host    <->  MYSQL_HOST
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 当前文件位于 app/core/config.py,parents[2] 即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """项目全部配置项的统一入口。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── 运行环境 ──────────────────────────────────────────
    app_env: str = "dev"
    app_name: str = "deepsearch-agents"

    # ─── 大模型 ────────────────────────────────────────────
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_api_key: str = ""
    # 兼容两套历史变量:优先 LLM_DEEPSEEK,回退 LLM_QWEN_MAX
    llm_deepseek: str = ""
    llm_qwen_max: str = "qwen-max"
    # 推理模型思维链开关:disabled/enabled/空(不传,跟随服务端默认)。
    # 思维链与正文共享 max_tokens,RAG 问答属「依据检索结果抽取摘要」任务,
    # 不需要长链推理,默认关掉(见 docs/rag-self-built.md §5.20)。
    llm_thinking: str = "disabled"

    # ─── Tavily ───────────────────────────────────────────
    tavily_api_key: str = ""

    # ─── RAGFlow(已下线,保留字段仅用于回滚旧链路) ──────────
    ragflow_api_url: str = ""
    ragflow_api_key: str = ""

    # ─── 自研 RAG ─────────────────────────────────────────
    # 知识库数据根目录,每个知识库是其中一个子目录。
    # 放在项目根独立的 rag_data/ 下,与代码目录分离:便于备份/迁移,也不会被缓存或打包流程误删。
    rag_data_dir: str = str(PROJECT_ROOT / "rag_data")
    # 默认知识库名称(CLI 与 Agent 工具未显式指定时使用)
    rag_default_kb: str = "default"
    # 向量化后端:local(sentence-transformers 本地模型) / openai(OpenAI 兼容接口) / hash(离线兜底)
    rag_embedding_provider: str = "local"
    # 本地模型名;BGE-small-zh-v1.5 中文效果好、体积小(约 95MB)、CPU 可跑
    rag_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    # 向量维度上限:仅用于 hash 兜底与接口后端,本地模型以模型自身维度为准
    rag_embedding_dim: int = 512
    # 本地模型推理设备,auto 表示有 CUDA 用 GPU,否则 CPU
    rag_embedding_device: str = "auto"
    # 单个 chunk 目标字符数:中文场景 400~600 之间召回与上下文完整性较平衡
    rag_chunk_size: int = 500
    # 相邻 chunk 重叠字符数,避免答案被切断在边界上
    rag_chunk_overlap: int = 80
    # 粗排召回条数(送入重排与生成的候选池大小)
    rag_top_k: int = 10
    # 重排/精排后最终送入大模型的条数
    rag_top_n: int = 4
    # 是否开启重排:lexical(本地词面重排,零成本) / llm(用大模型做列表式重排,更准更慢) / off
    rag_rerank_mode: str = "lexical"
    # 混合检索中稀疏路(BM25)与稠密路的 RRF 融合常数
    rag_rrf_k: int = 60
    # 是否启用 BM25 稀疏路。超大知识库使用远程 Qdrant 时可关闭，避免首查加载全部 chunk。
    rag_sparse_enabled: bool = True
    # 低于该相似度视为「知识库中没有依据」,直接拒答而不是让模型编造
    rag_min_score: float = 0.2
    # ─── 自研 RAG:向量存储后端 ─────────────────────────────
    # numpy(默认,零依赖,全量内存+磁盘持久化,万级规模够快) / qdrant(本地嵌入或远程 server)
    # 换后端必须重新入库(不同后端不自动迁移,向量指纹校验会拦住不匹配)。
    rag_vectorstore: str = "numpy"
    # qdrant 远程 server 地址;留空 = 本地嵌入模式(数据落在 <kb>/qdrant/)
    rag_qdrant_url: str = ""
    # ─── 自研 RAG:查询改写(Query Rewrite) ────────────────
    # off(关闭) / local(本地规则,零成本) / llm(大模型,失败降级本地) / local+llm(两级都用)
    # 应用级默认 local:CLI 与 Agent 工具都经 load_rag_config() 读配置,所以默认开启改写。
    # (app/rag/config.py 里 RagConfig 的数据类默认值是 off,保证直接构造配置的调用方行为不变)
    rag_query_rewrite: str = "local"
    # 变体总数上限(含原始查询——原始查询永远保留,所以 >= 1)
    rag_query_variants: int = 3
    # 首轮判定分(向量余弦)低于 rag_min_score 时,自动用改写变体重试一次再判定
    rag_retry_on_low_score: bool = True
    # 补充同义词表,格式 "key=alt1|alt2;key2=alt"(内置词表见 app/rag/rewrite.py)
    rag_rewrite_synonyms: str = ""

    # ─── OCR(图片/扫描件兜底;方案 A:默认 Windows 内置引擎,零 pip 依赖) ───
    # 总开关。关闭后:图片文件不会被解析、扫描件页只保留文本层 —— 链路仍然可用。
    rag_ocr_enabled: bool = True
    # 后端:windows(系统内置 Windows.Media.Ocr,仅 Windows) / rapidocr(需装 rapidocr-onnxruntime) / none
    # 后端不可用时自动降级为 none 并 warning,**不影响入库**(OCR 是兜底,不是主路径)
    rag_ocr_backend: str = "windows"
    # PDF 单页文本层字符数低于该值时对该页做 OCR 兜底(0 = 关闭页级 OCR)
    rag_ocr_min_text_chars: int = 50
    # OCR 语言(BCP-47);Windows 需已安装对应语言包,否则引擎创建失败并降级
    rag_ocr_language: str = "zh-Hans-CN"

    # ─── 自研 RAG:缓存(embedding / 查询结果 / 生成结果) ───
    # 默认关闭:查询与生成缓存一旦陈旧就是"返回过期答案",不该由默认配置承担这个风险;
    # embedding 缓存是内容寻址(不陈旧)但共用这个总开关,保证"一个开关解释全部行为"。
    # 缓存文件落在 rag_data/<kb>/.cache/(已加入 .gitignore)
    rag_cache_enabled: bool = False

    # ─── 自研 RAG:审计日志(JSONL;每次 search/query 一条,不含原文) ───
    # 默认开启;写入失败只 warning,绝不影响查询
    rag_audit_enabled: bool = True
    # 审计目录;留空 = 跟随知识库数据根目录下的 .audit/(默认即 rag_data/.audit/)
    rag_audit_dir: str = ""

    # ─── MySQL ────────────────────────────────────────────
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_database: str = "deepsearch_db"
    mysql_host: str = "localhost"
    mysql_port: int = 3307
    mysql_charset: str = "utf8mb4"
    mysql_collation: str = "utf8mb4_unicode_ci"
    mysql_sql_mode: str = "TRADITIONAL"

    # ─── 存储路径 ─────────────────────────────────────────
    output_dir: str = str(PROJECT_ROOT / "output")
    updated_dir: str = str(PROJECT_ROOT / "updated")
    memory_db_path: str = str(PROJECT_ROOT / "app" / "data" / "memory.db")
    sessions_db_path: str = str(PROJECT_ROOT / "app" / "sessions.db")

    # ─── 记忆模块 ─────────────────────────────────────────
    memory_recall_limit: int = 5
    memory_decay_enabled: bool = False

    # ─── 上下文模块 ───────────────────────────────────────
    context_max_tokens: int = 30000
    context_history_messages: int = 40
    context_max_history_tokens: int = 20000

    # ─── 安全 ─────────────────────────────────────────────
    auth_enabled: bool = False
    auth_api_key: str = ""
    # 服务端绑定凭证与身份，不能信任客户端的 X-User-Id/X-Tenant-Id。
    auth_users: dict[str, dict[str, str]] = {}
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    rag_shared_kbs: list[str] = []

    # ─── 并发与限流 ───────────────────────────────────────
    max_concurrent_tasks: int = 8
    agent_recursion_limit: int = 60
    rate_limit_per_minute: int = 60

    # ─── 日志 ─────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "text"  # text | json

    @property
    def llm_model(self) -> str:
        """实际使用的大模型名,兼容 LLM_DEEPSEEK 与 LLM_QWEN_MAX 两套变量。"""
        return self.llm_deepseek or self.llm_qwen_max

    @property
    def mysql_connect_args(self) -> dict:
        """拼成 mysql.connector.connect 可直接使用的连接参数。"""
        return {
            "host": self.mysql_host,
            "port": self.mysql_port,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "database": self.mysql_database,
            "charset": self.mysql_charset,
            "collation": self.mysql_collation,
            "autocommit": True,
            "sql_mode": self.mysql_sql_mode,
        }


@lru_cache
def get_settings() -> Settings:
    """返回全局共享的 Settings 单例(进程内缓存)。"""
    return Settings()


# 模块级便捷单例,保持与项目其它模块「import 即用」的风格一致
settings = get_settings()
