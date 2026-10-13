"""pytest 共享夹具（t2 测试套件）。

本文件承担三件事，三件都是环境硬约束而非风格偏好（依据 docs/rag-agent-test-plan.md 6.1.1 /
R1，以及 captain 的环境情报）：

1. **仓库根可导入**：测试放在仓库根的 `tests/`，而沙箱不允许在 `app/` 下新建任何东西，
   所以用「把仓库根插进 sys.path」而不是「把测试塞进 app/rag/tests/」。

2. **沙箱安全 tmp_path**：pyproject 的 addopts 传了 `-p no:tmpdir`，pytest 不再创建 basetemp
   （它的 `os.mkdir(..., 0o700)` 在本沙箱下会造出连自己都读不了的目录）。因此 `tmp_path`
   由本文件提供：一律**用默认 mode** 建目录。

3. **知识库目录隔离**：`app.core.config.settings` 是模块级 `lru_cache` 单例，**import 之后再改
   `os.environ` 无效**。所以这里既在 import app.* 之前 setdefault 环境变量，又用 autouse 夹具
   重建 settings 并把 `app.rag.config.settings` 指向 tmp 目录——任何用例都不可能写到真实
   `rag_data/`（实测后果：传不存在的 kb 会凭空造出 rag_data/ghost/ 空库）。

全程离线：向量后端固定注入 HashEmbedder，不加载 sentence-transformers，不调用网络与大模型 API。
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

# ── 0. 仓库根必须在 sys.path 上（沙箱下 app/ 不可写，测试只能放 tests/）──
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── 1. 临时根：默认 mode 创建，绝不用 mkdtemp/TemporaryDirectory ──
# 默认放 tmpdir/pytest-tmp-tester（已被 .gitignore 忽略）。
# 用「本会话独有的目录名」而不是共享的 tmpdir/pytest-tmp：被别的会话用 0o700 建过的目录
# 在此沙箱下不可读也不可删（跨会话所有权），复用它会让 pytest 直接崩。
_TMP_ROOT = Path(
    os.environ.get("RAG_TEST_TMP", str(REPO_ROOT / "tmpdir" / "pytest-tmp-tester"))
)
_CREATED_DIRS: list[Path] = []


def _ensure_root() -> Path:
    if not _TMP_ROOT.exists():
        _TMP_ROOT.mkdir(parents=True, exist_ok=True)  # 默认 mode
    return _TMP_ROOT


# ── 2. RAG_DATA_DIR 必须在任何 app.* import 之前落位 ──
os.environ.setdefault("RAG_DATA_DIR", str(_ensure_root() / "_isolated_rag_data"))
# 默认就别去碰 sentence-transformers：tool 层会走 load_rag_config()
os.environ.setdefault("RAG_EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("SESSIONS_DB_PATH", str(_ensure_root() / "_isolated_sessions.db"))
os.environ.setdefault("MEMORY_DB_PATH", str(_ensure_root() / "_isolated_memory.db"))


# ── 3. 夹具 ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_path():
    """沙箱安全版 tmp_path（替代 pytest 自带、已被 addopts 禁用的那个）。

    目录名唯一、每次调用新建，行为与内置 tmp_path 一致（函数级、互相隔离）。
    """
    d = _ensure_root() / f"t{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)  # 默认 mode → 沙箱可读写
    _CREATED_DIRS.append(d)
    return d


@pytest.fixture(autouse=True)
def _isolate_rag_data(tmp_path, monkeypatch):
    """把 RAG 数据根指向 tmp_path，保证没有用例能写到真实 rag_data/。

    同时清掉 pipeline 的进程内缓存，避免用例之间通过缓存串味。
    """
    import app.core.config as core_config
    import app.rag.config as rag_config
    import app.rag.pipeline as rag_pipeline

    data_dir = tmp_path / "_rag_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    # 强制离线：本机 .env 里配了真实的 DashScope key（openai_api_key/llm_model 非空），
    # 而工具层的 ask_knowledge_base 会经 pipeline._llm_client() 拿到真实客户端 ——
    # 不关掉就会出现「测试偷偷联网调用大模型」：不确定、要花钱、违背"离线可跑"的要求。
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # settings 是 lru_cache 单例：清缓存再重建，load_rag_config 才会看到新路径与新 key
    core_config.get_settings.cache_clear()
    monkeypatch.setattr(rag_config, "settings", core_config.get_settings(), raising=False)
    rag_pipeline.clear_pipeline_cache()
    yield
    rag_pipeline.clear_pipeline_cache()


@pytest.fixture
def rag_data_dir(tmp_path):
    """本用例的隔离数据根目录。"""
    return tmp_path / "_rag_data"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_created_dirs():
    """尽力回收本会话创建的临时目录；失败不影响测试结果（跨会话目录本就删不掉）。"""
    yield
    import shutil

    for d in _CREATED_DIRS:
        shutil.rmtree(d, ignore_errors=True)


# ── 领域夹具 ─────────────────────────────────────────────────────────


@pytest.fixture
def hash_embedder():
    """离线向量后端。测试一律注入它，绝不加载真实模型。"""
    from app.rag.embedding import HashEmbedder

    return HashEmbedder(dim=512)


@pytest.fixture
def make_chunk():
    """构造 Chunk 的小工厂（chunk_id 默认按内容派生，便于测幂等）。"""
    from app.rag.types import Chunk, make_chunk_id

    def _make(
        text: str = "内容",
        index: int = 0,
        source: str = "doc.md",
        source_path: str = "doc.md",
        chunk_id: str | None = None,
        doc_id: str = "",
        heading: str = "",
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id or make_chunk_id(source_path, index, text),
            text=text,
            source=source,
            source_path=source_path,
            chunk_index=index,
            heading=heading,
            doc_id=doc_id,
        )

    return _make


@pytest.fixture
def rag_config(tmp_path):
    """隔离的 RAG 配置：数据目录在 tmp_path、hash 向量、小 chunk 便于构造用例。"""
    from app.rag.config import RagConfig

    return RagConfig(
        data_dir=tmp_path / "_rag_data",
        default_kb="test",
        embedding_provider="hash",
        chunk_size=200,
        chunk_overlap=20,
        top_k=10,
        top_n=4,
        rerank_mode="lexical",
        min_score=0.2,
    )


@pytest.fixture
def pipeline(rag_config, hash_embedder):
    """注入了 HashEmbedder 的隔离流水线（不写 rag_data/、不加载模型）。"""
    from app.rag.pipeline import RagPipeline

    return RagPipeline(kb="test", config=rag_config, embedder=hash_embedder)


@pytest.fixture
def build_pipeline(tmp_path, hash_embedder):
    """按需构造多个流水线（例如测「换 embedder 后指纹不匹配」）。"""
    from app.rag.config import RagConfig
    from app.rag.pipeline import RagPipeline

    def _build(kb: str = "test", embedder=None, **overrides):
        cfg = RagConfig(
            data_dir=tmp_path / "_rag_data",
            default_kb=kb,
            embedding_provider="hash",
            chunk_size=200,
            chunk_overlap=20,
            rerank_mode="lexical",
            min_score=0.2,
            **overrides,
        )
        return RagPipeline(kb=kb, config=cfg, embedder=embedder or hash_embedder)

    return _build


class StubChatClient:
    """假的大模型客户端（generator 用）：可配置返回文本、是否可用、是否抛异常。"""

    def __init__(self, reply: str = "根据参考资料，准确率为 91.5%[1]。", available: bool = True, error: Exception | None = None):
        self.reply = reply
        self._available = available
        self.error = error
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return self._available

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.reply


class StubOpenAIChat:
    """假的 OpenAI 兼容客户端（LLMReranker 用）：按预设顺序返回编号列表。"""

    def __init__(self, content: str = "2,1,3", error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer.error is not None:
                    raise outer.error

                class _Msg:
                    content = outer.content

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


@pytest.fixture
def stub_client():
    return StubChatClient


@pytest.fixture
def stub_openai():
    return StubOpenAIChat


@pytest.fixture
def vector_of():
    """构造一个已归一化的定长向量（模拟 embedder 输出）。"""

    def _vector(seed: int = 0, dim: int = 8) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.random(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    return _vector
