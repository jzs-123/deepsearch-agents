"""Agent 装配层契约：prompts.yml 与子智能体装配必须一致。

背景（docs/rag-self-built.md 第八章）：知识库能力从 RAGFlow 切到自研 RAG 时，
`prompts.yml` 的段落由 `ragflow` 改名成 `rag`，`knowledge_base_agent` 的 import 也换成了
`app.tools.rag_tools`。**改名类重构最容易漏掉某一处引用**，所以这里用契约测试把它钉住：
键集合必须相等、旧链路不得残留。

本文件只读文件与轻量 import，不依赖 deepagents，因此 3.10 下可直接跑。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_YML = REPO_ROOT / "app" / "prompt" / "prompts.yml"
KB_AGENT_SRC = REPO_ROOT / "app" / "agent" / "subagents" / "knowledge_base_agent.py"
RAG_TOOL_NAMES = ("list_knowledge_bases", "search_knowledge_base", "ask_knowledge_base")


@pytest.fixture(scope="module")
def prompts_yml() -> dict:
    return yaml.safe_load(PROMPTS_YML.read_text(encoding="utf-8"))


def test_prompts_yml_exists():
    assert PROMPTS_YML.is_file()


def test_prompts_yml_has_main_and_sub_agents(prompts_yml):
    assert "main_agent" in prompts_yml
    assert "sub_agents" in prompts_yml


def test_sub_agent_keys_match_the_python_contract(prompts_yml):
    """★ 改名重构的核心回归锁：yml 的键集合必须与 `sub_agents_content` 完全一致。

    少一个键 → 子智能体拿不到 system_prompt；多一个键 → 说明有段落改了名却没人读。
    """
    prompts = pytest.importorskip("app.agent.prompts", reason="Agent 装配层在 3.10 下可能不可导入")
    assert set(prompts_yml["sub_agents"]) == set(prompts.sub_agents_content)


def test_rag_key_replaced_ragflow_key(prompts_yml):
    assert "rag" in prompts_yml["sub_agents"]
    assert "ragflow" not in prompts_yml["sub_agents"]


def test_every_sub_agent_has_name_and_prompts(prompts_yml):
    for key, section in prompts_yml["sub_agents"].items():
        assert section.get("name"), f"{key} 缺少 name"
        assert section.get("description"), f"{key} 缺少 description"
        assert section.get("system_prompt"), f"{key} 缺少 system_prompt"


def test_rag_agent_prompt_mentions_the_three_tools(prompts_yml):
    prompt = prompts_yml["sub_agents"]["rag"]["system_prompt"]
    for tool_name in RAG_TOOL_NAMES:
        assert tool_name in prompt, f"rag 子智能体提示词未提到工具 {tool_name}"


def test_knowledge_base_agent_wires_the_self_built_tools():
    src = KB_AGENT_SRC.read_text(encoding="utf-8")
    for tool_name in RAG_TOOL_NAMES:
        assert tool_name in src, f"knowledge_base_agent 未接入 {tool_name}"
    assert "rag_tools" in src


def test_knowledge_base_agent_imports_self_built_tools_not_ragflow():
    """只钉住**导入接线**：允许文档里解释"与旧 ragflow 版的区别"，但代码不得再引入它。"""
    src = KB_AGENT_SRC.read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in src.splitlines() if line.strip().startswith(("import ", "from "))
    )
    assert "rag_tools" in import_lines
    assert "ragflow" not in import_lines.lower()


def test_ragflow_tools_is_not_imported_anywhere_in_app():
    """旧链路必须彻底下线：全仓库不得再 import ragflow_tools（用 grep 级别的证据）。"""
    offenders: list[str] = []
    for path in (REPO_ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "ragflow_tools" in stripped and ("import" in stripped or "from" in stripped):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped}")
    assert offenders == [], f"仍存在对旧 RAGFlow 工具层的引用：{offenders}"


def test_rag_agent_is_registered_in_sub_agents_content():
    prompts = pytest.importorskip("app.agent.prompts", reason="Agent 装配层在 3.10 下可能不可导入")
    assert "rag" in prompts.sub_agents_content
    assert "ragflow" not in prompts.sub_agents_content
