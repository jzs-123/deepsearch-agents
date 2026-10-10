"""
知识库子智能体配置模块(自研 RAG 版)

把 app/prompt/prompts.yml 中的 rag 配置与本项目自研的 RAG 工具组装成 DeepAgents
可识别的字典式子智能体。主智能体根据 description 决定是否把企业内部非结构化文档
查询任务分派给它。

与旧 RAGFlow 版的区别:
- 不再依赖外部 RAGFlow 服务与 ragflow-sdk,检索链路全部在本项目内;
- 工具从「查助手列表 -> 创建临时会话提问」改为「列知识库 -> 混合检索 / 直接问答」,
  检索结果带匹配度与来源编号,便于模型判断信息是否充分;
- 检索无状态,不存在跨轮会话上下文污染。
"""

from app.agent.prompts import sub_agents_content
from app.tools.rag_tools import ask_knowledge_base, list_knowledge_bases, search_knowledge_base

# 知识库子智能体处理企业内部非结构化文档,与网络搜索助手、数据库查询助手互补
# tools 声明三件事:能发现有哪些知识库、能只做检索拿原文、也能一步拿到带引用的答案
knowledge_base_agent = {
    "name": sub_agents_content["rag"]["name"],
    "description": sub_agents_content["rag"]["description"],
    "system_prompt": sub_agents_content["rag"]["system_prompt"],
    "tools": [list_knowledge_bases, search_knowledge_base, ask_knowledge_base],
}
