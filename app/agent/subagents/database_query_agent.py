"""
数据库助手使用有界 Runnable，避免自主工具循环和重复读取样例。
"""

from app.agent.prompts import sub_agents_content
from app.agent.sql_query import run_database_query
from langchain_core.runnables import RunnableLambda

database_query_agent = {
    "name": sub_agents_content["db"]["name"],
    "description": sub_agents_content["db"]["description"],
    "runnable": RunnableLambda(run_database_query),
}
