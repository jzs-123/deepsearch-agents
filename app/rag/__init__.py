"""
自研 RAG(检索增强生成)模块

替代原 RAGFlow 外部服务,把「文档解析 -> 切块 -> 向量化 -> 存储 -> 混合检索 -> 重排 -> 生成」
整条链路收进本项目,便于二次开发和排障。

模块分层(自下而上):
    types     数据模型(Chunk / RetrievedChunk / Answer)
    config    RAG 配置装配(从 Settings 映射为运行时配置)
    loaders   文档解析(多格式统一成纯文本 + 结构信息)
    chunking  文本切块(结构感知 + 重叠窗口)
    embedding 向量化(本地模型 / 接口 / 离线兜底,可降级)
    vectorstore 向量库持久化 + 相似度检索
    bm25      稀疏检索(中文分词 + BM25)
    hybrid    稠密 + 稀疏 RRF 融合
    rerank    重排(词面重排 / 大模型列表式重排)
    generator 基于检索上下文的答案生成(带引用,支持拒答)
    pipeline  端到端编排(入库 / 查询)
    cli       命令行入口(建库 / 入库 / 检索 / 问答 / 评测)
"""

from app.rag.types import Answer, Chunk, RetrievedChunk

__all__ = ["Answer", "Chunk", "RetrievedChunk"]
