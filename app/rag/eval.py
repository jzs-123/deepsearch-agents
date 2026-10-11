"""
检索质量评测

没有评测的 RAG 调参全靠感觉。这里提供一个最小可用的评测脚本:给一份「问题 -> 期望来源」
的评测集,统计命中率与排序质量,用来回答三个具体问题:
1. 混合检索比只用向量检索好多少?(关闭 BM25 对比)
2. 重排到底有没有把最相关的片段提到前面?(切换 rerank 模式对比)
3. 切块参数改动是变好还是变坏?

指标定义:
    Hit@k   —— 期望来源出现在前 k 条结果里的问题占比(最直观的召回指标);
    MRR     —— 第一个正确结果的排名倒数均值,衡量「多快能命中」;
    Recall@k—— 期望来源的覆盖率(一个问题可能有多个来源);
    ｜  拒答率 —— 无依据问题的正确拒答比例(防止「什么都敢答」的模型刷分)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.pipeline import RagPipeline


@dataclass
class EvalItem:
    """一条评测样本。"""

    question: str
    expected_sources: list[str] = field(default_factory=list)
    # 期望「无依据」的问题(知识库不该回答),用于统计拒答率
    expect_no_answer: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "EvalItem":
        sources = data.get("expected_sources") or data.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        return cls(
            question=data["question"],
            expected_sources=list(sources),
            expect_no_answer=bool(data.get("expect_no_answer", False)),
        )


@dataclass
class EvalResult:
    """评测汇总结果。"""

    total: int = 0
    hit_at_k: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    refusal_accuracy: float = 0.0
    per_query: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    # 查询改写的可观测统计(默认采集,便于回答"改写到底起没起作用")
    rewrite_mode: str = "off"
    rewrite_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "hit_at_k": round(self.hit_at_k, 4),
            "mrr": round(self.mrr, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "config": self.config,
            "rewrite_mode": self.rewrite_mode,
            "rewrite_stats": self.rewrite_stats,
            "per_query": self.per_query,
        }


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """加载评测集,支持 JSON 数组或 JSONL。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在:{path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        records = data.get("items", []) if isinstance(data, dict) else data
    return [EvalItem.from_dict(item) for item in records]


def _norm_source(source: str) -> str:
    """来源归一化:去掉目录前缀,只留文件名。

    入库时的 source_path 是相对路径(如 sub_dir/报告.md),而评测集里通常只写文件名。
    不归一化会出现「检索其实完全正确却报 0 命中」的假失败(本项目实测踩过),
    对使用者也很不友好。
    """
    return str(source).replace("\\", "/").split("/")[-1].strip().lower()


def _matches(source: str, expected: list[str]) -> bool:
    """检索到的来源是否属于期望来源(按文件名比对,忽略目录与大小写)。"""
    norm = _norm_source(source)
    return any(norm == _norm_source(item) for item in expected)


def evaluate(
    pipeline: RagPipeline,
    items: list[EvalItem],
    k: int = 5,
    rewrite: bool | None = None,
) -> EvalResult:
    """对给定流水线跑一遍评测。

    注意:这里评测的是「检索层」,不调用生成模型 —— 生成质量受模型波动影响大,
    检索指标才是切块/向量/融合/重排这些自研部分能直接优化的对象。

    :param rewrite: 透传给 `pipeline.search`:None=按配置(默认 local,低分自动重试)、
        False=关闭改写(对比基线)、True=首轮就用全部变体
    """
    result = EvalResult(total=len(items))
    result.config = {
        "kb": pipeline.kb,
        "embedding": pipeline.embedder.fingerprint,
        "chunk_size": pipeline.config.chunk_size,
        "chunk_overlap": pipeline.config.chunk_overlap,
        "top_k": pipeline.config.top_k,
        "rerank_mode": pipeline.config.rerank_mode,
        "rrf_k": pipeline.config.rrf_k,
        "query_rewrite": pipeline.config.query_rewrite,
        "query_variants": pipeline.config.query_variants,
        "retry_on_low_score": pipeline.config.retry_on_low_score,
        "rewrite_override": rewrite,
        "k": k,
    }

    hits = 0
    reciprocal_sum = 0.0
    recall_sum = 0.0
    graded = 0
    refusal_correct = 0
    refusal_total = 0
    # 改写可观测性:默认采集(评测里看不到改写有没有生效,就没法解释指标变化)
    rewrite_stats: dict = {
        "mode": "off",
        "queries": len(items),
        "multi_variant_queries": 0,
        "retry_triggered": 0,
        "retry_improved": 0,
        "gate_before_vs_after": [],
        "variant_query_examples": [],
        "llm_used": 0,
        "llm_degraded": 0,
    }

    for item in items:
        chunks = pipeline.search(
            item.question,
            top_k=max(k, pipeline.config.top_k),
            top_n=k,
            rewrite=rewrite,
        )
        retrieved = [c.chunk.source for c in chunks]
        # 拒答判定与线上一致:一律用向量余弦(重排分的尺度不能直接和阈值比)
        top_score = pipeline.grounded_score(chunks)
        audit = pipeline.last_search_debug or {}
        rewrite_stats["mode"] = audit.get("rewrite_mode", rewrite_stats["mode"])
        variants = audit.get("variants") or []
        if len(variants) > 1:
            rewrite_stats["multi_variant_queries"] += 1
            if len(rewrite_stats["variant_query_examples"]) < 5:
                rewrite_stats["variant_query_examples"].append(
                    {"question": item.question, "variants": variants}
                )
        retry = audit.get("retry") or {}
        if retry.get("triggered"):
            rewrite_stats["retry_triggered"] += 1
            if retry.get("improved"):
                rewrite_stats["retry_improved"] += 1
            rewrite_stats["gate_before_vs_after"].append(
                {
                    "question": item.question,
                    "before": retry.get("gate_before"),
                    "after": retry.get("gate_after"),
                }
            )
        rewrite_meta = audit.get("rewrite") or {}
        if rewrite_meta.get("llm_used"):
            rewrite_stats["llm_used"] += 1
        if rewrite_meta.get("degraded"):
            rewrite_stats["llm_degraded"] += 1

        if item.expect_no_answer:
            refusal_total += 1
            # 无依据问题:检索不到内容,或最相关分数低于阈值,都算正确拒答
            passed = (not chunks) or top_score < pipeline.config.min_score
            if passed:
                refusal_correct += 1
            result.per_query.append(
                {
                    "question": item.question,
                    "expect_no_answer": True,
                    "retrieved": retrieved,
                    "top_score": round(top_score, 4),
                    "passed": bool(passed),
                    "variants": variants,
                    "retry_triggered": bool(retry.get("triggered")),
                }
            )
            continue

        graded += 1
        rank_of_first_hit = 0
        matched: list[str] = []
        for rank, source in enumerate(retrieved, start=1):
            if _matches(source, item.expected_sources) and source not in matched:
                matched.append(source)
                if not rank_of_first_hit:
                    rank_of_first_hit = rank

        hit = 1.0 if matched else 0.0
        hits += hit
        if rank_of_first_hit:
            reciprocal_sum += 1.0 / rank_of_first_hit
        if item.expected_sources:
            recall_sum += len(matched) / len(item.expected_sources)

        result.per_query.append(
            {
                "question": item.question,
                "expected": item.expected_sources,
                "retrieved": retrieved,
                "top_score": round(top_score, 4),
                "first_hit_rank": rank_of_first_hit,
                "hit": bool(hit),
                "variants": variants,
                "retry_triggered": bool(retry.get("triggered")),
            }
        )

    if graded:
        result.hit_at_k = hits / graded
        result.mrr = reciprocal_sum / graded
        result.recall_at_k = recall_sum / graded
    if refusal_total:
        result.refusal_accuracy = refusal_correct / refusal_total
    result.rewrite_mode = rewrite_stats["mode"]
    result.rewrite_stats = rewrite_stats
    return result


def format_report(result: EvalResult) -> str:
    """把评测结果格式化成可直接贴进文档的文本。"""
    lines = [
        "── RAG 检索评测报告 ──",
        f"样本数: {result.total}",
        f"Hit@{result.config.get('k')}: {result.hit_at_k:.2%}",
        f"MRR: {result.mrr:.4f}",
        f"Recall@{result.config.get('k')}: {result.recall_at_k:.2%}",
        f"拒答准确率: {result.refusal_accuracy:.2%}",
        f"查询改写: {result.rewrite_mode}"
        f"(多变体查询 {result.rewrite_stats.get('multi_variant_queries', 0)}/{result.total}"
        f" | 触发重试 {result.rewrite_stats.get('retry_triggered', 0)}"
        f" | 重试后改善 {result.rewrite_stats.get('retry_improved', 0)}"
        f" | 大模型改写 {result.rewrite_stats.get('llm_used', 0)} 次"
        f" / 降级 {result.rewrite_stats.get('llm_degraded', 0)} 次)",
        f"配置: {json.dumps(result.config, ensure_ascii=False)}",
        "",
        "逐条结果:",
    ]
    for row in result.per_query:
        variant_note = ""
        if len(row.get("variants") or []) > 1:
            variant_note = f" 变体={len(row['variants'])}"
        if row.get("retry_triggered"):
            variant_note += " 已重试"
        if row.get("expect_no_answer"):
            lines.append(
                f"  [{'OK' if row['passed'] else 'MISS'}] (应拒答) {row['question']}"
                f" -> 最高分 {row.get('top_score')}{variant_note}"
            )
        else:
            mark = "HIT" if row.get("hit") else "MISS"
            lines.append(
                f"  [{mark} rank={row.get('first_hit_rank')} score={row.get('top_score')}{variant_note}] "
                f"{row['question']} -> 命中来源 {row.get('retrieved')}"
            )
    return "\n".join(lines)


__all__ = ["EvalItem", "EvalResult", "load_eval_set", "evaluate", "format_report"]
