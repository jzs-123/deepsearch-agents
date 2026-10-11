"""零依赖离线「生成质量」近似评测 —— **不是 RAGAS**。

定位（详见 `docs/rag-eval-guide.md`）：
本项目的 `app/rag/eval.py` 测的是**检索层**（Hit@k / MRR / Recall@k / 拒答率），
它回答不了「答案本身是否忠于检索到的上下文」。测生成质量的正解是 RAGAS，
但 RAGAS 依赖 `datasets` / `sentence-transformers` 会牵出 torch 系依赖，
在本沙箱里装不上（pip 会空转）。

所以本模块提供一个**诚实的近似**：只用标准库，判断答案里出现的
**数字 / 百分比 / 金额 / 型号 / 编号**这些「可核验事实」能否在检索到的上下文中找到，
找不到的记为「疑似幻觉」。

    ⚠️ 它与 RAGAS 的 faithfulness **不是一回事**，不能互相替代、也不得混为一谈：
    - 本近似只做**字面包含**判断，无法识别「同义改写」「语义蕴含」；
    - 它对「答案完全正确但换了说法」会误报为疑似幻觉（假阳性）；
    - 它对「答案通篇是空话、没有任何数字」会判为不可核验（`score=None`）。

请在报告里明确标注来源。RAGAS 的地位是正式指标，本模块只是"没有 RAGAS 时也能有个抓手"。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

#: 报告里必须携带的免责声明（任何下游消费者都应能一眼看到这不是 RAGAS）
OFFLINE_DISCLAIMER = (
    "离线近似，非 RAGAS：仅做「可核验事实的字面包含」判断，"
    "无法识别同义改写/语义蕴含，存在假阳性与假阴性；"
    "正式生成质量评测请使用 RAGAS（见 docs/rag-eval-guide.md）。"
)

#: RAGAS 官方指标名，用于提醒使用者本报告不含这些指标
RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# ── 事实抽取 ────────────────────────────────────────────────────────
# 顺序有意为之：越具体的模式越靠前，否则 "91.5%" 会先被「小数编号」模式吃掉一半。
_FACT_RE = re.compile(
    r"""
      \d+(?:[.,]\d+)?\s*(?:%|％)                          # 百分比 91.5%
    | \d+(?:[.,]\d+)?\s*(?:亿元|万元|亿|万|元|美元|美元)     # 金额 1.5 亿元
    | [A-Za-z]{1,10}\d+[A-Za-z0-9\-]*                     # 型号 A13 / B7 / A13Pro
    | \d+(?:\.\d+){1,3}                                   # 编号 3.2.1
    | \d+(?:[.,]\d+)?                                     # 其他数字
    """,
    re.VERBOSE,
)

# 中文里几乎无区分度的功能字/停用字，用于把长中文串切成「词」
_FUNCTION_CHARS = "的了是在和与及或为对以并按从把被让等中这那有之其与所"
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_LATIN_RE = re.compile(r"[A-Za-z]{2,}[A-Za-z0-9\-]*")
_CJK_SPLIT_RE = re.compile(f"[{_FUNCTION_CHARS}]")


def _norm(text: str) -> str:
    """归一化用于「是否包含」比较：去空白、统一全角百分号、统一小写、统一千分位逗号。"""
    return (
        str(text)
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("％", "%")
        .replace("，", ",")
        .lower()
    )


def extract_facts(text: str) -> list[str]:
    """抽取答案里**可核验的事实**：数字、百分比、金额、型号、编号。

    这些是幻觉最容易暴露的地方（数字/金额/日期被模型悄悄改写），
    也是唯一能用字面包含稳定核验的东西。
    """
    seen: set[str] = set()
    facts: list[str] = []
    for match in _FACT_RE.finditer(text or ""):
        raw = match.group(0).strip()
        key = _norm(raw)
        if key and key not in seen:
            seen.add(key)
            facts.append(raw)
    return facts


def extract_key_terms(text: str) -> list[str]:
    """抽取答案的关键术语（比事实更宽）：事实 + 英文词 + 中文切出的词。

    中文没有分词器（本模块坚持零依赖），所以用「功能字切分 + 长度≥2」做粗糙切词。
    这只是覆盖度的**近似**，不是真正的实体识别。
    """
    seen: set[str] = set()
    terms: list[str] = []

    def _add(value: str) -> None:
        value = value.strip()
        key = _norm(value)
        if len(key) < 2 or key in seen:
            return
        seen.add(key)
        terms.append(value)

    for fact in extract_facts(text or ""):
        _add(fact)
    for word in _LATIN_RE.findall(text or ""):
        _add(word)
    for run in _CJK_RUN_RE.findall(text or ""):
        for segment in _CJK_SPLIT_RE.split(run):
            if len(segment) >= 2:
                _add(segment)
    return terms


# ── 上下文归一 ──────────────────────────────────────────────────────


def normalize_contexts(contexts: Iterable[Any] | None) -> list[str]:
    """把多种形态的上下文统一成 `list[str]`。

    接受：`list[str]`、`list[dict]`（取 `text` 字段，即 `Answer.contexts` 的形态）、
    或单个字符串。
    """
    if contexts is None:
        return []
    if isinstance(contexts, str):
        return [contexts] if contexts.strip() else []
    out: list[str] = []
    for item in contexts:
        if isinstance(item, str):
            if item.strip():
                out.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            if str(text).strip():
                out.append(str(text))
        else:
            if item is None:  # None 表示"没有上下文"，不能变成字符串 "None"
                continue
            text = str(item)
            if text.strip():
                out.append(text)
    return out


# ── 两个近似指标 ────────────────────────────────────────────────────


def context_faithfulness_approx(answer: str, contexts: Iterable[Any] | None) -> dict[str, Any]:
    """`context_faithfulness_approx`：答案里的可核验事实能否在上下文中找到。

    :return: dict，含 `score`（支持比例；答案没有可核验事实时为 ``None``）、
             `checkable`、`supported`、`suspected_hallucinations`。
    """
    ctx_list = normalize_contexts(contexts)
    norm_ctx = _norm("\n".join(ctx_list))
    facts = extract_facts(answer or "")

    supported: list[str] = []
    hallucinated: list[str] = []
    for fact in facts:
        (supported if _norm(fact) in norm_ctx else hallucinated).append(fact)

    score = (len(supported) / len(facts)) if facts else None
    return {
        "score": score,
        "checkable": bool(facts),
        "fact_count": len(facts),
        "supported": supported,
        "suspected_hallucinations": hallucinated,
        "context_count": len(ctx_list),
    }


def context_coverage_approx(answer: str, contexts: Iterable[Any] | None) -> dict[str, Any]:
    """`context_coverage_approx`：上下文中包含答案关键术语的比例。

    与 faithfulness 的区别：术语集合比「事实」宽（含英文词与中文词），
    衡量的是上下文对答案的**支撑广度**，而不是精确数字是否被篡改。
    """
    ctx_list = normalize_contexts(contexts)
    norm_ctx = _norm("\n".join(ctx_list))
    terms = extract_key_terms(answer or "")

    covered = [t for t in terms if _norm(t) in norm_ctx]
    missing = [t for t in terms if _norm(t) not in norm_ctx]
    score = (len(covered) / len(terms)) if terms else None
    return {
        "score": score,
        "term_count": len(terms),
        "covered": covered,
        "missing": missing,
        "context_count": len(ctx_list),
    }


# ── 评测集 ──────────────────────────────────────────────────────────

#: 允许的评测集条目字段
_ITEM_FIELDS = ("question", "ground_truth", "expected_sources", "sources", "expect_no_answer", "contexts")


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """加载评测集，支持 JSON 数组 / `{"items": [...]}` / JSONL。

    条目字段（全部可选，除 question）：
        question          必填
        ground_truth      参考答案（RAGAS 的 context_recall 需要；离线近似不强制）
        expected_sources  期望来源文件名（与 eval.py 的检索评测同一套写法）
        expect_no_answer  该题应当拒答
        contexts          直接给定上下文（跳过检索，便于离线复算）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测集不存在:{path}")
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        records = data.get("items", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f"评测集格式不正确:{path}（应为数组或 {{'items': [...]}}）")

    items: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not str(record.get("question", "")).strip():
            raise ValueError(f"评测集第 {index + 1} 条缺少 question 字段")
        item = {k: record[k] for k in _ITEM_FIELDS if k in record}
        item["question"] = str(record["question"]).strip()
        item.setdefault("expect_no_answer", False)
        sources = item.get("expected_sources") or item.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        item["expected_sources"] = list(sources)
        items.append(item)
    return items


# ── 答案来源 ────────────────────────────────────────────────────────


def build_extractive_answer(chunks: Sequence[Any], max_chars: int = 400) -> str:
    """离线抽取式「答案」基线：直接拼接检索到的原文片段，**不调用任何大模型**。

    它的用途是：在没有大模型、也不想联网的环境里，让生成质量指标有一个
    确定性的被测量对象（也便于把「检索好但生成差」与「检索本身就差」区分开）。
    它不是产品答案，报告里会标明 `answer_mode=extractive`。
    """
    pieces: list[str] = []
    total = 0
    for chunk in chunks:
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text", "")
        text = str(text or "").strip()
        if not text:
            continue
        if total + len(text) > max_chars and pieces:
            break
        pieces.append(text)
        total += len(text)
    return "\n".join(pieces)


def _answer_and_contexts(
    pipeline: Any,
    question: str,
    answer_mode: str,
    top_n: int | None,
) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]:
    """返回 (answer, contexts, citations, extra_debug)。"""
    if answer_mode == "extractive":
        chunks = pipeline.search(question, top_n=top_n) if top_n else pipeline.search(question)
        contexts = [c.chunk.text for c in chunks]
        answer = build_extractive_answer([c.chunk for c in chunks])
        citations = [
            {
                "index": i,
                "source": c.chunk.source,
                "chunk_index": c.chunk.chunk_index,
                "heading": c.chunk.heading,
                "score": round(float(c.score), 4),
            }
            for i, c in enumerate(chunks, start=1)
        ]
        return answer, contexts, citations, {"grounded": None, "reason": "extractive_baseline"}

    answer_obj = pipeline.query(question)
    contexts = normalize_contexts(getattr(answer_obj, "contexts", None))
    if not contexts:
        # 拒答（或没有依据）时 Answer.contexts 为空；此时退回检索结果，
        # 让「答案是否忠于上下文」这件事在拒答场景下也仍然可核验。
        contexts = [c.chunk.text for c in pipeline.search(question)]
    citations = list(getattr(answer_obj, "citations", []) or [])
    debug = dict(getattr(answer_obj, "debug", {}) or {})
    debug["grounded"] = bool(getattr(answer_obj, "grounded", False))
    return str(getattr(answer_obj, "answer", "") or ""), contexts, citations, debug


# ── 评测主流程 ──────────────────────────────────────────────────────


def evaluate_offline(
    items: Sequence[dict[str, Any]],
    pipeline: Any,
    *,
    answer_mode: str = "pipeline",
    top_n: int | None = None,
    eval_set: str | None = None,
) -> dict[str, Any]:
    """对每条问题跑一遍，产出**离线近似**报告（结构与 RAGAS 报告同形，便于对比）。"""
    if answer_mode not in ("pipeline", "extractive"):
        raise ValueError(f"未知的 answer_mode {answer_mode!r}，可选:pipeline / extractive")

    started = time.time()
    per_query: list[dict[str, Any]] = []
    faith_scores: list[float] = []
    cover_scores: list[float] = []
    uncheckable = 0
    answers_with_hallucination = 0
    checkable_answers = 0
    refusal_total = 0
    refusal_correct = 0

    for item in items:
        question = item["question"]
        answer, contexts, citations, debug = _answer_and_contexts(pipeline, question, answer_mode, top_n)
        faith = context_faithfulness_approx(answer, contexts)
        cover = context_coverage_approx(answer, contexts)

        row: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "answer_chars": len(answer),
            "context_count": len(contexts),
            "citations": citations,
            "expected_sources": item.get("expected_sources", []),
            "ground_truth": item.get("ground_truth", ""),
            "context_faithfulness_approx": faith["score"],
            "context_coverage_approx": cover["score"],
            "suspected_hallucinations": faith["suspected_hallucinations"],
            "missing_terms": cover["missing"][:20],
            "debug": debug,
        }

        if item.get("expect_no_answer"):
            refusal_total += 1
            # 与 eval.py 同一口径：拒答判定看 grounded（无依据）或答案里带拒答标记
            refused = (debug.get("grounded") is False) or (not answer.strip())
            row["kind"] = "no_answer"
            row["refused"] = bool(refused)
            if refused:
                refusal_correct += 1
        else:
            row["kind"] = "answerable"
            if faith["checkable"]:
                checkable_answers += 1
                faith_scores.append(float(faith["score"]))
                if faith["suspected_hallucinations"]:
                    answers_with_hallucination += 1
                if cover["score"] is not None:
                    cover_scores.append(float(cover["score"]))
            else:
                uncheckable += 1

        per_query.append(row)

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    notes: list[str] = []
    if answer_mode == "extractive":
        # 抽取式基线的答案就是从检索原文拼出来的，所以 faithfulness/coverage 必然接近 100%。
        # 这只能证明「评测链路是通的」，不能说明生成质量好 —— 必须说清楚，否则会误导读者。
        notes.append(
            "answer_mode=extractive：答案由检索到的原文直接拼接，因此两个指标天然接近 100%，"
            "只能证明评测链路可用，不能说明生成质量好。"
        )
        # 抽取式基线不会拒答，拒答准确率在此模式下无意义 —— 记 N/A，而不是记 0%（后者会被误读成失败）
        notes.append("answer_mode=extractive 不会拒答，故拒答准确率记 N/A。")
        refusal_accuracy: float | None = None
    else:
        refusal_accuracy = round(refusal_correct / refusal_total, 4) if refusal_total else None

    report: dict[str, Any] = {
        "report_type": "offline_approximation",
        "is_ragas": False,
        "not_ragas": True,
        "disclaimer": OFFLINE_DISCLAIMER,
        "ragas_metrics_not_included": list(RAGAS_METRIC_NAMES),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 3),
        "eval_set": eval_set,
        "kb": getattr(pipeline, "kb", None),
        "answer_mode": answer_mode,
        "item_count": len(items),
        "metrics": {
            "context_faithfulness_approx": _avg(faith_scores),
            "context_coverage_approx": _avg(cover_scores),
            "suspected_hallucination_rate": (
                round(answers_with_hallucination / checkable_answers, 4) if checkable_answers else None
            ),
            "refusal_accuracy": refusal_accuracy,
        },
        "notes": notes,
        "counts": {
            "answerable": sum(1 for r in per_query if r["kind"] == "answerable"),
            "no_answer": refusal_total,
            "checkable_answers": checkable_answers,
            "uncheckable_answers": uncheckable,
        },
        "config": {
            "embedding": getattr(getattr(pipeline, "embedder", None), "fingerprint", None),
            "min_score": getattr(getattr(pipeline, "config", None), "min_score", None),
            "top_n": top_n,
        },
        "per_query": per_query,
    }
    return report


def format_offline_report(report: dict[str, Any]) -> str:
    """把报告格式化成可直接贴进文档的控制台摘要。"""
    metrics = report.get("metrics", {})
    lines = [
        "── 生成质量离线近似评测（非 RAGAS）──",
        f"知识库: {report.get('kb')}   答案来源: {report.get('answer_mode')}",
        f"样本数: {report.get('item_count')}   耗时: {report.get('elapsed_seconds')}s",
        "",
        f"context_faithfulness_approx : {_fmt(metrics.get('context_faithfulness_approx'))}",
        f"context_coverage_approx     : {_fmt(metrics.get('context_coverage_approx'))}",
        f"疑似幻觉答案比例            : {_fmt(metrics.get('suspected_hallucination_rate'))}",
        f"拒答准确率                  : {_fmt(metrics.get('refusal_accuracy'), na='N/A（本模式不适用）')}",
        "",
        f"⚠️  {report.get('disclaimer')}",
        "",
    ]
    for note in report.get("notes", []):
        lines.append(f"注意: {note}")
    if report.get("notes"):
        lines.append("")
    lines.append("逐条结果:")
    for row in report.get("per_query", []):
        mark = "[拒答]" if row.get("kind") == "no_answer" else "[答案]"
        halluc = row.get("suspected_hallucinations") or []
        extra = f"  疑似幻觉: {halluc}" if halluc else ""
        lines.append(
            f"  {mark} faith={_fmt(row.get('context_faithfulness_approx'))} "
            f"cover={_fmt(row.get('context_coverage_approx'))} {row.get('question')}{extra}"
        )
    return "\n".join(lines)


def _fmt(value: Any, na: str = "N/A（无可核验内容）") -> str:
    if value is None:
        return na
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.eval_offline",
        description="生成质量「离线近似」评测（零依赖，不是 RAGAS）",
    )
    parser.add_argument("-e", "--eval-set", required=True, help="评测集 JSON / JSONL 路径")
    parser.add_argument("-k", "--kb", default=None, help="知识库名称，缺省用配置里的默认库")
    parser.add_argument("-o", "--out", default=None, help="报告输出路径（JSON）")
    parser.add_argument(
        "--answer-mode",
        choices=("pipeline", "extractive"),
        default="pipeline",
        help="pipeline=调用 RagPipeline.query（需要大模型）；extractive=只用检索原文拼答案（完全离线，默认作为无大模型环境的基线）",
    )
    parser.add_argument("--top-n", type=int, default=None, help="送入评测的片段条数")
    args = parser.parse_args(argv)

    from app.rag.pipeline import get_pipeline

    items = load_eval_set(args.eval_set)
    pipeline = get_pipeline(kb=args.kb)
    report = evaluate_offline(
        items, pipeline, answer_mode=args.answer_mode, top_n=args.top_n, eval_set=str(args.eval_set)
    )

    print(format_offline_report(report))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入:{out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
