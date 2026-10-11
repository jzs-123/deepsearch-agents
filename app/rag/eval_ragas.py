"""RAGAS 生成质量评测脚本（**可选依赖，不是项目核心依赖**）。

本脚本回答一个 `app/rag/eval.py` 回答不了的问题：

    「检索到的上下文是对的，但**模型生成的答案**是否忠于上下文、是否切题？」

`app/rag/eval.py` 测的是**检索层**（Hit@k / MRR / Recall@k / 拒答率）——它只看
"该召回的资料有没有被召回"，完全不看答案。RAGAS 补的正是**生成层**这一半。
两者互补，不要互相替代（分工见 `docs/rag-eval-guide.md`）。

依赖说明
--------
RAGAS 依赖 `datasets`，而 `datasets` 常会牵出 `sentence-transformers` / torch 系依赖，
在这些装不上的环境里（例如本项目开发所在的 Windows 沙箱，pip 会空转）**跑不了**。
因此：

* 本脚本**不把 ragas/datasets 写进 pyproject.toml**，属于"用户自己机器上可选安装"；
* 依赖缺失时给出中文安装指引，而不是裸抛 ImportError；
* 装不上时请改用零依赖近似：`python -m app.rag.eval_offline`（**它不是 RAGAS**，见其模块文档）。

用法（在装了 ragas 的机器上）
------------------------------
    pip install ragas datasets                     # 用户侧可选安装
    python -m app.rag.eval_ragas -e examples/ragas_eval_set.example.json -k fin -o output/ragas.json

安全
----
* **不硬编码任何密钥**：judge 的 base_url / api_key / 模型名一律读 `.env`（`settings`）
  或环境变量覆盖（`RAGAS_LLM_MODEL` / `RAGAS_JUDGE_BASE_URL` / `RAGAS_JUDGE_API_KEY`）；
* 报告与控制台**只输出脱敏后的 key**（如 `sk-1***cdef`），绝不打印原始密钥。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from app.rag.eval_offline import normalize_contexts

#: 必装依赖（pip 名）
REQUIRED_PACKAGES = ("datasets", "ragas")
#: 缺依赖时给出的安装命令
INSTALL_HINT = "pip install ragas datasets"
#: 默认评测指标
DEFAULT_METRICS = ("faithfulness", "answer_relevancy")
#: 需要 ground_truth 才能算的指标
GROUND_TRUTH_METRICS = ("context_precision", "context_recall")
ALL_METRICS = DEFAULT_METRICS + GROUND_TRUTH_METRICS

#: 允许用环境变量覆盖 judge 配置
ENV_LLM_MODEL = "RAGAS_LLM_MODEL"
ENV_JUDGE_BASE_URL = "RAGAS_JUDGE_BASE_URL"
ENV_JUDGE_API_KEY = "RAGAS_JUDGE_API_KEY"


# ── 依赖检查 ────────────────────────────────────────────────────────


def missing_packages() -> list[str]:
    """返回缺失的必装依赖名列表（可被测试 monkeypatch 影响，见 tests）。"""
    missing: list[str] = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - 任何导入失败都算"装不上"
            missing.append(name)
    return missing


def dependency_message(missing: Sequence[str] | None = None) -> str:
    """构造清晰的中文安装指引（缺依赖时唯一应该给用户看的东西）。"""
    missing = list(missing if missing is not None else missing_packages())
    listed = "、".join(missing) if missing else "（未知）"
    return (
        f"缺少依赖:{listed}。\n"
        f"RAGAS 属于「用户侧可选依赖」，本项目 pyproject.toml 不包含它。\n"
        f"请在你要跑评测的机器上安装:\n"
        f"    {INSTALL_HINT}\n"
        f"若 RAGAS 需要本地向量模型 / 中文分词，可能还要:\n"
        f"    pip install sentence-transformers\n"
        f"并在 .env 里配好 judge 大模型（OPENAI_BASE_URL / OPENAI_API_KEY / LLM_DEEPSEEK 或 LLM_QWEN_MAX）。\n"
        f"装好后重跑:\n"
        f"    python -m app.rag.eval_ragas -e examples/ragas_eval_set.example.json -k fin -o output/ragas.json\n"
        f"相关文档:docs/rag-eval-guide.md\n"
        f"暂时装不上的话，可先用零依赖近似指标（明确标注为非 RAGAS）:\n"
        f"    python -m app.rag.eval_offline -e examples/ragas_eval_set.example.json -k fin --answer-mode extractive\n"
    )


def check_ragas_available() -> tuple[bool, str]:
    """返回 (是否可用, 提示信息)。可用时提示信息是一句确认。"""
    missing = missing_packages()
    if missing:
        return False, dependency_message(missing)
    return True, "ragas 与 datasets 已安装，可以运行生成质量评测。"


def require_ragas() -> None:
    """依赖缺失时抛 RuntimeError（带中文安装指引），而不是让 ImportError 裸奔。"""
    missing = missing_packages()
    if missing:
        raise RuntimeError(dependency_message(missing))


# ── 密钥脱敏 ────────────────────────────────────────────────────────


def mask_secret(value: str | None) -> str:
    """脱敏展示密钥：只保留头 4 位与尾 4 位。绝不返回完整密钥。"""
    if not value:
        return "(未配置)"
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}{'*' * 6}{text[-4:]}"


def resolve_judge_config() -> dict[str, Any]:
    """解析 judge 大模型配置：环境变量优先，其次 `.env`（settings），最后给个保守默认。

    ⚠️ 返回值里含**原始 api_key**，仅供构造客户端使用；
    放进报告/日志前必须先过 :func:`public_judge_config`。
    """
    from app.core.config import settings

    model = os.environ.get(ENV_LLM_MODEL) or getattr(settings, "llm_model", "") or "gpt-4o-mini"
    base_url = os.environ.get(ENV_JUDGE_BASE_URL) or getattr(settings, "openai_base_url", "") or ""
    api_key = os.environ.get(ENV_JUDGE_API_KEY) or getattr(settings, "openai_api_key", "") or ""
    return {
        "model": str(model),
        "base_url": str(base_url),
        "api_key": str(api_key),
        "temperature": 0.0,
    }


def public_judge_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """脱敏后的 judge 配置，可以安全写进报告 / 日志 / 控制台。"""
    config = config if config is not None else resolve_judge_config()
    return {
        "model": config.get("model", ""),
        "base_url": config.get("base_url", ""),
        "api_key": mask_secret(config.get("api_key")),
        "api_key_configured": bool(config.get("api_key")),
    }


# ── 评测集 ──────────────────────────────────────────────────────────

#: 允许的条目字段
_ITEM_FIELDS = ("question", "ground_truth", "expected_sources", "sources", "expect_no_answer")


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    """加载 RAGAS 评测集：JSON 数组 / `{"items": [...]}` / JSONL。

    条目字段：
        question          必填
        ground_truth      参考答案（context_precision/context_recall 需要）
        expected_sources  期望来源（与 eval.py 同一套写法，便于两边共用一份评测集）
        expect_no_answer  该题应当拒答（拒答题不参与生成质量打分）
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
        item.setdefault("ground_truth", "")
        item.setdefault("expect_no_answer", False)
        sources = item.get("expected_sources") or item.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        item["expected_sources"] = list(sources)
        items.append(item)
    return items


# ── 采样：从 RagPipeline 取 answer + contexts ───────────────────────


def collect_rows(
    pipeline: Any,
    items: Sequence[dict[str, Any]],
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """对每条可答题调用 `RagPipeline.query()`，收集 RAGAS 需要的一行记录。

    拒答题（`expect_no_answer=True`）不参与生成质量打分，会被跳过并记录原因
    —— 拿"知识库中没有找到相关信息"去算 faithfulness 是没有意义的。
    """
    rows: list[dict[str, Any]] = []
    for item in items:
        if item.get("expect_no_answer"):
            rows.append(
                {
                    "question": item["question"],
                    "answer": "",
                    "contexts": [],
                    "ground_truth": item.get("ground_truth", ""),
                    "skipped_reason": "expect_no_answer：应拒答的题不参与生成质量评测",
                }
            )
            continue

        answer_obj = pipeline.query(item["question"])
        answer = str(getattr(answer_obj, "answer", "") or "")
        contexts = normalize_contexts(getattr(answer_obj, "contexts", None))
        if not contexts:
            # 拒答时 contexts 为空：退回检索结果，保证 faithfulness 仍有可核验的上下文
            contexts = [c.chunk.text for c in pipeline.search(item["question"])]
        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item.get("ground_truth", ""),
                "citations": list(getattr(answer_obj, "citations", []) or []),
                "grounded": bool(getattr(answer_obj, "grounded", False)),
            }
        )
    return rows


# ── 组装 RAGAS 记录（纯函数，可独立测试）────────────────────────────


@dataclass
class RagasRecords:
    """RAGAS 数据集 + 被跳过的条目。"""

    dataset: dict[str, list] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)
    has_ground_truth: bool = False

    @property
    def size(self) -> int:
        return len(self.dataset.get("question", []))


def build_ragas_records(rows: Sequence[dict[str, Any]]) -> RagasRecords:
    """把采样行转成 RAGAS 需要的数据集字典。

    规则（RAGAS 的硬性要求）：
    * `contexts` 必须是 `list[list[str]]`，且**每行都不能为空** —— 没有上下文就算不了
      faithfulness，所以这类行会被跳过并记录原因，而不是塞空列表让 RAGAS 报错；
    * `ground_truth` 只有在**所有保留行都提供**时才会带上（`Dataset.from_dict` 要求各列等长，
      缺列的条目会让它直接抛错）；缺 ground_truth 时自动降级为只用 faithfulness / answer_relevancy。
    """
    dataset: dict[str, list] = {"question": [], "answer": [], "contexts": []}
    ground_truths: list[str] = []
    skipped: list[dict[str, str]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        if row.get("skipped_reason"):
            skipped.append({"question": question, "reason": str(row["skipped_reason"])})
            continue
        if not question:
            skipped.append({"question": "", "reason": "缺少 question"})
            continue
        contexts = [c for c in normalize_contexts(row.get("contexts")) if c.strip()]
        if not contexts:
            skipped.append({"question": question, "reason": "没有检索到上下文，无法评测生成质量"})
            continue
        answer = str(row.get("answer", "") or "").strip()
        if not answer:
            skipped.append({"question": question, "reason": "答案为空"})
            continue

        dataset["question"].append(question)
        dataset["answer"].append(answer)
        dataset["contexts"].append(contexts)
        ground_truths.append(str(row.get("ground_truth", "") or "").strip())

    has_ground_truth = bool(dataset["question"]) and all(ground_truths)
    if has_ground_truth:
        dataset["ground_truth"] = ground_truths
    elif any(ground_truths):
        missing = sum(1 for g in ground_truths if not g)
        skipped.append({"question": "(评测集整体)", "reason": f"有 {missing} 条缺 ground_truth，已整体省略该列"})

    return RagasRecords(dataset=dataset, skipped=skipped, has_ground_truth=has_ground_truth)


def resolve_metric_names(requested: Sequence[str] | None, has_ground_truth: bool) -> list[str]:
    """按可用性裁剪指标：缺 ground_truth 时自动去掉 context_precision/context_recall。"""
    names = list(requested or DEFAULT_METRICS)
    unknown = [n for n in names if n not in ALL_METRICS]
    if unknown:
        raise ValueError(f"未知的指标 {unknown}，可选:{list(ALL_METRICS)}")
    if not has_ground_truth:
        names = [n for n in names if n not in GROUND_TRUTH_METRICS]
    if not names:
        names = list(DEFAULT_METRICS)
    return names


# ── 跑 RAGAS（依赖隔离在这一层）─────────────────────────────────────


def _load_metric_objects(names: Sequence[str]) -> tuple[list[Any], list[str]]:
    """加载 ragas 指标对象；不同版本可能缺个别指标，缺的记入 unavailable。"""
    from ragas import metrics as ragas_metrics  # 依赖缺失由调用方先检查

    loaded: list[Any] = []
    unavailable: list[str] = []
    for name in names:
        metric = getattr(ragas_metrics, name, None)
        if metric is None:
            unavailable.append(name)
        else:
            loaded.append(metric)
    return loaded, unavailable


def _build_judge(config: dict[str, Any]) -> tuple[Any, Any, str]:
    """尽力构造 RAGAS 的 judge LLM/embeddings，失败则交回 RAGAS 默认（读环境变量）。

    RAGAS 各版本构造方式差异较大，这里逐级尝试；**任何一种失败都不致命**，
    最差情况下 RAGAS 会用它自己的 OpenAI 默认客户端（从环境变量取 key）。
    """
    # 把配置落到环境变量里，是最跨版本的做法：ragas 默认客户端会自己读
    if config.get("api_key"):
        os.environ.setdefault("OPENAI_API_KEY", str(config["api_key"]))
    if config.get("base_url"):
        os.environ.setdefault("OPENAI_BASE_URL", str(config["base_url"]))

    judge_llm = None
    judge_embeddings = None
    note = "使用 RAGAS 默认 judge（读 OPENAI_* 环境变量）"
    try:  # 新版/旧版 ragas 的包装类都可能不带
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        chat = ChatOpenAI(
            model=config.get("model") or "gpt-4o-mini",
            api_key=config.get("api_key") or "not-needed",
            base_url=config.get("base_url") or None,
            temperature=float(config.get("temperature", 0.0)),
        )
        embeddings = OpenAIEmbeddings(
            model=os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-v3"),
            api_key=config.get("api_key") or "not-needed",
            base_url=config.get("base_url") or None,
        )
        try:
            from ragas.llms import LangchainLLMWrapper

            judge_llm = LangchainLLMWrapper(chat)
        except Exception:  # noqa: BLE001
            judge_llm = chat
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper

            judge_embeddings = LangchainEmbeddingsWrapper(embeddings)
        except Exception:  # noqa: BLE001
            judge_embeddings = embeddings
        note = "已用 .env / 环境变量构造 judge LLM 与 embeddings"
    except Exception as exc:  # noqa: BLE001
        note = f"构造 judge 失败，回退 RAGAS 默认 judge:{type(exc).__name__}"
    return judge_llm, judge_embeddings, note


def run_evaluation(
    records: RagasRecords,
    metric_names: Sequence[str],
    *,
    evaluator: Callable[..., Any] | None = None,
    judge_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 RAGAS 评测。

    :param evaluator: 注入点。默认用 `ragas.evaluate`；测试里可传 stub，**永不真的调用大模型**。
    """
    if records.size == 0:
        raise ValueError("没有可评测的记录（全部被跳过）：请检查检索是否返回了上下文。")

    judge_config = judge_config if judge_config is not None else resolve_judge_config()
    notes: list[str] = []
    unavailable: list[str] = []

    if evaluator is None:
        require_ragas()
        from datasets import Dataset

        from ragas import evaluate as evaluator  # type: ignore[no-redef]

        dataset = Dataset.from_dict(records.dataset)
        metric_objects, unavailable = _load_metric_objects(metric_names)
        judge_llm, judge_embeddings, judge_note = _build_judge(judge_config)
        notes.append(judge_note)
        kwargs: dict[str, Any] = {}
        if judge_llm is not None:
            kwargs["llm"] = judge_llm
        if judge_embeddings is not None:
            kwargs["embeddings"] = judge_embeddings
        result = evaluator(dataset, metrics=metric_objects, **kwargs)
    else:
        metric_objects = list(metric_names)
        result = evaluator(records.dataset, metric_names)

    return result_to_report(result, records, metric_names, notes, unavailable, judge_config)


def _as_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def result_to_report(
    result: Any,
    records: RagasRecords,
    metric_names: Sequence[str],
    notes: Sequence[str] = (),
    unavailable: Sequence[str] = (),
    judge_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 RAGAS 返回值归一化成报告（跨版本容错：dict / to_pandas / scores 都试）。"""
    scores: dict[str, Any] = {}

    if isinstance(result, dict):
        scores = dict(result)
    else:
        try:  # 新版 EvaluationResult 支持 dict 转换
            scores = dict(result)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            scores = {}
        if not scores:
            try:
                series = getattr(result, "scores", None)
                if isinstance(series, list):  # [{...}, ...] 逐条
                    keys = {k for row in series if isinstance(row, dict) for k in row}
                    for key in keys:
                        values = [_as_float(row.get(key)) for row in series if isinstance(row, dict)]
                        values = [v for v in values if v is not None]
                        scores[key] = sum(values) / len(values) if values else None
                elif isinstance(series, dict):
                    scores = dict(series)
            except Exception:  # noqa: BLE001
                pass
        if not scores:
            try:  # 退回 pandas（ragas 长期提供的出口）
                frame = result.to_pandas()
                for column in frame.columns:
                    if column in ("question", "answer", "contexts", "ground_truth"):
                        continue
                    scores[column] = _as_float(frame[column].mean())
            except Exception:  # noqa: BLE001
                pass

    metrics = {name: _as_float(scores.get(name)) for name in metric_names}
    metrics = {k: v for k, v in metrics.items() if v is not None}

    report: dict[str, Any] = {
        "report_type": "ragas",
        "is_ragas": True,
        "not_ragas": False,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluated_count": records.size,
        "metrics": metrics,
        "skipped": records.skipped,
        "has_ground_truth": records.has_ground_truth,
        "unavailable_metrics": list(unavailable),
        "notes": list(notes),
        "judge": public_judge_config(judge_config),
        "raw_scores": {k: (None if v is None else str(v)) for k, v in scores.items()},
    }
    if not metrics:
        report["notes"] = list(notes) + [
            "未能从 RAGAS 结果里解析出指标值：可能是 ragas 版本差异，"
            "请把 report['raw_scores'] 贴给维护者，或直接用 result.to_pandas() 自行查看。"
        ]
    return report


def format_summary(report: dict[str, Any]) -> str:
    """控制台摘要（已脱敏）。"""
    judge = report.get("judge", {})
    lines = [
        "── RAGAS 生成质量评测 ──",
        f"评测条数: {report.get('evaluated_count')}   跳过: {len(report.get('skipped', []))}",
        f"judge: {judge.get('model')} @ {judge.get('base_url')}  key={judge.get('api_key')}",
        "",
        "指标:",
    ]
    metrics = report.get("metrics", {})
    if metrics:
        for name, value in metrics.items():
            lines.append(f"  {name:<20} {value:.4f}")
    else:
        lines.append("  （无：见 report['notes']）")
    if report.get("unavailable_metrics"):
        lines.append(f"\n当前 ragas 版本不提供这些指标:{report['unavailable_metrics']}")
    if report.get("skipped"):
        lines.append("\n跳过的条目:")
        for item in report["skipped"][:10]:
            lines.append(f"  - {item['question']}: {item['reason']}")
    for note in report.get("notes", []):
        lines.append(f"\n提示: {note}")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.eval_ragas",
        description="RAGAS 生成质量评测（可选依赖：pip install ragas datasets）",
    )
    parser.add_argument("-e", "--eval-set", default=None, help="评测集 JSON / JSONL")
    parser.add_argument("-k", "--kb", default=None, help="知识库名称")
    parser.add_argument("-o", "--out", default=None, help="报告输出路径（JSON）")
    parser.add_argument("--top-n", type=int, default=None, help="送入评测的片段条数")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=f"逗号分隔，可选:{','.join(ALL_METRICS)}（缺 ground_truth 时自动省略需要参考答案的指标）",
    )
    parser.add_argument("--check", action="store_true", help="只检查依赖是否可用，不做评测")
    args = parser.parse_args(argv)

    # --check 必须先于「必填参数」校验：它的用途正是在装不上依赖时也能得到明确提示
    if args.check:
        available, message = check_ragas_available()
        print(message)
        return 0 if available else 2

    if not args.eval_set:
        parser.error("-e/--eval-set 是必需的（只有 --check 可以省略）")

    available, message = check_ragas_available()
    print(message)
    if not available:
        return 2

    from app.rag.pipeline import get_pipeline

    items = load_eval_set(args.eval_set)
    pipeline = get_pipeline(kb=args.kb)
    rows = collect_rows(pipeline, items, top_n=args.top_n)
    records = build_ragas_records(rows)
    metric_names = resolve_metric_names([m.strip() for m in args.metrics.split(",") if m.strip()], records.has_ground_truth)

    report = run_evaluation(records, metric_names)
    report["eval_set"] = str(args.eval_set)
    report["kb"] = getattr(pipeline, "kb", None)
    report["requested_metrics"] = metric_names

    print(format_summary(report))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入:{out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
