"""
审计日志(Audit Log):把每一次 `search` / `query` 的关键过程落成 JSONL,便于事后追责与排障

企业里最常见的三个问题是:
1. **"这条答案是谁查出来的、什么时候查的?"** —— 需要时间戳 + 知识库 + 查询 + 过滤条件;
2. **"为什么这次没查到?"** —— 需要改写变体、是否触发低分重试、判定分、检索结果摘要;
3. **"有没有把不该看的东西给出去?"** —— 需要记录 filter 与命中的来源(不含原文)。

设计取舍:

- **JSONL 追加写**:一条一行,随时可 `tail`,不需要额外依赖(标准库 json + 文件 io);
- **不含完整原文**:只记 chunk_id / source / chunk_index / 分数 / 通道 —— 审计日志一旦带原文,
  既会迅速膨胀,也会把敏感正文复制到另一个文件里(反而扩大泄露面);
- **写入失败绝不影响查询**:目录不可写、磁盘满、文件被锁……统一降级为 warning 并返回 False,
  调用方(检索链路)不感知失败;
- **目录可配置**:默认跟随知识库数据根目录(`<data_dir>/.audit/`,即默认的 `rag_data/.audit/`),
  这样测试与自测脚本用临时数据目录时不会往真实 `rag_data/` 里写东西,也可以通过
  `RAG_AUDIT_DIR` 指到独立的审计盘。
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIT_DIR_NAME = ".audit"

#: 单条日志各字段的长度上限(防止超长 query / 变体把日志写爆)
MAX_TEXT_CHARS = 500
MAX_VARIANTS = 8
MAX_RESULTS = 20


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def _clip(text: object, limit: int = MAX_TEXT_CHARS) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def audit_path(audit_dir: str | Path, kb: str) -> Path:
    """审计文件路径:`<audit_dir>/<kb>.jsonl`。"""
    return Path(audit_dir) / f"{kb}.jsonl"


class AuditLogger:
    """单个知识库的审计写入器(线程内串行追加,不做跨进程锁)。"""

    def __init__(
        self,
        audit_dir: str | Path,
        kb: str,
        enabled: bool = True,
    ):
        self.audit_dir = Path(audit_dir)
        self.kb = kb
        self.enabled = bool(enabled)
        self.failures = 0
        self.writes = 0

    @property
    def path(self) -> Path:
        return audit_path(self.audit_dir, self.kb)

    # ── 写 ──
    def record(self, entry: dict) -> bool:
        """追加一条审计记录;任何异常都只 warning(审计失败不能拖垮查询)。"""
        if not self.enabled:
            return False
        payload = {
            "ts": _now_iso(),
            "kb": self.kb,
            **entry,
        }
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.writes += 1
            return True
        except Exception as exc:  # noqa: BLE001 - 审计是旁路,绝不能抛出
            self.failures += 1
            logger.warning("审计日志写入失败(已忽略,不影响本次查询):%s", exc)
            return False

    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "dir": str(self.audit_dir),
            "path": str(self.path),
            "writes": self.writes,
            "failures": self.failures,
        }


# ────────────────────────────── 记录构造 ──────────────────────────────


def summarize_results(results: list) -> list[dict]:
    """检索结果摘要(不含原文):定位"命中了什么"足够,不把正文复制进日志。"""
    summary: list[dict] = []
    for item in results[:MAX_RESULTS]:
        chunk = getattr(item, "chunk", None)
        summary.append(
            {
                "rank": getattr(item, "rank", 0),
                "chunk_id": getattr(chunk, "chunk_id", ""),
                "source": getattr(chunk, "source", ""),
                "chunk_index": getattr(chunk, "chunk_index", None),
                "score": round(float(getattr(item, "score", 0.0) or 0.0), 4),
                "dense": None
                if getattr(item, "dense_score", None) is None
                else round(float(item.dense_score), 4),
                "sparse": None
                if getattr(item, "sparse_score", None) is None
                else round(float(item.sparse_score), 4),
                "channels": list(getattr(item, "channels", []) or []),
            }
        )
    return summary


def build_search_entry(
    question: str,
    debug: dict,
    filter_spec: dict | None,
    fingerprint: str,
    results: list | None = None,
    cache_info: dict | None = None,
    audit_dir: str = "",
) -> dict:
    """把一次 search 的 `last_search_debug` 折成一条审计记录。

    :param results: 本次返回的结果对象列表(只用于生成**不含原文**的摘要)
    """
    retry = debug.get("retry") or {}
    rewrite = debug.get("rewrite") or {}
    variants = [_clip(item, 200) for item in (debug.get("variants") or [])[:MAX_VARIANTS]]
    return {
        "kind": "search",
        "query": _clip(question),
        "filter": filter_spec or {},
        "rewrite_mode": debug.get("rewrite_mode", "off"),
        "variants": variants,
        "variant_hits": debug.get("variant_hits", [])[:MAX_VARIANTS],
        "llm_rewrite": {
            "used": bool(rewrite.get("llm_used")),
            "error": _clip(rewrite.get("llm_error") or "", 200),
        },
        "retry": {
            "enabled": bool(retry.get("enabled")),
            "triggered": bool(retry.get("triggered")),
            "gate_before": retry.get("gate_before"),
            "gate_after": retry.get("gate_after"),
            "improved": retry.get("improved"),
        },
        "gate_score": debug.get("gate_score"),
        "fused_count": debug.get("fused_count"),
        "filtered_out": debug.get("filtered_out", 0),
        "cache": cache_info or {"used": False},
        "results": summarize_results(results or []),
        "result_count": len(results or debug.get("results") or []),
        "llm": {"called": False},
        "embedding_fingerprint": fingerprint,
        "audit_dir": _clip(audit_dir, 200),
    }


def build_query_entry(
    search_entry: dict,
    answer,
    llm_called: bool,
    elapsed: float,
    cache_info: dict | None = None,
) -> dict:
    """在 search 记录的基础上补齐生成阶段信息(是否进大模型、耗时、拒答原因)。"""
    debug = getattr(answer, "debug", {}) or {}
    entry = dict(search_entry)
    entry["kind"] = "query"
    entry["answer_len"] = len(getattr(answer, "answer", "") or "")
    entry["grounded"] = bool(getattr(answer, "grounded", False))
    entry["citations"] = [
        {
            "index": cite.get("index"),
            "source": cite.get("source"),
            "chunk_index": cite.get("chunk_index"),
            "score": cite.get("score"),
        }
        for cite in (getattr(answer, "citations", []) or [])[:MAX_RESULTS]
    ]
    entry["llm"] = {
        "called": bool(llm_called),
        "elapsed": round(float(elapsed or 0.0), 3),
        "refusal_reason": debug.get("reason") or "",
    }
    entry["cache"] = cache_info or entry.get("cache") or {"used": False}
    return entry


# ────────────────────────────── 读 ──────────────────────────────


def read_audit(
    audit_dir: str | Path,
    kb: str,
    tail: int = 20,
) -> list[dict]:
    """读回最近 N 条审计记录(读取失败返回空列表,不抛异常)。"""
    path = audit_path(audit_dir, kb)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
    except OSError as exc:
        logger.warning("审计日志读取失败:%s", exc)
        return []
    records: list[dict] = []
    for line in lines[-max(0, int(tail)) :]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def format_audit_records(records: list[dict]) -> str:
    """把审计记录渲染成人能读的文本(CLI `audit` 子命令用)。"""
    if not records:
        return "(没有审计记录)"
    lines: list[str] = []
    for record in records:
        head = (
            f"{record.get('ts', '-')} [{record.get('kind', '?')}] kb={record.get('kb', '-')} "
            f"query={record.get('query', '')!r}"
        )
        lines.append(head)
        filter_text = record.get("filter") or {}
        lines.append(
            f"    filter={json.dumps(filter_text, ensure_ascii=False)}"
            f" 改写={record.get('rewrite_mode', '-')}"
            f" 变体={len(record.get('variants') or [])}"
            f" 重试={record.get('retry', {}).get('triggered')}"
            f" 判定分={record.get('gate_score')}"
        )
        cache = record.get("cache") or {}
        if cache.get("used"):
            lines.append(
                f"    缓存:{cache.get('type')} hit={cache.get('hit')} key={cache.get('key', '')[:12]}"
            )
        for item in (record.get("results") or [])[:5]:
            lines.append(
                f"    #{item.get('rank')} {item.get('source')}#{item.get('chunk_index')}"
                f" score={item.get('score')} dense={item.get('dense')}"
                f" 通道={item.get('channels')}"
            )
        llm = record.get("llm") or {}
        if record.get("kind") == "query" and llm:
            elapsed = llm.get("elapsed")
            lines.append(
                f"    生成:called={llm.get('called')}"
                f" 耗时={elapsed if elapsed is not None else '-'}s"
                f" 拒答原因={llm.get('refusal_reason') or '-'}"
            )
        lines.append(f"    指纹={record.get('embedding_fingerprint', '-')}")
    return "\n".join(lines)


__all__ = [
    "AUDIT_DIR_NAME",
    "AuditLogger",
    "audit_path",
    "read_audit",
    "format_audit_records",
    "summarize_results",
    "build_search_entry",
    "build_query_entry",
]
