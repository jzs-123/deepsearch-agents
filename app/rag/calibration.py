"""知识库级拒答阈值；仅匹配模型、数据版本和检索配置的标定可使用。"""

import hashlib
import json
import math
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

RETRIEVAL_FIELDS = (
    "chunk_size", "chunk_overlap", "top_k", "top_n", "rrf_k", "sparse_enabled",
    "rerank_mode", "query_rewrite", "query_variants", "retry_on_low_score", "rewrite_synonyms",
)


class CalibrationRequiredError(ValueError):
    pass


def retrieval_settings(config):
    return {field: getattr(config, field) for field in RETRIEVAL_FIELDS}


def apply_calibration(config, kb, meta, fingerprint):
    path = config.kb_dir(kb) / "calibration.json"
    if not config.use_calibration:
        return config, {"status": "disabled", "threshold": config.min_score}
    if not path.exists():
        return config, {"status": "global_default", "threshold": config.min_score}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        threshold = float(record["recommended_threshold"])
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("阈值必须是 [-1,1] 中的有限数字")
        if (record.get("schema_version") != "1.0" or record.get("kb") != kb
                or record.get("embedding_fingerprint") != fingerprint
                or record.get("data_version") != meta.get("data_version", 0)
                or record.get("retrieval_settings") != retrieval_settings(config)):
            raise ValueError("向量模型、数据版本或检索参数已变化")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationRequiredError(f"知识库 {kb} 的标定无效，请重新执行 eval --save-calibration：{exc}") from exc
    return replace(config, min_score=threshold), {"status": "calibrated", "threshold": threshold}


def save_calibration(pipeline, result, eval_set):
    summary = result.threshold_calibration
    if not summary.get("available"):
        raise ValueError("标定至少需要一个可答样例和一个应拒答样例")
    eval_path = Path(eval_set)
    record = {
        "schema_version": "1.0", "kb": pipeline.kb,
        "embedding_fingerprint": pipeline.embedder.fingerprint,
        "data_version": pipeline.store.data_version,
        "retrieval_settings": retrieval_settings(pipeline.config),
        "eval_set": eval_path.name,
        "eval_set_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **summary,
        "metrics": {key: value for key, value in result.to_dict().items() if key != "per_query"},
    }
    path = pipeline.kb_dir / "calibration.json"
    temporary = path.with_name(f".calibration-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return record
