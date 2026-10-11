"""
元数据(Metadata):自动抽取 + 检索期过滤

企业场景下「检索不到想要的片段」很多时候不是召回算法的问题,而是**权限/范围**问题:
同一个知识库里混着各部门、各密级的文档,查询时应该能把范围收窄到"我该看的那些"。
这一层做两件事:

1. **抽取**:入库时给每个 chunk 打上元数据
   - 文件级:`file`(文件名)、`path`(相对路径)、`suffix`(后缀)、`mtime`(修改时间)、`size`(字节数)
   - 内容级:Markdown 头部的 **YAML frontmatter**(`---\\n部门: 风控\\n---`),字段名随文档自定
   - 抽取失败(没有 frontmatter / YAML 语法错 / 不是字典)**只给 warning**,降级为空元数据,
     绝不让一个文档因为元数据问题进不了库。
2. **过滤**:`pipeline.search(..., filter={...})` 的匹配规则,见 `matches_filter()`。
   - 等值:`{"部门": "风控"}`
   - 列表值(命中其一即可):`{"类型": ["制度", "规范"]}`
   - 元数据值是列表时同样按"任一元素命中"处理(例如 `标签: [a, b]`)
   - 比较时忽略大小写与首尾空白;缺字段 = 不匹配(过滤是"白名单"语义)

为什么过滤放在「召回之后、重排之前」:过滤只是把不属于该范围的候选剔除,不改变向量/BM25
的召回机制本身(否则要重建索引),而且放在重排前可以让重排把名额花在真正可用的片段上。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 文件级元数据键;frontmatter 里出现同名键时会被忽略(避免和系统字段打架)
RESERVED_KEYS = ("file", "path", "suffix", "mtime", "size")

#: frontmatter 最大行数/字符数:超过就认为这不是 frontmatter(防止把整篇文档当 YAML 解析)
MAX_FRONTMATTER_LINES = 100
MAX_FRONTMATTER_CHARS = 8192

_OPEN_RE = re.compile(r"^\ufeff?---\s*$")
_CLOSE_RE = re.compile(r"^(?:---|\.\.\.)\s*$")


@dataclass
class MetadataResult:
    """一次元数据抽取的结果。"""

    fields: dict[str, Any] = field(default_factory=dict)
    body: str = ""
    frontmatter: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stripped: bool = False

    @property
    def filter_fields(self) -> list[str]:
        """可用于过滤的字段名(去掉系统保留字段,frontmatter 字段优先展示)。"""
        return sorted(k for k in self.fields if k not in RESERVED_KEYS)


def _load_yaml(text: str) -> Any:
    """解析 YAML;优先用 PyYAML(项目已有依赖),不可用时退化为简单的 key: value 解析。"""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:  # pragma: no cover - 环境确实没有 PyYAML 时的兜底
        return _simple_frontmatter(text)


def _simple_frontmatter(text: str) -> dict[str, Any]:
    """极简兜底解析器:只认 `key: value` 与 `- item` 列表,不认识嵌套结构。"""
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("- ") and current_list_key:
            result.setdefault(current_list_key, [])
            if isinstance(result[current_list_key], list):
                result[current_list_key].append(_clean_scalar(line.lstrip()[2:]))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().strip("\"'")
        value = value.strip()
        if not key:
            continue
        if not value:
            result[key] = []
            current_list_key = key
            continue
        result[key] = _clean_scalar(value)
        current_list_key = None
    return result


def _clean_scalar(value: str) -> Any:
    """去掉引号;能当数字/布尔就当数字/布尔(便于 `版本: 3` 这类过滤)。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def split_frontmatter(text: str) -> MetadataResult:
    """把「YAML frontmatter + 正文」拆开。

    严格判定:第一行必须是 `---`(允许 BOM 与尾随空白),且在前 `MAX_FRONTMATTER_LINES` 行内
    能找到 `---`/`...` 收尾。不满足就当普通正文,**不报错**——很多 Markdown 文章开头就是
    一条分隔线,把它当 frontmatter 会把正文吃掉。
    """
    result = MetadataResult(body=text or "")
    if not text:
        return result

    lines = text.splitlines()
    if not lines or not _OPEN_RE.match(lines[0]):
        return result

    end_index = -1
    for index in range(1, min(len(lines), MAX_FRONTMATTER_LINES + 1)):
        if _CLOSE_RE.match(lines[index]):
            end_index = index
            break
    if end_index < 0:
        result.warnings.append("检测到疑似 frontmatter 起始行,但未找到结束行(---),已按普通正文处理")
        return result

    raw = "\n".join(lines[1:end_index])
    if len(raw) > MAX_FRONTMATTER_CHARS:
        result.warnings.append(
            f"frontmatter 超过 {MAX_FRONTMATTER_CHARS} 字符,已按普通正文处理"
        )
        return result

    try:
        parsed = _load_yaml(raw)
    except Exception as exc:  # noqa: BLE001 - YAML 语法错必须降级,不能拦截入库
        result.warnings.append(f"frontmatter 解析失败({type(exc).__name__}: {exc}),已按普通正文处理")
        return result

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        result.warnings.append(
            f"frontmatter 不是键值对(解析出 {type(parsed).__name__}),已忽略元数据"
        )
        parsed = {}

    fields: dict[str, Any] = {}
    for key, value in parsed.items():
        key = str(key).strip()
        if not key:
            continue
        if key in RESERVED_KEYS:
            result.warnings.append(f"frontmatter 字段 {key!r} 与系统字段同名,已忽略")
            continue
        fields[key] = value

    result.frontmatter = fields
    result.fields = dict(fields)
    result.body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    result.stripped = True
    return result


def file_metadata(path: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """文件级元数据;取不到的字段直接省略(不抛异常)。"""
    path = Path(path)
    fields: dict[str, Any] = {"file": path.name, "suffix": path.suffix.lower()}
    try:
        if base_dir is not None:
            fields["path"] = str(path.relative_to(base_dir)).replace("\\", "/")
        else:
            fields["path"] = path.name
    except ValueError:
        fields["path"] = path.name
    try:
        stat = path.stat()
        fields["size"] = int(stat.st_size)
        fields["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
    except OSError as exc:  # pragma: no cover - 文件刚被删掉等极端情况
        logger.warning("读取文件元数据失败 %s:%s", path, exc)
    return fields


def extract_metadata(
    path: str | Path | None = None,
    text: str = "",
    base_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> MetadataResult:
    """抽取一份文档的完整元数据,并返回剥掉 frontmatter 的正文。"""
    result = split_frontmatter(text)
    fields: dict[str, Any] = {}
    if path is not None:
        fields.update(file_metadata(path, base_dir))
    fields.update(result.frontmatter)
    if extra:
        for key, value in extra.items():
            if key in RESERVED_KEYS:
                continue
            fields[key] = value
    result.fields = fields
    return result


# ────────────────────────────── 过滤 ──────────────────────────────


class InvalidFilterError(ValueError):
    """filter 结构非法(必须是「字段 -> 标量/列表」的字典)。"""


def _norm_scalar(value: Any) -> str:
    """比较用的归一化:转字符串 + 去首尾空白 + 忽略大小写。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value).strip().casefold()


def normalize_filter(filter_spec: dict | None) -> dict[str, list[str]]:
    """把用户给的 filter 规范成 `{字段: [可接受值…]}`。

    - 值是标量 → 单元素列表
    - 值是列表/元组/集合 → 多元素列表(命中其一即可)
    - 空列表 → 视为"该字段不可满足",任何片段都不匹配(而不是被忽略)
    """
    if not filter_spec:
        return {}
    if not isinstance(filter_spec, dict):
        raise InvalidFilterError(
            f"filter 必须是 JSON 对象(如 {{\"部门\": \"风控\"}}),当前是 {type(filter_spec).__name__}"
        )
    normalized: dict[str, list[str]] = {}
    for raw_key, raw_value in filter_spec.items():
        key = str(raw_key).strip()
        if not key:
            raise InvalidFilterError("filter 里存在空字段名")
        if isinstance(raw_value, (list, tuple, set)):
            values = [_norm_scalar(item) for item in raw_value]
        else:
            values = [_norm_scalar(raw_value)]
        normalized[key] = values
    return normalized


def matches_filter(metadata: dict | None, normalized: dict[str, list[str]]) -> bool:
    """片段元数据是否满足过滤条件(白名单语义:缺字段或值不匹配都不算通过)。"""
    if not normalized:
        return True
    metadata = metadata or {}
    for key, accepted in normalized.items():
        if not accepted:
            return False
        if key not in metadata:
            return False
        actual = metadata.get(key)
        actual_values = actual if isinstance(actual, (list, tuple, set)) else [actual]
        if not any(_norm_scalar(item) in accepted for item in actual_values):
            return False
    return True


def apply_filter(candidates: list, normalized: dict[str, list[str]]) -> list:
    """对候选(任何带 `.chunk.metadata` 的对象)做过滤;不过滤时原样返回。"""
    if not normalized:
        return candidates
    return [item for item in candidates if matches_filter(getattr(item.chunk, "metadata", None), normalized)]


def canonical_filter(filter_spec: dict | None) -> str:
    """把 filter 归一化成稳定字符串(缓存键 + 审计日志都用它,保证顺序不影响命中)。"""
    normalized = normalize_filter(filter_spec)
    return json.dumps(
        {key: sorted(values) for key, values in sorted(normalized.items())},
        ensure_ascii=False,
        sort_keys=True,
    )


def parse_filter_json(raw: str | dict | None) -> dict:
    """解析 CLI/工具传来的 filter(JSON 字符串或已是 dict)。非法输入抛 InvalidFilterError。"""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        normalize_filter(raw)  # 触发校验
        return raw
    if not isinstance(raw, str):
        raise InvalidFilterError(f"filter 只接受 JSON 字符串或对象,当前是 {type(raw).__name__}")
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidFilterError(f"filter 不是合法 JSON:{exc}") from exc
    if not isinstance(parsed, dict):
        raise InvalidFilterError(
            f"filter 必须是 JSON 对象(如 {{\"部门\": \"风控\"}}),当前是 {type(parsed).__name__}"
        )
    normalize_filter(parsed)
    return parsed


def filter_fields_of(chunks: list) -> list[str]:
    """扫描所有 chunk 的元数据,汇总"可过滤字段"(去重排序;排除系统保留字段)。"""
    fields: set[str] = set()
    for chunk in chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        if isinstance(metadata, dict):
            fields.update(str(key) for key in metadata)
    return sorted(fields - set(RESERVED_KEYS))


def format_filter_hint(normalized: dict[str, list[str]], available: list[str]) -> str:
    """过滤没命中时给调用方/模型看的提示(把"可选字段"直接写出来)。"""
    wanted = ", ".join(f"{key}={values}" for key, values in sorted(normalized.items()))
    available_text = "、".join(available) if available else "(该知识库没有任何文档元数据字段)"
    return (
        f"没有符合过滤条件的片段(过滤条件:{wanted})。"
        f"当前可用于过滤的字段:{available_text}。"
        f"请检查字段名与取值,或先执行 python -m app.rag.cli list 查看各库的过滤字段。"
    )


__all__ = [
    "RESERVED_KEYS",
    "MetadataResult",
    "InvalidFilterError",
    "split_frontmatter",
    "file_metadata",
    "extract_metadata",
    "normalize_filter",
    "matches_filter",
    "apply_filter",
    "canonical_filter",
    "parse_filter_json",
    "filter_fields_of",
    "format_filter_hint",
]
