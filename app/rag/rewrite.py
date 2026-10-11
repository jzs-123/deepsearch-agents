"""
查询改写(Query Rewrite):本地规则 + 大模型两级改写

解决的问题是「用户问法」和「文档写法」对不上,而不是检索算法本身有问题:

1. **查询过短**:只问「达标率」「限额」,关键词路只有两个词,向量路(尤其 hash 兜底层)几乎没有信息量;
2. **用词不一致**:用户问「成交比例」,文档写的是「转化率」,词面完全对不上;
3. **口语化**:「请问一下夜间要是没人回咋办」里的「请问/一下/咋办」对检索毫无贡献,还会稀释关键词权重。

两级设计(与配置 `RAG_QUERY_REWRITE` 一一对应):

- 第 1 级 `local`(零成本、默认开启):口语归一化 + 关键词/同义词扩展 + 专有名词与数字强化。
  用 jieba 分词(没装 jieba 时退化为单字+二元组,与 `app/rag/bm25.py` 同一套兜底),词表内置在本模块,
  也可通过 `RAG_REWRITE_SYNONYMS` 从配置补充。
- 第 2 级 `llm`(默认关闭,要花钱):一次调用让大模型给出 N 个检索变体,并附带一段"可能的答案片段"
  (轻量 HyDE 思路:用答案的用词去检索,通常比原问题更接近文档措辞)。

三条硬约束(决定了下面的实现方式):

1. **只增不减召回**:返回列表第一项永远是**原始查询**,变体只往后追加,所以改写开启后召回集合
   一定是原查询结果的超集(最终排序可能变,但不会"因为改写而召回不到")。
2. **绝不阻断查询**:大模型不可用、超时、返回格式乱、变体全被过滤掉……任何异常都只是降级回本地结果,
   由 `ChatClient` 的 timeout + 这里的 try/except 兜住,不向调用方抛异常。
3. **变体数量可控**:`max_variants` 上限(默认 3,含原始查询),避免每个查询都打 3 路稠密 + 3 路稀疏。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from app.rag.bm25 import tokenize
from app.rag.generator import ChatClient

logger = logging.getLogger(__name__)

#: 合法取值:`off` 关闭 / `local` 只用本地规则 / `llm` 只用大模型(失败降级本地) / `local+llm` 两级都用
REWRITE_MODES = ("off", "local", "llm", "local+llm")

# ────────────────────────────── 词表 ──────────────────────────────

#: 口语/疑问句噪声词:去掉后不影响检索语义,却会稀释关键词权重
COLLOQUIAL_NOISE = (
    "请问一下", "请问", "麻烦问一下", "麻烦问下", "麻烦", "想问一下", "想问", "问一下",
    "帮忙看一下", "帮忙看下", "帮忙", "帮我", "看一下", "看下",
    "一下", "到底是什么", "到底", "是多少呢", "是多少", "是什么样", "是什么样的", "是什么",
    "怎么样", "怎么弄", "咋办呢", "咋办", "咋整", "该怎么办", "怎么办", "如何处理", "怎么处理",
    "有没有", "能不能", "可不可以", "可以吗", "行吗", "对吗", "是吗", "呢", "吗", "啊", "呀", "吧",
    "谢谢", "多谢", "感谢", "哈", "哦",
)

#: 同义词/领域词扩展表:key 命中查询时,把 tuple 里的词追加进关键词变体。
#: 设计原则:只放**确定成立**的业务等价词,不放"可能相关"的词——错扩比不扩更伤精度。
SYNONYMS: dict[str, tuple[str, ...]] = {
    # 电商/运营
    "成交比例": ("转化率", "成交转化率"),
    "成交率": ("转化率",),
    "转化": ("转化率",),
    "复购": ("复购率", "回购"),
    "客单价": ("平均订单金额", "单均金额"),
    "拉新": ("新客获取", "获客"),
    # 风控/额度
    "达标率": ("目标完成率", "完成率", "达成率"),
    "达标": ("目标完成率", "完成率", "达成率"),
    "达成率": ("目标完成率", "完成率"),
    "限额": ("授信额度", "额度上限", "单笔上限"),
    "额度": ("授信额度",),
    "准确率": ("精度",),
    "召回率": ("查全率",),
    "坏账": ("不良贷款", "逾期"),
    # 客服/服务
    "没人回": ("无人应答", "未响应"),
    "没人应答": ("无人值守", "无人应答"),
    "没人管": ("无人值守", "无人应答"),
    "夜间": ("非工作时间", "夜间时段"),
    "响应": ("响应时限", "响应时间"),
    "超时": ("超出时限", "升级处理"),
    # 制度/流程
    "报销": ("报销制度", "费用报销"),
    "流程": ("业务流程", "办理流程"),
    "制度": ("管理制度", "规定"),
}

# 专有名词/数字:型号、编号、金额、比例、日期——这些词关键词路最敏感,单独构成一个变体
_PROPER_NOUN_PATTERNS = (
    re.compile(r"[A-Za-z]{1,8}[-_]?\d{2,}[A-Za-z0-9]*"),          # A13 / RTX4090 / IP-2026
    re.compile(r"\d+(?:\.\d+){1,3}"),                              # 3.2.1
    re.compile(r"\d+(?:\.\d+)?\s*(?:%|％)"),                        # 91.5%
    re.compile(r"\d+(?:\.\d+)?\s*(?:万元|亿元|元|美元|万|亿)"),       # 5000 元 / 1.2 亿元
    re.compile(r"\d{4}\s*年(?:\s*\d{1,2}\s*月)?"),                  # 2026 年 3 月
    re.compile(r"\d+(?:\.\d+)?"),                                   # 兜底:裸数字
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|\d+\))\s*")
_WS_RE = re.compile(r"\s+")
# 中文之间不该有空格,但英文/数字之间要有;这里只压缩多空格与首尾空白
_HYDE_PREFIX = ("答案:", "答案：", "假设答案:", "假设答案：", "可能答案:", "可能答案：")

MAX_VARIANT_CHARS = 120

#: 大模型改写的 token 预算。**必须给足**:本项目 .env 里配置的是推理型模型
#: (deepseek-v4-pro),推理过程会先吃掉 completion tokens。实测 max_tokens=400 时
#: finish_reason=length 且 content 为空字符串(全部预算花在 reasoning_content 上),
#: 于是每次都"看起来像大模型坏了"、实际是预算不够。见 docs/rag-query-rewrite.md 第 5 节。
LLM_MAX_TOKENS = 1500


def _clean(text: str) -> str:
    """归一化:压空白、去首尾标点。"""
    text = _WS_RE.sub(" ", (text or "").strip())
    return text.strip(" ,,;；。.、:")


def parse_synonyms(spec: str | None) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """解析配置里的补充同义词表。

    格式:`key=alt1|alt2;key2=alt`(分号分隔条目,竖线分隔同义词,等号分隔 key 与扩展词)。
    解析失败的条目会被忽略并记录原因——配置脏了不该让整条检索链路起不来。

    :return: (合并了内置表的词表, 被忽略条目的说明)
    """
    merged = {key: tuple(values) for key, values in SYNONYMS.items()}
    warnings: list[str] = []
    if not spec:
        return merged, warnings

    for raw_item in str(spec).split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            warnings.append(f"忽略同义词配置条目(缺少 '='):{item}")
            continue
        key, _, values = item.partition("=")
        key = _clean(key)
        alts = tuple(_clean(v) for v in values.split("|") if _clean(v))
        if not key or not alts:
            warnings.append(f"忽略同义词配置条目(key 或扩展词为空):{item}")
            continue
        merged[key] = tuple(dict.fromkeys((*merged.get(key, ()), *alts)))
    return merged, warnings


# ────────────────────────────── 结果对象 ──────────────────────────────


@dataclass
class RewriteResult:
    """一次改写的完整结果(含审计信息)。

    `variants` 是给检索链路用的;其余字段用于 CLI / 日志 / 测试,回答"改写到底做了什么、花了多久"。
    """

    variants: list[str]
    mode: str = "off"
    local_variants: list[str] = field(default_factory=list)
    llm_variants: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_error: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def degraded(self) -> bool:
        """是否发生了「本可以调大模型但失败了,退回本地」的降级。"""
        return bool(self.llm_error) and self.mode in ("llm", "local+llm")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "variants": list(self.variants),
            "variant_count": len(self.variants),
            "local_variants": list(self.local_variants),
            "llm_variants": list(self.llm_variants),
            "llm_used": self.llm_used,
            "llm_error": self.llm_error,
            "degraded": self.degraded,
            "warnings": list(self.warnings),
            "elapsed": round(self.elapsed, 4),
        }


# ────────────────────────────── 改写器 ──────────────────────────────


class QueryRewriter:
    """两级查询改写器:统一入口 `rewrite(question) -> list[str]`。"""

    def __init__(
        self,
        mode: str = "local",
        max_variants: int = 3,
        synonyms: dict[str, tuple[str, ...]] | None = None,
        synonym_spec: str | None = None,
        llm_client: ChatClient | None = None,
        llm_model: str = "",
    ):
        """
        :param mode: off / local / llm / local+llm,非法值会抛 ValueError(与其它配置项一致的"早失败"风格)
        :param max_variants: 返回变体总数上限(**含原始查询**),至少 1
        :param synonyms: 直接注入的词表(测试用)
        :param synonym_spec: 配置里的补充词表字符串,与内置表合并
        :param llm_client: 复用的 `ChatClient`(不可用/为 None 时只能走本地)
        :param llm_model: 大模型名,仅用于日志
        """
        mode = (mode or "off").lower()
        if mode not in REWRITE_MODES:
            raise ValueError(f"未知的查询改写模式 {mode!r},可选:{REWRITE_MODES}")
        if max_variants < 1:
            raise ValueError("max_variants 至少为 1(原始查询本身也算一个变体)")

        self.mode = mode
        self.max_variants = int(max_variants)
        self.llm_client = llm_client
        self.llm_model = llm_model

        parsed, warnings = parse_synonyms(synonym_spec)
        if synonyms:
            for key, values in synonyms.items():
                parsed[key] = tuple(dict.fromkeys((*parsed.get(key, ()), *values)))
        self.synonyms = parsed
        self.config_warnings = warnings

    # ── 对外统一接口 ──

    def rewrite(self, question: str) -> list[str]:
        """返回「原始查询 + 各变体」,原始查询永远在第一项。"""
        return self.rewrite_detailed(question).variants

    def rewrite_detailed(self, question: str) -> RewriteResult:
        """带审计信息的改写。任何异常都在内部收敛,不抛给调用方。"""
        started = time.time()
        question = (question or "").strip()
        result = RewriteResult(variants=[], mode=self.mode, warnings=list(self.config_warnings))

        if not question:
            result.variants = [question]
            result.elapsed = time.time() - started
            return result

        # 原始查询永远第一:保证改写"只增不减召回"
        result.variants = [question]
        if self.mode == "off":
            result.elapsed = time.time() - started
            return result

        local_variants = self._local_variants(question)
        result.local_variants = local_variants

        llm_variants: list[str] = []
        if self.mode in ("llm", "local+llm"):
            llm_variants, llm_error = self._llm_variants(question)
            result.llm_variants = llm_variants
            result.llm_error = llm_error
            result.llm_used = bool(llm_variants)

        # 组装:off 之外一路是「原始 + 本地 + 大模型」,llm 模式下若大模型失败就用本地兜底
        if self.mode == "llm" and not llm_variants:
            merged = [question, *local_variants]
        else:
            merged = [question, *local_variants, *llm_variants]

        result.variants = self._dedupe_cap(merged)
        result.elapsed = time.time() - started
        return result

    # ── 第 1 级:本地规则 ──

    def _local_variants(self, question: str) -> list[str]:
        """本地变体:口语归一化 → 关键词+同义词扩展 → 专有名词/数字强化。"""
        base = _clean(question)
        normalized = self._normalize_colloquial(base)

        candidates: list[str] = []
        if normalized and normalized != base:
            candidates.append(normalized)

        keyword = self._keyword_variant(normalized or base)
        if keyword:
            candidates.append(keyword)

        proper = self._proper_noun_variant(base)
        if proper:
            candidates.append(proper)

        return self._dedupe_cap(candidates, keep=[])  # 只去重,不在这里截断总数

    def _normalize_colloquial(self, text: str) -> str:
        """去掉口语/疑问句噪声词。

        按词长倒序替换,避免「请问一下」被「请问」先切掉后只剩「一下」这种残留;
        替换后可能出现连续空格或悬空标点,统一再清洗一次。
        """
        result = text
        for noise in sorted(COLLOQUIAL_NOISE, key=len, reverse=True):
            if noise in result:
                result = result.replace(noise, " ")
        return _clean(result)

    def _keyword_variant(self, text: str) -> str:
        """关键词 + 同义词扩展变体。

        分词用 `app/rag/bm25.tokenize`(jieba 缺席时自动退化为单字+二元组,与本项目 BM25 同一套逻辑),
        再把命中的同义词扩展词并进来。保留原始词序,便于 BM25 与 hash 向量两边都能匹配。
        """
        tokens = [t for t in tokenize(text) if t]
        expansions: list[str] = []
        lowered = text.lower()
        for key, values in self.synonyms.items():
            if key.lower() in lowered:
                expansions.extend(values)
        if not tokens and not expansions:
            return ""
        # 优先保留多字词;若全是单字(未装 jieba 时的退化态),把它们拼回连续串,
        # 避免产出「成 交 比 例」这种既难读、关键词路也没收益的变体。
        words = [t for t in tokens if len(t) > 1]
        if not words:
            words = ["".join(tokens)]
        joined = " ".join(dict.fromkeys([*words, *expansions]))
        return _clean(joined)

    def _proper_noun_variant(self, text: str) -> str:
        """专有名词/数字强化变体:型号、编号、比例、金额、日期单独成一条查询。"""
        hits: list[str] = []
        for pattern in _PROPER_NOUN_PATTERNS:
            for match in pattern.findall(text):
                value = _clean(match)
                if value and value not in hits:
                    hits.append(value)
        # 全是"裸数字"且只有一个时信息量太低(如「30」),补上它周围的字符片段
        if len(hits) == 1 and re.fullmatch(r"\d+(?:\.\d+)?", hits[0]):
            index = text.find(hits[0])
            window = text[max(0, index - 6) : index + len(hits[0]) + 3]
            window = _clean(window)
            if window and window != hits[0]:
                return window
        if not hits:
            return ""
        return " ".join(hits)

    # ── 第 2 级:大模型 ──

    def _llm_variants(self, question: str) -> tuple[list[str], str]:
        """调用大模型生成变体;返回 (变体, 错误说明)。任何异常都转成错误说明返回。"""
        if self.llm_client is None or not getattr(self.llm_client, "available", False):
            return [], "大模型未配置或不可用(缺少 api_key / model)"

        want = max(1, self.max_variants - 1)
        try:
            content = self.llm_client.complete(
                LLM_SYSTEM_PROMPT,
                LLM_USER_TEMPLATE.format(question=question, count=want),
                max_tokens=LLM_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 - 大模型任何问题都必须降级,不能阻断查询
            logger.warning("大模型查询改写失败,降级为本地改写:%s", exc)
            return [], f"{type(exc).__name__}: {exc}"

        variants = self._parse_llm_output(content, question)
        if not variants:
            logger.warning("大模型查询改写返回内容无法解析,降级为本地改写:%r", (content or "")[:120])
            return [], f"返回内容无法解析:{_clean(content or '')[:80]}"
        return variants, ""

    def _parse_llm_output(self, content: str, question: str) -> list[str]:
        """解析大模型输出:一行一个变体,`答案:` 前缀的那行当作 HyDE 补充。

        容忍常见的格式噪声(项目符号、编号、`查询:` 前缀、代码块围栏、解释性长句)。
        """
        raw = (content or "").replace("```", "\n")
        variants: list[str] = []
        for line in raw.splitlines():
            line = _clean(_BULLET_RE.sub("", line))
            if not line:
                continue
            for prefix in ("查询:", "查询：", "变体:", "变体："):
                if line.startswith(prefix):
                    line = _clean(line[len(prefix) :])
            if not line or line == _clean(question):
                continue
            lowered = line.lower()
            if lowered.startswith(tuple(prefix.lower() for prefix in _HYDE_PREFIX)):
                line = _clean(line.split(":", 1)[-1].split("：", 1)[-1])
                if not line:
                    continue
            # 纯解释性长句(带"因为/所以/因此"且很长)不算变体
            if len(line) > MAX_VARIANT_CHARS and any(word in line for word in ("因为", "因此", "所以")):
                continue
            if line not in variants:
                variants.append(line)
        return variants

    # ── 工具 ──

    def _dedupe_cap(self, items: list[str], keep: list[str] | None = None) -> list[str]:
        """去重 + 按上限截断(保持顺序,第一项是原始查询时不会被截掉)。"""
        result = list(keep or [])
        for item in items:
            item = _clean(item)
            if not item or item in result:
                continue
            result.append(item)
        return result[: self.max_variants]


# ────────────────────────────── 大模型提示词 ──────────────────────────────

LLM_SYSTEM_PROMPT = """你是检索系统的查询改写助手。用户的问题可能很短、很口语化,或者用词与资料不一致。
请把它改写成更适合在企业内部文档库里做关键词/向量检索的形式。

要求:
1. 保留原问题里的专有名词、型号、编号、数字与比例,一个都不能丢;
2. 补充同义说法、领域术语、可能出现在文档里的正式表述(例如「成交比例」→「转化率」);
3. 去掉「请问」「一下」「怎么办」这类对检索没有贡献的口语成分;
4. 最后给出一段"可能的答案片段"(假设你已经在文档里读到了答案,把它写成 1 句话),用于按答案用词检索;
5. 每行输出一条,不要编号、不要解释、不要输出多余文字。"""

LLM_USER_TEMPLATE = """用户问题:{question}

请输出最多 {count} 行检索变体,每行一条;最后一行以「答案:」开头,给出可能的答案片段。"""


def build_rewriter(
    mode: str = "local",
    max_variants: int = 3,
    synonym_spec: str | None = None,
    llm_client: ChatClient | None = None,
    llm_model: str = "",
) -> QueryRewriter:
    """按配置构造改写器(与 `build_reranker` / `build_embedder` 同样的工厂风格)。"""
    return QueryRewriter(
        mode=mode,
        max_variants=max_variants,
        synonym_spec=synonym_spec,
        llm_client=llm_client,
        llm_model=llm_model,
    )


def rewrite_query(
    question: str,
    mode: str = "local",
    max_variants: int = 3,
    synonym_spec: str | None = None,
    llm_client: ChatClient | None = None,
) -> list[str]:
    """一行调用:返回「原始查询 + 变体」,失败也绝不抛异常(返回 [原始查询])。"""
    try:
        rewriter = build_rewriter(
            mode=mode,
            max_variants=max_variants,
            synonym_spec=synonym_spec,
            llm_client=llm_client,
        )
        return rewriter.rewrite(question)
    except Exception as exc:  # noqa: BLE001 - 改写出任何问题都不该让检索失败
        logger.warning("查询改写失败,使用原始查询:%s", exc)
        return [(question or "").strip()]


__all__ = [
    "REWRITE_MODES",
    "SYNONYMS",
    "COLLOQUIAL_NOISE",
    "RewriteResult",
    "QueryRewriter",
    "build_rewriter",
    "rewrite_query",
    "parse_synonyms",
]
