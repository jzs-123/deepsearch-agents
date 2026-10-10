"""
重排(Rerank)层

混合检索的 Top-K 是「召回」,名次由粗略相似度决定;重排是「精排」,用更精细的信号
对候选重新排序,并把最相关的少数几条送进大模型。这是低成本提升 RAG 答案精度最有效的一步。

提供三种策略,都可以单独开关,便于做 A/B 对比:
1. off      —— 不重排,直接用融合名次(作为 baseline);
2. lexical  —— 本地词面重排:查询特征词覆盖率 + 出现位置 + 片段紧密度。零成本、零延迟;
3. llm      —— 用大模型做列表式(listwise)重排,对语义相关但无字面重叠的候选更敏感,但慢且有成本。

关于 lexical 的取舍:它和 BM25 有相关性(BM25 已经用过词频),所以这里刻意只使用
「查询中区分度高的特征词」「是否完整出现」「出现在片段前部」「多个特征词是否彼此靠近」
这些 BM25 没建模的信号,尽量降低两者的同质化,让重排真正带来增量而不是重复排序。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from app.rag.bm25 import STOPWORDS, tokenize
from app.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """重排器统一接口。"""

    name = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """对候选重排并截断到 top_n。"""


class NoopReranker(Reranker):
    """不重排:保持混合检索的融合名次,作为对照基线。"""

    name = "off"

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        result = candidates[:top_n]
        for rank, item in enumerate(result, start=1):
            item.rank = rank
        return result


class LexicalReranker(Reranker):
    """本地词面重排器。

    打分组成(权重见 WEIGHTS,均为「正向信号」,便于人工解释):
    - coverage:查询特征词(分词结果 + 中文二元组)被覆盖的比例
    - char    :中文有效字符覆盖率,用于兜底分词差异(如查询"准确率" vs 文档"准确率为")
    - exact   :完整短语(相邻特征词原样相邻)出现
    - position:命中词出现在片段靠前的位置
    - density :命中词在片段中的分布密度
    权重改动请同步更新本注释,避免文档与实际打分不一致。
    """

    name = "lexical"

    WEIGHTS = {"coverage": 0.40, "char": 0.25, "exact": 0.20, "position": 0.10, "density": 0.05}
    # 只取前若干个特征词,过多会让 coverage 被无关词稀释
    MAX_FEATURES = 12
    # 中文问句里这些字几乎无区分度,做字符级兜底时要剔除
    CHAR_STOPWORDS = set("的了是在有和与及或吗呢吧啊请问我你他它们这那哪什么如何怎为多少几个条些会可以对被让从到就都也很更最要能需是否以及并且但是如果")

    def __init__(self, stopwords: set[str] | None = None):
        self.stopwords = STOPWORDS if stopwords is None else stopwords

    def _features(self, query: str) -> list[str]:
        """抽取查询特征词:分词结果 + 中文相邻二元组。

        为什么要加二元组:中文分词有歧义,查询里的「准确率」可能被切成「准确/率」,
        而文档里写的是「准确率为 91.5%」,只按分词 token 做子串匹配会一条都命中不了,
        重排分数全为 0、名次完全不变(本项目实测踩到过)。二元组能绕开分词结果的分歧。
        """
        tokens = tokenize(query, self.stopwords)
        seen: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", query):
            for first, second in zip(segment, segment[1:]):
                bigram = first + second
                if bigram not in seen:
                    seen.append(bigram)
        # 长特征在前,截断时保留信息量更大的
        seen.sort(key=len, reverse=True)
        return seen[: self.MAX_FEATURES]

    def _char_coverage(self, query: str, text: str) -> tuple[int, int]:
        """中文有效字符覆盖率(去掉无区分度的单字)。

        token 匹配失败时的兜底信号:即使分词不同,只要字符大面积重合,语义大概率相关。
        """
        chars = [ch for ch in query if "\u4e00" <= ch <= "\u9fff" and ch not in self.CHAR_STOPWORDS]
        if not chars:
            return 0, 0
        return sum(1 for ch in chars if ch in text), len(chars)

    def score(self, query: str, text: str) -> float:
        features = self._features(query)
        if not features:
            return 0.0
        lowered = text.lower()
        length = max(len(lowered), 1)

        hit_positions: list[int] = []
        hit_count = 0
        matched = 0
        for feature in features:
            pos = lowered.find(feature)
            if pos >= 0:
                matched += 1
                hit_count += 1
                hit_positions.append(pos)
                # 同一特征在片段里多次出现按一次覆盖算,但计入密度
                hit_count += lowered.count(feature) - 1

        coverage = matched / len(features)
        char_hit, char_total = self._char_coverage(query, lowered)
        char_coverage = (char_hit / char_total) if char_total else 0.0

        # 完整短语:相邻特征词是否在片段中原样相邻出现(顺序敏感)
        exact = 0.0
        for first, second in zip(features, features[1:]):
            if f"{first}{second}" in lowered:
                exact = 1.0
                break
        # 单特征查询没有相邻对,用「是否包含完整查询串」兜底
        if not exact and len(features) == 1:
            exact = 1.0 if features[0] in lowered else 0.0

        # 位置:命中越靠前越好(片段靠前的内容在实践中更常是主题句)
        if hit_positions:
            position = 1.0 - min(min(hit_positions), length) / length
        elif char_total:
            first_char_pos = min(
                (pos for pos in (lowered.find(ch) for ch in query) if pos >= 0), default=-1
            )
            position = 0.0 if first_char_pos < 0 else 1.0 - first_char_pos / length
        else:
            position = 0.0

        # 密度:命中总次数相对于片段长度,做饱和处理避免长片段刷分
        density = min(hit_count / (len(features) * 2), 1.0)

        return (
            self.WEIGHTS["coverage"] * coverage
            + self.WEIGHTS["char"] * char_coverage
            + self.WEIGHTS["exact"] * exact
            + self.WEIGHTS["position"] * position
            + self.WEIGHTS["density"] * density
        )

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scored: list[RetrievedChunk] = []
        for item in candidates:
            item.score = self.score(query, item.chunk.text)
            scored.append(item)
        # 同分时保留融合名次(稳定排序),避免重排把双路共识的候选无意义地打乱
        scored.sort(key=lambda item: (-item.score, item.rank))
        result = scored[:top_n]
        for rank, item in enumerate(result, start=1):
            item.rank = rank
        return result


class LLMReranker(Reranker):
    """大模型列表式重排。

    相比逐条打分(pointwise,需要 N 次调用),列表式一次给出全部候选的排序,调用次数从 N 降到 1,
    代价是受上下文长度限制、候选不能太多(这里默认最多 20 条)。
    解析失败时保持原顺序(fail-open):重排是优化项,不能因为模型输出格式问题让整个问答失败。
    """

    name = "llm"

    PROMPT = """你是一个检索结果重排专家。请根据用户问题,对下列候选片段按相关性从高到低排序。

用户问题：
{question}

候选片段：
{candidates}

要求：
1. 只输出候选编号,用英文逗号分隔,例如：3,1,5,2,4
2. 必须包含全部编号,不得增删、不得重复
3. 不要输出任何解释文字
"""

    MAX_CANDIDATES = 20

    def __init__(self, client, model: str, temperature: float = 0.0):
        self._client = client
        self._model = model
        self._temperature = temperature

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if len(candidates) <= 1:
            result = candidates[:top_n]
            for rank, item in enumerate(result, start=1):
                item.rank = rank
            return result

        pool = candidates[: self.MAX_CANDIDATES]
        listing = "\n".join(
            f"[{i}] {item.chunk.text[:400]}" for i, item in enumerate(pool, start=1)
        )
        prompt = self.PROMPT.format(question=query, candidates=listing)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=128,
            )
            content = response.choices[0].message.content or ""
            order = self._parse_order(content, len(pool))
        except Exception as exc:
            logger.warning("LLM 重排失败,回退到原顺序:%s", exc)
            order = []

        if order:
            reordered = [pool[i - 1] for i in order]
            # 模型没排到的候选补在后面,保证不丢候选
            reordered.extend(item for i, item in enumerate(pool, start=1) if i not in order)
            pool = reordered

        result = pool[:top_n]
        for rank, item in enumerate(result, start=1):
            item.rank = rank
            item.channels.append("llm_rerank")
        return result

    @staticmethod
    def _parse_order(content: str, size: int) -> list[int]:
        """从模型输出里提取编号顺序,非法编号直接丢弃。"""
        numbers = [int(n) for n in re.findall(r"\d+", content)]
        order: list[int] = []
        for number in numbers:
            if 1 <= number <= size and number not in order:
                order.append(number)
        return order


def build_reranker(
    mode: str = "lexical",
    llm_client=None,
    llm_model: str = "",
) -> Reranker:
    """按配置构造重排器;llm 模式缺少可用客户端时自动退回 lexical。"""
    mode = (mode or "off").lower()
    if mode == "off":
        return NoopReranker()
    if mode == "lexical":
        return LexicalReranker()
    if mode == "llm":
        if llm_client is None or not llm_model:
            logger.warning("llm 重排需要可用的大模型客户端,已回退到 lexical 重排")
            return LexicalReranker()
        return LLMReranker(client=llm_client, model=llm_model)
    raise ValueError(f"未知的重排模式 {mode!r},可选:off / lexical / llm")


__all__ = ["Reranker", "NoopReranker", "LexicalReranker", "LLMReranker", "build_reranker"]
