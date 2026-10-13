"""
答案生成层

把检索到的 top-N 片段拼成上下文,交给大模型生成带引用的回答。这里处理的是 RAG 落地的
三个真实痛点:

1. 幻觉 —— 提示词强制「只用给定资料作答」,并要求标注引用编号;检索不到依据时直接拒答
   (宁可回答「知识库中没有」,也不要让模型用参数化知识编一个像样的答案);
2. 引用可追溯 —— 每个片段编号 [来源#序号],答案里的引用能对应回原始 chunk,用户可以核对;
3. 不可用降级 —— 没有配置大模型 / 调用失败时,返回「检索片段 + 明确提示」而不是抛异常,
   这样即便 LLM 挂了,检索能力仍然可用(前端可以只展示依据)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.types import Answer, RetrievedChunk

logger = logging.getLogger(__name__)

# 拒答文案:检索不到依据时使用
NO_EVIDENCE_ANSWER = "知识库中没有找到与该问题相关的资料,无法依据内部文档回答。"
# 模型调用不可用时的兜底文案
LLM_UNAVAILABLE_ANSWER = "（大模型调用不可用,以下仅返回检索到的原文依据,请参考下方引用。）"

# 模型输出里出现这些字样视为「承认没有依据」
_REFUSAL_MARKERS = (
    "没有相关信息",
    "未找到相关",
    "没有找到相关",
    "无法回答",
    "无法确定",
    "资料中没有",
    "文档中没有",
    "知识库中没有",
    "不包含相关",
    "无法依据",
)


@dataclass
class ChatClient:
    """OpenAI 兼容的对话客户端(与主项目共用同一套 .env 配置)。"""

    model: str
    api_key: str
    base_url: str = ""
    temperature: float = 0.1
    timeout: float = 120.0
    # 推理模型思维链开关:"disabled" 关闭 / "enabled" 开启 / "" 不传(跟随服务端默认)。
    # 思维链与正文共享 max_tokens,复杂上下文下思考可耗尽预算致 content 为空(§5.20)。
    thinking: str = "disabled"

    def __post_init__(self) -> None:
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model)

    def _ensure(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("缺少 openai 依赖,请执行: pip install openai") from exc
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url or None, timeout=self.timeout
            )
        return self._client

    def _extra_body(self) -> dict:
        """思维链开关走 extra_body 注入(`thinking: {"type": ...}`,DeepSeek 原生参数)。

        放 extra_body 而不是顶层字段:openai SDK 不认识这个参数,直接传会报
        "unexpected keyword";extra_body 会原样并入请求 JSON,不支持的网关最多
        忽略它(探测验证过 chat_template_kwargs 等参数就是被静默忽略的)。
        """
        if self.thinking in ("disabled", "enabled"):
            return {"thinking": {"type": self.thinking}}
        return {}

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str:
        """发起一次对话补全。"""
        client = self._ensure()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body(),
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        # 推理模型(如 deepseek-v4-pro)的思维链会与正文共享 max_tokens 预算:
        # 上下文复杂/噪声多时,思考就能耗光 1200 tokens(finish_reason=length 且
        # reasoning_tokens=1199),content 一个字没生成就被截断。生成层拿到空文本
        # 会套上拒答模板,被误判成「模型拒答」——必须在这里放大预算重试一次。
        # (若 thinking 已设为 disabled,正常不会走到这里;重试是最后一道保险。)
        if not text and choice.finish_reason == "length":
            retry_tokens = max(4096, max_tokens * 4)
            logger.warning(
                "模型思维链耗尽 %s tokens 致正文为空,放大至 %s 重试", max_tokens, retry_tokens
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=retry_tokens,
                extra_body=self._extra_body(),
            )
            text = (response.choices[0].message.content or "").strip()
        return text

    @property
    def raw(self):
        """暴露底层 client,供重排等模块复用同一连接。"""
        return self._ensure()


SYSTEM_PROMPT = """你是一个企业内部知识库问答助手。你只能依据【参考资料】中的内容回答问题。

铁律：
1. 只使用【参考资料】里出现的信息,绝对不要使用你自己的先验知识补充、推测或编造任何内容;
2. 每一条结论后面必须标注来源编号,格式为 [编号],例如 [1] 或 [1][3];
3. 如果【参考资料】不足以回答问题,直接回答"知识库中没有找到相关信息",不要勉强作答,也不要解释推理过程;
4. 资料之间如果有冲突,要同时指出冲突并分别标注来源;
5. 回答使用中文,结构清晰;涉及数字、金额、比例、日期时必须与原文完全一致,禁止改写或估算。"""


USER_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请依据上述参考资料作答,并在每条结论后标注来源编号。"""


def build_context(chunks: list[RetrievedChunk], max_chars: int = 8000) -> tuple[str, list[RetrievedChunk]]:
    """把检索片段拼成带编号的上下文,返回 (上下文文本, 实际使用的片段列表)。

    max_chars 是二次防线:即使上游 top_n 配得过大,也不会把上千字塞进一次生成,
    截断以「整条片段」为粒度,不会切出半个句子。
    """
    used: list[RetrievedChunk] = []
    blocks: list[str] = []
    total = 0
    for idx, item in enumerate(chunks, start=1):
        heading = f"（{item.chunk.heading}）" if item.chunk.heading else ""
        block = f"[{idx}] 来源:{item.chunk.source}#{item.chunk.chunk_index}{heading}\n{item.chunk.text}"
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(item)
        total += len(block)
    return "\n\n".join(blocks), used


class AnswerGenerator:
    """基于检索结果生成答案。"""

    def __init__(self, client: ChatClient | None = None, min_score: float = 0.2, max_context_chars: int = 8000):
        self.client = client
        self.min_score = min_score
        self.max_context_chars = max_context_chars

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        gate_score: float | None = None,
    ) -> Answer:
        """生成答案。

        :param question: 用户问题
        :param chunks: 重排后的检索结果(按相关性降序)
        :param gate_score: 用于「是否有依据」判定的分数(应为向量余弦相似度)。
            不传则退回用排序分,但开启重排后排序分尺度会变,阈值需要重新标定。
        """
        answer = Answer(question=question, answer="", citations=[], contexts=[])

        if not chunks:
            answer.answer = NO_EVIDENCE_ANSWER
            answer.grounded = False
            return answer

        # 相似度门槛:低于阈值说明「知识库没有这方面的资料」,直接拒答比让模型硬答更安全
        effective_score = gate_score if gate_score is not None else chunks[0].score
        if effective_score < self.min_score:
            answer.answer = NO_EVIDENCE_ANSWER
            answer.grounded = False
            answer.debug = {
                "reason": "score_below_threshold",
                "gate_score": round(float(effective_score), 4),
                "rank_score": round(chunks[0].score, 4),
                "min_score": self.min_score,
            }
            return answer

        context, used = build_context(chunks, self.max_context_chars)
        answer.contexts = [
            {
                "index": idx,
                "source": item.chunk.source,
                "chunk_index": item.chunk.chunk_index,
                "heading": item.chunk.heading,
                "score": round(item.score, 4),
                "text": item.chunk.text,
            }
            for idx, item in enumerate(used, start=1)
        ]
        answer.citations = [
            {
                "index": idx,
                "source": item.chunk.source,
                "chunk_index": item.chunk.chunk_index,
                "heading": item.chunk.heading,
                "score": round(item.score, 4),
                "channels": item.channels,
            }
            for idx, item in enumerate(used, start=1)
        ]
        answer.debug["gate_score"] = round(float(effective_score), 4)
        answer.debug["rank_score"] = round(chunks[0].score, 4)
        answer.debug["used_chunks"] = len(used)

        if self.client is None or not self.client.available:
            # 无大模型时退化为「只给依据」,检索价值仍然保留
            answer.answer = LLM_UNAVAILABLE_ANSWER
            answer.grounded = False
            answer.debug["reason"] = "llm_unavailable"
            answer.debug["llm_called"] = False
            return answer

        try:
            answer.debug["llm_called"] = True
            text = self.client.complete(
                SYSTEM_PROMPT, USER_TEMPLATE.format(context=context, question=question)
            )
        except Exception as exc:
            logger.warning("生成答案失败:%s", exc)
            answer.answer = f"（生成回答时调用大模型失败:{exc}）\n\n已检索到以下依据,请参考引用。"
            answer.grounded = False
            answer.debug["reason"] = "llm_error"
            answer.debug["error"] = str(exc)
            answer.debug["llm_called"] = False
            return answer

        answer.answer = text or NO_EVIDENCE_ANSWER
        # 模型自己承认没依据时,标记为不 grounded(评测脚本据此统计拒答率)
        if any(marker in answer.answer for marker in _REFUSAL_MARKERS) and len(answer.answer) < 200:
            answer.grounded = False
            answer.debug["reason"] = "model_refused"
        return answer


__all__ = [
    "AnswerGenerator",
    "ChatClient",
    "build_context",
    "NO_EVIDENCE_ANSWER",
    "LLM_UNAVAILABLE_ANSWER",
    "SYSTEM_PROMPT",
]
