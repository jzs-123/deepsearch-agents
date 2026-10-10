"""RAG 核心链路自测脚本(不依赖大模型与本地向量模型,可离线运行)。

用 HashEmbedder 作为向量后端,验证「解析 -> 切块 -> 入库 -> 混合检索 -> 重排 -> 引用」
全链路,并覆盖几个容易出错的边界:重复入库幂等、向量后端切换报错、重叠切块、拒答。

用法:
    python -m app.rag.selftest
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rag.config import load_rag_config  # noqa: E402
from app.rag.embedding import HashEmbedder  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.rag.vectorstore import EmbeddingMismatchError  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK]   {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name} {detail}")


SAMPLE_DOC = """# 2026 年电商运营策略

## 一、品类策略

公司将重点投入小家电与宠物用品两个品类,预算占比分别为 35% 和 25%。
小家电的毛利率目标是 28%,宠物用品为 32%。

## 二、风控策略

风控采取三档授信规则:
1. 新客首单额度不超过 500 元;
2. 连续履约 6 个月以上的用户可提额至 5000 元;
3. 出现逾期 2 次以上的用户直接冻结额度。
风控模型 A13 的准确率为 91.5%,召回率为 87.2%。

## 三、AI 应用路线图

2026 年 Q1 上线智能客服,目标是把人工转接率降到 18%;
Q2 上线商品描述生成,覆盖 60% 的新品;
Q3 试运行数字人直播,单场成本目标 3000 元以内。
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "rag"
        config = load_rag_config(data_dir=data_dir, default_kb="test", chunk_size=180, chunk_overlap=40)
        embedder = HashEmbedder(dim=512)
        pipeline = RagPipeline(kb="test", config=config, embedder=embedder)

        print("\n[1] 入库(手动文本)")
        added = pipeline.add_texts([SAMPLE_DOC], source="运营策略.md")
        check("文本入库成功", added > 0, f"新增 {added} chunks")
        check("chunk 数量随切块参数增长", added >= 3, f"{added} 段")

        print("\n[2] 重复入库幂等")
        first_count = len(pipeline.store)
        pipeline.add_texts([SAMPLE_DOC], source="运营策略.md")
        check("重复入库后总量不变(整篇替换)", len(pipeline.store) == first_count, f"{len(pipeline.store)} 条")

        print("\n[3] chunk 结构信息")
        chunks = list(pipeline.store.all_chunks())
        check("chunk 带标题路径", all(c.heading for c in chunks), f"示例:{chunks[0].heading!r}")
        check("chunk 长度受控", all(len(c.text) <= 400 for c in chunks), f"max={max(len(c.text) for c in chunks)}")
        check("长文档切出多个 chunk", len(chunks) >= 3, f"{len(chunks)} 个")

        print("\n[4] 混合检索(稠密 + BM25 + RRF)")
        results = pipeline.search("风控模型的准确率是多少", top_k=6, top_n=3)
        joined = " ".join(r.chunk.text for r in results)
        check("检索命中相关片段", "91.5" in joined or "A13" in joined, f"命中 {len(results)} 条")
        check("召回结果带分数", all(r.score > 0 for r in results))
        check("记录了命中通道", all(r.channels for r in results), str(results[0].channels))

        print("\n[5] 稀疏路对专有名词的作用")
        sparse_only = pipeline._bm25_index().search("A13", top_k=3)
        check("BM25 能精确命中型号 A13", any("A13" in chunk.text for chunk, _ in sparse_only))

        print("\n[6] 重排开关对比")
        with_rerank = pipeline.search("数字人直播的单场成本目标", top_k=8, top_n=2, rerank=True)
        without = pipeline.search("数字人直播的单场成本目标", top_k=8, top_n=2, rerank=False)
        check("重排返回指定条数", len(with_rerank) == 2 and len(without) == 2)
        check(
            "重排把包含答案的片段排到第一",
            "3000" in with_rerank[0].chunk.text or "直播" in with_rerank[0].chunk.text,
            with_rerank[0].chunk.text[:40].replace("\n", " "),
        )

        print("\n[7] 无依据拒答")
        answer = pipeline.query("今天上海天气怎么样")
        check("无依据时不编造答案", not answer.grounded, answer.answer[:40])

        print("\n[8] 向量后端指纹校验")
        try:
            RagPipeline(kb="test", config=config, embedder=HashEmbedder(dim=256))
            check("维度不一致应报错", False, "未报错")
        except EmbeddingMismatchError as exc:
            check("维度不一致会显式报错", True, str(exc)[:60] + "...")

        print("\n[9] 持久化与重新加载")
        reloaded = RagPipeline(kb="test", config=config, embedder=embedder)
        check("重新加载 chunk 数一致", len(reloaded.store) == len(chunks), f"{len(reloaded.store)} 条")
        check(
            "重新加载后 BM25 可用",
            len(reloaded._bm25_index().search("宠物用品", top_k=2)) > 0,
        )

        print("\n[10] 文件入库(Markdown / 纯文本)")
        docs_dir = Path(tmp) / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "制度.md").write_text("# 报销制度\n\n差旅费报销上限为每日 800 元,需附发票。", encoding="utf-8")
        (docs_dir / "说明.txt").write_text("客服响应时限为 30 分钟,超时需升级处理。", encoding="utf-8")
        report = pipeline.ingest(docs_dir)
        check("目录入库成功", report.files_indexed == 2, f"入库 {report.files_indexed} 个文件")
        hit = pipeline.search("差旅费报销上限", top_k=5, top_n=2)
        check("新文档可被检索到", any("800" in r.chunk.text for r in hit))

    print(f"\n──── 自测结果:{len(PASSED)} 通过 / {len(FAILED)} 失败 ────")
    for name in FAILED:
        print(f"  失败:{name}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
