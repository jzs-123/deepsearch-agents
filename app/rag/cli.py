"""
RAG 命令行工具

一条命令跑通全流程,替代原来 RAGFlow 的页面操作:
    python -m app.rag.cli list                                   # 看有哪些知识库
    python -m app.rag.cli ingest -k default -p docs/knowledge_base  # 入库一个目录
    python -m app.rag.cli search -k default -q "风控策略有哪些"    # 只看检索结果(不调用大模型)
    python -m app.rag.cli ask    -k default -q "风控策略有哪些"    # 检索 + 生成带引用的答案
    python -m app.rag.cli eval   -k default -e examples/rag_eval_set.jsonl
    python -m app.rag.cli stats  -k default                       # 看库的规模与配置
    python -m app.rag.cli reset  -k default                       # 清空(谨慎)

设计说明:search 与 ask 分成两个命令,是为了把「检索」和「生成」解耦。
调检索质量时用 search(不受大模型波动影响,快且免费),验证端到端体验时用 ask。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 支持 `python app/rag/cli.py ...` 直接运行:把项目根目录加进 sys.path
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rag.config import load_rag_config  # noqa: E402
from app.rag.eval import evaluate, format_report, load_eval_set  # noqa: E402
from app.rag.pipeline import get_pipeline  # noqa: E402
from app.rag.vectorstore import EmbeddingMismatchError, list_knowledge_bases  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.rag.cli",
        description="自研 RAG 知识库命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="WARNING", help="日志级别,默认 WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, with_kb: bool = True) -> None:
        if with_kb:
            p.add_argument("-k", "--kb", default=None, help="知识库名称,默认取 RAG_DEFAULT_KB")
        p.add_argument(
            "--embedding-provider",
            choices=["local", "openai", "hash"],
            default=None,
            help="临时覆盖向量后端(便于对比不同后端的效果)",
        )

    p_list = sub.add_parser("list", help="列出所有知识库")
    p_list.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_create = sub.add_parser("create", help="创建空知识库")
    add_common(p_create)
    p_create.add_argument("--chunk-size", type=int, default=None, help="切块字符数")
    p_create.add_argument("--chunk-overlap", type=int, default=None, help="切块重叠字符数")

    p_ingest = sub.add_parser("ingest", help="把文件或目录入库")
    add_common(p_ingest)
    p_ingest.add_argument("-p", "--path", required=True, help="文件或目录路径")
    p_ingest.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="检索(不调用大模型)")
    add_common(p_search)
    p_search.add_argument("-q", "--question", required=True)
    p_search.add_argument("-n", "--top-n", type=int, default=None, help="返回条数")
    p_search.add_argument("--top-k", type=int, default=None, help="粗排候选数")
    p_search.add_argument(
        "--no-rerank", action="store_true", help="关闭重排(用于对比重排效果)"
    )
    p_search.add_argument("--json", action="store_true")

    p_ask = sub.add_parser("ask", help="检索 + 大模型生成带引用的答案")
    add_common(p_ask)
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.add_argument("-n", "--top-n", type=int, default=None)
    p_ask.add_argument("--json", action="store_true")

    p_eval = sub.add_parser("eval", help="用评测集评估检索质量")
    add_common(p_eval)
    p_eval.add_argument("-e", "--eval-set", required=True, help="评测集 JSON/JSONL 路径")
    p_eval.add_argument("-n", "--k", type=int, default=5, help="评测的 top-k,默认 5")
    p_eval.add_argument(
        "--no-rerank", action="store_true", help="关闭重排(用于对比重排带来的收益)"
    )
    p_eval.add_argument("--json", action="store_true")

    p_stats = sub.add_parser("stats", help="查看知识库规模与配置")
    add_common(p_stats)
    p_stats.add_argument("--json", action="store_true")

    p_reset = sub.add_parser("reset", help="清空知识库数据(不可恢复)")
    add_common(p_reset)
    p_reset.add_argument("--yes", action="store_true", help="跳过确认")

    return parser


def _build_config(args) -> object:
    overrides: dict = {}
    if getattr(args, "embedding_provider", None):
        overrides["embedding_provider"] = args.embedding_provider
    if getattr(args, "chunk_size", None):
        overrides["chunk_size"] = args.chunk_size
    if getattr(args, "chunk_overlap", None) is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if getattr(args, "top_n", None):
        overrides["top_n"] = args.top_n
    if getattr(args, "top_k", None):
        overrides["top_k"] = args.top_k
    if getattr(args, "no_rerank", False):
        overrides["rerank_mode"] = "off"
    return load_rag_config(**overrides)


def cmd_list(args) -> int:
    config = load_rag_config()
    bases = list_knowledge_bases(config.data_path)
    if args.json:
        print(json.dumps(bases, ensure_ascii=False, indent=2))
        return 0
    if not bases:
        print(f"还没有任何知识库。数据目录:{config.data_path}")
        print("先执行: python -m app.rag.cli ingest -k default -p docs/knowledge_base")
        return 0
    print(f"数据目录:{config.data_path}")
    for item in bases:
        print(
            f"  - {item['name']}: {item['chunk_count']} chunks"
            f" | 向量:{item['embedding'] or '未初始化'}"
            f" | 更新:{item['updated_at'] or '-'}"
        )
    return 0


def cmd_create(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    pipeline.store.meta.update({"kb": pipeline.kb, "chunk_size": config.chunk_size})
    pipeline.store.embedding_fingerprint = pipeline.store.embedding_fingerprint or pipeline.embedder.fingerprint
    pipeline.store.dim = pipeline.store.dim or pipeline.embedder.dim
    pipeline.store.save()
    print(f"知识库已就绪:{pipeline.kb_dir}")
    print(json.dumps(pipeline.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_ingest(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    report = pipeline.ingest(args.path)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"知识库:{report.kb}(累计 {report.total_chunks} chunks)")
        print(
            f"文件:{report.files_indexed}/{report.files_total} 入库,"
            f"跳过 {report.files_skipped},新增 chunk {report.chunks_added},"
            f"替换旧 chunk {report.chunks_replaced},用时 {report.elapsed:.2f}s"
        )
        for detail in report.details:
            print(f"  + {detail['file']}({detail['chars']} 字符 -> {detail['chunks']} chunks)")
        for warning in report.warnings:
            print(f"  ! {warning}")
    return 0 if report.files_indexed or not report.files_total else 1


def cmd_search(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    if not len(pipeline.store):
        print(f"知识库 {pipeline.kb} 是空的,请先 ingest 文档", file=sys.stderr)
        return 1
    results = pipeline.search(args.question, top_k=config.top_k, top_n=config.top_n)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "rank": item.rank,
                        "score": round(item.score, 4),
                        "rrf": round(item.rrf_score, 4),
                        "dense": None if item.dense_score is None else round(item.dense_score, 4),
                        "sparse": None if item.sparse_score is None else round(item.sparse_score, 4),
                        "channels": item.channels,
                        "source": item.chunk.source,
                        "chunk_index": item.chunk.chunk_index,
                        "heading": item.chunk.heading,
                        "text": item.chunk.text,
                    }
                    for item in results
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"问题:{args.question}")
    print(f"命中 {len(results)} 条(向量后端:{pipeline.embedder.fingerprint},重排:{config.rerank_mode})")
    # 阈值提示:search 只做检索、不做拒答,但要让使用者知道「这些片段够不够作为依据」
    gate = pipeline.grounded_score(results)
    if results and gate < config.min_score:
        print(
            f"提示:向量余弦 {gate:.4f} 低于拒答阈值 {config.min_score},"
            f"ask 会判定为「知识库无依据」并拒答;以下片段仅供排查参考。"
        )
    for item in results:
        print(
            f"\n[{item.rank}] score={item.score:.4f} rrf={item.rrf_score:.4f}"
            f" dense={item.dense_score if item.dense_score is None else round(item.dense_score, 4)}"
            f" sparse={item.sparse_score if item.sparse_score is None else round(item.sparse_score, 4)}"
            f" 通道={item.channels}"
        )
        print(f"    来源:{item.chunk.source}#{item.chunk.chunk_index} {item.chunk.heading}")
        print(f"    内容:{item.chunk.text[:300]}{'...' if len(item.chunk.text) > 300 else ''}")
    return 0


def cmd_ask(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    answer = pipeline.query(args.question, top_n=config.top_n)
    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"问题:{args.question}\n")
    print(f"回答:{answer.answer}\n")
    if answer.citations:
        print("引用:")
        for cite in answer.citations:
            print(
                f"  [{cite['index']}] {cite['source']}#{cite['chunk_index']}"
                f" {cite['heading']} (score={cite['score']})"
            )
    print(f"\n调试:{json.dumps(answer.debug, ensure_ascii=False)}")
    return 0


def cmd_eval(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    items = load_eval_set(args.eval_set)
    result = evaluate(pipeline, items, k=args.k)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(result))
    return 0


def cmd_stats(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    stats = pipeline.stats()
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    for key, value in stats.items():
        print(f"{key:16}: {value}")
    return 0


def cmd_reset(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    if not args.yes:
        confirm = input(f"确认清空知识库 {pipeline.kb}({pipeline.kb_dir})?输入 yes 继续:")
        if confirm.strip().lower() != "yes":
            print("已取消")
            return 1
    pipeline.reset()
    print(f"知识库 {pipeline.kb} 已清空")
    return 0


HANDLERS = {
    "list": cmd_list,
    "create": cmd_create,
    "ingest": cmd_ingest,
    "search": cmd_search,
    "ask": cmd_ask,
    "eval": cmd_eval,
    "stats": cmd_stats,
    "reset": cmd_reset,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return HANDLERS[args.command](args)
    except EmbeddingMismatchError as exc:
        # 向量不兼容是最容易踩的坑,单独给出可执行的处理建议
        print(f"向量不兼容:{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"文件错误:{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
