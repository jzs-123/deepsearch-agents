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

from app.rag.audit import format_audit_records, read_audit  # noqa: E402
from app.rag.config import load_rag_config  # noqa: E402
from app.rag.eval import evaluate, format_report, load_eval_set  # noqa: E402
from app.rag.metadata import (  # noqa: E402
    InvalidFilterError,
    format_filter_hint,
    normalize_filter,
    parse_filter_json,
)
from app.rag.pipeline import format_search_audit, get_pipeline  # noqa: E402
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

    def add_rewrite_args(p: argparse.ArgumentParser, with_always: bool = False) -> None:
        """查询改写相关开关(检索类命令共用)。

        存在的意义就是「能对比」:同一份语料、同一个问题,分别用关闭 / 低分重试 / 始终改写跑一遍,
        差异才有解释力。
        """
        p.add_argument(
            "--rewrite-mode",
            choices=["off", "local", "llm", "local+llm"],
            default=None,
            help="查询改写模式,默认取 RAG_QUERY_REWRITE(应用默认 local)",
        )
        p.add_argument("--no-rewrite", action="store_true", help="关闭查询改写(= --rewrite-mode off)")
        p.add_argument(
            "--variants",
            type=int,
            default=None,
            help="变体总数上限(含原始查询),默认取 RAG_QUERY_VARIANTS",
        )
        p.add_argument("--no-retry", action="store_true", help="低分时不自动用变体重试")
        if with_always:
            p.add_argument(
                "--rewrite-always",
                action="store_true",
                help="首轮就用全部变体(不等低分触发重试)",
            )

    def add_filter_args(p: argparse.ArgumentParser) -> None:
        """元数据过滤开关(检索类命令共用)。

        filter 是 JSON 字符串:`--filter '{"部门": "风控"}'`,值可以是列表表示"命中其一即可"。
        可用字段用 `list` / `stats` 查看,也可以先 `ingest` 带 frontmatter 的文档再看入库报告。
        """
        p.add_argument(
            "--filter",
            default=None,
            help='元数据过滤,JSON 字符串(如 {"部门": "风控"} 或 {"类型": ["制度", "规范"]});'
            "缺字段或值不匹配的片段会被剔除",
        )

    def add_cache_args(p: argparse.ArgumentParser) -> None:
        """缓存开关(检索类命令共用);默认取 RAG_CACHE_ENABLED(默认关)。"""
        p.add_argument("--cache", dest="cache_enabled", action="store_true", default=None,
                       help="本次查询启用结果缓存(默认取 RAG_CACHE_ENABLED)")
        p.add_argument("--no-cache", dest="cache_enabled", action="store_false",
                       help="本次查询关闭结果缓存")

    def add_min_score_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--min-score",
            type=float,
            default=None,
            help="临时覆盖拒答阈值，便于按评测建议复验",
        )

    p_list = sub.add_parser("list", help="列出所有知识库")
    p_list.add_argument("--json", action="store_true", help="以 JSON 输出")

    p_create = sub.add_parser("create", help="创建空知识库")
    add_common(p_create)
    p_create.add_argument("--chunk-size", type=int, default=None, help="切块字符数")
    p_create.add_argument("--chunk-overlap", type=int, default=None, help="切块重叠字符数")

    p_ingest = sub.add_parser("ingest", help="把文件或目录入库")
    add_common(p_ingest)
    add_cache_args(p_ingest)
    p_ingest.add_argument("-p", "--path", required=True, help="文件或目录路径")
    p_ingest.add_argument(
        "--document-key",
        default=None,
        help="单文件的稳定业务身份；再次使用同一值会整篇替换旧版本",
    )
    p_ingest.add_argument("--revision-id", default=None, help="单文件版本标识，默认取内容指纹")
    p_ingest.add_argument("--effective-from", default=None, help="单文件业务生效时间，仅作为元数据保存")
    p_ingest.add_argument("--json", action="store_true")

    p_search = sub.add_parser("search", help="检索(不调用大模型)")
    add_common(p_search)
    add_rewrite_args(p_search, with_always=True)
    add_filter_args(p_search)
    add_cache_args(p_search)
    add_min_score_arg(p_search)
    p_search.add_argument("-q", "--question", required=True)
    p_search.add_argument("-n", "--top-n", type=int, default=None, help="返回条数")
    p_search.add_argument("--top-k", type=int, default=None, help="粗排候选数")
    p_search.add_argument(
        "--no-rerank", action="store_true", help="关闭重排(用于对比重排效果)"
    )
    p_search.add_argument("--json", action="store_true")

    p_ask = sub.add_parser("ask", help="检索 + 大模型生成带引用的答案")
    add_common(p_ask)
    add_rewrite_args(p_ask)
    add_filter_args(p_ask)
    add_cache_args(p_ask)
    add_min_score_arg(p_ask)
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.add_argument("-n", "--top-n", type=int, default=None)
    p_ask.add_argument("--json", action="store_true")

    p_eval = sub.add_parser("eval", help="用评测集评估检索质量")
    add_common(p_eval)
    add_rewrite_args(p_eval, with_always=True)
    add_filter_args(p_eval)
    add_cache_args(p_eval)
    add_min_score_arg(p_eval)
    p_eval.add_argument("-e", "--eval-set", required=True, help="评测集 JSON/JSONL 路径")
    p_eval.add_argument("-n", "--k", type=int, default=5, help="评测的 top-k,默认 5")
    p_eval.add_argument(
        "--no-rerank", action="store_true", help="关闭重排(用于对比重排带来的收益)"
    )
    p_eval.add_argument("--json", action="store_true")

    p_stats = sub.add_parser("stats", help="查看知识库规模与配置")
    add_common(p_stats)
    p_stats.add_argument("--json", action="store_true")

    p_audit = sub.add_parser("audit", help="查看审计日志(只读,最近 N 条)")
    add_common(p_audit)
    p_audit.add_argument("--tail", type=int, default=20, help="展示最近 N 条,默认 20")
    p_audit.add_argument(
        "--kind", choices=["all", "search", "query"], default="all", help="只看某一类记录"
    )
    p_audit.add_argument("--json", action="store_true", help="以 JSON Lines 原样输出")

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
    if getattr(args, "min_score", None) is not None:
        overrides["min_score"] = args.min_score
    if getattr(args, "no_rerank", False):
        overrides["rerank_mode"] = "off"
    # 查询改写开关(只有 search/ask/eval 传了这些参数)
    if getattr(args, "no_rewrite", False):
        overrides["query_rewrite"] = "off"
    elif getattr(args, "rewrite_mode", None):
        overrides["query_rewrite"] = args.rewrite_mode
    if getattr(args, "variants", None):
        overrides["query_variants"] = args.variants
    if getattr(args, "no_retry", False):
        overrides["retry_on_low_score"] = False
    # 缓存开关(search/ask/eval)
    if getattr(args, "cache_enabled", None) is not None:
        overrides["cache_enabled"] = bool(args.cache_enabled)
    return load_rag_config(**overrides)


def _parse_filter(args) -> dict:
    """把 CLI 的 --filter 解析成 dict;非法时抛 InvalidFilterError(由 main 统一转成退出码 2)。"""
    return parse_filter_json(getattr(args, "filter", None))


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
        fields = item.get("filter_fields") or []
        fields_text = f" | 可过滤字段:{'、'.join(fields)}" if fields else " | 可过滤字段:(无)"
        print(
            f"  - {item['name']}: {item['chunk_count']} chunks"
            f" | 向量:{item['embedding'] or '未初始化'}"
            f" | 更新:{item['updated_at'] or '-'}"
            f" | 数据版本:{item.get('data_version', 0)}{fields_text}"
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
    if Path(args.path).is_dir() and any(
        (args.document_key, args.revision_id, args.effective_from)
    ):
        print(
            "参数错误:--document-key/--revision-id/--effective-from 只适用于单文件入库",
            file=sys.stderr,
        )
        return 2
    report = pipeline.ingest(
        args.path,
        document_key=args.document_key,
        revision_id=args.revision_id,
        effective_from=args.effective_from,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"知识库:{report.kb}(累计 {report.total_chunks} chunks)")
        print(
            f"文件:{report.files_indexed}/{report.files_total} 入库,"
            f"跳过 {report.files_skipped},新增 chunk {report.chunks_added},"
            f"替换旧 chunk {report.chunks_replaced},用时 {report.elapsed:.2f}s"
        )
        cache_info = report.embedding_cache or {}
        if cache_info.get("enabled"):
            print(
                f"  · 向量化缓存:命中 {cache_info.get('hits', 0)}/{cache_info.get('texts', 0)} 条"
                f"(未命中 {cache_info.get('misses', 0)} 条需要重新编码)"
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
    filter_spec = _parse_filter(args)
    results = pipeline.search(
        args.question,
        top_k=config.top_k,
        top_n=config.top_n,
        rewrite=True if getattr(args, "rewrite_always", False) else None,
        filter=filter_spec,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "filter": filter_spec,
                    "filter_fields": pipeline.last_search_debug.get("filter_fields", []),
                    "filtered_out": pipeline.last_search_debug.get("filtered_out", 0),
                    "cache": pipeline.last_search_debug.get("cache"),
                    "cached": pipeline.last_search_debug.get("cached", False),
                    "rewrite": pipeline.last_search_debug.get("rewrite"),
                    "retry": pipeline.last_search_debug.get("retry"),
                    "variants": pipeline.last_search_debug.get("variants", []),
                    "variant_hits": pipeline.last_search_debug.get("variant_hits", []),
                    "results": [
                        {
                            "rank": item.rank,
                            "score": round(item.score, 4),
                            "rrf": round(item.rrf_score, 4),
                            "dense": None if item.dense_score is None else round(item.dense_score, 4),
                            "sparse": None if item.sparse_score is None else round(item.sparse_score, 4),
                            "channels": item.channels,
                            "cached": item.cached,
                            "metadata": item.chunk.metadata,
                            "source": item.chunk.source,
                            "chunk_index": item.chunk.chunk_index,
                            "heading": item.chunk.heading,
                            "text": item.chunk.text,
                        }
                        for item in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(f"问题:{args.question}")
    print(f"命中 {len(results)} 条(向量后端:{pipeline.embedder.fingerprint},重排:{config.rerank_mode})")
    # 改写可见性(默认输出,不是 debug 开关):用了哪些变体、各自命中几条、是否重试、重试后判定分
    audit = format_search_audit(pipeline.last_search_debug)
    if audit:
        print(audit)
    if filter_spec and not results:
        # 过滤把候选清空了:必须明确说是"过滤条件"的问题,不能只报"没检索到"
        print(format_filter_hint(normalize_filter(filter_spec), pipeline.filter_fields))
        return 0
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
            f"{' [缓存]' if item.cached else ''}"
        )
        print(f"    来源:{item.chunk.source}#{item.chunk.chunk_index} {item.chunk.heading}")
        if item.chunk.metadata:
            print(f"    元数据:{json.dumps(item.chunk.metadata, ensure_ascii=False)}")
        print(f"    内容:{item.chunk.text[:300]}{'...' if len(item.chunk.text) > 300 else ''}")
    return 0


def cmd_ask(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    filter_spec = _parse_filter(args)
    answer = pipeline.query(args.question, top_n=config.top_n, filter=filter_spec)
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
    if filter_spec:
        print(
            f"\n元数据过滤:{json.dumps(filter_spec, ensure_ascii=False)}"
            f"(过滤掉 {answer.debug.get('filtered_out', 0)} 条候选)"
        )
        if not answer.citations:
            print(format_filter_hint(normalize_filter(filter_spec), pipeline.filter_fields))
    if answer.debug.get("cache_hit"):
        print(f"缓存:{answer.debug.get('cache_type')} 命中(本次未重新检索/生成)")
    print(f"\n调试:{json.dumps(answer.debug, ensure_ascii=False)}")
    return 0


def cmd_eval(args) -> int:
    config = _build_config(args)
    pipeline = get_pipeline(kb=args.kb, config=config)
    items = load_eval_set(args.eval_set)
    result = evaluate(
        pipeline,
        items,
        k=args.k,
        rewrite=True if getattr(args, "rewrite_always", False) else None,
        filter=_parse_filter(args),
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(result))
    return 0


def cmd_audit(args) -> int:
    """只读查看审计日志(最近 N 条),便于人工排查"谁在什么时候查了什么、为什么没查到"。"""
    config = _build_config(args)
    audit_dir = config.audit_path(args.kb or config.default_kb)
    kb_name = args.kb or config.default_kb
    records = read_audit(audit_dir, kb_name, tail=args.tail)
    if getattr(args, "kind", "all") != "all":
        records = [item for item in records if item.get("kind") == args.kind]
    if args.json:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
        return 0
    print(f"审计目录:{audit_dir}")
    print(f"知识库:{kb_name}(最近 {len(records)} 条)\n")
    print(format_audit_records(records))
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
    "audit": cmd_audit,
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
    except InvalidFilterError as exc:
        print(f"过滤条件错误:{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"文件错误:{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
