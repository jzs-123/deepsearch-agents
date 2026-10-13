from dataclasses import replace

from app.rag.eval import recommend_refusal_threshold


def test_recommend_refusal_threshold_separates_positive_and_negative_scores():
    rows = [
        {"top_score": 0.62, "expect_no_answer": False},
        {"top_score": 0.55, "expect_no_answer": False},
        {"top_score": 0.18, "expect_no_answer": True},
        {"top_score": 0.12, "expect_no_answer": True},
    ]
    result = recommend_refusal_threshold(rows, current_threshold=0.2)
    assert result["available"] is True
    assert 0.18 < result["recommended_threshold"] < 0.55
    assert result["balanced_accuracy"] == 1.0


def test_dense_only_search_does_not_build_bm25(pipeline, monkeypatch):
    pipeline.add_texts(["支付清算系统支持实时到账。"], source="payment.md")
    pipeline.config = replace(pipeline.config, sparse_enabled=False)

    def fail_if_called():
        raise AssertionError("关闭稀疏检索时不应构建 BM25")

    monkeypatch.setattr(pipeline, "_bm25_index", fail_if_called)
    results = pipeline.search("支付清算", top_n=1, rewrite=False)
    assert results


def test_cli_min_score_override_is_available_for_eval():
    from app.rag.cli import _build_config, build_parser

    args = build_parser().parse_args(
        ["eval", "-k", "fin", "-e", "examples/rag_eval_set_finance.json", "--min-score", "0.61"]
    )
    assert _build_config(args).min_score == 0.61
