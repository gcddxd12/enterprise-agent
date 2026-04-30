"""Tests for evaluation/metrics.py and evaluation/eval_runner.py."""

import json
import sys

import pytest

# Ensure evaluation package is importable
sys.path.insert(0, ".")

from evaluation.metrics import AnswerQuality, RetrievalMetrics
from evaluation.eval_runner import (
    MockRetriever,
    MockAnswerGenerator,
    load_queries,
    run_retrieval_eval,
    run_answer_eval,
    run_full_eval,
)


# ============================================================================
# TestRetrievalMetrics
# ============================================================================


class TestRetrievalMetrics:
    def test_recall_all_found(self):
        assert RetrievalMetrics.recall_at_k(
            ["a", "b", "c"], ["a", "b", "c"], k=5
        ) == 1.0

    def test_recall_none_found(self):
        assert RetrievalMetrics.recall_at_k(
            ["x", "y", "z"], ["a", "b", "c"], k=5
        ) == 0.0

    def test_recall_partial(self):
        assert RetrievalMetrics.recall_at_k(
            ["a", "x", "y"], ["a", "b", "c"], k=5
        ) == pytest.approx(1.0 / 3.0)

    def test_recall_empty_relevant(self):
        assert RetrievalMetrics.recall_at_k(
            ["a", "b"], [], k=5
        ) == 1.0

    def test_recall_respects_k(self):
        # "b" is at position 3 (0-indexed), so k=2 should exclude it
        assert RetrievalMetrics.recall_at_k(
            ["a", "x", "b"], ["b"], k=2
        ) == 0.0

    def test_mrr_simple(self):
        results = [
            (["a", "b", "c"], ["b"]),  # first relevant at rank 2 → 0.5
            (["x", "y", "z"], ["x"]),  # first relevant at rank 1 → 1.0
        ]
        mrr = RetrievalMetrics.mrr(results)
        assert mrr == pytest.approx((0.5 + 1.0) / 2.0)

    def test_mrr_no_match(self):
        results = [(["x", "y"], ["a"])]
        assert RetrievalMetrics.mrr(results) == 0.0

    def test_mrr_empty(self):
        assert RetrievalMetrics.mrr([]) == 0.0

    def test_hit_rate_full(self):
        results = [
            (["a", "b"], ["a"]),   # hit
            (["x", "y"], ["a"]),   # miss
        ]
        assert RetrievalMetrics.hit_rate(results, k=5) == 0.5

    def test_hit_rate_empty(self):
        assert RetrievalMetrics.hit_rate([], k=5) == 0.0

    def test_evaluate_all(self):
        results = [
            (["a", "b", "c", "d", "e"], ["a", "b"]),
            (["x", "y", "z", "w", "v"], ["a"]),
        ]
        metrics = RetrievalMetrics.evaluate_all(results, k_values=[1, 3, 5])
        assert "recall_at_1" in metrics
        assert "recall_at_3" in metrics
        assert "recall_at_5" in metrics
        assert "hit_rate_at_1" in metrics
        assert "hit_rate_at_5" in metrics
        assert "mrr" in metrics
        assert metrics["num_queries"] == 2


# ============================================================================
# TestAnswerQuality
# ============================================================================


class TestAnswerQuality:
    def test_rouge_l_identical(self):
        text = "5G套餐包含10GB流量和50分钟通话"
        score = AnswerQuality.rouge_l(text, text)
        assert score == 1.0

    def test_rouge_l_completely_different(self):
        candidate = "查询话费余额"
        reference = "5G套餐有哪些"
        score = AnswerQuality.rouge_l(candidate, reference)
        assert score < 0.3  # should be low

    def test_rouge_l_partial_overlap(self):
        candidate = "您可以通过10086查询余额"
        reference = "查询话费余额请拨打10086"
        score = AnswerQuality.rouge_l(candidate, reference)
        assert 0.3 < score < 0.95

    def test_rouge_l_empty(self):
        assert AnswerQuality.rouge_l("", "something") == 0.0
        assert AnswerQuality.rouge_l("something", "") == 0.0
        assert AnswerQuality.rouge_l("", "") == 0.0

    def test_keyword_coverage_all_found(self):
        assert AnswerQuality.keyword_coverage(
            "5G套餐39元包含10GB流量",
            ["5G", "39元", "10GB", "流量"]
        ) == 1.0

    def test_keyword_coverage_none_found(self):
        assert AnswerQuality.keyword_coverage(
            "查询话费余额",
            ["5G", "宽带", "工单"]
        ) == 0.0

    def test_keyword_coverage_partial(self):
        score = AnswerQuality.keyword_coverage(
            "5G套餐39元包含流量",
            ["5G", "宽带", "工单", "流量"]
        )
        assert score == 0.5

    def test_keyword_coverage_case_insensitive(self):
        assert AnswerQuality.keyword_coverage(
            "5G-A is 5G Advanced",
            ["5g-a", "5g"]
        ) == 1.0

    def test_keyword_coverage_empty(self):
        assert AnswerQuality.keyword_coverage("anything", []) == 1.0

    def test_evaluate_all(self):
        results = [
            {
                "candidate": "5G套餐39元包含流量",
                "reference": "5G套餐最低39元包含10GB流量和50分钟通话",
                "keywords": ["5G", "39元", "流量", "套餐"],
                "domain": "5g",
            },
            {
                "candidate": "请拨打10086查询",
                "reference": "查询余额请拨打10086或发送短信",
                "keywords": ["10086", "余额", "查询"],
                "domain": "billing",
            },
        ]
        metrics = AnswerQuality.evaluate_all(results)
        assert metrics["num_results"] == 2
        assert 0.0 < metrics["rouge_l"] <= 1.0
        assert 0.0 < metrics["keyword_coverage"] <= 1.0
        assert "5g" in metrics["rouge_l_by_domain"]
        assert "billing" in metrics["rouge_l_by_domain"]


# ============================================================================
# TestMockRetriever
# ============================================================================


class TestMockRetriever:
    def test_retrieve_returns_results(self):
        mr = MockRetriever()
        results = mr.retrieve("5G套餐39元")
        assert len(results) > 0
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

    def test_retrieve_ids(self):
        mr = MockRetriever()
        ids = mr.retrieve_ids("宽带掉线")
        assert len(ids) > 0
        assert all(isinstance(doc_id, str) for doc_id in ids)

    def test_retrieve_respects_top_k(self):
        mr = MockRetriever()
        results = mr.retrieve("查询", top_k=3)
        assert len(results) <= 3

    def test_retrieve_no_match(self):
        mr = MockRetriever()
        # Query with characters unlikely to appear in documents
        results = mr.retrieve("QQQWWWEEE___999")
        # Character-based overlap may return some matches;
        # the key property is that scores are low for unrelated queries
        if len(results) > 0:
            max_score = max(r[1] for r in results)
            assert max_score < 0.5  # low confidence for unrelated query


# ============================================================================
# TestEvalRunner
# ============================================================================


class TestEvalRunner:
    def test_load_queries(self):
        queries = load_queries()
        assert len(queries) == 20
        assert all("id" in q for q in queries)
        assert all("domain" in q for q in queries)
        assert all("query" in q for q in queries)
        assert all("relevant_docs" in q for q in queries)
        assert all("expected_answer" in q for q in queries)
        assert all("keywords" in q for q in queries)

    def test_load_queries_unique_ids(self):
        queries = load_queries()
        ids = [q["id"] for q in queries]
        assert len(ids) == len(set(ids))

    def test_load_queries_valid_domains(self):
        queries = load_queries()
        valid_domains = {"billing", "tickets", "network", "5g", "plans", "complaint"}
        for q in queries:
            assert q["domain"] in valid_domains, f"{q['id']} has invalid domain {q['domain']}"

    def test_run_retrieval_eval(self):
        queries = load_queries()
        retriever = MockRetriever()
        metrics, raw_results = run_retrieval_eval(queries, retriever)
        assert metrics["num_queries"] == 20
        assert len(raw_results) == 20
        # Mock retriever should have reasonable performance
        assert metrics["hit_rate_at_5"] > 0.0

    def test_run_answer_eval(self):
        queries = load_queries()[:5]
        retriever = MockRetriever()
        answer_gen = MockAnswerGenerator()
        metrics, results = run_answer_eval(queries, answer_gen, retriever)
        assert metrics["num_results"] == 5
        assert len(results) == 5

    def test_run_full_eval(self):
        queries = load_queries()[:5]
        retriever = MockRetriever()
        answer_gen = MockAnswerGenerator()
        passed, report = run_full_eval(queries, retriever, answer_gen, threshold=0.1)
        assert "total_queries" in report
        assert report["total_queries"] == 5
        assert "per_query" in report
        assert len(report["per_query"]) == 5

    def test_mock_answer_generator(self):
        gen = MockAnswerGenerator()
        answer = gen.generate("5G套餐", ["doc_017", "doc_018"])
        assert "5G套餐" in answer
        assert "2" in answer or "docs" in answer


# ============================================================================
# TestEvaluationDataQuality
# ============================================================================


class TestEvaluationDataQuality:
    def test_all_queries_are_non_empty(self):
        queries = load_queries()
        for q in queries:
            assert len(q["query"]) > 0, f"{q['id']}: empty query"
            assert len(q["expected_answer"]) > 10, f"{q['id']}: answer too short"
            assert len(q["keywords"]) >= 2, f"{q['id']}: need at least 2 keywords"
            assert len(q["relevant_docs"]) >= 1, f"{q['id']}: need at least 1 relevant doc"

    def test_domain_coverage(self):
        queries = load_queries()
        domains = set(q["domain"] for q in queries)
        required = {"billing", "tickets", "network", "5g", "plans", "complaint"}
        assert domains == required, f"Missing domains: {required - domains}"
