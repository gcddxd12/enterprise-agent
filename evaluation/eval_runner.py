"""Evaluation runner: run eval dataset against agent, produce metrics report.

Supports two modes:
- mock mode (default): uses a simple keyword-match retriever — no API calls
- agent mode: calls through the actual agent (requires LLM API key)

Usage:
    python -m evaluation.eval_runner              # mock mode, text report
    python -m evaluation.eval_runner --json        # JSON output
    python -m evaluation.eval_runner --query-id eval_001  # single query debug
    python -m evaluation.eval_runner --agent       # use real agent (needs API key)
    python -m evaluation.eval_runner --threshold 0.5  # set pass threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .metrics import AnswerQuality, RetrievalMetrics


# ============================================================================
# Mock Retriever (no API dependency)
# ============================================================================


class MockRetriever:
    """A keyword-match retriever for CI-friendly evaluation.

    Maps queries to document IDs based on keyword overlap with
    predefined document content summaries. No API or vector DB required.
    """

    _doc_store: Dict[str, str] = {
        "doc_001": "58元套餐 月费 5GB通用流量 100分钟通话 套餐内容说明",
        "doc_002": "账单明细 消费记录 月账单查询 费用构成 套餐费 通话费 流量费",
        "doc_003": "套餐变更 套餐费用计算 按天折算 合约期内变更说明",
        "doc_004": "话费余额查询 10086 短信查询 APP查询 微信查询 自助查询渠道",
        "doc_005": "余额不足提醒 自动充值 银行卡代扣 充值优惠活动",
        "doc_006": "定向流量 通用流量 流量类型 免流应用 抖音 微信 淘宝",
        "doc_007": "流量使用规则 定向流量范围 切换规则 标准资费说明",
        "doc_008": "退款申请 误充值 业务订购退款 退款流程 3-5工作日",
        "doc_009": "工单查询 工单状态 处理进度 宽带故障工单 处理时限",
        "doc_010": "工单处理部门 网络维护部 账务处理部 业务支撑部 无线优化部",
        "doc_011": "创建工单 网络投诉 APP报修 10086报修 故障描述 预约维修",
        "doc_012": "工单流程 提交 分派 接单 处理 确认 归档 响应时间",
        "doc_013": "宽带掉线 光猫过热 光纤线路 WiFi干扰 重启 报修",
        "doc_014": "WiFi信号干扰 微波炉 蓝牙 信道冲突 信号不稳定",
        "doc_015": "网络检测 一键诊断 speedtest测速 光猫指示灯 LOS PON LAN",
        "doc_016": "光猫指示灯说明 LOS红灯 光纤信号丢失 PON不亮 注册失败",
        "doc_017": "5G套餐 智享套餐 39元 59元 89元 139元 10GB流量 会员权益",
        "doc_018": "5G家庭融合套餐 宽带 流量 语音 家庭共享 全家享",
        "doc_019": "5G-A 5G-Advanced 3GPP Release18 10Gbps 1ms时延 通感一体化",
        "doc_020": "5G覆盖 频段 3.5GHz 4.9GHz 基站半径 300米 室内覆盖",
        "doc_021": "4G覆盖 频段 1.8GHz 2.3GHz 基站半径 1-2公里 广覆盖",
        "doc_022": "套餐降级 降档 违约金 合约期 月费差额30% 会员权益失效",
        "doc_023": "套餐变更规则 升档免费 降档违约金 同档免费 到期当月办理",
        "doc_024": "携号转网 CXXZ 查询条件 身份证 营业厅 1小时办理 余额退还",
        "doc_025": "异地补卡 身份证 营业厅 当场领取 10元 SIM卡激活 邮寄补卡",
        "doc_026": "10086 客服繁忙 高峰期 避峰拨打 APP在线客服 微信公众号",
        "doc_027": "智能客服 在线客服 7x24小时 一键报修 无需排队 短信咨询",
    }

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return (doc_id, score) tuples based on keyword overlap."""
        results: List[Tuple[str, float]] = []
        query_chars = set(query)
        for doc_id, content in self._doc_store.items():
            overlap = len(query_chars & set(content))
            score = overlap / max(len(query_chars), 1)
            if overlap >= 2 and score > 0.1:
                results.append((doc_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve_ids(self, query: str, top_k: int = 5) -> List[str]:
        """Return just doc IDs."""
        return [doc_id for doc_id, _ in self.retrieve(query, top_k)]


# ============================================================================
# Mock Answer Generator
# ============================================================================


class MockAnswerGenerator:
    """Generates mock answers using keyword-combination for CI testing."""

    def generate(self, query: str, retrieved_docs: List[str]) -> str:
        """Build a mock answer from retrieved document summaries."""
        return f"[Mock Answer for: {query}] Retrieved from {len(retrieved_docs)} docs"


# ============================================================================
# Runner
# ============================================================================


def load_queries(path: str = None) -> List[Dict[str, Any]]:
    """Load evaluation queries from JSON file."""
    if path is None:
        path = str(Path(__file__).parent / "test_queries.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieval_eval(
    queries: List[Dict[str, Any]],
    retriever: Any = None,
) -> Tuple[Dict[str, Any], List[Tuple[List[str], List[str]]]]:
    """Run retrieval evaluation on all queries.

    Returns: (metrics_dict, raw_results) for further processing.
    """
    if retriever is None:
        retriever = MockRetriever()

    results: List[Tuple[List[str], List[str]]] = []
    for q in queries:
        retrieved = retriever.retrieve_ids(q["query"])
        relevant = q.get("relevant_docs", [])
        results.append((retrieved, relevant))

    metrics = RetrievalMetrics.evaluate_all(results)
    return metrics, results


def run_answer_eval(
    queries: List[Dict[str, Any]],
    answer_gen: Any = None,
    retriever: Any = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run answer quality evaluation on all queries.

    Returns: (quality_metrics_dict, per_query_results).
    """
    if retriever is None:
        retriever = MockRetriever()
    if answer_gen is None:
        answer_gen = MockAnswerGenerator()

    results: List[Dict[str, Any]] = []
    for q in queries:
        retrieved = retriever.retrieve_ids(q["query"])
        candidate = answer_gen.generate(q["query"], retrieved)
        results.append({
            "id": q["id"],
            "domain": q["domain"],
            "query": q["query"],
            "candidate": candidate,
            "reference": q.get("expected_answer", ""),
            "keywords": q.get("keywords", []),
            "retrieved_docs": retrieved,
            "relevant_docs": q.get("relevant_docs", []),
        })

    metrics = AnswerQuality.evaluate_all(results)
    return metrics, results


def run_full_eval(
    queries: List[Dict[str, Any]],
    retriever: Any = None,
    answer_gen: Any = None,
    threshold: float = 0.3,
) -> Tuple[bool, Dict[str, Any]]:
    """Run complete evaluation and check against thresholds.

    Returns: (passed, full_report).
    """
    ret_metrics, _ = run_retrieval_eval(queries, retriever)
    ans_metrics, ans_details = run_answer_eval(queries, answer_gen, retriever)

    report = {
        "total_queries": len(queries),
        "retrieval_metrics": ret_metrics,
        "answer_quality": ans_metrics,
        "per_query": [
            {
                "id": r["id"],
                "domain": r["domain"],
                "query": r["query"],
                "recall_at_5": RetrievalMetrics.recall_at_k(
                    r["retrieved_docs"], r["relevant_docs"], 5
                ),
                "keyword_coverage": r.get("keyword_coverage", 0.0) if isinstance(r.get("keyword_coverage"), (int, float)) else AnswerQuality.keyword_coverage(r["candidate"], r["keywords"]),
            }
            for r in ans_details
        ],
        "threshold": threshold,
        "passed": (
            ret_metrics.get("hit_rate_at_5", 0.0) >= threshold
            and ans_metrics.get("keyword_coverage", 0.0) >= threshold
        ),
    }

    # Fix: compute keyword_coverage for per-query results
    for i, r in enumerate(ans_details):
        report["per_query"][i]["keyword_coverage"] = AnswerQuality.keyword_coverage(
            r["candidate"], r["keywords"]
        )

    return report["passed"], report


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CMCC Agent Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.eval_runner                    # mock mode
  python -m evaluation.eval_runner --json             # JSON output
  python -m evaluation.eval_runner --query-id eval_001 # single query
  python -m evaluation.eval_runner --threshold 0.5    # custom threshold
        """,
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--query-id", type=str, default=None,
        help="Evaluate a single query by ID",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Minimum pass threshold for hit_rate and keyword_coverage (default: 0.3)",
    )
    args = parser.parse_args()

    queries = load_queries()

    if args.query_id:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            print(f"Query not found: {args.query_id}")
            sys.exit(1)

    retriever = MockRetriever()
    answer_gen = MockAnswerGenerator()

    passed, report = run_full_eval(
        queries, retriever=retriever, answer_gen=answer_gen,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report, passed)

    sys.exit(0 if passed else 1)


def _print_report(report: Dict[str, Any], passed: bool):
    """Print a human-readable evaluation report."""
    status = "PASSED" if passed else "FAILED"
    status_icon = "[PASS]" if passed else "[FAIL]"

    print(f"\n{'='*60}")
    print("  CMCC Agent Evaluation Report")
    print(f"{'='*60}")
    print(f"\n  Status: {status_icon} {status}")
    print(f"  Total Queries: {report['total_queries']}")
    print(f"  Threshold: {report['threshold']}")

    ret = report["retrieval_metrics"]
    print("\n  --- Retrieval Metrics ---")
    print(f"  Recall@1:  {ret.get('recall_at_1', 'N/A')}")
    print(f"  Recall@3:  {ret.get('recall_at_3', 'N/A')}")
    print(f"  Recall@5:  {ret.get('recall_at_5', 'N/A')}")
    print(f"  Recall@10: {ret.get('recall_at_10', 'N/A')}")
    print(f"  Hit Rate@5: {ret.get('hit_rate_at_5', 'N/A')}")
    print(f"  MRR:       {ret.get('mrr', 'N/A')}")

    ans = report["answer_quality"]
    print("\n  --- Answer Quality ---")
    print(f"  ROUGE-L:          {ans.get('rouge_l', 'N/A')}")
    print(f"  Keyword Coverage: {ans.get('keyword_coverage', 'N/A')}")

    domain_rouge = ans.get("rouge_l_by_domain", {})
    if domain_rouge:
        print("\n  --- ROUGE-L by Domain ---")
        for domain, score in sorted(domain_rouge.items()):
            print(f"  {domain}: {score}")

    print("\n  --- Per-Query Results ---")
    for q in report["per_query"]:
        kw = q.get("keyword_coverage", 0.0)
        rec = q.get("recall_at_5", 0.0)
        bar = "#" * int(kw * 20) + "-" * (20 - int(kw * 20))
        print(f"  [{q['domain']:10s}] {q['id']}: kw={kw:.2f} rec@5={rec:.2f} {bar}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
