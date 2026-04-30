"""Evaluation metrics for CMCC Agent.

Zero-dependency implementations:
- Retrieval metrics: Recall@k, MRR, Hit Rate
- Answer quality: ROUGE-L (LCS-based), keyword coverage
"""

from typing import Any, Dict, List, Tuple


# ============================================================================
# Retrieval Metrics
# ============================================================================


class RetrievalMetrics:
    """Information Retrieval quality metrics."""

    @staticmethod
    def recall_at_k(
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int = 5,
    ) -> float:
        """Recall@k: fraction of relevant docs found in top-k retrieved results.

        Args:
            retrieved_doc_ids: IDs of retrieved documents (ordered by relevance).
            relevant_doc_ids: IDs of known-relevant documents (ground truth).
            k: Consider only top-k retrieved documents.

        Returns:
            recall value in [0.0, 1.0].
        """
        if not relevant_doc_ids:
            return 1.0  # nothing to find
        top_k = retrieved_doc_ids[:k]
        found = sum(1 for doc_id in top_k if doc_id in set(relevant_doc_ids))
        return found / len(relevant_doc_ids)

    @staticmethod
    def mrr(
        queries_results: List[Tuple[List[str], List[str]]],
    ) -> float:
        """Mean Reciprocal Rank: average of 1/rank of first relevant doc.

        For each query (retrieved_ids, relevant_ids), finds the position
        of the first relevant document in the ranked list.

        Args:
            queries_results: List of (retrieved_doc_ids, relevant_doc_ids) tuples.

        Returns:
            MRR value in [0.0, 1.0].
        """
        if not queries_results:
            return 0.0

        reciprocal_ranks = []
        for retrieved, relevant in queries_results:
            relevant_set = set(relevant)
            rank = 1
            found = False
            for doc_id in retrieved:
                if doc_id in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    break
                rank += 1
            if not found:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def hit_rate(
        queries_results: List[Tuple[List[str], List[str]]],
        k: int = 5,
    ) -> float:
        """Hit Rate @ k: fraction of queries where at least one relevant doc is in top-k.

        Args:
            queries_results: List of (retrieved_doc_ids, relevant_doc_ids) tuples.
            k: Consider only top-k retrieved documents.

        Returns:
            Hit rate in [0.0, 1.0].
        """
        if not queries_results:
            return 0.0

        hits = 0
        for retrieved, relevant in queries_results:
            relevant_set = set(relevant)
            if any(doc_id in relevant_set for doc_id in retrieved[:k]):
                hits += 1
        return hits / len(queries_results)

    @staticmethod
    def evaluate_all(
        queries_results: List[Tuple[List[str], List[str]]],
        k_values: List[int] = None,
    ) -> Dict[str, Any]:
        """Compute all retrieval metrics at once.

        Args:
            queries_results: List of (retrieved_doc_ids, relevant_doc_ids) tuples.
            k_values: k values for Recall@k and Hit Rate. Default: [1, 3, 5, 10].

        Returns:
            Dict with 'recall@k', 'hit_rate@k', 'mrr', and 'num_queries'.
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics: Dict[str, Any] = {
            "num_queries": len(queries_results),
        }
        # Workaround: "recall@k" is not a valid list key in old Python.
        for k in k_values:
            r = [
                RetrievalMetrics.recall_at_k(r_ids, g_ids, k)
                for r_ids, g_ids in queries_results
            ]
            r_avg = sum(r) / len(r) if r else 0.0
            metrics[f"recall_at_{k}"] = round(r_avg, 4)

            h = RetrievalMetrics.hit_rate(queries_results, k)
            metrics[f"hit_rate_at_{k}"] = round(h, 4)

        metrics["mrr"] = round(RetrievalMetrics.mrr(queries_results), 4)
        return metrics


# ============================================================================
# Answer Quality Metrics
# ============================================================================


class AnswerQuality:
    """Answer quality evaluation using zero-dependency scoring."""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into tokens.

        For Chinese-dominant text, uses character-level tokens.
        For mixed text, splits on word boundaries.
        """
        tokens: List[str] = []
        for ch in text:
            if ch.isalpha():
                tokens.append(ch.lower())
            elif ch.isdigit():
                tokens.append(ch)
            elif ch.isspace():
                if tokens and tokens[-1] != " ":
                    tokens.append(" ")
            else:
                if tokens and tokens[-1] != " ":
                    tokens.append(" ")
        return [t for t in tokens if t != " "]

    @staticmethod
    def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
        """Longest Common Subsequence length using dynamic programming."""
        if not seq_a or not seq_b:
            return 0
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def rouge_l(candidate: str, reference: str) -> float:
        """ROUGE-L: Longest Common Subsequence based F1 score.

        Uses character-level LCS for Chinese text compatibility.
        No external NLP dependency required.

        Args:
            candidate: Generated answer text.
            reference: Reference (ground truth) answer text.

        Returns:
            F1 score in [0.0, 1.0].
        """
        if not candidate or not reference:
            return 0.0

        cand_tokens = AnswerQuality._tokenize(candidate)
        ref_tokens = AnswerQuality._tokenize(reference)

        if not cand_tokens or not ref_tokens:
            return 0.0

        lcs_len = AnswerQuality._lcs_length(cand_tokens, ref_tokens)
        precision = lcs_len / len(cand_tokens)
        recall = lcs_len / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def keyword_coverage(candidate: str, expected_keywords: List[str]) -> float:
        """Fraction of expected keywords found in the candidate answer.

        Case-insensitive matching.

        Args:
            candidate: Generated answer text.
            expected_keywords: List of keywords that should appear.

        Returns:
            Coverage fraction in [0.0, 1.0].
        """
        if not expected_keywords:
            return 1.0

        candidate_lower = candidate.lower()
        found = sum(
            1 for kw in expected_keywords
            if kw.lower() in candidate_lower
        )
        return found / len(expected_keywords)

    @staticmethod
    def evaluate_all(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute all answer quality metrics across an evaluation dataset.

        Args:
            results: List of {candidate, reference, keywords, ...} dicts.

        Returns:
            Dict with 'rouge_l', 'keyword_coverage', and per-domain breakdown.
        """
        if not results:
            return {"num_results": 0, "rouge_l": 0.0, "keyword_coverage": 0.0}

        rouge_scores = []
        kw_scores = []
        domain_scores: Dict[str, List[float]] = {}

        for r in results:
            candidate = r.get("candidate", "")
            reference = r.get("reference", "")
            keywords = r.get("keywords", [])
            domain = r.get("domain", "unknown")

            rl = AnswerQuality.rouge_l(candidate, reference)
            kw = AnswerQuality.keyword_coverage(candidate, keywords)

            rouge_scores.append(rl)
            kw_scores.append(kw)

            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(rl)

        domain_avg = {
            d: round(sum(s) / len(s), 4) for d, s in domain_scores.items()
        }

        return {
            "num_results": len(results),
            "rouge_l": round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0.0,
            "keyword_coverage": round(sum(kw_scores) / len(kw_scores), 4) if kw_scores else 0.0,
            "rouge_l_by_domain": domain_avg,
        }
