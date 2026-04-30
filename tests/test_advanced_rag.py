"""Tests for advanced_rag_system.py — cache, query expansion, retrieval instantiation."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, ".")

from advanced_rag_system import (
    QueryExpander,
    VectorCache,
    AdvancedRAGRetriever,
    create_advanced_rag_system,
)


_MOCK_EMBED = lambda text: [0.1, 0.2, 0.3]
_MOCK_SEARCH = lambda query: []


# ============================================================================
# TestVectorCache
# ============================================================================


class TestVectorCache:
    def test_init_creates_cache_dir(self, tmp_path):
        cache_dir = str(tmp_path / "test_cache")
        vc = VectorCache(cache_dir=cache_dir)
        assert os.path.exists(cache_dir)

    def test_get_embedding_no_cache(self, tmp_path):
        vc = VectorCache(cache_dir=str(tmp_path / "cache"))
        # With no cached embedding, should return the computed value
        result = vc.get_embedding("hello world", _MOCK_EMBED)
        assert result is not None
        assert isinstance(result, list)

    def test_get_embedding_returns_consistent(self, tmp_path):
        vc = VectorCache(cache_dir=str(tmp_path / "cache"))
        r1 = vc.get_embedding("hello", _MOCK_EMBED)
        r2 = vc.get_embedding("hello", _MOCK_EMBED)
        # Same call should return same result
        assert r1 == r2

    def test_get_search_results_no_cache(self, tmp_path):
        vc = VectorCache(cache_dir=str(tmp_path / "cache"))
        result = vc.get_search_results("test query", _MOCK_SEARCH)
        assert isinstance(result, list)

    def test_get_stats_returns_dict(self, tmp_path):
        vc = VectorCache(cache_dir=str(tmp_path / "cache"))
        stats = vc.get_stats()
        assert isinstance(stats, dict)

    def test_clear_cache(self, tmp_path):
        vc = VectorCache(cache_dir=str(tmp_path / "cache"))
        vc.get_embedding("test", _MOCK_EMBED)
        vc.clear_cache()
        # After clear, should still work but stats reset
        stats = vc.get_stats()
        assert isinstance(stats, dict)

    def test_get_embedding_different_texts(self, tmp_path):
        vc = VectorCache(cache_dir=str(tmp_path / "cache"))
        r1 = vc.get_embedding("text one", _MOCK_EMBED)
        r2 = vc.get_embedding("text two", _MOCK_EMBED)
        # Both should return valid results
        assert r1 is not None
        assert r2 is not None


# ============================================================================
# TestQueryExpander (mock mode)
# ============================================================================


class TestQueryExpander:
    def test_mock_expand_generates_variants(self):
        expander = QueryExpander(use_mock=True)
        variants = expander.expand_query("话费查询")
        assert len(variants) > 0
        assert isinstance(variants, list)

    def test_synonym_dict_non_empty(self):
        expander = QueryExpander(use_mock=True)
        assert len(expander.synonyms) > 0

    def test_max_variants_limit(self):
        expander = QueryExpander(use_mock=True)
        variants = expander.expand_query("查询", max_variants=3)
        assert len(variants) <= 3

    def test_empty_query(self):
        expander = QueryExpander(use_mock=True)
        variants = expander.expand_query("")
        assert isinstance(variants, list)

    def test_expand_returns_strings(self):
        expander = QueryExpander(use_mock=True)
        variants = expander.expand_query("5G套餐有哪些")
        assert all(isinstance(v, str) for v in variants)

    def test_question_patterns_defined(self):
        expander = QueryExpander(use_mock=True)
        assert len(expander.question_patterns) > 0


# ============================================================================
# TestAdvancedRAGRetrieverInstantiation
# ============================================================================


class TestAdvancedRAGRetrieverInstantiation:
    def test_instantiate_without_retrievers(self):
        retriever = AdvancedRAGRetriever(
            query_expander=QueryExpander(use_mock=True),
            use_cache=False,
            cache_dir=tempfile.mkdtemp(),
        )
        assert retriever is not None
        assert retriever.query_expander is not None

    def test_instantiate_with_cache(self, tmp_path):
        retriever = AdvancedRAGRetriever(
            query_expander=QueryExpander(use_mock=True),
            use_cache=True,
            cache_dir=str(tmp_path / "rag_cache"),
        )
        assert retriever.use_cache is True

    def test_get_relevant_documents_empty(self):
        retriever = AdvancedRAGRetriever(
            query_expander=QueryExpander(use_mock=True),
            use_cache=False,
            cache_dir=tempfile.mkdtemp(),
        )
        results = retriever.get_relevant_documents("anything")
        assert isinstance(results, list)


# ============================================================================
# TestCreateAdvancedRAGSystem
# ============================================================================


class TestCreateAdvancedRAGSystem:
    def test_create_without_retrievers(self):
        system = create_advanced_rag_system()
        assert system is not None
        assert isinstance(system, AdvancedRAGRetriever)
