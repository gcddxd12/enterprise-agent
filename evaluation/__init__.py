"""Evaluation module — retrieval and answer quality metrics for the CMCC Agent.

Provides:
- Labeled test queries across 6 business domains
- Zero-dependency IR metrics (Recall@k, MRR, Hit Rate)
- Zero-dependency answer quality scoring (ROUGE-L, keyword coverage)
- CI-friendly eval runner (no API calls required in mock mode)
"""
