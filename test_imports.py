#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LangChain 1.2.15的导入
"""

import sys

def test_imports():
    print("测试LangChain导入...")

    # 测试不同的导入路径
    imports_to_test = [
        ("langchain.retrievers", "EnsembleRetriever"),
        ("langchain.retrievers.contextual_compression", "ContextualCompressionRetriever"),
        ("langchain.retrievers.document_compressors", "LLMChainExtractor"),
        ("langchain_community.retrievers", "BM25Retriever"),
        ("langchain.schema", "Document"),
        ("langchain_core.documents", "Document"),
        ("langchain_core.retrievers", "BaseRetriever"),
        ("langchain_core.embeddings", "Embeddings"),
        ("langchain_core.language_models", "BaseLanguageModel"),
    ]

    for module_name, class_name in imports_to_test:
        try:
            exec(f"from {module_name} import {class_name}")
            print(f"[OK] {module_name}.{class_name}")
        except ImportError as e:
            print(f"[FAIL] {module_name}.{class_name}: {e}")
        except Exception as e:
            print(f"[WARN] {module_name}.{class_name}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_imports()