#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试langchain_community.retrievers内容
"""

import importlib

def test_retrievers_module():
    print("测试langchain_community.retrievers模块...")

    try:
        module = importlib.import_module("langchain_community.retrievers")
        print(f"[OK] langchain_community.retrievers")

        # 列出导出的类
        if hasattr(module, '__all__'):
            print("导出的类:")
            for name in module.__all__:
                print(f"  - {name}")
        else:
            # 尝试查看模块属性
            print("模块属性 (前20个):")
            attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            for attr in attrs[:20]:
                print(f"  - {attr}")

        # 尝试导入BM25Retriever来确认
        from langchain_community.retrievers import BM25Retriever
        print(f"\nBM25Retriever导入成功")

        # 检查是否有EnsembleRetriever
        if hasattr(module, 'EnsembleRetriever'):
            print("EnsembleRetriever在模块中")
        else:
            print("EnsembleRetriever不在模块中")

    except ImportError as e:
        print(f"[FAIL] langchain_community.retrievers: {e}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_retrievers_module()