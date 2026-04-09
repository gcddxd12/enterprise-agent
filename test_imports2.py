#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LangChain 1.2.15的高级导入
"""

import sys
import pkgutil
import importlib

def test_community_imports():
    print("测试langchain_community模块...")

    # 检查langchain_community有哪些子模块
    try:
        import langchain_community
        print(f"langchain_community版本: {getattr(langchain_community, '__version__', '未知')}")

        # 列出所有子模块
        print("\n可用的子模块:")
        for _, name, _ in pkgutil.iter_modules(langchain_community.__path__):
            print(f"  - {name}")
    except ImportError as e:
        print(f"无法导入langchain_community: {e}")
        return

def test_specific_imports():
    print("\n测试特定导入...")

    imports_to_test = [
        "langchain_community.retrievers.ensemble",
        "langchain_community.retrievers.contextual_compression",
        "langchain_community.document_compressors",
        "langchain_community.retrievers.document_compressors",
    ]

    for module_name in imports_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"[OK] {module_name}")
            # 列出模块中的内容
            if hasattr(module, '__all__'):
                print(f"    导出: {module.__all__}")
        except ImportError as e:
            print(f"[FAIL] {module_name}: {e}")

if __name__ == "__main__":
    test_community_imports()
    test_specific_imports()