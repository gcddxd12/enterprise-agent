#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试BM25Retriever的方法
"""

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# 创建测试文档
test_docs = [
    Document(page_content="如何重置密码？您可以通过登录页面点击'忘记密码'链接重置密码。"),
    Document(page_content="产品价格信息：企业版每年10,000元，包含技术支持。"),
    Document(page_content="技术支持时间：工作日9:00-18:00，电话400-123-4567。"),
]

# 创建BM25Retriever
retriever = BM25Retriever.from_documents(test_docs)

print("BM25Retriever类型:", type(retriever))
print("\nBM25Retriever方法:")
for attr in dir(retriever):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# 测试检索
print("\n测试检索:")
try:
    results = retriever.invoke("密码")
    print(f"invoke结果: {results}")
except Exception as e:
    print(f"invoke失败: {e}")

print("\n尝试get_relevant_documents:")
try:
    results = retriever.get_relevant_documents("密码")
    print(f"get_relevant_documents结果: {results}")
except Exception as e:
    print(f"get_relevant_documents失败: {e}")

print("\n尝试__call__:")
try:
    results = retriever("密码")
    print(f"__call__结果: {results}")
except Exception as e:
    print(f"__call__失败: {e}")