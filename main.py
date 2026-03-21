import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 获取API密钥（需要先在Fireworks注册获取）
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")


def get_qwen_embedding(text):
    url = "https://api.fireworks.ai/inference/v1/embeddings"

    payload = {
        "input": text,
        "model": "fireworks/qwen3-embedding-8b",
        "dimensions": 1024  # 可选：可以降低维度节省存储
    }

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()


# 测试
embedding_result = get_qwen_embedding("Python中列表和元组的区别是什么？")
print(embedding_result)