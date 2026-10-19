"""
共享 Embedding 模型（惰性单例）。

bge-large-zh-v1.5 加载一次要几秒钟、占约 1GB 内存，
全项目只加载一份，记忆层和知识库共用——和 ingest.py 入库
policies 集合时用的是同一个模型，向量空间才一致。

模型文件已在本地缓存（入库时下载过），HF_HUB_OFFLINE=1
让它不联网直接读缓存。
"""
import os

# 必须在 sentence_transformers 被 import 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_EF = None


def get_embedding_function():
    """返回全项目共享的向量模型（第一次调用时才真正加载）。"""
    global _EF
    if _EF is None:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        _EF = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-zh-v1.5",
            device="cpu",
        )
    return _EF