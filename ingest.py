# ingest.py —— 预案入库脚本（Policies 库）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 必须在 import 前
os.environ["HF_HUB_OFFLINE"] = "1"                    # 离线加载本地缓存的 bge

import fitz  # pymupdf；会有 deprecation 警告，不影响
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PDF_DIR = "./data/pdfs"        # 预案存放处
CHROMA_DIR = "./data/chroma"   # 向量库落盘处（这个文件夹就是向量库本身）
CHUNK_SIZE, OVERLAP = 300, 50  # 沿用你项目已确立的切块配置

ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-large-zh-v1.5", device="cpu")
client = chromadb.PersistentClient(path=CHROMA_DIR,
                                   settings=chromadb.Settings(anonymized_telemetry=False))
coll = client.get_or_create_collection("policies", embedding_function=ef)

import re

def chunk_page(text):
    """一页文字切成 300 字、重叠 50 字的块"""
    out, i = [], 0
    while i < len(text):
        piece = text[i:i + CHUNK_SIZE]
        if piece.strip():
            out.append(piece)
        i += CHUNK_SIZE - OVERLAP
    return out

for fname in sorted(os.listdir(PDF_DIR)):
    if not fname.endswith(".pdf"):
        continue
    doc = fitz.open(os.path.join(PDF_DIR, fname))
    n = 0
    for pno in range(doc.page_count):
        text = re.sub(r"—\s*\d+\s*—", "", doc[pno].get_text()).replace("\n", "")
        if len(text.strip()) < 20:   # 近空白页（如纯图片的体系图页）直接跳过
            continue
        for j, c in enumerate(chunk_page(text)):
            cid = f"{fname}_p{pno+1}_c{j}"
            coll.upsert(ids=[cid], documents=[c],
                        metadatas=[{"plan": fname, "page": pno + 1}])
            n += 1
    print(f"{fname} 入库 {n} 块")

print("总切块数:", coll.count())