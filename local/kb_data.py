"""Nạp dataset của nhóm (vectors.f32 + chunks.json do scripts/export_data.py tạo) và tìm kiếm cosine chính xác.
Tương đương faiss.IndexFlatIP trong faiss_merged của nhóm (4 nghìn vector nên không cần ANN)."""
import json, os
import numpy as np
import config

DIM = 1024

class KnowledgeBase:
    def __init__(self, data_dir=config.DATA_DIR, embedder=None):
        self.chunks = json.load(open(os.path.join(data_dir, "chunks.json"), encoding="utf-8"))
        self.vec = np.fromfile(os.path.join(data_dir, "vectors.f32"), dtype="float32").reshape(-1, DIM)
        assert len(self.vec) == len(self.chunks)
        self._embedder = embedder

    def embed(self, text):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder.encode(text[:config.MAX_QUERY_CHARS], normalize_embeddings=True).astype("float32")

    def search_vec(self, q, k):
        s = self.vec @ q
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [{**self.chunks[i], "score": float(s[i])} for i in top]

    def search(self, query, k=config.TOP_K):
        return self.search_vec(self.embed(query), k)
