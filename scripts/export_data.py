"""Chuyển dataset KLTN_Knowledge_Base (faiss_merged) sang định dạng nhẹ dùng chung cho Vercel + Python.
Dùng: python scripts/export_data.py <đường_dẫn_KLTN_Knowledge_Base>
Kết quả: data/vectors.f32 (float32, n x 1024, đã chuẩn hóa), data/chunks.json, data/ground_truth.csv"""
import sys, re, json, shutil
import numpy as np, pandas as pd

root = sys.argv[1].rstrip("/\\") + "/"
chunks = pd.read_csv(root + "faiss_merged/kb_chunks_merged.csv")
a = np.load(root + "kb_embeddings_bge_m3.npy")                              # 494 vector Sysmon + MITRE cũ
b = np.load(root + "mitre_attack/enterprise_stix/mitre_embeddings_bge_m3.npy")  # 3814 vector MITRE Enterprise
V = np.vstack([a, b]).astype("float32")
assert len(V) == len(chunks), "số vector và số chunk không khớp"
V /= np.linalg.norm(V, axis=1, keepdims=True)                               # inner product = cosine

MOJI = {"â\x80\x93": "-", "â\x80\x99": "'", "â\x80\x9c": '"', "â\x80\x9d": '"', "â\x80\x94": "-"}

def clean(t):
    for k, v in MOJI.items():
        t = t.replace(k, v)
    t = re.sub(r"\(Citation:[^)]*(?:\)|$)", "", t)                    # bỏ mã trích dẫn MITRE
    t = re.sub(r"\[([^\]]+)\]\(https?://[^)]*\)", r"\1", t)     # [text](url) -> text
    t = re.sub(r"</?code>", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+\.", ".", t)
    return t.strip()

def empty_detection(t):                                          # "Detection of X:" không có nội dung
    parts = t.split("\n", 2)
    return len(parts) > 1 and parts[1].startswith("Detection Strategy:") and len(parts[1]) < 60

keep, out = [], []
for i, r in chunks.iterrows():
    txt = clean(str(r.chunk_text))
    if empty_detection(txt):
        continue
    keep.append(i)
    out.append({"id": r.chunk_id, "type": r.knowledge_type, "ref": r.reference_id, "title": r.title,
                "url": r.source_url if isinstance(r.source_url, str) else "", "text": txt})

V = V[keep]
V.tofile("data/vectors.f32")
json.dump(out, open("data/chunks.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
shutil.copy(root + "retrieval_evaluation_ground_truth_valid.csv", "data/ground_truth.csv")
print(f"{len(chunks)} chunks gốc -> giữ {len(out)} (bỏ {len(chunks)-len(out)} chunk rỗng) | vectors {V.shape}")
