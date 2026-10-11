"""RQ3 - chất lượng truy xuất trên ground truth của nhóm: Hit@1/3/5 và MRR.
python eval_retrieval.py"""
import os, pandas as pd, numpy as np
import config
from kb_data import KnowledgeBase

kb = KnowledgeBase()
gt = pd.read_csv(os.path.join(config.DATA_DIR, "ground_truth.csv"))
rows = []
for _, r in gt.iterrows():
    hits = kb.search(r["query"], 10)
    refs = [h["ref"] for h in hits]
    rank = refs.index(r["expected_reference_id"]) + 1 if r["expected_reference_id"] in refs else None
    rows.append({"query_id": r["query_id"], "expected": r["expected_reference_id"], "rank": rank,
                 "hit@1": rank == 1, "hit@3": bool(rank and rank <= 3), "hit@5": bool(rank and rank <= 5),
                 "rr": 1 / rank if rank else 0.0})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nHit@1={df['hit@1'].mean():.3f}  Hit@3={df['hit@3'].mean():.3f}  Hit@5={df['hit@5'].mean():.3f}  MRR={df.rr.mean():.3f}")
df.to_csv("retrieval_eval_results.csv", index=False)
