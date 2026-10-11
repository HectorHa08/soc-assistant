"""python evaluate.py compare            -> Baseline vs RAG (RQ1, RQ2, RQ4)
   python evaluate.py ablation          -> Top-K 1/3/5 (RQ3)
   Thêm --ragas để chấm RAGAS (cần judge LLM; xem ghi chú cuối file)."""
import json, re, sys, statistics as st
import pandas as pd
from rag import SOCAssistant

TECH = re.compile(r"T\d{4}(?:\.\d{3})?")


def prf(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run(bot, tests, use_rag):
    rows = []
    for t in tests:
        res = bot.answer(t["question"], use_rag=use_rag)
        pred = TECH.findall(res["answer"])
        p, r, f = prf(pred, t["techniques"])
        rows.append({"id": t["id"], "task": t["task"], "precision": p, "recall": r, "f1": f,
                     "task_success": float(r == 1.0 and p > 0),
                     "retrieval_s": res["retrieval_s"], "generation_s": res["generation_s"],
                     "total_s": res["total_s"], "answer": res["answer"],
                     "contexts": res["contexts"], "question": t["question"],
                     "ground_truth": t["ground_truth"]})
    return pd.DataFrame(rows)


def summarize(df):
    return {"P": df.precision.mean(), "R": df.recall.mean(), "F1": df.f1.mean(),
            "success": df.task_success.mean(),
            "lat_mean": df.total_s.mean(), "lat_std": df.total_s.std(),
            "retr_mean": df.retrieval_s.mean(), "gen_mean": df.generation_s.mean()}


def ragas_score(df):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (faithfulness, answer_relevancy, answer_correctness,
                               context_precision, context_recall)
    ds = Dataset.from_dict({"question": df.question.tolist(), "answer": df.answer.tolist(),
                            "contexts": df.contexts.tolist(), "ground_truth": df.ground_truth.tolist()})
    # Ghi chú: mặc định RAGAS dùng OpenAI làm judge. Muốn dùng Qwen local, truyền
    # llm=LangchainLLMWrapper(ChatOllama(...)), embeddings=LangchainEmbeddingsWrapper(...) vào evaluate().
    return evaluate(ds, metrics=[faithfulness, answer_relevancy, answer_correctness,
                                 context_precision, context_recall]).to_pandas()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    tests = json.load(open("testset.json", encoding="utf-8"))
    if mode == "compare":
        bot = SOCAssistant()
        out = {}
        for name, flag in [("Baseline", False), ("Proposed", True)]:
            df = run(bot, tests, flag)
            df.to_csv(f"results_{name}.csv", index=False)
            out[name] = summarize(df)
            if "--ragas" in sys.argv and flag:
                ragas_score(df).to_csv("ragas_proposed.csv", index=False)
        print(pd.DataFrame(out).round(3))
    elif mode == "ablation":
        rows = []
        for k in (1, 3, 5):   # chunk đã dựng sẵn trong dataset của nhóm nên chỉ thay đổi Top-K
            s = summarize(run(SOCAssistant(top_k=k), tests, True))
            rows.append({"top_k": k, **s})
        df = pd.DataFrame(rows).round(3)
        df.to_csv("ablation.csv", index=False)
        print(df)
