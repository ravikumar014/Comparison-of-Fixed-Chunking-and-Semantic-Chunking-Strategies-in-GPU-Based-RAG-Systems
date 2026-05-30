from src.retrieval.index_builders import build_semantic_index
from experiments.stage5_retrieval_evaluation import evaluate
import json

thresholds = [0.4, 0.5, 0.6, 0.7]

with open("data/eval/queries.json") as f:
    queries = json.load(f)

ground_truth = {
    q["query_id"]: q["relevant_docs"]
    for q in queries
}

for t in thresholds:
    print(f"\n=== Threshold: {t} ===")

    index = build_semantic_index(use_cache=False, threshold=t)

    recall, mrr = evaluate(index, queries, ground_truth)

    print(f"Recall@5: {recall:.3f}, MRR: {mrr:.3f}")