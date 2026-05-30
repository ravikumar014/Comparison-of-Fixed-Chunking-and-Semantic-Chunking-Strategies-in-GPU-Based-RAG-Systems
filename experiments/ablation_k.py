from src.retrieval.index_builders import (
    build_fixed_index,
    build_semantic_index
)
from src.evaluation.retrieval_metrics import recall_at_k
from src.embeddings.embedder import Embedder
from src.evaluation.experiment_logger import ExperimentLogger
import json

k_values = [1, 3, 5, 10]

logger = ExperimentLogger()
logger.reset_experiment("k_ablation")

with open("data/eval/queries.json") as f:
    queries = json.load(f)

ground_truth = {
    q["query_id"]: q["relevant_docs"]
    for q in queries
}

print("\n=== Top-k Sensitivity Analysis ===")

fixed_index = build_fixed_index()
semantic_index = build_semantic_index()

embedder = Embedder("all-mpnet-base-v2")

def get_results(index):
    results = {}
    for q in queries:
        qid = q["query_id"]

        q_emb = embedder.model.encode(
            [q["query"]],
            normalize_embeddings=True
        )

        retrieved = index.search(q_emb, k=10)  # max k
        results[qid] = retrieved

    return results

fixed_results = get_results(fixed_index)
semantic_results = get_results(semantic_index)

print("\n=== Top-k Ablation ===")

for k in k_values:
    fixed_r = recall_at_k(fixed_results, ground_truth, k)
    sem_r = recall_at_k(semantic_results, ground_truth, k)

    print(f"\nk={k}")
    print(f"Fixed: {fixed_r:.3f}, Semantic: {sem_r:.3f}")

    logger.log("k_ablation", {
        "k": k,
        "fixed_recall": round(fixed_r, 4),
        "semantic_recall": round(sem_r, 4)
    })