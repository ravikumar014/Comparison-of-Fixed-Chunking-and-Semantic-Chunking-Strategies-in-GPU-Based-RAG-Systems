from src.retrieval.index_builders import build_fixed_index
from experiments.stage5_retrieval_evaluation import evaluate
from src.evaluation.experiment_logger import ExperimentLogger
import json

chunk_sizes = [256, 512, 768]

logger = ExperimentLogger()
logger.reset_experiment("chunk_size_ablation")

with open("data/eval/queries.json") as f:
    queries = json.load(f)

ground_truth = {
    q["query_id"]: q["relevant_docs"]
    for q in queries
}

print("\n=== Chunk Size Ablation (Fixed Chunking) ===")

for size in chunk_sizes:
    print(f"\n--- Chunk Size: {size} ---")

    index = build_fixed_index(
        use_cache=False,  # important
        chunk_size=size
    )

    recall5, mrr, recall1, precision5, ndcg5, results = evaluate(index, queries, ground_truth)

    print(f"Recall@5: {recall5:.3f}")
    print(f"MRR: {mrr:.3f}")

    logger.log("chunk_size_ablation", {
        "chunk_size": size,
        "recall@5": round(recall5, 4),
        "mrr": round(mrr, 4)
    })