import json
from src.embeddings.embedder import Embedder
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    recall_at_1,
    precision_at_k,
    ndcg_at_k
)
from src.evaluation.save_metrics import save_metrics


def evaluate(index, queries, ground_truth, k=5):
    embedder = Embedder("all-mpnet-base-v2")
    results = {}

    for q in queries:
        qid = q["query_id"]
        query_text = q["query"]

        query_emb = embedder.model.encode(
            [query_text],
            normalize_embeddings=True
        )

        retrieved = index.search(query_emb, k=k)
        results[qid] = retrieved

    # recall = recall_at_k(results, ground_truth, k)
    # mrr = mean_reciprocal_rank(results, ground_truth)
    recall5 = recall_at_k(results, ground_truth, k=5)
    mrr = mean_reciprocal_rank(results, ground_truth)
    recall1 = recall_at_1(results, ground_truth)
    precision5 = precision_at_k(results, ground_truth, k=5)
    ndcg5 = ndcg_at_k(results, ground_truth, k=5)


    return recall5, mrr, recall1, precision5, ndcg5, results


def main():

    with open("data/eval/queries.json") as f:
        queries = json.load(f)

    ground_truth = {
        q["query_id"]: q["relevant_docs"]
        for q in queries
    }

    # from experiments.stage4_fixed_embeddings_faiss import index as fixed_index
    # from experiments.stage4_semantic_embeddings_faiss import index as semantic_index

    from src.retrieval.index_builders import (
        build_fixed_index,
        build_semantic_index
    )

    fixed_index = build_fixed_index()
    semantic_index = build_semantic_index()


    fixed_recall5, fixed_mrr, fixed_recall1, fixed_precision5, fixed_ndcg5, fixed_results = evaluate(fixed_index, queries, ground_truth)
    semantic_recall5, semantic_mrr, semantic_recall1, semantic_precision5, semantic_ndcg5, semantic_results = evaluate(semantic_index, queries, ground_truth)

    print("\n=== Retrieval Evaluation ===")
    print(f"Fixed Chunking  | Recall@5: {fixed_recall5:.3f} | MRR: {fixed_mrr:.3f} | Recall@1: {fixed_recall1:.3f} | Precision: {fixed_precision5:.3f} | NDCG5: {fixed_ndcg5:.3f}")
    print(f"Semantic Chunk | Recall@5: {semantic_recall5:.3f} | MRR: {semantic_mrr:.3f} | Recall@1: {semantic_recall1:.3f} | Precision: {semantic_precision5:.3f} | NDCG5: {semantic_ndcg5:.3f}")
    
    save_metrics("fixed", {
        "recall@1": fixed_recall1,
        "recall@5": fixed_recall5,
        "mrr": fixed_mrr,
        "precision@5": fixed_precision5,
        "ndcg@5": fixed_ndcg5
    },)
    save_metrics("semantic", {
        "recall@1": semantic_recall1,
        "recall@5": semantic_recall5,
        "mrr": semantic_mrr,
        "precision@5": semantic_precision5,
        "ndcg@5": semantic_ndcg5
    },)

if __name__ == "__main__":
    main()
