from typing import List, Dict
import math


def recall_at_k(results: Dict[str, List[Dict]], ground_truth: Dict[str, List[str]], k: int) -> float:
    hits = 0
    for qid, retrieved_chunks in results.items():
        retrieved_docs = {
            chunk["chunk"]["doc_id"]
            for chunk in retrieved_chunks[:k]
        }
        if any(doc in retrieved_docs for doc in ground_truth[qid]):
            hits += 1
    return hits / len(results)


def mean_reciprocal_rank(results: Dict[str, List[Dict]], ground_truth: Dict[str, List[str]]) -> float:
    rr_sum = 0.0
    for qid, retrieved_chunks in results.items():
        for rank, item in enumerate(retrieved_chunks, start=1):
            if item["chunk"]["doc_id"] in ground_truth[qid]:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(results)

def recall_at_1(results, ground_truth):
    hits = 0
    for qid, retrieved in results.items():
        if len(retrieved) == 0:
            continue
        top_doc = retrieved[0]["chunk"]["doc_id"]
        if top_doc in ground_truth[qid]:
            hits += 1
    return hits / len(results)

def precision_at_k(results, ground_truth, k=5):
    total = 0
    for qid, retrieved in results.items():
        retrieved_docs = [
            item["chunk"]["doc_id"] for item in retrieved[:k]
        ]
        hits = sum(1 for d in retrieved_docs if d in ground_truth[qid])
        total += hits / k
    return total / len(results)

def ndcg_at_k(results, ground_truth, k=5):
    def dcg(rels):
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))

    total = 0
    for qid, retrieved in results.items():
        rels = [
            1 if item["chunk"]["doc_id"] in ground_truth[qid] else 0
            for item in retrieved[:k]
        ]
        ideal = sorted(rels, reverse=True)
        if sum(ideal) == 0:
            continue
        total += dcg(rels) / dcg(ideal)

    return total / len(results)
