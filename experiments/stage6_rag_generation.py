import json
from src.generation.rag_generator import RAGGenerator
from src.generation.context_builder import build_context
from src.embeddings.embedder import Embedder


def run_rag(index, queries, label):
    embedder = Embedder("all-mpnet-base-v2")
    generator = RAGGenerator("microsoft/phi-3-mini-4k-instruct")

    outputs = []

    for q in queries:
        query = q["query"]
        qid = q["query_id"]

        q_emb = embedder.model.encode(
            [query],
            normalize_embeddings=True
        )

        retrieved = index.search(q_emb, k=5)
        context = build_context(retrieved)

        answer = generator.generate(context, query)

        outputs.append({
            "query_id": qid,
            "query": query,
            "answer": answer,
            "retrieved_docs": [r["chunk"]["doc_id"] for r in retrieved]
        })

    with open(f"results/rag_answers_{label}.json", "w") as f:
        json.dump(outputs, f, indent=2)


def main():
    with open("data/eval/queries.json") as f:
        queries = json.load(f)

    # from experiments.stage4_fixed_embeddings_faiss import index as fixed_index
    # from experiments.stage4_semantic_embeddings_faiss import index as semantic_index
    from src.retrieval.index_builders import (
        build_fixed_index,
        build_semantic_index
    )

    fixed_index = build_fixed_index()
    semantic_index = build_semantic_index()


    run_rag(fixed_index, queries, label="fixed")
    run_rag(semantic_index, queries, label="semantic")


if __name__ == "__main__":
    main()
