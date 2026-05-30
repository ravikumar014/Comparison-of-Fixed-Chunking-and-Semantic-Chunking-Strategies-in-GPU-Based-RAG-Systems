from src.retrieval.index_builders import build_fixed_index, build_semantic_index
from experiments.stage6_rag_generation import run_rag
import json


def main():
    with open("data/eval/queries.json") as f:
        queries = json.load(f)

    fixed_index = build_fixed_index()
    semantic_index = build_semantic_index()

    run_rag(fixed_index, queries, "fixed")
    run_rag(semantic_index, queries, "semantic")


if __name__ == "__main__":
    main()