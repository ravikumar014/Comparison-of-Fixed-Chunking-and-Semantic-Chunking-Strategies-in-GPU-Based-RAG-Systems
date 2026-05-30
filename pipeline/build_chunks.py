from src.retrieval.index_builders import build_fixed_index, build_semantic_index

def main():
    build_fixed_index(use_cache=False)
    build_semantic_index(use_cache=False)
    print("Chunks built and cached.")


if __name__ == "__main__":
    main()