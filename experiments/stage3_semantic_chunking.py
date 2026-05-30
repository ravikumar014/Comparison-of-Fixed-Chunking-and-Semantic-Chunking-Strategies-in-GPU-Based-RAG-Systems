from src.loaders.document_loader import DocumentLoader
from src.preprocessing.clean_text import clean_text
from src.chunking.semantic import SemanticChunker


def main():
    loader = DocumentLoader("data/raw")
    documents = loader.load_all()

    documents = [
        {
            **doc,
            "text": clean_text(doc["text"])
        }
        for doc in documents
    ]

    chunker = SemanticChunker(
        embedding_model_name="all-mpnet-base-v2",
        similarity_threshold=0.5
    )

    all_chunks = []

    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"Documents loaded: {len(documents)}")
    print(f"Total semantic chunks created: {len(all_chunks)}")

    sample = all_chunks[0]
    print("\nSample semantic chunk:\n")
    print("Chunk ID:", sample["chunk_id"])
    print("Sentences:", sample["num_sentences"])
    print(sample["text"][:500])


if __name__ == "__main__":
    main()
