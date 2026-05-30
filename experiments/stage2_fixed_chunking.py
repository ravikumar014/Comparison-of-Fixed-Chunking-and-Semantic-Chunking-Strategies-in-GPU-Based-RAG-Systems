from src.loaders.document_loader import DocumentLoader
from src.preprocessing.clean_text import clean_text
from src.chunking.fixed import FixedChunker


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

    chunker = FixedChunker(
        tokenizer_name="bert-base-uncased",
        chunk_size=512,
        overlap=50
    )

    all_chunks = []

    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(all_chunks)}")

    print("\nSample chunk:\n")
    sample = all_chunks[0]
    print("Chunk ID:", sample["chunk_id"])
    print("Tokens:", sample["num_tokens"])
    print(sample["text"][:500])


if __name__ == "__main__":
    main()

#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