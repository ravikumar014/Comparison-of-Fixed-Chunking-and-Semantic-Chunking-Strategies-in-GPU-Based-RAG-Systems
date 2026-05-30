from src.loaders.document_loader import DocumentLoader
from src.preprocessing.clean_text import clean_text
from src.chunking.fixed import FixedChunker
from src.embeddings.embedder import Embedder
from src.retrieval.faiss_index import FaissIndex


def main():
    loader = DocumentLoader("data/raw")
    documents = loader.load_all()

    documents = [
        {**doc, "text": clean_text(doc["text"])}
        for doc in documents
    ]

    chunker = FixedChunker(
        tokenizer_name="bert-base-uncased",
        chunk_size=512,
        overlap=50
    )

    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))

    print(f"Total fixed chunks: {len(chunks)}")

    embedder = Embedder("all-mpnet-base-v2")
    embeddings = embedder.embed_chunks(chunks)

    index = FaissIndex(embedding_dim=embeddings.shape[1])
    index.add(embeddings, chunks)

    print("FAISS index built (fixed chunking).")

    # Sanity query
    query = "What is the main contribution of the transformer architecture?"
    query_emb = embedder.model.encode(
        [query],
        normalize_embeddings=True
    )

    results = index.search(query_emb, k=3)
    for r in results:
        print("\nScore:", r["score"])
        print(r["chunk"]["text"][:300])


if __name__ == "__main__":
    main()
