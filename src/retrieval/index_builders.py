# from src.loaders.document_loader import DocumentLoader
# from src.preprocessing.clean_text import clean_text
# from src.chunking.fixed import FixedChunker
# from src.embeddings.embedder import Embedder
# from src.retrieval.faiss_index import FaissIndex


# def build_fixed_index(data_dir="data/raw"):
#     loader = DocumentLoader(data_dir)
#     documents = loader.load_all()

#     documents = [
#         {**doc, "text": clean_text(doc["text"])}
#         for doc in documents
#     ]

#     chunker = FixedChunker(
#         tokenizer_name="bert-base-uncased",
#         chunk_size=512,
#         overlap=50
#     )

#     chunks = []
#     for doc in documents:
#         chunks.extend(chunker.chunk_document(doc))

#     embedder = Embedder("all-mpnet-base-v2")
#     embeddings = embedder.embed_chunks(chunks)

#     index = FaissIndex(embedding_dim=embeddings.shape[1])
#     index.add(embeddings, chunks)

#     return index


# from src.chunking.semantic import SemanticChunker


# def build_semantic_index(data_dir="data/raw"):
#     loader = DocumentLoader(data_dir)
#     documents = loader.load_all()

#     documents = [
#         {**doc, "text": clean_text(doc["text"])}
#         for doc in documents
#     ]

#     chunker = SemanticChunker(
#         embedding_model_name="all-mpnet-base-v2",
#         similarity_threshold=0.75
#     )

#     chunks = []
#     for doc in documents:
#         chunks.extend(chunker.chunk_document(doc))

#     embedder = Embedder("all-mpnet-base-v2")
#     embeddings = embedder.embed_chunks(chunks)

#     index = FaissIndex(embedding_dim=embeddings.shape[1])
#     index.add(embeddings, chunks)

#     return index

from pathlib import Path
import numpy as np
import faiss

from src.loaders.document_loader import DocumentLoader
from src.preprocessing.clean_text import clean_text, remove_noise_lines
from src.chunking.fixed import FixedChunker
from src.chunking.semantic import SemanticChunker
from src.embeddings.embedder import Embedder
from src.retrieval.faiss_index import FaissIndex
from src.utils.io_utils import save_json, load_json


ARTIFACTS_DIR = Path("artifacts")
CHUNKS_DIR = ARTIFACTS_DIR / "chunks"
EMB_DIR = ARTIFACTS_DIR / "embeddings"
INDEX_DIR = ARTIFACTS_DIR / "indexes"

for d in [CHUNKS_DIR, EMB_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _build_index(
    data_dir,
    chunk_type,
    chunker,
    use_cache=True
    ):
    chunk_path = CHUNKS_DIR / f"{chunk_type}_chunks.json"
    emb_path = EMB_DIR / f"{chunk_type}_embeddings.npy"
    index_path = INDEX_DIR / f"{chunk_type}.faiss"

    if use_cache and chunk_path.exists():
        print(f"[Cache] Loading {chunk_type} chunks...")
        chunks = load_json(chunk_path)
    else:
        print(f"[Build] Creating {chunk_type} chunks...")

        loader = DocumentLoader(data_dir)
        documents = loader.load_all()

        documents = [
            {
                **doc,
                "text": remove_noise_lines(clean_text(doc["text"]))
            }
            for doc in documents
        ]

        chunks = []
        for doc in documents:
            chunks.extend(chunker.chunk_document(doc))

        save_json(chunks, chunk_path)

    print(f"[Info] Total {chunk_type} chunks: {len(chunks)}")

    if use_cache and emb_path.exists():
        print(f"[Cache] Loading {chunk_type} embeddings...")
        embeddings = np.load(emb_path)
    else:
        print(f"[Build] Creating {chunk_type} embeddings...")
        embedder = Embedder("all-mpnet-base-v2")
        embeddings = embedder.embed_chunks(chunks)
        np.save(emb_path, embeddings)

    if use_cache and index_path.exists():
        print(f"[Cache] Loading {chunk_type} FAISS index...")
        index = FaissIndex(embedding_dim=embeddings.shape[1])
        index.index = faiss.read_index(str(index_path))
        index.chunks = chunks
    else:
        print(f"[Build] Creating {chunk_type} FAISS index...")
        index = FaissIndex(embedding_dim=embeddings.shape[1])
        index.add(embeddings, chunks)
        faiss.write_index(index.index, str(index_path))

    return index


def build_fixed_index(data_dir="data/raw", use_cache=True, chunk_size=512):
    chunker = FixedChunker(
        tokenizer_name="bert-base-uncased",
        chunk_size=chunk_size,
        overlap=50
    )

    return _build_index(
        data_dir=data_dir,
        chunk_type="fixed",
        chunker=chunker,
        use_cache=use_cache
    )


def build_semantic_index(data_dir="data/raw", use_cache=True, threshold=0.5):
    chunker = SemanticChunker(
        embedding_model_name="all-mpnet-base-v2",
        similarity_threshold=threshold
    )

    return _build_index(
        data_dir=data_dir,
        chunk_type="semantic",
        chunker=chunker,
        use_cache=use_cache
    )