from typing import List, Dict
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.device import get_device
MAX_CHARS = 1200

class SemanticChunker:
    def __init__(
        self,
        embedding_model_name: str = "all-mpnet-base-v2",
        similarity_threshold: float = 0.5,
        min_sentences: int = 3,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_sentences = min_sentences

        self.device = get_device()
        self.embedder = SentenceTransformer(
            embedding_model_name,
            device=str(self.device)
        )

        self.nlp = spacy.load("en_core_web_sm")

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        document: {
            'doc_id': str,
            'text': str
        }
        """
        sentences = [sent.text.strip() for sent in self.nlp(document["text"]).sents]
        sentences = [s for s in sentences if len(s) > 8]
        # paragraphs = document["text"].split(". ")

        # sentences = [
        #     p.strip()
        #     for p in paragraphs
        #     if len(p.split()) > 10
        # ]

        if len(sentences) == 0:
            return []

        embeddings = self.embedder.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        chunks = []
        current_chunk = [sentences[0]]
        start_idx = 0
        chunk_id = 0

        for i in range(1, len(sentences)):
            current_text = " ".join(current_chunk)
            sim = cosine_similarity(
                embeddings[i - 1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]

            if ((sim < self.similarity_threshold and len(current_chunk) >= self.min_sentences)
            or len(current_text) > MAX_CHARS):
                chunks.append(self._make_chunk(
                    document, current_chunk, chunk_id, start_idx, i
                ))
                chunk_id += 1
                current_chunk = [sentences[i]]
                start_idx = i
            else:
                current_chunk.append(sentences[i])

        # Final chunk
        chunks.append(self._make_chunk(
            document, current_chunk, chunk_id, start_idx, len(sentences)
        ))

        return chunks

    def _make_chunk(
        self,
        document: Dict,
        sentences: List[str],
        chunk_id: int,
        start_sent: int,
        end_sent: int
    ) -> Dict:
        text = " ".join(sentences)
        return {
            "chunk_id": f"{document['doc_id']}_sem_chunk_{chunk_id}",
            "doc_id": document["doc_id"],
            "text": text,
            "start_sentence": start_sent,
            "end_sentence": end_sent,
            "num_sentences": len(sentences),
            "num_chars": len(text)
        }
