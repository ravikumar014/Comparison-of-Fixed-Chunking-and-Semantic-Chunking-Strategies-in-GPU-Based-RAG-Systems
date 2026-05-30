from typing import List, Dict
from transformers import AutoTokenizer


class FixedChunker:
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        assert chunk_size > overlap, "chunk_size must be > overlap"

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        document: {
            'doc_id': str,
            'text': str
        }
        """
        tokens = self.tokenizer.encode(
            document["text"],
            add_special_tokens=False
        )

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append({
                "chunk_id": f"{document['doc_id']}_chunk_{chunk_id}",
                "doc_id": document["doc_id"],
                "text": chunk_text,
                "start_token": start,
                "end_token": min(end, len(tokens)),
                "num_tokens": len(chunk_tokens)
            })

            start += self.chunk_size - self.overlap
            chunk_id += 1

        return chunks
