from pathlib import Path
import fitz  # pymupdf


class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all(self):
        documents = []

        for file_path in self.data_dir.iterdir():
            if file_path.suffix.lower() == ".pdf":
                text = self._load_pdf(file_path)
            elif file_path.suffix.lower() in [".txt", ".md"]:
                text = self._load_text(file_path)
            else:
                continue

            documents.append({
                "doc_id": file_path.stem,
                "source": file_path.name,
                "text": text,
                "num_chars": len(text)
            })

        return documents

    def _load_pdf(self, path: Path) -> str:
        doc = fitz.open(path)
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)

    def _load_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="ignore")
