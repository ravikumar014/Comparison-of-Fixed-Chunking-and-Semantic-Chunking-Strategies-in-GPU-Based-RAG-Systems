from src.loaders.document_loader import DocumentLoader
from src.preprocessing.clean_text import clean_text


def main():
    loader = DocumentLoader(r".\data\raw")
    print(loader)
    documents = loader.load_all()

    cleaned_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        doc["text"] = cleaned_text
        doc["num_chars"] = len(cleaned_text)
        cleaned_docs.append(doc)

    print(f"Loaded documents: {len(cleaned_docs)}")

    for d in cleaned_docs[:3]:
        print("-" * 40)
        print("Doc ID:", d["doc_id"])
        print("Characters:", d["num_chars"])
        print(d["text"][:300])


if __name__ == "__main__":
    main()
