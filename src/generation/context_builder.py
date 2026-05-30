def build_context(retrieved_chunks, max_chars: int = 3000):
    context = ""
    for item in retrieved_chunks:
        text = item["chunk"]["text"]
        if len(context) + len(text) > max_chars:
            break
        context += text + "\n\n"
    return context.strip()
