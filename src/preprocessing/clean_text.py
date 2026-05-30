import re


def clean_text(text: str) -> str:
    import re

    text = text.replace("\x00", " ")

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove citations
    text = re.sub(r"\[\d+(,\s*\d+)*\]", " ", text)

    # Fix broken words
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Preserve structure
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()

def remove_noise_lines(text):
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        if len(line) < 20:
            continue

        if "figure" in line.lower():
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines)