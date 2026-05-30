import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.device import get_device, device_name


class RAGGenerator:
    def __init__(self, model_name: str):
        self.device = get_device()
        print(f"[Generator] Using device: {device_name(self.device)}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32
        ).to(self.device)

        self.model.eval()

    def generate(self, context: str, question: str, max_new_tokens: int = 150):
        prompt = f"""
You are an assistant answering questions using ONLY the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Use only the context above.
- If the answer is not present, say: "Not found in the provided context."
- Do not add external knowledge.

Answer:
""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).split("Answer:")[-1].strip()
