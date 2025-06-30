from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_model = None
_tokenizer = None

def load_mistral_model():
    global _model, _tokenizer
    if _model is None:
        print("⏬ Loading Mistral 7B Instruct model...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        _tokenizer = AutoTokenizer.from_pretrained(model_name)

        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True
        )
    return _model, _tokenizer

def generate_answer(question, context_chunks, max_tokens=512):
    model, tokenizer = load_mistral_model()

    # 🧠 Add structured context and source attribution
    context = "\n\n".join(
        f"[{c['source']} - {c['section']}]\n{c['chunk']}" for c in context_chunks
    )

    prompt = f"""
You are a helpful AI research assistant. Use the provided context from academic papers to answer the user's question.
Only answer using the information from the context. If the answer is not found, say "The answer is not available in the papers."

### Context:
{context}

### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt.strip(), return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("### Answer:")[-1].strip()
