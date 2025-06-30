from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Global cache for model and tokenizer
_model = None
_tokenizer = None

def load_mistral_model():
    global _model, _tokenizer
    if _model is None:
        print("⏬ Loading Mistral 7B Instruct model (CPU fallback)...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # ✅ Use float32 for CPU
        ).to("cpu")
    return _model, _tokenizer

def generate_answer(question, context_chunks, max_tokens=512):
    model, tokenizer = load_mistral_model()

    # Combine chunks + question into prompt
    context = "\n\n".join(context_chunks)
    prompt = f"""You are an AI assistant reading policy documents. Based on the context below, answer the user question.

### Context:
{context}

### Question:
{question}

### Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Cut off the prompt to show only generated answer
    return answer.split("### Answer:")[-1].strip()