import os
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import list_models

load_dotenv()

# --------------------------
# OpenAI Models
# --------------------------
def list_openai_models():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        models = client.models.list()
        print("\n🧠 OpenAI Models:")
        for model in models.data[:10]:
            print("-", model.id)
    except Exception as e:
        print("\n⚠️ Could not fetch OpenAI models:", e)

# --------------------------
# Hugging Face Models
# --------------------------
def list_huggingface_models():
    try:
        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        print("\\n🤗 Hugging Face Models (top 10):")
        for model in list_models(limit=10, use_auth_token=hf_token):
            print(" -", model.modelId)
    except Exception as e:
        print("\\n⚠️ Could not fetch Hugging Face models:", e)

if __name__ == "__main__":
    list_openai_models()
    list_huggingface_models()