# 🤖 Q5: List Available LLM Models (ChatGPT, Gemini, Hugging Face)

## ✅ Problem Statement

> Create a simple Python program to print the list of models in ChatGPT (OpenAI), Gemini, and Hugging Face.

---

## 📦 Project Overview

This script displays available models from 3 major GenAI providers:

| Provider     | Access Method                    |
|--------------|----------------------------------|
| OpenAI       | API call (`openai>=1.0.0`)       |
| Gemini       | Static list (API does not support listing yet) |
| Hugging Face | API call via `huggingface_hub`  |

---

## 🧪 Example Output
🧠 OpenAI Models(Top 10):
    - dall-e-3
    - dall-e-2
    - gpt-4o-audio-preview-2024-10-01
    - text-embedding-3-small
    - babbage-002
    - text-embedding-ada-002
    - gpt-4o-mini-audio-preview
    - gpt-4o-audio-preview
    - gpt-4.1-nano
    - gpt-3.5-turbo-instruct-0914
🤗 Hugging Face Models (top 10):
    - mistralai/Devstral-Small-2505
    - google/gemma-3n-E4B-it-litert-preview
    - ByteDance-Seed/BAGEL-7B-MoT
    - multimodalart/isometric-skeumorphic-3d-bnb
    - nvidia/parakeet-tdt-0.6b-v2
    - nari-labs/Dia-1.6B
    - Wan-AI/Wan2.1-VACE-14B
    - IndexTeam/Index-anisora
    - google/medgemma-4b-it
    - lodestones/Chroma