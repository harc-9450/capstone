import sys
import torch
from transformers import AutoTokenizer
sys.path.append('src')
from inference import generate_text
from evaluation import evaluate
from train import train
from model import TransformerLM
from dataset import WikiTextDataset
from torch.utils.data import DataLoader

# ✅ Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer_name = 'gpt2'
block_size = 128
batch_size = 16
epochs = 8
learning_rate = 1e-4

# ✅ Load tokenizer and datasets
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = WikiTextDataset(split='train', tokenizer_name=tokenizer_name, block_size=block_size)
val_dataset = WikiTextDataset(split='validation', tokenizer_name=tokenizer_name, block_size=block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ✅ Initialize model and optimizer
model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=384, n_heads=6, num_layers=4, block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # <-- Add this line

# ✅ Training
train(model, train_loader, optimizer, device, scheduler, epochs=epochs)

# ✅ Evaluation
evaluate(model, val_loader, device=device)

# ✅ Inference
prompt = "The scientist opened the ancient journal and found"
output = generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_p=0.92, device=device)
print("\\n📘 Generated Text:")
print(output)