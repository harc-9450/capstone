import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

def train(model, dataloader, optimizer, device, scheduler=None, epochs=3, log_interval=100):
    model.train()
    model.to(device)
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for step, batch in loop:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            # Reshape for loss: (B*T, vocab_size) vs (B*T)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if step % log_interval == 0:
                loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        if scheduler is not None:
            scheduler.step()