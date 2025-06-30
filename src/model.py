import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        T = x.size(1)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, 0.0)

        attn_output, _ = self.attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.mlp(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, num_layers=4, dropout=0.1, block_size=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        x = self.token_emb(input_ids) + self.pos_emb[:, :T, :]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits