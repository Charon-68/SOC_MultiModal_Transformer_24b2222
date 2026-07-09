import sys
from pathlib import Path

# Find project root dynamically
def get_project_root() -> Path:
    try:
        path = Path(__file__).resolve()
        for parent in [path] + list(path.parents):
            if (parent / "requirements.txt").exists() or (parent / "project").exists():
                return parent
    except NameError:
        pass
    path = Path.cwd().resolve()
    for parent in [path] + list(path.parents):
        if (parent / "requirements.txt").exists() or (parent / "project").exists():
            return parent
    return path

ROOT = get_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from project.models.layers.attention import AttentionHead
from project.tokenizers import CharTokenizer
from project.datasets import get_char_batch

# Parameters
batch_size = 32       # 32 sequences at once
block_size = 8        # length of each seq = 8
steps = 5000          # 5000 training updates
learning_rate = 1e-3  # size of parameter updates
n_embd = 32           # no. of features associated with each token
head_size = 32

# Load dataset using pathlib for directory-agnostic execution
input_path = ROOT / "TASK 1" / "input.txt"
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer
tokenizer = CharTokenizer(text)
vocab_size = len(tokenizer)

# Encode entire dataset
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Train - test split
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# Batching helper
def get_batch(split):
    data_split = train_data if split == "train" else test_data
    return get_char_batch(
        data=data_split,
        block_size=block_size,
        batch_size=batch_size,
        device=torch.device("cpu")
    )

# Model definition using refactored AttentionHead
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Self-attention head from project layers
        self.sa_head = AttentionHead(
            n_embd=n_embd,
            head_size=head_size,
            causal=True,
            block_size=block_size
        )
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        
        x = token_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize model
model = BigramLanguageModel(vocab_size)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(steps):
    # Sample batch
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Clear gradients
    optimizer.zero_grad(set_to_none=True)

    # Backprop
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print loss
    if step % 300 == 0:
        print(f"step {step}: loss {loss.item():.4f}")

# Generate text
context = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = model.generate(context, max_new_tokens=500)
generated_text = tokenizer.decode(generated_tokens[0].tolist())

print("\nGenerated Text:\n")
print(generated_text)