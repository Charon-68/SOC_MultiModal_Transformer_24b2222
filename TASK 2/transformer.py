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
from project.configs import TransformerConfig
from project.models.transformer import TransformerLanguageModel
from project.tokenizers import CharTokenizer
from project.datasets import get_char_batch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Parameters
batch_size = 64          # number of sequences processed together
block_size = 64          # maximum context length
max_iters = 500         # training iterations for verification
eval_interval = 300
learning_rate = 3e-4

n_embd = 128             # embedding dimension
n_head = 4               # number of attention heads
n_layer = 4              # number of transformer blocks
dropout = 0.2            # dropout probability

# Resolve dataset location
input_path = ROOT / "TASK 1" / "input.txt"

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer
tokenizer = CharTokenizer(text)
vocab_size = len(tokenizer)

# Encode full dataset
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
        device=torch.device(device)
    )

# Model configuration and initialization
config = TransformerConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
)
model = TransformerLanguageModel(config).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):
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
    if step % eval_interval == 0:
        print(f"step {step}: loss {loss.item():.4f}")

# Autoregressive generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=500)
generated_text = tokenizer.decode(generated_tokens[0].tolist())

print("\nGenerated Text:\n")
print(generated_text)