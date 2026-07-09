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
from project.models.transformer import BigramLanguageModel
from project.tokenizers import CharTokenizer
from project.datasets import get_char_batch

# Parameters
batch_size = 32       # 32 sequences at once
block_size = 8        # length of each seq = 8
steps = 3000          # 3000 training updates
learning_rate = 1e-2  # size of parameter updates

# Load dataset using pathlib for directory-agnostic execution
input_path = ROOT / "TASK 1" / "input.txt"
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Creating vocabulary and tokenizer
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

# Initialize model from project
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