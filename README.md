# Multimodal Vision-Language Transformer from Scratch

Author: Shourya Saxena, IIT Bombay Seasons of Code

This repository builds a small vision-language learning stack in PyTorch, progressing from tensor basics and attention experiments to a CLIP-style dual encoder and multimodal cross-attention models. The assignment notebooks are preserved in `TASK 0/` through `TASK 6/`, while reusable code lives in the `project/` package.

## What Is Inside

- `project/models/`: reusable model code for attention blocks, ViT encoders, transformer language models, CLIP text/image encoders, and multimodal models.
- `project/datasets/`: Flickr8k, synthetic fallback, and character-text dataset utilities.
- `project/tokenizers/`: character and word tokenizers.
- `project/losses/`: InfoNCE contrastive loss.
- `project/training/`: trainer and learning-rate scheduler helpers.
- `project/tests/`: unit tests for datasets and model behavior.
- `TASK 0/` to `TASK 6/`: notebooks, scripts, writeups, samples, and assignment-specific experiments.
- `evaluate.py`: evaluation script for checkpoints, retrieval metrics, plots, and exported metrics.

## Repository Layout

```text
.
├── TASK 0/                  # Tensor fundamentals and introductory notebooks
├── TASK 1/                  # Bigram model and attention experiments
├── TASK 2/                  # Decoder-only transformer language model
├── TASK 3/                  # Vision Transformer experiments
├── TASK 4/                  # Multimodal cross-attention experiments
├── TASK 5/                  # CLIP loss/model experiments
├── TASK 6/                  # End-to-end training script and setup notebook
├── project/                 # Reusable Python package
│   ├── configs/
│   ├── datasets/
│   ├── losses/
│   ├── models/
│   ├── tests/
│   ├── tokenizers/
│   ├── training/
│   └── utils/
├── evaluate.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most commands should be run from the repository root with `PYTHONPATH=.` so imports resolve consistently.

## Run Tests

```bash
PYTHONPATH=. python -m unittest discover -s project/tests -p "test_*.py"
```

## Run Training

Task 6 is the main end-to-end CLIP-style training entry point.

```bash
PYTHONPATH=. python "TASK 6/train.py"
```

The script looks for Flickr8k data at:

```text
data/Flickr8k/captions.txt
data/Flickr8k/Images/
```

If those files are not present, it falls back to `SyntheticFlickrDataset` so the full pipeline can still be smoke-tested locally.

Training writes a checkpoint to `TASK 6/checkpoint.pt`. Checkpoints are ignored by Git because they are generated binary artifacts.

## Run Evaluation

```bash
PYTHONPATH=. python evaluate.py
```

Useful options:

```bash
PYTHONPATH=. python evaluate.py --checkpoint "TASK 6/checkpoint.pt"
PYTHONPATH=. python evaluate.py --dataset synthetic
PYTHONPATH=. python evaluate.py --dataset cifar10
PYTHONPATH=. python evaluate.py --batch_size 64 --output_dir my_evaluation_results
```

By default, evaluation writes generated metrics, CSVs, numpy arrays, and plots to `evaluation_outputs/`. That directory is ignored by Git.

## Tasks

| Task | Focus | Key files |
| --- | --- | --- |
| Task 0 | Tensor operations and foundations | `TASK 0/` |
| Task 1 | Bigram model and causal attention | `TASK 1/bigram.py`, `TASK 1/attention.py` |
| Task 2 | Decoder-only transformer | `TASK 2/transformer.py`, `project/models/transformer/` |
| Task 3 | Vision Transformer | `TASK 3/`, `project/models/vit/` |
| Task 4 | Multimodal cross-attention | `TASK 4/`, `project/models/multimodal.py` |
| Task 5 | CLIP and contrastive loss | `TASK 5/`, `project/models/clip/`, `project/losses/` |
| Task 6 | End-to-end training | `TASK 6/train.py`, `project/training/` |

## Git Hygiene

The `.gitignore` is configured to keep source code, notebooks, writeups, sample text, and committed figures trackable while ignoring local/generated files such as:

- `.venv/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `.pytest_cache/`
- `data/`
- `evaluation_outputs/`
- model checkpoints such as `*.pt`, `*.pth`, and `*.ckpt`

If ignored files were already staged before this `.gitignore` was added, unstage them from Git's index without deleting local files:

```bash
git rm -r --cached .venv __pycache__ .ipynb_checkpoints .pytest_cache evaluation_outputs
git rm -r --cached "TASK 6/checkpoint.pt" "TASK 6/__pycache__" project/**/__pycache__
```

Then check the result:

```bash
git status --short
```
