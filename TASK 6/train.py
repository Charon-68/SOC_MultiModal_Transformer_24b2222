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
import csv
import time
import random
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from project.tokenizers.word_tokenizer import WordTokenizer as Tokenizer
from project.datasets.flickr8k import Flickr8kDataset, collate_fn
from project.models.vit.vit_encoder import ViTEncoder
from project.models.clip.text_encoder import TextEncoder
from project.models.clip.clip_model import CLIPStyleModel
from project.datasets.synthetic import SyntheticFlickrDataset
from project.training.lr_scheduler import get_warmup_cosine_scheduler
from project.utils import set_seed, get_device, get_logger

logger = get_logger("Task6-CLIP-Training")
logger.info(f"Working directory: {os.getcwd()}")

# ==========================================================
# Configuration
# ==========================================================
CONFIG = {
    "image_size": 64,
    "max_text_len": 32,
    "embed_dim": 192,
    "projection_dim": 128,
    "patch_size": 8,
    "vit_depth": 4,
    "text_depth": 4,
    "n_head": 6,
    "dropout": 0.1,
    "batch_size": 32,  # Reduced for standard environment execution
    "epochs": 2,      # Reduced from 20 to 2 for quick validation verification
    "lr": 5e-4,
    "weight_decay": 0.05,
    "warmup_steps": 10,
    "total_steps": 100,
    "grad_clip": 1.0,
    "val_every": 20,
    "seed": 42,
}

set_seed(CONFIG["seed"])
device = get_device()
logger.info(f"Using device: {device}")

# ==========================================================
# Dataset Detection & Loading
# ==========================================================
dataset_file = ROOT / "data" / "Flickr8k" / "captions.txt"
image_dir = ROOT / "data" / "Flickr8k" / "Images"

use_synthetic = not (os.path.exists(dataset_file) and os.path.exists(image_dir))

if use_synthetic:
    logger.warning("Flickr8k dataset files not found. Falling back to SyntheticFlickrDataset.")
    # Initialize WordTokenizer with a dummy corpus to build vocabulary
    tokenizer = Tokenizer()
    dummy_captions = [
        "a black dog running on grass",
        "a white cat sitting on the couch",
        "a child playing with a red ball",
        "two dogs fighting in the street"
    ]
    tokenizer.build_vocab(dummy_captions)
    
    # Create datasets using synthetic generator
    train_dataset = SyntheticFlickrDataset(num_samples=128, img_size=(3, 64, 64), vocab_size=len(tokenizer))
    val_dataset = SyntheticFlickrDataset(num_samples=32, img_size=(3, 64, 64), vocab_size=len(tokenizer))
    test_dataset = SyntheticFlickrDataset(num_samples=32, img_size=(3, 64, 64), vocab_size=len(tokenizer))
else:
    logger.info("Flickr8k dataset found. Loading files...")
    captions_df = pd.read_csv(dataset_file)
    logger.info(f"Loaded {len(captions_df)} captions.")
    
    tokenizer = Tokenizer()
    tokenizer.build_vocab(captions_df["caption"])
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_names = sorted(captions_df["image"].unique())
    random.shuffle(image_names)
    
    train_images = set(image_names[:6000])
    val_images   = set(image_names[6000:7000])
    test_images  = set(image_names[7000:8000])
    
    train_df = captions_df[captions_df["image"].isin(train_images)].reset_index(drop=True)
    val_df   = captions_df[captions_df["image"].isin(val_images)].reset_index(drop=True)
    test_df  = captions_df[captions_df["image"].isin(test_images)].reset_index(drop=True)
    
    train_dataset = Flickr8kDataset(image_dir, train_df, tokenizer, transform)
    val_dataset = Flickr8kDataset(image_dir, val_df, tokenizer, transform)
    test_dataset = Flickr8kDataset(image_dir, test_df, tokenizer, transform)

logger.info(f"Vocabulary size: {len(tokenizer)}")

# ==========================================================
# DataLoader
# ==========================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=0,  # 0 is safer for cross-platform/interactive runs
    collate_fn=collate_fn,
)
print(f"Batches per epoch: {len(train_loader)}")

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
)

# ==========================================================
# Model Creation
# ==========================================================
vit_encoder = ViTEncoder(
    img_size=CONFIG["image_size"],
    patch_size=CONFIG["patch_size"],
    n_embd=CONFIG["embed_dim"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["vit_depth"],
    dropout=CONFIG["dropout"],
)

text_encoder = TextEncoder(
    vocab_size=len(tokenizer),
    max_len=CONFIG["max_text_len"],
    n_embd=CONFIG["embed_dim"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["text_depth"],
)

model = CLIPStyleModel(
    vit_encoder=vit_encoder,
    text_encoder=text_encoder,
    embed_dim=CONFIG["embed_dim"],
    projection_dim=CONFIG["projection_dim"],
    init_temperature=0.07,
).to(device)

logger.info("Model created successfully!")

# ==========================================================
# Optimizer & Scheduler
# ==========================================================
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

scheduler = get_warmup_cosine_scheduler(
    optimizer=optimizer,
    warmup_steps=CONFIG["warmup_steps"],
    total_steps=CONFIG["total_steps"]
)

# ==========================================================
# Forward Pass Verification & Training Loop
# ==========================================================
images, captions, masks = next(iter(train_loader))
images = images.to(device)
captions = captions.to(device)
masks = masks.to(device)

loss, img_feats, txt_feats = model(images, captions, masks)
logger.info("Forward pass successful!")
logger.info(f"Verification loss: {loss.item():.4f}")

# Reusable Training Loop
logger.info("Starting training loop...")
model.train()
step = 0
best_val_loss = float("inf")
checkpoint_path = ROOT / "TASK 6" / "checkpoint.pt"

for epoch in range(1, CONFIG["epochs"] + 1):
    epoch_loss = 0.0
    for batch in train_loader:
        img_b, cap_b, mask_b = batch
        img_b, cap_b, mask_b = img_b.to(device), cap_b.to(device), mask_b.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = model(img_b, cap_b, mask_b)
        loss.backward()
        
        if CONFIG["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            
        optimizer.step()
        scheduler.step()
        
        step += 1
        epoch_loss += loss.item()
        
        if step % CONFIG["val_every"] == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    v_img, v_cap, v_mask = val_batch
                    v_img, v_cap, v_mask = v_img.to(device), v_cap.to(device), v_mask.to(device)
                    v_loss, _, _ = model(v_img, v_cap, v_mask)
                    val_loss += v_loss.item()
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Step {step} | Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": CONFIG,
                    "vocab": tokenizer.vocab,
                    "epoch": epoch,
                    "best_loss": best_val_loss,
                    "temperature": 1.0 / model.loss_fn.log_inv_tau.exp().item(),
                    "training_metadata": {
                        "device": str(device),
                        "total_steps": step,
                        "val_loss": avg_val_loss
                    }
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved new best checkpoint with validation loss {best_val_loss:.4f} to {checkpoint_path}")
            
            model.train()
            
    logger.info(f"Epoch {epoch} finished | Avg Loss: {epoch_loss / len(train_loader):.4f}")

# Save final checkpoint at end of training
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": CONFIG,
    "vocab": tokenizer.vocab,
    "epoch": CONFIG["epochs"],
    "best_loss": best_val_loss,
    "temperature": 1.0 / model.loss_fn.log_inv_tau.exp().item(),
    "training_metadata": {
        "device": str(device),
        "total_steps": step,
        "final": True
    }
}
torch.save(checkpoint, checkpoint_path)
logger.info(f"Saved final checkpoint to {checkpoint_path}")

logger.info("CLIP training validation complete!")
