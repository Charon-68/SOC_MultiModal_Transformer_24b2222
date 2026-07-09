#!/usr/bin/env python3
"""
Production-quality evaluation script for CLIP-style Vision Language Models (VLMs).
Computes validation loss, Recall@K (Image->Text and Text->Image), Zero-shot Classification,
model parameters/FLOPs, exports key arrays/CSV metrics, and plots embedding spaces (PCA, t-SNE, heatmaps).
"""

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
import math
import argparse
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

# Headless matplotlib config
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from project.tokenizers.word_tokenizer import WordTokenizer
from project.models.vit.vit_encoder import ViTEncoder
from project.models.clip.text_encoder import TextEncoder
from project.models.clip.clip_model import CLIPStyleModel
from project.datasets.flickr8k import Flickr8kDataset, collate_fn
from project.datasets.synthetic import SyntheticFlickrDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM CLIPStyleModel Performance")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the PyTorch checkpoint.pt file. (Defaults to TASK 6/checkpoint.pt)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        choices=["auto", "flickr8k", "synthetic", "cifar10"],
        help="Dataset type to run evaluation on. (auto: Flickr8k if found, else Synthetic)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for dataloader batches."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on ('cuda', 'mps', 'cpu', or 'auto')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save metrics, CSV outputs, and plots. (Defaults to project_root/evaluation_outputs)"
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def estimate_flops(config: dict, batch_size: int = 1) -> float:
    """Estimates the total FLOPs of a single forward pass analytically."""
    vit_depth = config.get("vit_depth", 4)
    img_size = config.get("image_size", 64)
    patch_size = config.get("patch_size", 8)
    in_chans = 3
    n_embd = config.get("embed_dim", 192)
    projection_dim = config.get("projection_dim", 128)
    
    num_patches = (img_size // patch_size) ** 2
    T_v = num_patches + 1  # Patches + CLS Token
    
    # ViT Patch projection (Conv2d)
    flops_patch = 2 * num_patches * in_chans * (patch_size ** 2) * n_embd
    
    # ViT Transformer blocks
    flops_vit_layers = vit_depth * (2 * T_v * (12 * (n_embd ** 2) + 2 * T_v * n_embd))
    flops_vit = flops_patch + flops_vit_layers
    
    # Text Encoder Transformer blocks
    text_depth = config.get("text_depth", 4)
    max_text_len = config.get("max_text_len", 32)
    T_t = max_text_len
    flops_text_layers = text_depth * (2 * T_t * (12 * (n_embd ** 2) + 2 * T_t * n_embd))
    
    # Projection heads
    flops_projections = 2 * n_embd * projection_dim + 2 * n_embd * projection_dim
    
    total_flops_single = flops_vit + flops_text_layers + flops_projections
    return total_flops_single * batch_size


def run_evaluation():
    args = parse_args()
    
    # 1. Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "evaluation_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Determine device
    device = get_device(args.device)
    print(f"Using evaluation device: {device}")
    
    # 3. Resolve and load checkpoint
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else ROOT / "TASK 6" / "checkpoint.pt"
    if not checkpoint_path.exists():
        sys.exit(
            f"\n[ERROR] Checkpoint file not found at: {checkpoint_path}\n"
            f"Please run model training first to generate 'checkpoint.pt' using:\n"
            f"  python \"TASK 6/train.py\""
        )
        
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 4. Restore vocabulary and config
    config_data = checkpoint.get("config", {})
    if not config_data:
        config_data = {
            "image_size": 64,
            "max_text_len": 32,
            "embed_dim": 192,
            "projection_dim": 128,
            "patch_size": 8,
            "vit_depth": 4,
            "text_depth": 4,
            "n_head": 6,
            "dropout": 0.1,
            "init_temperature": 0.07
        }
        
    tokenizer = WordTokenizer()
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
    print(f"Restored vocabulary with {len(tokenizer)} tokens.")
    
    # 5. Recreate model architecture
    vit_encoder = ViTEncoder(
        img_size=config_data.get("image_size", 64),
        patch_size=config_data.get("patch_size", 8),
        n_embd=config_data.get("embed_dim", 192),
        n_head=config_data.get("n_head", 6),
        n_layer=config_data.get("vit_depth", 4),
        dropout=config_data.get("dropout", 0.1),
    )
    
    text_encoder = TextEncoder(
        vocab_size=len(tokenizer),
        max_len=config_data.get("max_text_len", 32),
        n_embd=config_data.get("embed_dim", 192),
        n_head=config_data.get("n_head", 6),
        n_layer=config_data.get("text_depth", 4),
        dropout=config_data.get("dropout", 0.1),
    )
    
    model = CLIPStyleModel(
        vit_encoder=vit_encoder,
        text_encoder=text_encoder,
        embed_dim=config_data.get("embed_dim", 192),
        projection_dim=config_data.get("projection_dim", 128),
        init_temperature=config_data.get("init_temperature", 0.07),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print("Model architecture built and state dictionary loaded successfully.")
    
    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_m = total_params / 1e6
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model Size: {model_size_m:.2f}M parameters")
    
    # Checkpoint metadata reporting
    epoch = checkpoint.get("epoch", "N/A")
    best_loss = checkpoint.get("best_loss", "N/A")
    ckpt_temp = checkpoint.get("temperature", "N/A")
    training_metadata = checkpoint.get("training_metadata", {})
    
    print(f"Loaded checkpoint trained for {epoch} epochs.")
    print(f"Checkpoint Best Validation Loss: {best_loss}")
    if ckpt_temp != "N/A":
        print(f"Checkpoint scaling temperature: {ckpt_temp:.4f}")
    if training_metadata:
        print(f"Checkpoint training metadata: {training_metadata}")
    
    # FLOPs Estimation
    flops_est = estimate_flops(config_data, batch_size=1)
    print(f"Estimated forward pass computation: {flops_est / 1e9:.3f} GFLOPs per sample")
    
    # Learned temperature parameter
    log_inv_tau = model.loss_fn.log_inv_tau.clamp(0, 4.6052)
    inv_tau = log_inv_tau.exp().item()
    current_temp = 1.0 / inv_tau
    print(f"Learned log_inv_tau: {model.loss_fn.log_inv_tau.item():.4f}")
    print(f"Current evaluation temperature: {current_temp:.4f}")
    
    # 6. Load evaluation dataset
    flickr_captions_file = ROOT / "data" / "Flickr8k" / "captions.txt"
    flickr_image_dir = ROOT / "data" / "Flickr8k" / "Images"
    
    dataset_type = args.dataset.lower()
    if dataset_type == "auto":
        if flickr_captions_file.exists() and flickr_image_dir.exists():
            dataset_type = "flickr8k"
        else:
            dataset_type = "synthetic"
            
    print(f"Loading data from dataset type: {dataset_type}")
    
    if dataset_type == "flickr8k":
        captions_df = pd.read_csv(flickr_captions_file)
        image_names = sorted(captions_df["image"].unique())
        random.seed(42)
        random.shuffle(image_names)
        
        # Test split mirroring TASK 6 splits: 7000 to 8000 index
        test_images = set(image_names[7000:8000]) if len(image_names) >= 8000 else set(image_names)
        test_df = captions_df[captions_df["image"].isin(test_images)].reset_index(drop=True)
        
        transform = transforms.Compose([
            transforms.Resize((config_data.get("image_size", 64), config_data.get("image_size", 64))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_dataset = Flickr8kDataset(flickr_image_dir, test_df, tokenizer, transform)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
    elif dataset_type == "cifar10":
        from torchvision.datasets import CIFAR10
        transform = transforms.Compose([
            transforms.Resize((config_data.get("image_size", 64), config_data.get("image_size", 64))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_dataset = CIFAR10(root=str(ROOT / "data"), train=False, download=True, transform=transform)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
    else:  # synthetic
        eval_dataset = SyntheticFlickrDataset(
            num_samples=128,
            img_size=(3, config_data.get("image_size", 64), config_data.get("image_size", 64)),
            vocab_size=len(tokenizer),
            max_seq_len=config_data.get("max_text_len", 32)
        )
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
    # 7. Core evaluation loops
    all_img_features = []
    all_text_features = []
    val_losses = []
    
    # Differentiate loader unpacking based on dataset class
    is_cifar = (dataset_type == "cifar10")
    
    print("Running forward pass and extracting features...")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation"):
            if not is_cifar:
                images, captions, masks = batch
                images = images.to(device)
                captions = captions.to(device)
                masks = masks.to(device)
                
                loss, img_embeds, text_embeds = model(images, captions, masks)
                val_losses.append(loss.item())
            else:
                images, labels = batch
                images = images.to(device)
                # Compute image features only for classification evaluation
                img_embeds = model.encode_image(images)
                text_embeds = torch.zeros(img_embeds.shape[0], config_data.get("projection_dim", 128), device=device)
                
            all_img_features.append(img_embeds.cpu())
            all_text_features.append(text_embeds.cpu())
            
    # Stack features
    image_embeddings = torch.cat(all_img_features, dim=0)  # (N, D)
    text_embeddings = torch.cat(all_text_features, dim=0)   # (N, D)
    
    # 8. Compute primary metrics (Recall & Cosine Similarity)
    avg_loss = np.mean(val_losses) if val_losses else float("nan")
    
    # Cosine similarities
    image_embeddings_norm = F.normalize(image_embeddings, dim=-1)
    text_embeddings_norm = F.normalize(text_embeddings, dim=-1)
    
    similarity_matrix = (image_embeddings_norm @ text_embeddings_norm.T).numpy()
    
    # Matched cosine similarity (diagonal elements)
    matched_similarities = np.diag(similarity_matrix)
    avg_cosine_sim = float(np.mean(matched_similarities))
    
    # Computations for Recall@K (I2T and T2I)
    recall_metrics = {}
    if not is_cifar:
        N = len(image_embeddings)
        targets = np.arange(N)
        
        # Image-to-Text retrieval ranks
        # Sort indices row-wise
        i2t_sorted_indices = np.argsort(-similarity_matrix, axis=1)
        # Text-to-Image retrieval ranks
        # Sort indices column-wise (transpose matrix)
        t2i_sorted_indices = np.argsort(-similarity_matrix.T, axis=1)
        
        def calculate_recall_k(sorted_indices, k):
            correct_ranks = [np.where(sorted_indices[i] == targets[i])[0][0] for i in range(N)]
            hits = sum(1 for rank in correct_ranks if rank < k)
            return hits / N
            
        recall_metrics["i2t_recall_1"] = calculate_recall_k(i2t_sorted_indices, 1)
        recall_metrics["i2t_recall_5"] = calculate_recall_k(i2t_sorted_indices, 5)
        recall_metrics["i2t_recall_10"] = calculate_recall_k(i2t_sorted_indices, 10)
        
        recall_metrics["t2i_recall_1"] = calculate_recall_k(t2i_sorted_indices, 1)
        recall_metrics["t2i_recall_5"] = calculate_recall_k(t2i_sorted_indices, 5)
        recall_metrics["t2i_recall_10"] = calculate_recall_k(t2i_sorted_indices, 10)
        
        print("\n--- Retrieval Metrics ---")
        print(f"Image->Text Recall@1:  {recall_metrics['i2t_recall_1'] * 100:.2f}%")
        print(f"Image->Text Recall@5:  {recall_metrics['i2t_recall_5'] * 100:.2f}%")
        print(f"Image->Text Recall@10: {recall_metrics['i2t_recall_10'] * 100:.2f}%")
        print(f"Text->Image Recall@1:  {recall_metrics['t2i_recall_1'] * 100:.2f}%")
        print(f"Text->Image Recall@5:  {recall_metrics['t2i_recall_5'] * 100:.2f}%")
        print(f"Text->Image Recall@10: {recall_metrics['t2i_recall_10'] * 100:.2f}%")
    else:
        print("\nRetrieval metrics skipped for CIFAR-10 classification.")
        
    # 9. Zero-shot Classification (on CIFAR-10 or label-compatible datasets)
    zero_shot_accuracy = None
    has_labels = hasattr(eval_dataset, "classes") and (hasattr(eval_dataset, "targets") or hasattr(eval_dataset, "data"))
    
    if has_labels:
        print("\nRunning Zero-shot classification...")
        classes = eval_dataset.classes
        prompts = [f"a photo of a {c}" for c in classes]
        
        # Build prompt tokens and masks
        tokenized = []
        max_prompt_len = 0
        for prompt in prompts:
            toks = tokenizer.encode(prompt)
            tokenized.append(toks)
            max_prompt_len = max(max_prompt_len, len(toks))
            
        padded_prompts = []
        masks_prompts = []
        for toks in tokenized:
            padded_prompts.append(toks + [0] * (max_prompt_len - len(toks)))
            masks_prompts.append([1] * len(toks) + [0] * (max_prompt_len - len(toks)))
            
        prompts_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=device)
        masks_tensor = torch.tensor(masks_prompts, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            prompt_embeddings = model.encode_text(prompts_tensor, masks_tensor)
            prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)  # (C, D)
            
        # Re-batch to check matching prediction
        correct = 0
        total = 0
        for batch in eval_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                img_embeds = model.encode_image(images)
                img_embeds = F.normalize(img_embeds, dim=-1)  # (B, D)
                
                # similarities between batch images and all prompt classes
                logits = img_embeds @ prompt_embeddings.T  # (B, C)
                preds = logits.argmax(dim=-1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        zero_shot_accuracy = correct / total
        print(f"Zero-shot classification accuracy: {zero_shot_accuracy * 100:.2f}%")
    else:
        print("\nZero-shot classification skipped gracefully (no labels found in dataset).")
        
    # 10. Save and export metrics
    metrics = {
        "val_infonce_loss": avg_loss if not math.isnan(avg_loss) else None,
        "avg_cosine_similarity": avg_cosine_sim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_millions": model_size_m,
        "estimated_gflops_per_sample": flops_est / 1e9,
        "learned_temperature": current_temp,
        "embedding_dim": config_data.get("embed_dim", 192),
        "projection_dim": config_data.get("projection_dim", 128),
        "dataset_evaluated": dataset_type,
        "zero_shot_accuracy": zero_shot_accuracy
    }
    metrics.update(recall_metrics)
    
    # Save JSON metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Save similarity matrix
    sim_path = output_dir / "similarity_matrix.npy"
    np.save(sim_path, similarity_matrix)
    print(f"Similarity matrix saved to {sim_path}")
    
    # Save retrieval rank mappings (retrieval.csv)
    retrieval_path = output_dir / "retrieval.csv"
    with open(retrieval_path, "w", newline="") as f:
        writer = csv.writer(f)
        if not is_cifar:
            writer.writerow(["query_index", "correct_index", "retrieved_top_1", "top_5_match"])
            for idx in range(len(similarity_matrix)):
                top_1 = i2t_sorted_indices[idx, 0]
                top_5 = i2t_sorted_indices[idx, :5].tolist()
                writer.writerow([idx, idx, top_1, top_5])
        else:
            writer.writerow(["cifar10_image_index", "class_predicted", "similarity_scores"])
            # Save classification logits
            for idx in range(min(100, len(similarity_matrix))):
                writer.writerow([idx, similarity_matrix[idx].argmax(), similarity_matrix[idx].tolist()])
    print(f"Retrieval CSV log saved to {retrieval_path}")
    
    # 11. Plot Heatmap
    plt.figure(figsize=(10, 8))
    # Display the first 50x50 elements of the similarity matrix
    sub_matrix = similarity_matrix[:50, :50]
    plt.imshow(sub_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.title("Cosine Similarity Heatmap (First 50 Pairs)")
    plt.xlabel("Text Query Index")
    plt.ylabel("Image Query Index")
    heatmap_path = output_dir / "similarity_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Similarity Heatmap plot saved to {heatmap_path}")
    
    # 12. Plot Recall bar chart
    if not is_cifar:
        plt.figure(figsize=(8, 5))
        x_lbls = ["Recall@1", "Recall@5", "Recall@10"]
        i2t_vals = [recall_metrics["i2t_recall_1"], recall_metrics["i2t_recall_5"], recall_metrics["i2t_recall_10"]]
        t2i_vals = [recall_metrics["t2i_recall_1"], recall_metrics["t2i_recall_5"], recall_metrics["t2i_recall_10"]]
        
        x = np.arange(len(x_lbls))
        width = 0.35
        
        plt.bar(x - width/2, i2t_vals, width, label="Image -> Text", color="#2b5c8f")
        plt.bar(x + width/2, t2i_vals, width, label="Text -> Image", color="#d95f02")
        
        plt.ylabel("Recall Ratio")
        plt.title("VLM CLIP Evaluation Recall Metrics")
        plt.xticks(x, x_lbls)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        recall_plot_path = output_dir / "recall_plots.png"
        plt.savefig(recall_plot_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Recall plots saved to {recall_plot_path}")
        
    # 13. PCA Visualization
    print("Computing PCA dimensionality reduction...")
    pca = PCA(n_components=2)
    # Join image and text embeddings for consistent projection
    stacked_embeds = np.concatenate([image_embeddings.numpy(), text_embeddings.numpy()], axis=0)
    projected = pca.fit_transform(stacked_embeds)
    
    img_projected = projected[:len(image_embeddings)]
    text_projected = projected[len(image_embeddings):]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(img_projected[:, 0], img_projected[:, 1], color="#2b5c8f", alpha=0.7, label="Images", edgecolors="w")
    plt.scatter(text_projected[:, 0], text_projected[:, 1], color="#d95f02", alpha=0.7, label="Texts", edgecolors="w")
    
    # Draw connections for a subset to prevent clutter
    for i in range(min(15, len(img_projected))):
        plt.plot(
            [img_projected[i, 0], text_projected[i, 0]],
            [img_projected[i, 1], text_projected[i, 1]],
            color="gray", linestyle="--", alpha=0.5
        )
        
    plt.title("PCA Projection of Image and Text Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    pca_plot_path = output_dir / "pca_visualization.png"
    plt.savefig(pca_plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"PCA visualization saved to {pca_plot_path}")
    
    # 14. t-SNE Visualization
    print("Computing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(image_embeddings) - 1))
    projected_tsne = tsne.fit_transform(stacked_embeds)
    
    img_tsne = projected_tsne[:len(image_embeddings)]
    text_tsne = projected_tsne[len(image_embeddings):]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(img_tsne[:, 0], img_tsne[:, 1], color="#2b5c8f", alpha=0.7, label="Images", edgecolors="w")
    plt.scatter(text_tsne[:, 0], text_tsne[:, 1], color="#d95f02", alpha=0.7, label="Texts", edgecolors="w")
    
    # Draw connections for a subset
    for i in range(min(15, len(img_tsne))):
        plt.plot(
            [img_tsne[i, 0], text_tsne[i, 0]],
            [img_tsne[i, 1], text_tsne[i, 1]],
            color="gray", linestyle="--", alpha=0.5
        )
        
    plt.title("t-SNE Projection of Image and Text Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    tsne_plot_path = output_dir / "tsne_visualization.png"
    plt.savefig(tsne_plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"t-SNE visualization saved to {tsne_plot_path}")
    
    print("\n==============================================================")
    print("Evaluation completed successfully! Outputs written to:")
    print(f"  {output_dir}")
    print("==============================================================")


if __name__ == "__main__":
    run_evaluation()
