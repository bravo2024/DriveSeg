from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch

def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#1a1f2e", "figure.facecolor": "#1a1f2e",
        "axes.edgecolor": "#4a5568", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "text.color": "white", "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": "#4a5568",
    })

COLORMAP = np.array([
    [0, 0, 0], [128, 64, 128], [0, 0, 142], [220, 20, 60]
], dtype=np.uint8)

CLASS_NAMES = ["background", "road", "vehicle", "pedestrian"]

def decode_mask(mask):
    return COLORMAP[mask]

def plot_sample(image_tensor, mask_tensor, pred_tensor=None):
    _style()
    n = 2 if pred_tensor is None else 3
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    ax1.imshow(img); ax1.set_title("Input", color="white"); ax1.axis("off")
    gt_rgb = decode_mask(mask_tensor.cpu().numpy())
    ax2.imshow(gt_rgb); ax2.set_title("Ground Truth", color="white"); ax2.axis("off")
    if pred_tensor is not None:
        pred_rgb = decode_mask(pred_tensor.cpu().numpy())
        ax3.imshow(pred_rgb); ax3.set_title("Prediction", color="white"); ax3.axis("off")
    plt.tight_layout()
    return fig

def plot_training_history(history):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train", color="#22d3ee")
    ax1.plot(epochs, history["val_loss"], label="Val", color="#f97316")
    ax1.set_title("Loss", color="white"); ax1.legend(); ax1.grid(True, alpha=0.2)
    ax2.plot(epochs, history["val_miou"], color="#a78bfa")
    ax2.set_title("mIoU", color="white"); ax2.grid(True, alpha=0.2)
    ax3.plot(epochs, history["val_dice"], color="#22c55e")
    ax3.set_title("Dice", color="white"); ax3.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig

def plot_metrics_radar(metrics):
    _style()
    categories = ["mIoU", "Dice", "Pixel Acc"]
    values = [metrics.get("miou", 0), metrics.get("dice", 0), metrics.get("pixel_acc", 0)]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.fill(angles, values, alpha=0.25, color="#22d3ee")
    ax.plot(angles, values, color="#22d3ee", linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, color="white")
    ax.set_ylim(0, 1); ax.set_title("Segmentation Quality", color="white", pad=20)
    ax.grid(True, alpha=0.3)
    return fig
