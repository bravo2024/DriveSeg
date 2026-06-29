from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from src.data import make_synthetic, create_dataloaders, CLASS_NAMES, NUM_CLASSES
from src.model import build_deeplab
from src.core import compute_seg_metrics
from src.visualizations import plot_sample, plot_training_history, plot_metrics_radar, _style

st.set_page_config(page_title="DriveSeg | AImotive Perception", layout="wide", page_icon="\U0001f697")

@st.cache_resource
def load_model():
    m = build_deeplab(num_classes=NUM_CLASSES, pretrained_backbone=False)
    p = Path("models/best_model.pt")
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m

with st.sidebar:
    st.header("\u2699 Config")
    n_samples = st.slider("Samples", 100, 2000, 500, 50)
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_preds = st.checkbox("Show Predictions", True)
    st.caption("AImotive | Drive Segmentation | Autonomous Perception")

data = make_synthetic(n=n_samples, seed=42)
_, val_loader = create_dataloaders(data, batch_size=4, val_split=0.2, seed=42)
val_images, val_masks = next(iter(val_loader))
model = load_model()

with torch.no_grad():
    val_logits = model(val_images)["out"]
    val_probs = torch.softmax(val_logits, dim=1)
    val_preds = val_logits.argmax(dim=1)
    val_conf = val_probs.max(dim=1).values

metrics = compute_seg_metrics(val_preds.numpy(), val_masks.numpy(), NUM_CLASSES)

c1, c2, c3, c4 = st.columns(4)
c1.metric("mIoU", f"{metrics['miou']:.4f}")
c2.metric("Dice", f"{metrics['dice']:.4f}")
c3.metric("Pixel Acc", f"{metrics['pixel_acc']:.4f}")
c4.metric("Samples", f"{n_samples:,}")

t1, t2, t3, t4 = st.tabs(["\U0001f4ca Explorer", "\U0001f52c Model Lab", "\U0001f30d Segmentation Quality", "\U0001f6a8 Safety"])

with t1:
    st.subheader("Sample Predictions")
    cols = st.columns(4)
    for i in range(min(4, len(val_images))):
        with cols[i]:
            fig = plot_sample(val_images[i], val_masks[i], val_preds[i] if show_preds else None)
            st.pyplot(fig)
            conf = val_conf[i].mean().item()
            st.caption(f"Mean confidence: {conf:.2%}")

with t2:
    rows = []
    for cls in range(NUM_CLASSES):
        rows.append({
            "Class": CLASS_NAMES[cls],
            "IoU": f"{metrics['per_class_iou'][cls]:.4f}",
            "Dice": f"{metrics['per_class_dice'][cls]:.4f}",
        })
    st.dataframe(rows, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_metrics_radar(metrics))
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4)); _style()
        classes = CLASS_NAMES
        x = np.arange(len(classes))
        ax.bar(x - 0.2, metrics["per_class_iou"], 0.4, label="IoU", color="#22d3ee")
        ax.bar(x + 0.2, metrics["per_class_dice"], 0.4, label="Dice", color="#a78bfa")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45)
        ax.set_title("Per-Class Metrics", color="white"); ax.legend(); ax.grid(True, alpha=.2)
        st.pyplot(fig)

with t3:
    st.subheader("Semantic Segmentation Quality")
    st.latex(r"\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \quad \text{mIoU} = \frac{1}{C}\sum_{c=1}^C \text{IoU}_c")
    st.caption("Intersection over Union per class. mIoU > 0.65 is production-grade for autonomous driving.")
    st.latex(r"\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}")
    st.caption("Dice coefficient (F1 at pixel level). Less sensitive to class imbalance than IoU.")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4)); _style()
        ax.hist(val_conf.flatten().numpy(), bins=50, color="#22d3ee", alpha=0.6, edgecolor="#1a1f2e")
        ax.axvline(threshold, color="#f43f5e", ls="--", lw=2, label=f"Threshold={threshold}")
        ax.set_title("Confidence Distribution", color="white"); ax.legend(); ax.grid(True, alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4)); _style()
        conf_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        accuracies = []
        for lo, hi in zip(conf_bins[:-1], conf_bins[1:]):
            mask = (val_conf >= lo) & (val_conf < hi)
            if mask.sum() > 0:
                correct = (val_preds[mask] == val_masks[mask]).float().mean().item()
            else:
                correct = 0.0
            accuracies.append(correct)
        ax.plot(conf_bins[:-1], accuracies, marker="o", color="#22c55e")
        ax.set_title("Accuracy vs Confidence", color="white")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.grid(True, alpha=.2)
        st.pyplot(fig)

with t4:
    st.subheader("Safety Metrics & BEV Perception")
    st.latex(r"\text{Risk Score} = \frac{1}{1+e^{-(\beta_0 + \beta_1 \cdot (1-\text{mIoU}) + \beta_2 \cdot \text{hazard\_prob})}}")
    st.latex(r"P_{\text{BEV}} = K^{-1} \cdot [u, v, 1]^\top \cdot z \quad (\text{inverse perspective mapping})")
    st.caption("Bird's Eye View transform: projects segmentation mask pixels (u,v) at depth z onto the ground plane using camera intrinsics K.")
    hazard_probs = 1 - val_probs[:, 0, :, :].numpy()
    high_risk = (hazard_probs > 0.7).sum()
    med_risk = ((hazard_probs > 0.4) & (hazard_probs <= 0.7)).sum()
    low_risk = (hazard_probs <= 0.4).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk Pixels", f"{high_risk:,}")
    c2.metric("Medium Risk", f"{med_risk:,}")
    c3.metric("Low Risk", f"{low_risk:,}")
    fig, ax = plt.subplots(figsize=(8, 4)); _style()
    ax.hist(hazard_probs.ravel(), bins=50, color="#22d3ee", alpha=0.6, edgecolor="#1a1f2e")
    ax.axvline(0.4, color="#fbbf24", ls="--", lw=2, label="Medium Threshold")
    ax.axvline(0.7, color="#f43f5e", ls="--", lw=2, label="High Threshold")
    ax.set_title("Pixel Risk Distribution", color="white"); ax.legend(); ax.grid(True, alpha=.2)
    st.pyplot(fig)
