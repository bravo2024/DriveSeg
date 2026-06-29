from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import argparse
import torch
from src.data import make_synthetic, create_dataloaders, NUM_CLASSES
from src.model import build_deeplab, train_model
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = make_synthetic(n=a.n, seed=a.seed)
    print(f"Generated {data['n_samples']} synthetic samples, "
          f"classes: {data['class_names']}")

    train_loader, val_loader = create_dataloaders(
        data, batch_size=a.batch_size, val_split=0.2, seed=a.seed
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = build_deeplab(num_classes=NUM_CLASSES, pretrained_backbone=True)
    print(f"Model: DeepLabV3+ ResNet-101 ({sum(p.numel() for p in model.parameters()):,} params)")

    model, history = train_model(
        model, train_loader, val_loader,
        epochs=a.epochs, lr=a.lr, device=a.device, num_classes=NUM_CLASSES
    )

    print_report(history)
    save_model(model)
    save_metrics({
        "best_miou": history["best_miou"],
        "val_miou": history["val_miou"][-1],
        "val_dice": history["val_dice"][-1],
        "val_pixel_acc": history.get("val_pixel_acc", [0])[-1] if "val_pixel_acc" in history else 0,
        "epochs": a.epochs,
        "n_samples": a.n,
        "device": str(device),
    })
    print("Saved models/best_model.pt and models/metrics.json")

if __name__ == "__main__":
    main()
