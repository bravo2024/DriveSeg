from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from src.data import make_synthetic, create_dataloaders, NUM_CLASSES
from src.model import build_deeplab
from src.core import compute_seg_metrics

def test_data():
    data = make_synthetic(50, seed=42)
    assert data["images"].shape == (50, 3, 128, 128)
    assert data["masks"].shape == (50, 128, 128)
    assert data["num_classes"] == 4

def test_dataloader():
    data = make_synthetic(50, seed=42)
    tl, vl = create_dataloaders(data, batch_size=16)
    batch = next(iter(tl))
    assert batch[0].shape == (16, 3, 128, 128)
    assert batch[1].shape == (16, 128, 128)

def test_model():
    model = build_deeplab(num_classes=4, pretrained_backbone=False)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)["out"]
    assert out.shape == (2, 4, 128, 128)

def test_metrics():
    pred = torch.randint(0, 4, (10, 128, 128)).numpy()
    true = torch.randint(0, 4, (10, 128, 128)).numpy()
    m = compute_seg_metrics(pred, true, 4)
    assert 0 <= m["miou"] <= 1
    assert 0 <= m["dice"] <= 1
    assert 0 <= m["pixel_acc"] <= 1

def test_forward_pass():
    model = build_deeplab(num_classes=4, pretrained_backbone=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 128, 128))["out"]
    pred = out.argmax(dim=1)
    assert pred.shape == (1, 128, 128)
    assert pred.min() >= 0 and pred.max() < 4
