from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["background", "road", "vehicle", "pedestrian"]
NUM_CLASSES = 4
IMG_SIZE = 128

def _draw_road(mask, rng):
    h, w = mask.shape
    road_top = int(h * rng.uniform(0.35, 0.55))
    road_left = int(w * rng.uniform(0.0, 0.15))
    road_right = int(w * rng.uniform(0.85, 1.0))
    rr, cc = np.mgrid[road_top:h, road_left:road_right]
    mask[rr, cc] = 1

def _draw_vehicle(mask, rng):
    h, w = mask.shape
    for _ in range(rng.integers(1, 5)):
        vh = rng.integers(12, 25)
        vw = rng.integers(20, 35)
        vy = rng.integers(int(h * 0.25), h - vh - 5)
        vx = rng.integers(0, w - vw)
        rr, cc = np.mgrid[vy:vy + vh, vx:vx + vw]
        region = mask[rr, cc]
        region[(region != 1)] = 2

def _draw_pedestrian(mask, rng):
    h, w = mask.shape
    for _ in range(rng.integers(0, 4)):
        ph = rng.integers(8, 16)
        pw = rng.integers(4, 8)
        py = rng.integers(int(h * 0.35), h - ph - 3)
        px = rng.integers(0, w - pw)
        rr, cc = np.mgrid[py:py + ph, px:px + pw]
        region = mask[rr, cc]
        region[(region != 1)] = 3

def _render_image(mask, rng):
    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {0: (87, 87, 87), 1: (80, 80, 80), 2: (60, 60, 140), 3: (140, 60, 60)}
    for cls, color in colors.items():
        img[mask == cls] = color
    noise = rng.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def make_synthetic(n=1000, seed=42, img_size=IMG_SIZE):
    rng = np.random.default_rng(seed)
    images, masks = [], []
    for _ in range(n):
        mask = np.zeros((img_size, img_size), dtype=np.int64)
        _draw_road(mask, rng)
        _draw_vehicle(mask, rng)
        _draw_pedestrian(mask, rng)
        img = _render_image(mask, rng)
        images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0)
        masks.append(torch.from_numpy(mask).long())
    data = {
        "images": torch.stack(images),
        "masks": torch.stack(masks),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "n_samples": n,
    }
    return data

class SegmentationDataset(Dataset):
    def __init__(self, data, split="train", val_split=0.2, seed=42):
        n = data["n_samples"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split_n = int(n * (1 - val_split))
        indices = idx[:split_n] if split == "train" else idx[split_n:]
        self.images = data["images"][indices]
        self.masks = data["masks"][indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

def create_dataloaders(data, batch_size=16, val_split=0.2, seed=42):
    train_ds = SegmentationDataset(data, "train", val_split, seed)
    val_ds = SegmentationDataset(data, "val", val_split, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
