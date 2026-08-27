import argparse
import os
import random
from pathlib import Path

import numpy as np
import pydicom
import certifi
import ssl
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dicom_to_float01(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        lo, hi = np.percentile(arr, [0.5, 99.5])

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return arr


class DicomBinaryDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], image_size: int, augment: bool):
        self.items = items
        base = [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
        if augment:
            base += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
            ]
        base += [
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.tfm = T.Compose(base)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        p, y = self.items[idx]
        img = dicom_to_float01(p)
        x = self.tfm(img)
        return x, torch.tensor(y, dtype=torch.long)


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    model_name = model_name.lower().strip()
    if model_name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m
    if model_name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m
    raise ValueError("Unsupported model. Use resnet18 or resnet50.")


def configure_https_with_certifi() -> None:
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=cafile)  # type: ignore[attr-defined]


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg_dir", default="Negative CXR")
    ap.add_argument("--pos_dir", default="Positive CXR")
    ap.add_argument("--model", default="resnet18", help="resnet18 | resnet50")
    ap.add_argument("--pretrained", action="store_true", help="Requires internet to download weights")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.0, help="0 = train only (no internal split)")
    ap.add_argument("--max_files_per_class", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="cnn_stage1.pt", help="Output weights (.pt)")
    args = ap.parse_args()

    seed_everything(args.seed)

    if args.pretrained:
        configure_https_with_certifi()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    neg_files = sorted(Path(args.neg_dir).rglob("*.dcm"))
    pos_files = sorted(Path(args.pos_dir).rglob("*.dcm"))
    if args.max_files_per_class and args.max_files_per_class > 0:
        neg_files = neg_files[: args.max_files_per_class]
        pos_files = pos_files[: args.max_files_per_class]

    items = [(p, 0) for p in neg_files] + [(p, 1) for p in pos_files]
    random.shuffle(items)

    if args.val_ratio and args.val_ratio > 0:
        n_val = int(len(items) * args.val_ratio)
        val_items = items[:n_val]
        train_items = items[n_val:]
    else:
        val_items = []
        train_items = items

    train_ds = DicomBinaryDataset(train_items, args.image_size, augment=True)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_ld = None
    if val_items:
        val_ds = DicomBinaryDataset(val_items, args.image_size, augment=False)
        val_ld = DataLoader(val_ds, batch_size=max(args.batch_size, 32), shuffle=False, num_workers=0)

    model = build_model(args.model, args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_ld, desc=f"epoch {epoch}/{args.epochs}", unit="batch")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        if val_ld is not None:
            acc = accuracy(model, val_ld, device)
            print(f"val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), out_path)
                print(f"saved(best): {out_path}")
        else:
            torch.save(model.state_dict(), out_path)
            print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

