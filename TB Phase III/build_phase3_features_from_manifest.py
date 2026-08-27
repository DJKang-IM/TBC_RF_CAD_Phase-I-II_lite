import argparse
import csv
import json
import os
import ssl
from pathlib import Path

import certifi
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm


def configure_https_with_certifi() -> None:
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=cafile)  # type: ignore[attr-defined]


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
    return arr.astype(np.float32, copy=False)


def default_transform(image_size: int) -> T.Compose:
    def to_3ch(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected tensor (C,H,W). Got shape={tuple(x.shape)}")
        c = int(x.shape[0])
        if c == 1:
            return x.repeat(3, 1, 1)
        if c == 3:
            return x
        x1 = x[:1, :, :]
        return x1.repeat(3, 1, 1)

    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(to_3ch),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_backbone(model_name: str, pretrained: bool, weights: str | None) -> tuple[nn.Module, int]:
    model_name = model_name.lower().strip()
    if model_name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        feat_dim = 512
    elif model_name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        feat_dim = 2048
    else:
        raise ValueError("Unsupported model. Use resnet18/resnet50.")

    if weights:
        state = torch.load(weights, map_location="cpu")
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

    backbone = nn.Sequential(*(list(m.children())[:-1]))
    backbone.eval()
    return backbone, feat_dim


@torch.no_grad()
def extract_one(backbone: nn.Module, tfm: T.Compose, p: Path, device: str) -> np.ndarray:
    img = dicom_to_float01(p)
    x = tfm(img).unsqueeze(0).to(device)
    feat = backbone(x).squeeze().detach().cpu().numpy().astype(np.float32)
    return feat


def load_manifest(manifest_csv: Path, split: str) -> tuple[list[tuple[Path, list[int]]], str | None]:
    items: list[tuple[Path, list[int]]] = []
    dataset_label: str | None = None
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("split") or "").strip().lower() != split:
                continue
            if dataset_label is None:
                lab = (row.get("dataset_label") or "").strip()
                dataset_label = lab if lab else None
            p = Path((row.get("abs_path") or "").strip())
            raw5 = row.get("D5")
            d5 = 0
            if raw5 is not None and str(raw5).strip() != "":
                d5 = int(raw5)
            vec = [int(row["D1"]), int(row["D2"]), int(row["D3"]), int(row["D4"]), d5, int(row["D6"])]
            items.append((p, vec))
    if not items:
        raise SystemExit(f"No rows for split={split} in {manifest_csv}")
    return items, dataset_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--split", default="train", help="train|val|test")
    ap.add_argument("--out", default="phase3_features.npz")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_files", type=int, default=0, help="0=all (debug)")
    args = ap.parse_args()

    split = args.split.strip().lower()
    if split not in {"train", "val", "test"}:
        raise SystemExit("--split must be train|val|test")

    if args.pretrained:
        configure_https_with_certifi()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = Path(args.manifest_csv)
    items, dataset_label = load_manifest(manifest, split)
    if args.max_files and args.max_files > 0:
        items = items[: args.max_files]

    backbone, feat_dim = build_backbone(args.model, args.pretrained, args.weights)
    backbone = backbone.to(device)
    tfm = default_transform(args.image_size)

    X = np.zeros((len(items), feat_dim), dtype=np.float32)
    Y = np.zeros((len(items), 5), dtype=np.int64)
    D5 = np.zeros((len(items),), dtype=np.int64)
    paths: list[str] = []
    kept = 0
    skipped = 0

    for p, vec in tqdm(items, desc=f"extract-{split}", unit="file"):
        try:
            X[kept] = extract_one(backbone, tfm, p, device)
            d1, d2, d3, d4, d5, d6 = vec
            Y[kept] = np.array([d1, d2, d3, d4, d6], dtype=np.int64)
            D5[kept] = d5
            paths.append(str(p))
            kept += 1
        except Exception as e:  # noqa: BLE001
            skipped += 1
            print(f"[skip] {p} err={type(e).__name__}: {e}")

    X = X[:kept]
    Y = Y[:kept]
    D5 = D5[:kept]
    paths_arr = np.array(paths, dtype=object)

    meta = {
        "manifest_csv": str(manifest),
        "dataset_label": dataset_label,
        "split": split,
        "model": args.model,
        "weights": args.weights,
        "image_size": args.image_size,
        "device": device,
        "feat_dim": int(feat_dim),
        "n_rows": int(len(items)),
        "n_kept": int(kept),
        "n_skipped": int(skipped),
        "y_schema": "Y: [D1,D2,D3,D4,D6]; D5: cavitary lesion 0/1",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y, D5=D5, paths=paths_arr, meta=json.dumps(meta))
    print(f"saved: {out_path}  X={X.shape} Y={Y.shape} skipped={skipped}")


if __name__ == "__main__":
    main()

