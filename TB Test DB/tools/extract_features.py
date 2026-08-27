import argparse
import json
import os
from pathlib import Path
import csv

import numpy as np
import pydicom
import certifi
import ssl
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm


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
        raise ValueError(f"Unsupported model: {model_name}. Use resnet18/resnet50.")

    if weights:
        state = torch.load(weights, map_location="cpu")
        # allow either full model state_dict or only backbone
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

    backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> (B, C, 1, 1)
    backbone.eval()
    return backbone, feat_dim


def default_transform(image_size: int) -> T.Compose:
    def to_3ch(x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W). Some DICOMs are already RGB (C=3). Keep C=3 to match Normalize.
        if x.ndim != 3:
            raise ValueError(f"Expected tensor (C,H,W). Got shape={tuple(x.shape)}")
        c = int(x.shape[0])
        if c == 1:
            return x.repeat(3, 1, 1)
        if c == 3:
            return x
        # Fallback: collapse to 1ch then repeat (handles weird multi-sample frames)
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


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    tfm: T.Compose,
    files: list[Path],
    label: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ps: list[str] = []

    for p in tqdm(files, desc=f"label={label}", unit="file"):
        try:
            img = dicom_to_float01(p)
            x = tfm(img).unsqueeze(0).to(device)
            feat = backbone(x).squeeze().detach().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {p}  err={type(e).__name__}: {e}")
            continue
        xs.append(feat.astype(np.float32))
        ys.append(label)
        ps.append(str(p))

    X = np.stack(xs, axis=0) if xs else np.zeros((0, 0), dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, ps


def configure_https_with_certifi() -> None:
    """
    Torch/torchvision pretrained weights download uses urllib without passing an SSL context.
    On some Windows/proxy environments, the default trust store validation fails.
    This forces urllib's default HTTPS context to use certifi's CA bundle.
    """
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=cafile)  # type: ignore[attr-defined]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg_dir", default="Negative CXR", help="Directory with negative .dcm files")
    ap.add_argument("--pos_dir", default="Positive CXR", help="Directory with positive .dcm files")
    ap.add_argument(
        "--manifest_csv",
        default=None,
        help="Optional CSV created by tools/build_manifest_split.py (columns: abs_path,label,split,...)",
    )
    ap.add_argument("--manifest_split", default="train", help="train | val | test (used with --manifest_csv)")
    ap.add_argument("--out", default="features_stage1.npz", help="Output .npz path")
    ap.add_argument("--model", default="resnet18", help="resnet18 | resnet50")
    ap.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained weights (requires internet). Default: off (offline-safe).",
    )
    ap.add_argument("--weights", default=None, help="Optional .pt weights (state_dict)")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--max_files_per_class", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if args.pretrained:
        configure_https_with_certifi()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, feat_dim = build_backbone(args.model, args.pretrained, args.weights)
    backbone = backbone.to(device)
    tfm = default_transform(args.image_size)

    if args.manifest_csv:
        split = args.manifest_split.strip().lower()
        if split not in {"train", "val", "test"}:
            raise SystemExit("--manifest_split must be one of: train, val, test")
        neg_files: list[Path] = []
        pos_files: list[Path] = []
        with open(args.manifest_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("split") or "").strip().lower() != split:
                    continue
                p = Path((row.get("abs_path") or "").strip())
                y = int(row.get("label") or 0)
                if y == 0:
                    neg_files.append(p)
                else:
                    pos_files.append(p)
        neg_files = sorted(neg_files)
        pos_files = sorted(pos_files)
    else:
        neg_files = sorted(Path(args.neg_dir).rglob("*.dcm"))
        pos_files = sorted(Path(args.pos_dir).rglob("*.dcm"))

    if args.max_files_per_class and args.max_files_per_class > 0:
        neg_files = neg_files[: args.max_files_per_class]
        pos_files = pos_files[: args.max_files_per_class]

    x0, y0, p0 = extract_features(backbone, tfm, neg_files, 0, device)
    x1, y1, p1 = extract_features(backbone, tfm, pos_files, 1, device)

    X = np.concatenate([x0, x1], axis=0) if x0.size or x1.size else np.zeros((0, feat_dim), dtype=np.float32)
    y = np.concatenate([y0, y1], axis=0) if y0.size or y1.size else np.zeros((0,), dtype=np.int64)
    paths = np.array(p0 + p1, dtype=object)

    meta = {
        "neg_dir": args.neg_dir,
        "pos_dir": args.pos_dir,
        "model": args.model,
        "weights": args.weights,
        "image_size": args.image_size,
        "device": device,
        "feat_dim": int(feat_dim),
        "n_negative": int(len(neg_files)),
        "n_positive": int(len(pos_files)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, paths=paths, meta=json.dumps(meta))
    print(f"saved: {out_path}  X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()

