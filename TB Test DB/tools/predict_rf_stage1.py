import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import certifi
import ssl
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm
import joblib


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


def configure_https_with_certifi() -> None:
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=cafile)  # type: ignore[attr-defined]


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

    backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> (B,C,1,1)
    backbone.eval()
    return backbone, feat_dim


def default_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass(frozen=True)
class Row:
    rel_folder: str
    rel_path: str
    proba_pos: float
    pred: int


@torch.no_grad()
def extract_one(backbone: nn.Module, tfm: T.Compose, p: Path, device: str) -> np.ndarray:
    img = dicom_to_float01(p)
    x = tfm(img).unsqueeze(0).to(device)
    feat = backbone(x).squeeze().detach().cpu().numpy().astype(np.float32)
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder containing DICOMs (recursive)")
    ap.add_argument("--rf_model", default="artifacts/rf_stage1.joblib", help="RF joblib path")
    ap.add_argument("--out_csv", default="ext_stage1_predictions.csv", help="Output CSV path")
    ap.add_argument("--out_summary", default="ext_stage1_summary.json", help="Output summary JSON path")
    ap.add_argument("--model", default="resnet18", help="resnet18 | resnet50 (must match feature dim used for RF)")
    ap.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights (requires internet)")
    ap.add_argument("--cnn_weights", default=None, help="Optional state_dict for backbone (offline)")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()

    root = Path(args.in_dir)
    rf = joblib.load(args.rf_model)

    if args.pretrained:
        configure_https_with_certifi()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, feat_dim = build_backbone(args.model, args.pretrained, args.cnn_weights)
    backbone = backbone.to(device)
    tfm = default_transform(args.image_size)

    files = sorted(root.rglob("*.dcm"))
    if not files:
        raise SystemExit(f"No .dcm files found under: {root}")

    X = np.zeros((len(files), feat_dim), dtype=np.float32)
    rel_paths: list[str] = []
    rel_folders: list[str] = []
    for i, p in enumerate(tqdm(files, desc="extract+predict", unit="file")):
        X[i] = extract_one(backbone, tfm, p, device)
        rel = p.relative_to(root)
        rel_paths.append(str(rel))
        rel_folders.append(str(rel.parent))

    proba_pos = rf.predict_proba(X)[:, 1].astype(float)
    pred = (proba_pos >= 0.5).astype(int)

    rows = [
        Row(rel_folder=rel_folders[i], rel_path=rel_paths[i], proba_pos=proba_pos[i], pred=int(pred[i]))
        for i in range(len(files))
    ]

    # Folder-level summary
    summary: dict[str, dict[str, int]] = {}
    for r in rows:
        key = r.rel_folder
        if key not in summary:
            summary[key] = {"n": 0, "pred_negative": 0, "pred_positive": 0}
        summary[key]["n"] += 1
        if r.pred == 1:
            summary[key]["pred_positive"] += 1
        else:
            summary[key]["pred_negative"] += 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "dicom_rel_path", "proba_positive", "pred_label"])
        for r in rows:
            w.writerow([r.rel_folder, r.rel_path, f"{r.proba_pos:.6f}", r.pred])

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "n_files": len(files),
        "rf_model": str(Path(args.rf_model)),
        "feature_extractor": {"model": args.model, "cnn_weights": args.cnn_weights, "image_size": args.image_size},
        "device": device,
        "by_folder": summary,
    }
    out_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print a readable summary (sorted)
    print(f"saved csv: {out_csv}")
    print(f"saved summary: {out_summary}")
    print("")
    print("Folder summary (predicted):")
    for folder in sorted(summary.keys()):
        s = summary[folder]
        print(f"- {folder or '.'}: n={s['n']}  neg={s['pred_negative']}  pos={s['pred_positive']}")


if __name__ == "__main__":
    main()

