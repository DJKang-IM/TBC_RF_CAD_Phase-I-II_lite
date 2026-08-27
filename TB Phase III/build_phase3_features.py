import argparse
import json
import os
import ssl
import sys
from pathlib import Path

import certifi
import numpy as np
import pydicom
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pydicom.tag import Tag
from torchvision import models
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Must match embed_tb_labels_into_dicom / tb_cad_dicom_private_layout
from tb_cad_dicom_private_layout import TAG_D1, TAG_D2, TAG_D3, TAG_D4, TAG_D5, TAG_D6


def configure_https_with_certifi() -> None:
    cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cafile)
    ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(cafile=cafile)  # type: ignore[attr-defined]


def _pixel_array_to_hu(ds: pydicom.Dataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def _window_lo_hi(ds: pydicom.Dataset, hu: np.ndarray) -> tuple[float, float]:
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
        lo, hi = [float(x) for x in np.percentile(hu, [0.5, 99.5])]
    return lo, hi


def _hu_to_float01(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.clip(hu.astype(np.float32, copy=False), lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return arr.astype(np.float32, copy=False)


def crop_to_lung_bbox(
    hu: np.ndarray,
    lung_inferer,
    *,
    margin_ratio: float,
    min_mask_frac: float = 5e-4,
) -> np.ndarray:
    """
    lungmask LMInferer on HU slice → bbox → expand each edge by margin_ratio × bbox height (vertical)
    or × bbox width (horizontal). Typical CXR padding: 0.10–0.15 (10–15% per edge); default CLI 0.15.
    Falls back to full image if segmentation empty or too small.
    """
    H, W = hu.shape
    try:
        seg = lung_inferer.apply(hu.astype(np.float32))
    except Exception:
        return hu
    mask = np.asarray(seg, dtype=np.int32) > 0
    if mask.sum() < max(64, min_mask_frac * H * W):
        return hu
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    my = int(round(margin_ratio * bh))
    mx = int(round(margin_ratio * bw))
    y0n = max(0, y0 - my)
    y1n = min(H - 1, y1 + my)
    x0n = max(0, x0 - mx)
    x1n = min(W - 1, x1 + mx)
    return hu[y0n : y1n + 1, x0n : x1n + 1].copy()


def dicom_to_float01(
    dcm_path: Path,
    *,
    lung_inferer=None,
    lung_crop: bool = False,
    lung_margin: float = 0.15,
) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    hu = _pixel_array_to_hu(ds)
    if lung_crop and lung_inferer is not None:
        hu = crop_to_lung_bbox(hu, lung_inferer, margin_ratio=float(lung_margin))
    lo, hi = _window_lo_hi(ds, hu)
    return _hu_to_float01(hu, lo, hi)


def apply_clahe_float01(arr: np.ndarray, *, clip_limit: float, kernel_size: int | tuple[int, int] | None) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization on a single channel image in [0, 1].
    Uses scikit-image (same cohort as other preprocessing must stay explicit/versioned).
    """
    try:
        from skimage import exposure  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "CLAHE requires scikit-image. Install with: pip install scikit-image"
        ) from e

    if arr.ndim != 2:
        raise ValueError(f"CLAHE expects HxW grayscale, got shape {arr.shape}")
    # equalize_adapthist expects finite values in [0, 1]
    a = np.clip(arr.astype(np.float64, copy=False), 0.0, 1.0)
    ks = kernel_size
    if isinstance(ks, int):
        ks = (ks, ks)
    out = exposure.equalize_adapthist(a, kernel_size=ks, clip_limit=float(clip_limit), nbins=256)
    return out.astype(np.float32, copy=False)

def robust_norm_per_image(
    arr: np.ndarray,
    *,
    method: str = "iqr",
    eps: float = 1e-6,
    clip: float | None = 5.0,
) -> np.ndarray:
    """
    Per-image robust normalization on a single-channel image.
    - iqr: (x - median) / (p75 - p25)
    Returns float32. Values are typically centered around 0 (not in [0,1]).
    """
    if arr.ndim != 2:
        raise ValueError(f"robust_norm expects HxW grayscale, got shape {arr.shape}")
    a = arr.astype(np.float32, copy=False)
    m = (method or "").strip().lower()
    if m in ("", "none", "off"):
        return a
    if m != "iqr":
        raise ValueError(f"Unsupported robust norm method: {method}. Use iqr.")
    med = float(np.median(a))
    q25, q75 = [float(x) for x in np.percentile(a, [25.0, 75.0])]
    scale = max(q75 - q25, float(eps))
    z = (a - med) / scale
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z.astype(np.float32, copy=False)


def default_transform(image_size: int, *, use_imagenet_norm: bool = True) -> T.Compose:
    steps: list = [
        T.ToTensor(),
        T.Resize((image_size, image_size), antialias=True),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]
    if use_imagenet_norm:
        steps.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(steps)

def _letterbox_chw(
    x: torch.Tensor,
    *,
    target: int,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    x: (C,H,W) float tensor.
    Resize with aspect ratio preserved so max(H,W)->target, then pad to (C,target,target).
    """
    if x.ndim != 3:
        raise ValueError(f"letterbox expects CHW tensor, got shape {tuple(x.shape)}")
    c, h, w = x.shape
    if h <= 0 or w <= 0:
        raise ValueError("Invalid H/W for letterbox")
    t = int(target)
    if t <= 0:
        raise ValueError("Invalid target for letterbox")
    # scale so that the longer side becomes target
    scale = t / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    x2 = TF.resize(x, [nh, nw], antialias=True)
    pad_h = t - nh
    pad_w = t - nw
    # center pad: left, top, right, bottom
    pl = pad_w // 2
    pr = pad_w - pl
    pt = pad_h // 2
    pb = pad_h - pt
    x3 = TF.pad(x2, [pl, pt, pr, pb], fill=float(pad_value))
    # Safety: TF.pad should yield exact size but guard rounding.
    if x3.shape[-2:] != (t, t):
        x3 = TF.resize(x3, [t, t], antialias=True)
    return x3


def default_transform_with_resize_mode(
    image_size: int,
    *,
    resize_mode: str = "stretch",
    letterbox_pad_value: float = 0.0,
    use_imagenet_norm: bool = True,
) -> T.Compose:
    """
    resize_mode:
      - stretch: Resize((S,S)) (legacy behavior)
      - letterbox: aspect-preserving resize + pad to (S,S)
    """
    m = (resize_mode or "").strip().lower()
    if m not in ("stretch", "letterbox"):
        raise ValueError(f"Invalid resize_mode={resize_mode}. Use stretch|letterbox.")
    steps: list = [T.ToTensor()]
    if m == "stretch":
        steps.append(T.Resize((image_size, image_size), antialias=True))
    else:
        steps.append(
            T.Lambda(
                lambda x: _letterbox_chw(
                    x,
                    target=int(image_size),
                    pad_value=float(letterbox_pad_value),
                )
            )
        )
    steps.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))
    if use_imagenet_norm:
        steps.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(steps)


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
    elif model_name == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=w)
        feat_dim = 1024
    elif model_name == "efficientnet_v2_s":
        w = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_v2_s(weights=w)
        feat_dim = 1280
    elif model_name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=w)
        feat_dim = 768
    elif model_name == "swin_t":
        w = models.Swin_T_Weights.DEFAULT if pretrained else None
        m = models.swin_t(weights=w)
        feat_dim = 768
    else:
        raise ValueError(
            "Unsupported model. Use resnet18/resnet50/densenet121/"
            "efficientnet_v2_s/convnext_tiny/swin_t."
        )

    if weights:
        state = torch.load(weights, map_location="cpu")
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

    if model_name.startswith("resnet"):
        backbone = nn.Sequential(*(list(m.children())[:-1]))
    elif model_name == "densenet121":
        backbone = nn.Sequential(
            m.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    elif model_name == "efficientnet_v2_s":
        backbone = nn.Sequential(
            m.features,
            m.avgpool,
        )
    elif model_name == "convnext_tiny":
        backbone = nn.Sequential(
            m.features,
            m.avgpool,
        )
    elif model_name == "swin_t":
        backbone = nn.Sequential(
            m.features,
            m.norm,
            m.permute,
            m.avgpool,
        )
    else:
        raise ValueError("Unsupported model type after construction.")
    backbone.eval()
    return backbone, feat_dim


def read_vec_from_header(p: Path) -> list[int] | None:
    try:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
    except Exception:
        return None

    def _get_us(tag: Tag) -> int | None:
        if tag not in ds:
            return None
        try:
            return int(ds.get(tag).value)
        except Exception:
            return None

    d1 = _get_us(TAG_D1)
    d2 = _get_us(TAG_D2)
    d3 = _get_us(TAG_D3)
    d4 = _get_us(TAG_D4)
    d5 = _get_us(TAG_D5)
    d6 = _get_us(TAG_D6)
    if None in (d1, d2, d3, d4, d6):
        return None
    if d5 is None:
        d5 = 0
    return [int(d1), int(d2), int(d3), int(d4), int(d5), int(d6)]


@torch.no_grad()
def extract_one(
    backbone: nn.Module,
    tfm: T.Compose,
    p: Path,
    device: str,
    *,
    lung_inferer=None,
    lung_crop: bool = False,
    lung_margin: float = 0.15,
    use_clahe: bool = False,
    clahe_clip_limit: float = 0.03,
    clahe_kernel_size: int | None = None,
    robust_norm: str = "none",
    robust_norm_eps: float = 1e-6,
    robust_norm_clip: float | None = 5.0,
    robust_norm_order: str = "post_clahe",
) -> np.ndarray:
    img = dicom_to_float01(
        p,
        lung_inferer=lung_inferer,
        lung_crop=lung_crop,
        lung_margin=lung_margin,
    )
    rn_on = (robust_norm or "").strip().lower() not in ("", "none", "off")
    rn_order = (robust_norm_order or "").strip().lower()
    if rn_order not in ("pre_clahe", "post_clahe"):
        raise ValueError(f"Invalid robust_norm_order={robust_norm_order}. Use pre_clahe|post_clahe.")

    if rn_on and rn_order == "pre_clahe":
        img = robust_norm_per_image(img, method=robust_norm, eps=float(robust_norm_eps), clip=robust_norm_clip)
        # CLAHE expects [0,1] so re-range after robust norm (keep monotonicity).
        mn = float(np.min(img))
        mx = float(np.max(img))
        img = (img - mn) / max(mx - mn, 1e-6)

    if use_clahe:
        ks = clahe_kernel_size if clahe_kernel_size is not None else None
        img = apply_clahe_float01(img, clip_limit=clahe_clip_limit, kernel_size=ks)

    if rn_on and rn_order == "post_clahe":
        img = robust_norm_per_image(img, method=robust_norm, eps=float(robust_norm_eps), clip=robust_norm_clip)
    x = tfm(img).unsqueeze(0).to(device)
    feat = torch.flatten(backbone(x), 1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Root folder containing EMBED DICOMs (recursive)")
    ap.add_argument("--out", default="phase3_features.npz", help="Output .npz path")
    ap.add_argument(
        "--model",
        default="resnet18",
        help="resnet18 | resnet50 | densenet121 | efficientnet_v2_s | convnext_tiny | swin_t",
    )
    ap.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights (requires internet)")
    ap.add_argument("--weights", default=None, help="Optional .pt weights (state_dict)")
    ap.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Square resize before backbone (default 448 for finer CXR detail; use 224 for legacy/ImageNet-224 comparisons).",
    )
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--resize_mode",
        type=str,
        default="stretch",
        help="Resize mode before backbone: stretch (legacy Resize(S,S)) or letterbox (aspect-preserving + pad).",
    )
    ap.add_argument(
        "--letterbox_pad_value",
        type=float,
        default=0.0,
        help="Padding value for letterbox (in [0,1] image space). Default 0.0.",
    )
    ap.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE (scikit-image equalize_adapthist) on [0,1] image before resize/backbone.",
    )
    ap.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=0.03,
        help="CLAHE clip_limit (skimage exposure.equalize_adapthist). Typical 0.01–0.05.",
    )
    ap.add_argument(
        "--clahe-kernel-size",
        type=int,
        default=0,
        help="CLAHE tile size (odd int, e.g. 64). 0 = skimage default (~1/8 of image size).",
    )
    ap.add_argument(
        "--pipeline-version",
        type=str,
        default="",
        help='Optional label stored in meta JSON (e.g. "2.1").',
    )
    ap.add_argument(
        "--lung-crop",
        action="store_true",
        help="Segment lungs with lungmask (LMInferer), bbox-expand margin crop on HU before window→01.",
    )
    ap.add_argument(
        "--lung-margin",
        type=float,
        default=0.15,
        help="Per-edge bbox expansion as fraction of box height/width (default 0.15 ≈ 15%% padding each side).",
    )
    ap.add_argument(
        "--lungmask-force-cpu",
        action="store_true",
        help="Force lungmask inference on CPU (even if CUDA available).",
    )
    ap.add_argument(
        "--robust-norm",
        type=str,
        default="none",
        help="Per-image robust normalization method on image after CLAHE. Use: none | iqr",
    )
    ap.add_argument("--robust-norm-eps", type=float, default=1e-6, help="Epsilon for robust normalization scale.")
    ap.add_argument(
        "--robust-norm-clip",
        type=float,
        default=5.0,
        help="Clip robust-normalized values to [-clip,+clip]. Use 0 to disable clipping.",
    )
    ap.add_argument(
        "--robust-norm-order",
        type=str,
        default="post_clahe",
        help="When robust norm is enabled: apply it pre_clahe or post_clahe (default post_clahe).",
    )
    args = ap.parse_args()

    if args.pretrained:
        configure_https_with_certifi()

    lung_inferer = None
    if args.lung_crop:
        try:
            from lungmask import LMInferer  # type: ignore[import-untyped]
        except ImportError as e:
            raise SystemExit(
                "--lung-crop requires lungmask. Install: pip install lungmask"
            ) from e
        use_lm_cuda = torch.cuda.is_available() and not args.lungmask_force_cpu
        lung_inferer = LMInferer(force_cpu=not use_lm_cuda, tqdm_disable=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.in_dir)
    files = sorted(root.rglob("*.dcm"))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No .dcm files found under: {root}")

    use_imagenet_norm = (args.robust_norm or "").strip().lower() in ("", "none", "off")
    backbone, feat_dim = build_backbone(args.model, args.pretrained, args.weights)
    backbone = backbone.to(device)
    tfm = default_transform_with_resize_mode(
        args.image_size,
        resize_mode=str(args.resize_mode),
        letterbox_pad_value=float(args.letterbox_pad_value),
        use_imagenet_norm=use_imagenet_norm,
    )

    X = np.zeros((len(files), feat_dim), dtype=np.float32)
    # Y = (N,5) for D1..D4,D6 — matches train_rf_phase3 / hybrid; D5 cavitary stored separately
    Y = np.zeros((len(files), 5), dtype=np.int64)
    D5 = np.zeros((len(files),), dtype=np.int64)
    kept_paths: list[str] = []
    kept = 0
    skipped = 0

    for p in tqdm(files, desc="extract-phase3", unit="file"):
        vec = read_vec_from_header(p)
        if vec is None:
            skipped += 1
            continue
        ks = int(args.clahe_kernel_size) if int(args.clahe_kernel_size) > 0 else None
        rclip = float(args.robust_norm_clip)
        rclip_v = None if rclip == 0.0 else rclip
        X[kept] = extract_one(
            backbone,
            tfm,
            p,
            device,
            lung_inferer=lung_inferer,
            lung_crop=bool(args.lung_crop),
            lung_margin=float(args.lung_margin),
            use_clahe=bool(args.clahe),
            clahe_clip_limit=float(args.clahe_clip_limit),
            clahe_kernel_size=ks,
            robust_norm=str(args.robust_norm),
            robust_norm_eps=float(args.robust_norm_eps),
            robust_norm_clip=rclip_v,
            robust_norm_order=str(args.robust_norm_order),
        )
        d1, d2, d3, d4, d5, d6 = vec
        Y[kept, 0] = d1
        Y[kept, 1] = d2
        Y[kept, 2] = d3
        Y[kept, 3] = d4
        Y[kept, 4] = d6
        D5[kept] = d5
        kept_paths.append(str(p))
        kept += 1

    X = X[:kept]
    Y = Y[:kept]
    D5 = D5[:kept]
    paths = np.array(kept_paths, dtype=object)

    preprocess: dict = {}
    if args.clahe:
        preprocess.update(
            {
                "clahe": True,
                "clahe_impl": "skimage.exposure.equalize_adapthist",
                "clahe_clip_limit": float(args.clahe_clip_limit),
                "clahe_kernel_size": int(args.clahe_kernel_size) if int(args.clahe_kernel_size) > 0 else None,
            }
        )
    else:
        preprocess["clahe"] = False
    if args.lung_crop:
        preprocess.update(
            {
                "lung_crop": True,
                "lung_crop_impl": "lungmask.LMInferer",
                "lung_margin_ratio": float(args.lung_margin),
                "lungmask_force_cpu": bool(args.lungmask_force_cpu),
                "note": "lungmask is CT-trained; on CXR masks may be weak—tiny masks skip crop (full field).",
            }
        )
    rn = (args.robust_norm or "").strip().lower()
    if rn not in ("", "none", "off"):
        preprocess.update(
            {
                "robust_norm": rn,
                "robust_norm_eps": float(args.robust_norm_eps),
                "robust_norm_clip": (None if float(args.robust_norm_clip) == 0.0 else float(args.robust_norm_clip)),
                "robust_norm_order": (args.robust_norm_order or "").strip().lower(),
                "transform_imagenet_norm": False,
            }
        )
    else:
        preprocess["transform_imagenet_norm"] = True

    meta = {
        "root": str(root),
        "model": args.model,
        "weights": args.weights,
        "image_size": args.image_size,
        "device": device,
        "feat_dim": int(feat_dim),
        "n_files_total": int(len(files)),
        "n_kept": int(kept),
        "n_skipped_missing_labels": int(skipped),
        "y_schema": "Y: [D1,D2,D3,D4,D6]; D5: cavitary lesion 0/1 (private tag 0x1105)",
        "preprocess": preprocess,
        "resize_mode": (args.resize_mode or "").strip().lower(),
        "letterbox_pad_value": float(args.letterbox_pad_value),
        "pipeline_version": (args.pipeline_version or "").strip() or None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y, D5=D5, paths=paths, meta=json.dumps(meta))
    print(f"saved: {out_path}  X={X.shape} Y={Y.shape} skipped={skipped}")


if __name__ == "__main__":
    main()

