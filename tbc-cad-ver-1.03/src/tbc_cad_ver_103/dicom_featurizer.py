"""ResNet pooled embedding from a single DICOM (shared Phase I-III feature space)."""

from __future__ import annotations

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


def _configure_https() -> None:
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


def _default_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size), antialias=True),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _build_backbone(name: str, pretrained: bool, weights: Path | None) -> tuple[nn.Module, int]:
    name = name.lower().strip()
    if name == "resnet18":
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
        feat_dim = 512
    elif name == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        feat_dim = 2048
    else:
        raise ValueError("Unsupported backbone. Use resnet18 or resnet50.")
    if weights is not None:
        state = torch.load(str(weights), map_location="cpu")
        m.load_state_dict(state, strict=False)
    backbone = nn.Sequential(*(list(m.children())[:-1]))
    backbone.eval()
    return backbone, feat_dim


class DicomFeaturizer:
    def __init__(
        self,
        backbone: str,
        *,
        image_size: int = 224,
        pretrained: bool = True,
        weights: Path | None = None,
        device: str | None = None,
    ) -> None:
        if pretrained:
            _configure_https()
        self.backbone_name = backbone
        self.image_size = image_size
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = dev
        self._backbone, self.feat_dim = _build_backbone(backbone, pretrained, weights)
        self._backbone = self._backbone.to(dev)
        self._tfm = _default_transform(image_size)

    @torch.no_grad()
    def embed_path(self, path: str | Path) -> np.ndarray:
        p = Path(path)
        img = dicom_to_float01(p)
        x = self._tfm(img).unsqueeze(0).to(self._device)
        feat = self._backbone(x).flatten(1).detach().cpu().numpy().astype(np.float32)
        return feat
