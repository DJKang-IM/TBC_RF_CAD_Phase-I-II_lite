import argparse
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm


def dicom_to_uint8(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        wc = float(wc[0] if isinstance(wc, pydicom.multival.MultiValue) else wc)
        ww = float(ww[0] if isinstance(ww, pydicom.multival.MultiValue) else ww)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        lo, hi = np.percentile(arr, [0.5, 99.5])

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return (arr * 255.0).round().astype(np.uint8)


def export_one(dcm_path: Path, out_path: Path) -> None:
    ds = pydicom.dcmread(str(dcm_path))
    img = dicom_to_uint8(ds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=str, help="Folder containing .dcm files")
    ap.add_argument("--out_dir", required=True, type=str, help="Output folder for .png")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    dcm_files = sorted(in_dir.rglob("*.dcm"))

    for p in tqdm(dcm_files, desc=f"Exporting from {in_dir.name}"):
        rel = p.relative_to(in_dir)
        out_path = out_dir / rel.with_suffix(".png")
        export_one(p, out_path)


if __name__ == "__main__":
    main()

