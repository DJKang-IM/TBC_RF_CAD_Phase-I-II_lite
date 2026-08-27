import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom


def dicom_to_float32(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    # Windowing (if present); otherwise robust percentiles
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
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dicom_path", type=str)
    args = ap.parse_args()

    path = Path(args.dicom_path)
    ds = pydicom.dcmread(str(path))
    img = dicom_to_float32(ds)

    title = f"{path.name}"
    if hasattr(ds, "PatientID"):
        title += f" | PatientID={getattr(ds, 'PatientID', '')}"
    if hasattr(ds, "StudyDate"):
        title += f" | StudyDate={getattr(ds, 'StudyDate', '')}"

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

