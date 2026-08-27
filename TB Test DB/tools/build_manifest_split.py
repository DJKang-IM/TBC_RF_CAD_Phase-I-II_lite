import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pydicom
from tqdm import tqdm


@dataclass(frozen=True)
class Item:
    path: Path
    label: int
    source: str
    group_id: str


def dicom_group_id(p: Path) -> str:
    """
    Split-leakage guard.
    Prefer patient/study identifiers if present; fall back to a stable path-based id.
    """
    try:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        pid = str(getattr(ds, "PatientID", "") or "").strip()
        study = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
        if pid and study:
            return f"{pid}::{study}"
        if pid:
            return f"{pid}"
        if study:
            return f"study::{study}"
    except Exception:
        pass
    return f"path::{p.resolve()}"


def split_of_group(group_id: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    if not (0 < train_ratio < 1) or not (0 <= val_ratio < 1) or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy 0 < train_ratio < 1, 0 <= val_ratio < 1, train+val < 1")
    h = hashlib.sha256(f"{seed}::{group_id}".encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF  # ~[0,1]
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def iter_dicoms(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"Not found: {root}")
    files = sorted(root.rglob("*.dcm"))
    if not files:
        raise SystemExit(f"No .dcm found under: {root}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="artifacts/manifest.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument(
        "--neg",
        action="append",
        default=[],
        help="Negative class root folder (repeatable). Used as label=0.",
    )
    ap.add_argument(
        "--pos",
        action="append",
        default=[],
        help="Positive class root folder (repeatable). Used as label=1.",
    )
    args = ap.parse_args()

    if not args.neg or not args.pos:
        raise SystemExit("Provide at least one --neg and one --pos folder.")

    items: list[Item] = []
    for root in args.neg:
        r = Path(root)
        for p in iter_dicoms(r):
            items.append(Item(path=p, label=0, source=str(r), group_id=dicom_group_id(p)))
    for root in args.pos:
        r = Path(root)
        for p in iter_dicoms(r):
            items.append(Item(path=p, label=1, source=str(r), group_id=dicom_group_id(p)))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["abs_path", "label", "split", "group_id", "source_root"])
        for it in tqdm(items, desc="write-manifest", unit="file"):
            split = split_of_group(it.group_id, args.seed, args.train_ratio, args.val_ratio)
            w.writerow([str(it.path.resolve()), it.label, split, it.group_id, it.source])

    print(f"saved: {out_path}  n={len(items)}")


if __name__ == "__main__":
    main()

