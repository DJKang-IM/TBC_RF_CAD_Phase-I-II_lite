"""Scan a folder of DICOMs for private TB label tags (0011,11xx) including D5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pydicom
from pydicom.tag import Tag

TAGS = {
    "D1": Tag(0x0011, 0x1101),
    "D2": Tag(0x0011, 0x1102),
    "D3": Tag(0x0011, 0x1103),
    "D4": Tag(0x0011, 0x1104),
    "D5": Tag(0x0011, 0x1105),
    "D6": Tag(0x0011, 0x1106),
    "PrivateCreator_0010": Tag(0x0011, 0x0010),
}

def read_tag(ds: pydicom.Dataset, t: Tag) -> str | int | None:
    if t not in ds:
        return None
    try:
        v = ds[t].value
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        if isinstance(v, (int, float)) or (hasattr(v, "__int__") and not isinstance(v, str)):
            return int(v)
        return str(v)[:32]
    except Exception as e:  # noqa: BLE001
        return f"<err: {e}>"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check D1-D6 (esp. D5) embedded in DICOM private tags")
    ap.add_argument("--root", type=Path, default=None, help="Folder to scan for *.dcm (recursive); required if not --one")
    ap.add_argument("--max", type=int, default=0, help="0 = all .dcm found")
    ap.add_argument("--one", type=Path, default=None, help="Only check this single file (skip walk)")
    args = ap.parse_args()

    if args.one is not None:
        files = [args.one] if args.one.is_file() else []
        if not files:
            print("Not a file:", args.one, file=sys.stderr)
            return 2
    else:
        if args.root is None:
            print("Need --root DIR or --one FILE", file=sys.stderr)
            return 2
        r = args.root
        if not r.is_dir():
            print("Not a directory:", r, file=sys.stderr)
            return 2
        files = sorted(r.rglob("*.dcm"))
        if args.max and args.max > 0:
            files = files[: args.max]

    n = len(files)
    if n == 0:
        print("No .dcm files under", args.root)
        return 1

    with_d5 = 0
    with_block = 0
    d5_ones = 0
    samples: list[str] = []

    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception as e:  # noqa: BLE001
            continue
        if TAGS["D1"] in ds or TAGS["PrivateCreator_0010"] in ds:
            with_block += 1
        if TAGS["D5"] in ds:
            with_d5 += 1
            try:
                v = int(ds[TAGS["D5"]].value)
                if v == 1:
                    d5_ones += 1
            except Exception:
                pass
            if len(samples) < 8:
                parts = [f"{k}={read_tag(ds, t)}" for k, t in TAGS.items() if k != "PrivateCreator_0010"]
                samples.append(f"{p.name}: " + " ".join(parts))

    print("=== DICOM private-label scan ===")
    print("root / mode:", args.one or args.root)
    print("files_scanned:", n)
    print("any_TB_block_tag_D1_or_0010:", with_block)
    print("has_D5_tag_0011_1105:", with_d5, f"({100.0*with_d5/n:.1f}%)")
    print("D5==1 count (parse ok):", d5_ones)
    if with_d5 < n and with_d5 > 0:
        print("NOTE: some files lack D5 (older embeds or not matched study).")
    if with_d5 == 0:
        print("RESULT: D5 is NOT found on any file — re-run embed with --d5-csv, or use dicom-out folder as input.")
    else:
        print("RESULT: D5 embedding detected on at least", with_d5, "file(s).")
    if samples:
        print("\n--- sample lines (D1..D6) ---")
        for s in samples:
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())