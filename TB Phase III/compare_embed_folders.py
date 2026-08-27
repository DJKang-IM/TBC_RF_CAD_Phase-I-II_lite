from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pydicom
from pydicom.tag import Tag


TAGS = [
    Tag(0x0011, 0x1101),  # D1
    Tag(0x0011, 0x1102),  # D2
    Tag(0x0011, 0x1103),  # D3
    Tag(0x0011, 0x1104),  # D4
    Tag(0x0011, 0x1105),  # D5
    Tag(0x0011, 0x1106),  # D6
]


def read_vec(path: Path) -> tuple[int | None, ...]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return (None, None, None, None, None, None)
    out: list[int | None] = []
    for t in TAGS:
        if t not in ds:
            out.append(None)
            continue
        try:
            out.append(int(ds[t].value))
        except Exception:
            out.append(None)
    return tuple(out)


def is_binary_tuple(vec: tuple[int | None, ...]) -> bool:
    for v in vec:
        if v not in (0, 1):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare DICOM count and D1~D6 embedding between two folders")
    ap.add_argument("--old_root", type=Path, default=Path(r"D:\[EMBED]Active Image"))
    ap.add_argument("--new_root", type=Path, default=Path(r"D:\[EMBED]Active Image + D5(Cavity)"))
    args = ap.parse_args()

    old_root = args.old_root
    new_root = args.new_root

    old_files = sorted(old_root.rglob("*.dcm"))
    new_files = sorted(new_root.rglob("*.dcm"))
    old_rel = {str(p.relative_to(old_root)): p for p in old_files}
    new_rel = {str(p.relative_to(new_root)): p for p in new_files}

    old_set = set(old_rel.keys())
    new_set = set(new_rel.keys())
    common = sorted(old_set & new_set)
    only_old = sorted(old_set - new_set)
    only_new = sorted(new_set - old_set)

    print("=== 1) DICOM count ===")
    print(f"old_root: {old_root}")
    print(f"new_root: {new_root}")
    print(f"old_count: {len(old_files)}")
    print(f"new_count: {len(new_files)}")
    print(f"common_count: {len(common)}")
    print(f"only_old_count: {len(only_old)}")
    print(f"only_new_count: {len(only_new)}")
    if only_old:
        print("only_old sample:", only_old[:10])
    if only_new:
        print("only_new sample:", only_new[:10])

    print("")
    print("=== 2) D1~D6 embedding check on NEW folder ===")
    new_missing_any = 0
    new_non_binary = 0
    d5_present = 0
    d5_one = 0
    for rel in new_set:
        v = read_vec(new_rel[rel])
        if v[4] is not None:
            d5_present += 1
            if v[4] == 1:
                d5_one += 1
        if any(x is None for x in v):
            new_missing_any += 1
            continue
        if not is_binary_tuple(v):
            new_non_binary += 1
    print(f"new_missing_any_tag(D1~D6): {new_missing_any}")
    print(f"new_non_binary_value(D1~D6): {new_non_binary}")
    print(f"new_d5_present: {d5_present} / {len(new_set)}")
    print(f"new_d5_eq_1: {d5_one}")

    print("")
    print("=== 3) OLD vs NEW vector comparison (common files) ===")
    old_missing = 0
    new_missing = 0
    d1_d4_d6_compared = 0
    d1_d4_d6_mismatch = 0
    d5_compared = 0
    d5_mismatch = 0
    mismatch_counter = Counter()
    for rel in common:
        vo = read_vec(old_rel[rel])
        vn = read_vec(new_rel[rel])
        if any(vo[i] is None for i in (0, 1, 2, 3, 5)):
            old_missing += 1
            continue
        if any(vn[i] is None for i in (0, 1, 2, 3, 5)):
            new_missing += 1
            continue
        d1_d4_d6_compared += 1
        for i in (0, 1, 2, 3, 5):
            if vo[i] != vn[i]:
                d1_d4_d6_mismatch += 1
                mismatch_counter[f"D{i+1 if i < 4 else 6}"] += 1
        if vo[4] is not None and vn[4] is not None:
            d5_compared += 1
            if vo[4] != vn[4]:
                d5_mismatch += 1

    print(f"compared_common_for_D1,D2,D3,D4,D6: {d1_d4_d6_compared}")
    print(f"old_missing_any_of_D1,D2,D3,D4,D6: {old_missing}")
    print(f"new_missing_any_of_D1,D2,D3,D4,D6: {new_missing}")
    print(f"total_mismatch_cells_on_D1,D2,D3,D4,D6: {d1_d4_d6_mismatch}")
    if mismatch_counter:
        print("mismatch_by_dim:", dict(mismatch_counter))
    else:
        print("mismatch_by_dim: none")
    print(f"d5_compared_when_both_present: {d5_compared}")
    print(f"d5_mismatch_when_both_present: {d5_mismatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
