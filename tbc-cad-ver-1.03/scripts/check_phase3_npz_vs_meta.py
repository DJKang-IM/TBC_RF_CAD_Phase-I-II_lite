# -*- coding: utf-8 -*-
"""
Cross-check phase3 .npz (X, Y, D5, paths) against:
  - optional manifest CSV (per-file D1-D4, D6; abs_path or basename match)
  - optional D5 Version 2 CSV (study_id -> D5) + KN/NE filename rules (same as embed)
Reports counts, mismatches, and D5 consistency per study (multiple DICOMs per study).
Run from anywhere:
  python "<REDACTED_PATH> Phase III/check_phase3_npz_vs_meta.py" --help
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# filename rules aligned with embed_tb_labels_into_dicom.py
def _parse_kn(name: str) -> tuple[str | None, bool]:
    base = Path(name).stem
    m = re.match(r"^(?P<id>\d+)(?:_(?P<suffix>.+))?$", base)
    if not m:
        return None, False
    study_id = m.group("id")
    suffix = (m.group("suffix") or "").lower()
    is_close = suffix == "close"
    return study_id, is_close


def _parse_ne(name: str) -> tuple[str | None, bool]:
    stem = Path(name).stem
    low = stem.lower()
    is_close = "close" in low
    m = re.match(r"^(\d+)", stem)
    if not m:
        return None, is_close
    digits = m.group(1)
    study_id = digits if len(digits) < 5 else digits[:5]
    return study_id, is_close


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _csv_study_key(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int,)):
        return str(int(v))
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def _d5_cell(v) -> int:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0
    s = str(v).strip()
    if s in {"1", "1.0", "true", "True", "YES", "Y"}:
        return 1
    if s in {"0", "0.0", "false", "False", "NO", "N", ""}:
        return 0
    try:
        return 1 if int(float(s)) == 1 else 0
    except ValueError:
        return 0


def load_d5_version2_map(path: Path) -> dict[str, int]:
    df = _read_csv(path)
    cols = list(df.columns)
    study_cands = [
        c
        for c in cols
        if str(c).strip().lower()
        in {"study no.", "study no", "study_id", "study id", "studyid", "study id."}
    ]
    study_col = study_cands[0] if study_cands else cols[0]
    d5_col = None
    for c in cols:
        if c and "판독" in c:
            d5_col = c
            break
    if d5_col is None:
        raise SystemExit(f"D5 CSV needs a '판독' column: {path}")
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        sid = _csv_study_key(row[study_col])
        if not sid:
            continue
        out[sid] = _d5_cell(row[d5_col])
    return out


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def check_one_npz(
    name: str,
    data: dict,
    manifest_by_name: dict[str, tuple] | None,
    d5_map: dict[str, int] | None,
    site: str,
    check_files: bool,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"=== {name} ===")
    need = ("X", "Y", "paths")
    for k in need:
        if k not in data:
            lines.append(f"  ERROR: missing key '{k}'")
            return lines
    if "D5" not in data:
        lines.append("  NOTE: key 'D5' absent — run build_phase3_features on DICOMs with D5 embedded in headers")
        d5i = None
    else:
        d5i = data["D5"].ravel().astype(int)

    X, Y, paths = data["X"], data["Y"], data["paths"]
    n = int(X.shape[0])
    if d5i is not None and (paths.shape[0] != n or d5i.shape[0] != n or Y.shape[0] != n):
        lines.append(
            f"  ERROR: length mismatch n={n} paths={paths.shape[0]} D5={d5i.shape[0] if d5i is not None else 'n/a'} Y={Y.shape[0]}"
        )
        return lines
    if paths.shape[0] != n or Y.shape[0] != n:
        lines.append(f"  ERROR: length mismatch n={n} paths={paths.shape[0]} Y={Y.shape[0]}")
        return lines

    lines.append(f"  n={n}  X dim={X.shape[1] if X.ndim>1 else 0}  Y shape={Y.shape}")

    parse = _parse_ne if site == "ne" else _parse_kn
    missing_disk = 0
    for i in range(n):
        p = Path(str(paths[i]))
        if check_files and not p.is_file():
            missing_disk += 1

    if d5i is None:
        lines.append("  D5: (not in npz) — rebuild features after D5 is embedded in DICOMs")
    else:
        bad_d5 = np.where((d5i != 0) & (d5i != 1))[0]
        lines.append(
            f"  D5: pos={int(d5i.sum())}  neg={n - int(d5i.sum())}  rate={100.0 * float(d5i.mean()):.2f}%"
        )
        if len(bad_d5):
            lines.append(
                f"  WARNING: D5 not in {{0,1}} for {len(bad_d5)} rows (first idx): {bad_d5[:5].tolist()}"
            )
        by_study: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for i in range(n):
            p = Path(str(paths[i]))
            sid, _ = parse(p.name)
            if sid:
                by_study[sid].append((p.name, int(d5i[i])))
        study_conflicts = 0
        for sid, pairs in by_study.items():
            vs = {d for _, d in pairs}
            if len(vs) > 1:
                study_conflicts += 1
        if by_study:
            lines.append(f"  unique study_id (from filename): {len(by_study)}  (site={site})")
            if study_conflicts:
                lines.append(
                    f"  WARNING: {study_conflicts} study_id have inconsistent D5 across files (embed should be per-study)"
                )

    if check_files:
        lines.append(f"  paths: missing on disk (is_file==False): {missing_disk} / {n}")

    if manifest_by_name:
        m_y_mismatch = 0
        m_missing = 0
        for i in range(n):
            p = Path(str(paths[i]))
            key = p.name
            if key not in manifest_by_name:
                m_missing += 1
                continue
            exp = manifest_by_name[key]
            d1, d2, d3, d4, d6 = exp
            yrow = Y[i].astype(int)
            if yrow[0] != d1 or yrow[1] != d2 or yrow[2] != d3 or yrow[3] != d4 or (yrow[4] if yrow.size > 4 else 0) != d6:
                m_y_mismatch += 1
        lines.append(
            f"  vs manifest (basename): no row in manifest: {m_missing}  Y vs D1-D4/D6 mismatch: {m_y_mismatch}"
        )

    if d5_map and d5i is not None:
        d5_csv_mis = 0
        d5_unmapped = 0
        for i in range(n):
            p = Path(str(paths[i]))
            sid, _ = parse(p.name)
            if not sid or sid not in d5_map:
                d5_unmapped += 1
                exp = 0
            else:
                exp = d5_map[sid]
            if int(d5i[i]) != int(exp):
                d5_csv_mis += 1
        lines.append(
            f"  vs D5 CSV (by study_id): D5 mismatch: {d5_csv_mis}  study not in CSV (compare as exp=0): {d5_unmapped}"
        )
    elif d5_map and d5i is None:
        lines.append("  vs D5 CSV: skipped (add D5 to npz to compare npz D5 to Version 2 판독문)")

    return lines


def build_manifest_index(manifest_path: Path) -> dict[str, tuple[int, int, int, int, int]]:
    df = _read_csv(manifest_path)
    if "D1" not in df.columns or "abs_path" not in df.columns:
        raise SystemExit("Manifest needs columns abs_path, D1, D2, D3, D4, D6 (D6 in manifest)")
    idx: dict[str, tuple[int, int, int, int, int]] = {}
    for _, row in df.iterrows():
        name = Path(str(row["abs_path"])).name
        d6 = int(row.get("D6", 0))
        idx[name] = (int(row["D1"]), int(row["D2"]), int(row["D3"]), int(row["D4"]), d6)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Check phase3 npz vs manifest and D5 Version2 CSV")
    ap.add_argument("--train_npz", type=Path, default=None)
    ap.add_argument("--val_npz", type=Path, default=None)
    ap.add_argument("--test_npz", type=Path, default=None)
    ap.add_argument("--p3_root", type=Path, default=Path(r"D:\TB Phase III"), help="If set, default train/val/test names here")
    ap.add_argument("--manifest", type=Path, default=None, help="e.g. phase3_manifest_*.csv with abs_path, D1..D4, D6")
    ap.add_argument("--d5-csv", type=Path, default=None, help="Version 2 meta with Study + 판독문 0/1")
    ap.add_argument("--site", choices=("kn", "ne"), default="kn", help="Filename parsing (same as embed)")
    ap.add_argument("--check-files", action="store_true", help="Verify each path exists on disk")
    ap.add_argument("--out", type=Path, default=None, help="Write report to this UTF-8 text file")
    args = ap.parse_args()

    root = args.p3_root
    paths: list[tuple[str, Path]] = []
    if args.train_npz:
        paths.append(("train", args.train_npz))
    if args.val_npz:
        paths.append(("val", args.val_npz))
    if args.test_npz:
        paths.append(("test", args.test_npz))
    if not paths:
        for split in ("train", "val", "test"):
            p = root / f"phase3_features_{split}.npz"
            if p.is_file():
                paths.append((split, p))
    if not paths:
        print("No npz found. Pass --train_npz / --p3_root with phase3_features_*.npz", file=sys.stderr)
        return 2

    manifest_by_name: dict[str, tuple[int, int, int, int, int]] | None = None
    if args.manifest and args.manifest.is_file():
        manifest_by_name = build_manifest_index(args.manifest)
        print(f"Loaded manifest: {args.manifest}  file rows (by basename): {len(manifest_by_name)}")

    d5_map: dict[str, int] | None = None
    if args.d5_csv and args.d5_csv.is_file():
        d5_map = load_d5_version2_map(args.d5_csv)
        print(f"Loaded D5 CSV: {args.d5_csv}  study keys: {len(d5_map)}")

    if manifest_by_name is None and d5_map is None:
        print("Tip: add --manifest and/or --d5-csv to compare labels; otherwise only npz structure is checked.")

    all_lines: list[str] = []
    for label, p in paths:
        if not p.is_file():
            all_lines.append(f"=== {label} missing file: {p} ===\n")
            continue
        data = load_npz(p)
        all_lines.extend(
            check_one_npz(
                f"{label}  {p}",
                data,
                manifest_by_name,
                d5_map,
                args.site,
                args.check_files,
            )
        )
        all_lines.append("")

    text = "\n".join(all_lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("Wrote", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
