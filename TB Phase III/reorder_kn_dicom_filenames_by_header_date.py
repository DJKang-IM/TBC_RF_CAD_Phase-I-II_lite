# -*- coding: utf-8 -*-
"""
Reorder DICOM filenames by header study date (oldest -> newest) per Study ID (KN or NE).

- KN: Study ID = leading digit run (embed_tb_labels_into_dicom KN).
- NE (나은): Study ID = leading digit run, then **first 5 digits** if length >= 5 (embed NE rule).
- Parses _close from filename (``close`` in stem, case-insensitive).
- Reads StudyDate (+ StudyTime when present) from DICOM header; fallbacks: SeriesDate, AcquisitionDate, file mtime.
- Non-close: sorted by header date within (folder, study_id) -> ``{study_id} (1).dcm``, ``{study_id} (2).dcm``, ...
- Close (완치): **separate** from trial numbering -> ``{study_id}_close.dcm`` (oldest close image).
  If several close DICOMs exist in the same folder, extras become ``{study_id}_close_2.dcm``, ``_3``, ... (collision-safe; see warnings).

Default: **rename in place** (you keep originals elsewhere). Use ``--dry-run`` to print plan only.

Two-phase rename (temp -> final) avoids name collisions on Windows.

Usage:
  python reorder_kn_dicom_filenames_by_header_date.py --root "<REDACTED_PATH> 결핵 DICOM - 복사본"
  python reorder_kn_dicom_filenames_by_header_date.py --root "<REDACTED_PATH> 결핵 DICOM - 복사본" --site ne
  python reorder_kn_dicom_filenames_by_header_date.py --root "..." --site auto   # 나은 in path -> ne
"""

from __future__ import annotations

import argparse
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import pydicom


def parse_kn_stem(name: str) -> tuple[str | None, bool]:
    """Leading digits = Study ID; 'close' anywhere in rest => is_close (KN rule)."""
    base = Path(name).stem.strip()
    m = re.match(r"^(?P<id>\d+)(?P<rest>.*)$", base)
    if not m:
        return None, False
    study_id = m.group("id")
    rest = (m.group("rest") or "").lower()
    is_close = "close" in rest
    return study_id, is_close


def parse_ne_stem(name: str) -> tuple[str | None, bool]:
    """NE (나은): leading digit run; if >=5 digits use first 5 as Study ID (same as embed NE)."""
    stem = Path(name).stem.strip()
    low = stem.lower()
    is_close = "close" in low
    m = re.match(r"^(\d+)", stem)
    if not m:
        return None, is_close
    digits = m.group(1)
    study_id = digits if len(digits) < 5 else digits[:5]
    return study_id, is_close


def resolve_site(root: Path, site: str) -> str:
    if site in ("kn", "ne"):
        return site
    s = str(root.resolve())
    if "나은" in s:
        return "ne"
    return "kn"


def _tag_str(ds: pydicom.dataset.FileDataset, tag: tuple[int, int]) -> str | None:
    if tag not in ds:
        return None
    try:
        v = ds[tag].value
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None
    except Exception:
        return None


def _normalize_da_tm(da: str | None, tm: str | None) -> tuple[str, str]:
    """Return (YYYYMMDD, HHMMSS.ffffff) strings for sorting."""
    d = (da or "19700101").replace("-", "")[:8]
    if len(d) < 8 or not d.isdigit():
        d = "19700101"
    t = (tm or "000000").replace(":", "")[:6]
    if len(t) < 6 or not t[:6].isdigit():
        t = "000000"
    return d, t


def sort_key_from_header(path: Path) -> tuple[str, str, str]:
    """
    Sort key (date, time, tie_breaker).
    tie_breaker = SOPInstanceUID or path string for stable order.
    """
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        m = int(path.stat().st_mtime_ns)
        return ("19700101", "000000", f"z_fallback_mtime_{m:020d}_{path.name}")

    da = _tag_str(ds, (0x0008, 0x0020))  # StudyDate
    tm = _tag_str(ds, (0x0008, 0x0030))  # StudyTime
    if not da:
        da = _tag_str(ds, (0x0008, 0x0021))  # SeriesDate
    if not da:
        da = _tag_str(ds, (0x0008, 0x0022))  # AcquisitionDate
    if not tm:
        tm = _tag_str(ds, (0x0008, 0x0031))  # SeriesTime
    if not tm:
        tm = _tag_str(ds, (0x0008, 0x0032))  # AcquisitionTime

    d, t = _normalize_da_tm(da, tm)
    inst = _tag_str(ds, (0x0020, 0x0013))  # InstanceNumber
    sop = _tag_str(ds, (0x0008, 0x0018))  # SOPInstanceUID
    tie = ""
    if inst is not None:
        try:
            tie += f"{int(float(inst)):010d}_"
        except Exception:
            tie += f"{inst}_"
    tie += sop or path.name
    return (d, t, tie)


def build_rename_plan(root: Path, site: str = "kn") -> tuple[list[tuple[Path, Path]], list[str]]:
    """
    Returns list of (src, dst) with dst in same directory as src, and warning lines.

    Numbering (1),(2),... is **per folder + Study ID**: files in different subfolders
    are independent, so the same Study ID can be ``(1)`` in two different directories.
    """
    warnings: list[str] = []
    files = sorted(root.rglob("*.dcm"), key=lambda p: str(p).lower())
    # key = (parent resolved, study_id)
    by_group: dict[tuple[str, str], list[tuple[Path, bool, tuple]]] = defaultdict(list)

    parse = parse_ne_stem if site == "ne" else parse_kn_stem
    for p in files:
        sid, is_close = parse(p.name)
        if not sid:
            warnings.append(f"skip_no_study_id: {p}")
            continue
        sk = sort_key_from_header(p)
        gkey = (str(p.parent.resolve()), sid)
        by_group[gkey].append((p, is_close, sk))

    plan: list[tuple[Path, Path]] = []
    seen_dst: set[str] = set()

    for (_parent, sid), rows in sorted(by_group.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        non_close = [(p, sk) for (p, ic, sk) in rows if not ic]
        close_only = [(p, sk) for (p, ic, sk) in rows if ic]
        non_close.sort(key=lambda x: (x[1][0], x[1][1], x[1][2]))
        close_only.sort(key=lambda x: (x[1][0], x[1][1], x[1][2]))

        parent = rows[0][0].parent if rows else None
        assert parent is not None

        for idx, (src, _sk) in enumerate(non_close, start=1):
            new_name = f"{sid} ({idx}).dcm"
            dst = parent / new_name
            key = str(dst.resolve()).lower()
            if key in seen_dst:
                warnings.append(f"duplicate_target_skip: {src} -> {dst}")
                continue
            seen_dst.add(key)
            if src.resolve() != dst.resolve():
                plan.append((src, dst))

        n_close = len(close_only)
        if n_close > 1:
            warnings.append(
                f"multiple_close_dicom_same_folder: study_id={sid} n={n_close} -> "
                f"{sid}_close.dcm plus {sid}_close_2.dcm ..."
            )
        for j, (src, _sk) in enumerate(close_only):
            if j == 0:
                new_name = f"{sid}_close.dcm"
            else:
                new_name = f"{sid}_close_{j + 1}.dcm"
            dst = parent / new_name
            key = str(dst.resolve()).lower()
            if key in seen_dst:
                warnings.append(f"duplicate_target_skip: {src} -> {dst}")
                continue
            seen_dst.add(key)
            if src.resolve() != dst.resolve():
                plan.append((src, dst))

    return plan, warnings


def apply_two_phase(plan: list[tuple[Path, Path]], token: str) -> None:
    """Rename src -> temp unique, then temp -> dst."""
    phase1: list[tuple[Path, Path]] = []
    phase2: list[tuple[Path, Path]] = []
    for i, (src, dst) in enumerate(plan):
        tmp = src.parent / f".__reorder_{token}_{i:06d}.dcm"
        phase1.append((src, tmp))
        phase2.append((tmp, dst))

    for src, tmp in phase1:
        src.rename(tmp)
    for tmp, dst in phase2:
        if dst.exists():
            raise FileExistsError(f"Target exists after phase1: {dst}")
        tmp.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename KN/NE DICOMs by StudyDate order per Study ID.")
    ap.add_argument("--root", type=Path, required=True, help=r'Root folder e.g. D:\[RAW][KN] 결핵 DICOM - 복사본')
    ap.add_argument(
        "--site",
        choices=("auto", "kn", "ne"),
        default="auto",
        help="Filename Study ID rule: kn=full leading digits, ne=first 5 digits if len>=5; auto=ne if path contains '나은'.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not rename (default is to rename in place).",
    )
    ap.add_argument("--log", type=Path, default=None, help="Write mapping + warnings to this UTF-8 file")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    resolved_site = resolve_site(root, args.site)
    plan, warnings = build_rename_plan(root, site=resolved_site)
    lines: list[str] = []
    lines.append(f"root={root.resolve()}")
    lines.append(f"site={resolved_site}")
    lines.append(f"n_renames={len(plan)}")
    lines.append("")
    for src, dst in plan:
        lines.append(f"{src}\t->\t{dst}")
    if warnings:
        lines.append("")
        lines.append("=== warnings ===")
        lines.extend(warnings)

    text = "\n".join(lines)
    print(text)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(text, encoding="utf-8")
        print(f"\nWrote log: {args.log}")

    if args.dry_run:
        print("\nDry-run (--dry-run): no files renamed.", file=sys.stderr)
        return 0

    token = uuid.uuid4().hex[:12]
    apply_two_phase(plan, token)
    print(f"\nApplied {len(plan)} renames (token={token}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
