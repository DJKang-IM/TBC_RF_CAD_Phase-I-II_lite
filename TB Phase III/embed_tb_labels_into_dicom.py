"""
D1–D6 = six *independent* data elements (VR=US, value 0 or 1 each). No packed blob required.

**Canonical (model / features):** private group 0x0011, one Private Creator + six labels
- (0011,0010) LO PrivateCreator = TB_PHASE3_LABELS
- (0011,1101) US  D1
- (0011,1102) US  D2
- (0011,1103) US  D3
- (0011,1104) US  D4
- (0011,1105) US  D5  (cavitary)
- (0011,1106) US  D6  (NTM)

**Optional (derived, for quick viewing only):** off if --no-derived-tags
- (0011,1110) US  sum(D1..D4)
- (0011,1111) CS  TB|NTM

**Why not "standard" Grayscale / CT top-level tags for 6 flags?**  
STANDARD: No well-defined public tag pair gives six empty binary flags on all modalities; reusing
any clinical tag (e.g. comments) is unsafe and PACS may strip or alter them.  
PRIVATE (this design): odd group 0x0011 + Private Creator is the normal DICOM pattern for
site-specific flags; US fits 0/1 in two bytes. Alternatives considered and rejected: single LO/SH
string (harder to index), one multi-value US (one element, not six headers), OB blob (not separate
headers), SQ (overkill). If 0x0011 conflicts on a specific PACS, switch **group** only (e.g. 0x7F01
with a new creator) and keep 0x11xx private block offsets in sync across readers.

Legacy duplicate LO (0011,1010) packed vector: only with --write-packed-vector-lo.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tb_cad_dicom_private_layout import (
    EMBED_GROUP as PRIVATE_GROUP,
    PHASE3_PRIVATE_CREATOR_ELEM as PRIVATE_CREATOR_ELEM,
    PHASE3_PRIVATE_CREATOR_VALUE as PRIVATE_CREATOR_VALUE,
    TAG_D1,
    TAG_D2,
    TAG_D3,
    TAG_D4,
    TAG_D5,
    TAG_D6,
    TAG_FINAL,
    TAG_PACKED_VEC_LO as TAG_VEC_STR,
    TAG_SCORE,
)

# TAG_VEC_STR = packed LO (0011,1010) — only if --write-packed-vector-lo; see layout module for P1/P2 reservation hints.


def _read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001 - intentional fallback
            last_err = e
    raise RuntimeError(f"Failed to read CSV with utf-8-sig/cp949/euc-kr: {path}") from last_err


def _cell_flags(v) -> tuple[int, bool]:
    """
    Convert a label cell into (tb_positive, ntm_present).
    Known values in KR datasets often include:
    - '양성', '음성'
    - '미결'/'결핍' 등 결측성 표기
    - 'NTM' 표기
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0, False
    s = str(v).strip()
    if s == "":
        return 0, False

    s_up = s.upper()
    if "NTM" in s_up:
        # When a cell explicitly indicates NTM, we treat it as NTM evidence.
        # We keep TB-positive for that test as 0 because it will be overridden anyway.
        return 0, True

    if s_up in {"1", "TRUE", "T", "Y", "YES", "POS", "POSITIVE"}:
        return 1, False
    if s_up in {"0", "FALSE", "F", "N", "NO", "NEG", "NEGATIVE"}:
        return 0, False

    # Korean common strings
    if "양성" in s:
        return 1, False
    if "음성" in s:
        return 0, False

    # Unknown tokens are treated as 0 by default (safer than forcing 1)
    return 0, False


def _infer_label_columns(df: pd.DataFrame) -> list[str]:
    """
    Try to infer label columns from the CSV (study column + at least 4 others).
    If we can't, fall back to "all columns except Study No." taking the first 6.
    """
    cols = list(df.columns)
    # study id column
    study_candidates = [
        c
        for c in cols
        if str(c).strip().lower()
        in {"study no.", "study no", "study_id", "study id", "studyid", "study id."}
    ]
    if not study_candidates:
        # some files may use "Study No." exactly (as seen), but in strange encodings this may differ;
        # fallback: first column as study id
        study_col = cols[0]
    else:
        study_col = study_candidates[0]

    # everything else are candidates
    label_cols = [c for c in cols if c != study_col]
    if len(label_cols) < 4:
        raise RuntimeError(f"Need at least 4 label columns besides Study ID. Found: {label_cols}")
    return [study_col] + label_cols[:6]


def _resolve_column(df: pd.DataFrame, token: str) -> str:
    """
    Resolve a column spec token into an existing column name.
    - If token is an int (e.g. "0", "3"): treat as 0-based column index.
    - Otherwise: treat as exact column name.
    """
    token = token.strip()
    if token == "":
        raise ValueError("Empty column token")

    if re.fullmatch(r"\d+", token):
        idx = int(token)
        cols = list(df.columns)
        if idx < 0 or idx >= len(cols):
            raise ValueError(f"Column index out of range: {idx} (0..{len(cols)-1})")
        return cols[idx]

    if token not in df.columns:
        raise ValueError(f"Column not found: {token!r}")
    return token


def _pick_columns(df: pd.DataFrame, study_col: str | None, label_cols: str | None) -> tuple[str, list[str]]:
    """
    Pick study id column + 4 label columns (D1..D4).
    D6(NTM) is derived: if any of D1..D4 cells includes "NTM", then D6=1.
    """
    if study_col is None and label_cols is None:
        inferred = _infer_label_columns(df)
        # use first 4 as D1..D4 by default; D6 derived
        return inferred[0], inferred[1:5]

    # Resolve study col
    if study_col is None:
        # default: infer study column only, but do not infer labels (labels must be provided)
        inferred = _infer_label_columns(df)
        resolved_study = inferred[0]
    else:
        resolved_study = _resolve_column(df, study_col)

    if not label_cols:
        raise ValueError("--label-cols is required when --study-col is provided (or when auto-infer is not desired).")

    parts = [p.strip() for p in label_cols.split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError(f"--label-cols must have exactly 4 columns for D1..D4. Got {len(parts)}: {parts}")

    resolved_labels = [_resolve_column(df, p) for p in parts]
    return resolved_study, resolved_labels


def _parse_study_id_from_filename(name: str) -> tuple[str | None, bool]:
    """
    Returns (study_id, is_close).
    Examples:
      10144_1.dcm -> ("10144", False)
      10144_close.dcm -> ("10144", True)
      10144_close -> ("10144", True)
    """
    base = Path(name).stem.strip()
    # KN filenames can appear as:
    # - 10144_1.dcm
    # - 10095_.dcm
    # - 10153 (1).dcm
    # We always map by the leading numeric Study ID.
    m = re.match(r"^(?P<id>\d+)(?P<rest>.*)$", base)
    if not m:
        return None, False
    study_id = m.group("id")
    rest = (m.group("rest") or "").lower()
    is_close = "close" in rest
    return study_id, is_close


def _parse_ne_filename(name: str) -> tuple[str | None, bool]:
    """
    나은(NE) 병원 파일명 규칙.
    - Study ID: 파일명 stem **맨 앞 연속 숫자의 앞 5자리만** (예: 301230_6 -> 30123).
    - 완치: stem에 'close' 포함 시 (예: 30125_close0, 30124_close) -> is_close True.
    """
    stem = Path(name).stem
    low = stem.lower()
    is_close = "close" in low
    m = re.match(r"^(\d+)", stem)
    if not m:
        return None, is_close
    digits = m.group(1)
    if len(digits) < 5:
        study_id = digits
    else:
        study_id = digits[:5]
    return study_id, is_close


def _d5_value(v) -> int:
    """
    Parse D5 (cavitary lesion) from either:
    - explicit binary tokens (0/1, true/false, yes/no), or
    - free-text reading containing cavity-related terms.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0
    s = str(v).strip()
    if s == "":
        return 0

    s_low = s.lower()
    if s_low in {"1", "1.0", "true", "yes", "y", "pos", "positive"}:
        return 1
    if s_low in {"0", "0.0", "false", "no", "n", "neg", "negative"}:
        return 0

    # Free-text 판독문 parsing:
    # User rule: if any cavity-related term appears, force D5=1.
    pos_pat = re.compile(r"(공동|공동성|cavity|cavitary|cavitary lesion|cavitation|cavit)", flags=re.IGNORECASE)
    if pos_pat.search(s):
        return 1

    try:
        return 1 if int(float(s)) == 1 else 0
    except ValueError:
        return 0


def _load_d5_map_from_csv(path: Path) -> dict[str, int]:
    """
    Load map study_id -> 0/1 for D5 (cavitary lesion) from a CSV such as *Version 2* meta
    (column 판독문 must be 0/1; study column: Study No. / Study ID or first column).
    """
    df = _read_csv_with_fallbacks(path)
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
        c_str = str(c).strip()
        c_low = c_str.lower()
        if ("판독" in c_str) or ("reading" in c_low) or ("소견" in c_str) or ("impression" in c_low):
            d5_col = c
            break
    if d5_col is None:
        raise RuntimeError(f"Need a D5 text column (e.g. '판독문' or 'Reading') in D5 CSV: {path}")
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        sid = _csv_study_key(row[study_col])
        if not sid:
            continue
        out[sid] = _d5_value(row[d5_col])
    return out


def _csv_study_key(v) -> str | None:
    """Normalize Study ID from CSV cell to string key (e.g. 30123.0 -> '30123')."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int,)):
        return str(int(v))
    if isinstance(v, float):
        if float(v).is_integer():
            return str(int(v))
        s = str(v).strip()
        return s if s and s.lower() != "nan" else None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def _study_key_candidates(study_id: str) -> list[str]:
    """
    Metadata-key candidates for filename-derived study IDs.
    Example: 301330 -> [301330, 30133] so it can match a 5-digit Study No.
    """
    sid = str(study_id).strip()
    if not sid:
        return []
    out: list[str] = [sid]
    if sid.isdigit() and len(sid) > 5:
        pref5 = sid[:5]
        if pref5 not in out:
            out.append(pref5)
        t = sid
        while len(t) > 5:
            t = t[:-1]
            if t not in out:
                out.append(t)
    return out


def _lookup_mapping_with_fallback(mapping: dict[str, list[int]], study_id: str) -> tuple[list[int] | None, str | None]:
    for key in _study_key_candidates(study_id):
        if key in mapping:
            return mapping[key], key
    return None, None


def _lookup_d5_with_fallback(d5_map: dict[str, int], study_id: str) -> int:
    for key in _study_key_candidates(study_id):
        if key in d5_map:
            return int(d5_map[key])
    return 0


def _log_all_elements_in_group(ds: Dataset, group: int, title: str) -> None:
    """Debug: list every data element in ``group`` (e.g. 0x0011). Does not modify ``ds``."""
    rows: list[str] = []
    for tag in sorted(ds.keys(), key=int):
        t = Tag(tag)
        if t.group != group:
            continue
        elem = ds[tag]
        v = elem.value
        s = repr(v) if v is not None and not isinstance(v, (bytes, bytearray)) else f"<{type(v).__name__} len={len(v) if v is not None else 0}>"
        if len(s) > 100:
            s = s[:97] + "..."
        rows.append(f"  {t}  {getattr(elem, 'name', '')}  =  {s}")
    print(f"--- {title}  group=0x{group:04X}  n={len(rows)} ---")
    print("\n".join(rows) if rows else "  (empty)")


def _ensure_private_creator(ds: Dataset) -> None:
    creator_tag = Tag(PRIVATE_GROUP, PRIVATE_CREATOR_ELEM)
    # pydicom: private creator is LO, but stored in the creator element
    if creator_tag not in ds:
        ds.add_new(creator_tag, "LO", PRIVATE_CREATOR_VALUE)
    else:
        # keep existing, but if different, append a second creator slot (0011,0011) etc.
        if str(ds.get(creator_tag).value) != PRIVATE_CREATOR_VALUE:
            # find first free creator element in the group (0x0010..0x00FF)
            for elem in range(0x0010, 0x0100):
                t = Tag(PRIVATE_GROUP, elem)
                if t not in ds:
                    ds.add_new(t, "LO", PRIVATE_CREATOR_VALUE)
                    break


def _write_labels(
    ds: Dataset,
    vec: list[int],
    *,
    write_packed_vector_lo: bool = False,
    write_derived_tags: bool = True,
) -> None:
    if len(vec) != 6:
        raise ValueError("Expected 6-length vector [D1..D4,D5,D6] (D5=cavitary lesion)")

    # Never ``del``/``clear`` the dataset. Phase I/II tags in other (group,element) stay unless
    # they reuse the same numbers as P3 (see tb_cad_dicom_private_layout).

    # If you plan to store Korean strings in header, set UTF-8.
    # NOTE: some legacy viewers ignore UTF-8; if so, prefer English in private tags.
    if "SpecificCharacterSet" not in ds:
        ds.SpecificCharacterSet = "ISO_IR 192"  # UTF-8

    _ensure_private_creator(ds)

    d1, d2, d3, d4, d5, d6 = [int(x) for x in vec]
    score = int(d1 + d2 + d3 + d4)
    final = "NTM" if d6 == 1 else "TB"

    # Six independent data elements (one DICOM attribute per class).
    ds.add_new(TAG_D1, "US", d1)
    ds.add_new(TAG_D2, "US", d2)
    ds.add_new(TAG_D3, "US", d3)
    ds.add_new(TAG_D4, "US", d4)
    ds.add_new(TAG_D5, "US", d5)
    ds.add_new(TAG_D6, "US", d6)
    if write_derived_tags:
        ds.add_new(TAG_SCORE, "US", score)
        ds.add_new(TAG_FINAL, "CS", final)

    # Optional legacy duplicate — off by default (avoids two representations of the same vector)
    if write_packed_vector_lo:
        ds.add_new(TAG_VEC_STR, "LO", "\\".join(str(x) for x in vec))


def _force_inactive_signature_for_close(ds: Dataset) -> None:
    """
    For *_close samples, force legacy active/inactive signature to inactive-style zeros.
    - Keep D1..D6 handling in _write_labels (already all-zero when close-all-zero path is used).
    - Also normalize legacy packed vector LO (0011,1010) to all-zero string.
    """
    zero5 = "0\\0\\0\\0\\0"
    if TAG_VEC_STR in ds:
        ds[TAG_VEC_STR].value = zero5
    else:
        ds.add_new(TAG_VEC_STR, "LO", zero5)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Embed TB/NTM multi-label vector (D1–D4, D5 cavitary, D6 NTM) into DICOM headers using private tags."
    )
    ap.add_argument("--csv", required=True, help="Path to metadata CSV (Study No. + 6 label columns).")
    ap.add_argument("--dicom-in", required=True, help="Input folder containing DICOM files.")
    ap.add_argument("--dicom-out", required=True, help="Output folder to write modified DICOM files.")
    ap.add_argument(
        "--site",
        choices=("kn", "ne"),
        default="kn",
        help="Filename / close rules: kn=강남성심(KN), ne=나은(NE, 5-digit ID + close*).",
    )
    ap.add_argument(
        "--study-col",
        default=None,
        help="Study ID column name or 0-based index. If omitted, script tries to infer (default).",
    )
    ap.add_argument(
        "--label-cols",
        default=None,
        help="Comma-separated 4 columns for D1..D4 (each can be exact name or 0-based index). "
        "D6(NTM) is derived: if any of D1..D4 cells contains 'NTM', then D6=1. "
        "Example: --label-cols \"도말검사,TB PCR,배양검사(고체),배양검사(액체)\" or --label-cols \"1,2,3,4\"",
    )
    ap.add_argument(
        "--print-columns",
        action="store_true",
        help="Print CSV columns (with 0-based indices) and exit.",
    )
    ap.add_argument(
        "--close-all-zero",
        action="store_true",
        help="If filename indicates cured (kn: *_close.*; ne: *close* in stem e.g. close0), force all-zero vector.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print what would change.",
    )
    ap.add_argument(
        "--d5-csv",
        default=None,
        help="Optional CSV (e.g. *Version 2* meta) with study ID + '판독문' 0/1 = D5 cavitary lesion. If omitted, D5=0.",
    )
    ap.add_argument(
        "--write-packed-vector-lo",
        action="store_true",
        help="Also write (0011,1010) LO with D1~D6 backslash-separated. Off by default; "
        "readers should use (0011,1101..1106) US only.",
    )
    ap.add_argument(
        "--no-derived-tags",
        action="store_true",
        help="Do not write (0011,1110) SCORE and (0011,1111) FINAL; only the six D1~D6 US elements.",
    )
    ap.add_argument(
        "--log-private-group",
        action="store_true",
        help="On the first successfully read DICOM, print all elements in P3 private group 0x0011 before/after P3 write (audit P1/P2 + P3).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    dicom_in = Path(args.dicom_in)
    dicom_out = Path(args.dicom_out)

    df = _read_csv_with_fallbacks(csv_path)
    if args.print_columns:
        for i, c in enumerate(list(df.columns)):
            print(f"[{i}] {c}")
        return 0

    # NE defaults:
    # Prefer explicit column names when present (robust to column order changes).
    # Fallback to legacy index rule only when these names are absent.
    label_cols_arg = args.label_cols
    if args.site == "ne" and not label_cols_arg:
        ne_name_cols = ["도말검사", "TB-PCR검사", "배양검사(고체)", "배양검사(액체)"]
        if all(c in df.columns for c in ne_name_cols):
            label_cols_arg = ",".join(ne_name_cols)
        else:
            label_cols_arg = "1,2,4,5"

    study_col, label_cols = _pick_columns(df, args.study_col, label_cols_arg)

    d5_map: dict[str, int] = {}
    if args.d5_csv:
        d5_map = _load_d5_map_from_csv(Path(args.d5_csv))

    # Build mapping: study_id(str) -> [D1,D2,D3,D4,D6] (D5 added at embed time)
    mapping: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        sid = _csv_study_key(row[study_col])
        if not sid:
            continue
        flags = [_cell_flags(row[c]) for c in label_cols]
        d1, ntm1 = flags[0]
        d2, ntm2 = flags[1]
        d3, ntm3 = flags[2]
        d4, ntm4 = flags[3]
        d6 = 1 if (ntm1 or ntm2 or ntm3 or ntm4) else 0
        vec = [int(d1), int(d2), int(d3), int(d4), int(d6)]
        mapping[sid] = vec

    if not dicom_in.exists():
        raise RuntimeError(f"dicom-in does not exist: {dicom_in}")
    dicom_out.mkdir(parents=True, exist_ok=True)

    # Iterate all files (some sites use no extension)
    in_files = [p for p in dicom_in.rglob("*") if p.is_file()]
    if not in_files:
        raise RuntimeError(f"No files found under dicom-in: {dicom_in}")

    processed = 0
    skipped = 0
    missing_label = 0
    missing_study_ids: Counter[str] = Counter()
    missing_label_files: list[tuple[str, str]] = []
    fallback_key_hits = 0
    _logged_group = False

    parse_fn = _parse_ne_filename if args.site == "ne" else _parse_study_id_from_filename

    skip_ext = {
        ".csv",
        ".tsv",
        ".txt",
        ".xlsx",
        ".xls",
        ".json",
        ".md",
        ".zip",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".pdf",
    }

    for p in tqdm(in_files, desc="Embedding labels"):
        if p.suffix.lower() in skip_ext:
            skipped += 1
            continue

        study_id, is_close = parse_fn(p.name)
        if not study_id:
            skipped += 1
            continue

        if args.close_all_zero and is_close:
            vec6 = [0, 0, 0, 0, 0, 0]
        else:
            v5, used_key = _lookup_mapping_with_fallback(mapping, study_id)
            if v5 is None:
                missing_label += 1
                missing_study_ids[str(study_id)] += 1
                missing_label_files.append((p.name, str(study_id)))
                continue
            if used_key is not None and used_key != study_id:
                fallback_key_hits += 1
            d1, d2, d3, d4, d6 = v5
            d5 = _lookup_d5_with_fallback(d5_map, study_id)
            vec6 = [d1, d2, d3, d4, d5, d6]

        try:
            ds = pydicom.dcmread(str(p), force=True)
        except Exception:
            skipped += 1
            continue

        if bool(args.log_private_group) and not _logged_group:
            _log_all_elements_in_group(ds, PRIVATE_GROUP, f"BEFORE P3 embed  file={p.name}")
        _write_labels(
            ds,
            vec6,
            write_packed_vector_lo=bool(args.write_packed_vector_lo),
            write_derived_tags=not bool(args.no_derived_tags),
        )
        if bool(args.close_all_zero) and is_close:
            _force_inactive_signature_for_close(ds)
        if bool(args.log_private_group) and not _logged_group:
            _log_all_elements_in_group(ds, PRIVATE_GROUP, f"AFTER P3 embed  file={p.name}")
            _logged_group = True

        out_path = dicom_out / p.relative_to(dicom_in)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            processed += 1
            continue

        # Preserve original transfer syntax as much as possible
        ds.save_as(str(out_path), write_like_original=True)
        processed += 1

    print(
        f"Done. processed={processed}, skipped={skipped}, missing_label={missing_label}, "
        f"fallback_key_hits={fallback_key_hits}. "
        f"Output: {dicom_out}"
    )
    if missing_study_ids:
        miss_path = dicom_out / "_missing_label_study_ids.csv"
        rows = ["study_id,n_files"]
        rows += [f"{sid},{int(n)}" for sid, n in missing_study_ids.most_common()]
        miss_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"Wrote: {miss_path}  (unique study_id={len(missing_study_ids)})")
    if missing_label_files:
        miss_file_path = dicom_out / "_missing_label_files.csv"
        rows = ["file_name,study_id"]
        rows += [f"{fn},{sid}" for fn, sid in missing_label_files]
        miss_file_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"Wrote: {miss_file_path}  (n_files={len(missing_label_files)})")
    print("Private tags (canonical: separate US per label):")
    print(f"  ({PRIVATE_GROUP:04X},{PRIVATE_CREATOR_ELEM:04X}) PrivateCreator = {PRIVATE_CREATOR_VALUE!r}")
    for name, t in [("D1", TAG_D1), ("D2", TAG_D2), ("D3", TAG_D3), ("D4", TAG_D4), ("D5", TAG_D5), ("D6", TAG_D6)]:
        print(f"  {t} {name} = 0/1 (US)")
    if not args.no_derived_tags:
        print(f"  {TAG_SCORE} score=sum(D1..D4)")
        print(f"  {TAG_FINAL} 'NTM' or 'TB' (D6=1 overrides)")
    else:
        print("  (not written) SCORE/FINAL -- omitted; use default or drop --no-derived-tags to add")
    if args.write_packed_vector_lo:
        print(f"  (legacy) {TAG_VEC_STR} packed LO D1~D6 backslash string")
    else:
        print(f"  (not written) {TAG_VEC_STR} packed LO -- pass --write-packed-vector-lo to add")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

