"""Find files where embed_tb_labels_into_dicom would increment missing_label (NE site)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pydicom


def load_mod():
    p = Path(__file__).resolve().parent / "embed_tb_labels_into_dicom.py"
    spec = importlib.util.spec_from_file_location("emb", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = load_mod()
    csv_path = Path(
        r"C:\Users\SEJONG_ENDO_3\Downloads\[RAW][나은] 결핵 DICOM\[RAW][나은] 결핵 DICOM\251221_NE_Meta(csv_new).csv"
    )
    dicom_in = Path(
        r"C:\Users\SEJONG_ENDO_3\Downloads\[RAW][나은] 결핵 DICOM\[RAW][나은] 결핵 DICOM"
    )
    df = mod._read_csv_with_fallbacks(csv_path)
    study_col, label_cols = mod._pick_columns(df, None, "1,2,4,5")
    mapping: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        sid = mod._csv_study_key(row[study_col])
        if not sid:
            continue
        flags = [mod._cell_flags(row[c]) for c in label_cols]
        d1, ntm1 = flags[0]
        d2, ntm2 = flags[1]
        d3, ntm3 = flags[2]
        d4, ntm4 = flags[3]
        d6 = 1 if (ntm1 or ntm2 or ntm3 or ntm4) else 0
        mapping[sid] = [int(d1), int(d2), int(d3), int(d4), int(d6)]

    missing: list[str] = []
    for p in sorted(dicom_in.rglob("*"), key=lambda x: str(x)):
        if not p.is_file():
            continue
        study_id, is_close = mod._parse_ne_filename(p.name)
        if not study_id:
            continue
        try:
            pydicom.dcmread(str(p), force=True)
        except Exception:
            continue
        if is_close:
            continue
        if study_id not in mapping:
            missing.append(f"{p}\tstudy_id={study_id}")

    print(f"missing_label_candidates={len(missing)}")
    for line in missing:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
