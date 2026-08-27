"""Emit lightweight Cursor canvases for Phase III Version 1.01 (pre–Vector Balanced).

Active vs inactive cohort: rf_phase3_active_vs_inactive.joblib + phase3_features_*.npz
Copies .canvas.tsx into <REDACTED_PATH> Phase III\\saved_canvases\\version_1_01\\ for archival.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "artifacts" / "metrics_phase3_version_1_01.json"
RF = ROOT / "rf_phase3_active_vs_inactive.joblib"
NPZ = {
    "val": ROOT / "phase3_features_val.npz",
    "test": ROOT / "phase3_features_test.npz",
}
OUT_DIR = Path(r"C:\Users\SEJONG_ENDO_3\.cursor\projects\d\canvases")
SAVE_DIR = ROOT / "saved_canvases" / "version_1_01"

VERSION_TITLE = "Phase III Version 1.01"
VERSION_NOTE = "Active vs inactive cohort (before Vector Balanced Version 1)."


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def annotate_metrics_json() -> None:
    if not METRICS.exists():
        return
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    data["dataset_version"] = "1.01"
    data["dataset_title"] = VERSION_TITLE
    data["dataset_note"] = VERSION_NOTE
    METRICS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def table_rows_from_metrics(per_label: list) -> list[list[str]]:
    rows: list[list[str]] = []
    for p in per_label:
        eer = p["eer"]
        m = p["confusion_matrix"]["matrix"]
        tn, fp = str(m[0][0]), str(m[0][1])
        fn, tp = str(m[1][0]), str(m[1][1])
        rows.append(
            [
                p["key"],
                p["name"],
                str(p["support"]["n0"]),
                str(p["support"]["n1"]),
                fmt4(float(eer["eer"])),
                fmt4(float(eer["threshold"])),
                fmt4(float(eer["fpr"])),
                fmt4(float(eer["fnr"])),
                tn,
                fp,
                fn,
                tp,
            ]
        )
    return rows


def emit_overview(auroc_val: list[float], auroc_test: list[float]) -> None:
    keys = ["D1", "D2", "D3", "D4", "D6"]
    names = ["AFB", "TB PCR", "Solid culture", "Liquid culture", "NTM"]
    rows = [[keys[i], names[i], fmt4(auroc_val[i]), fmt4(auroc_test[i])] for i in range(5)]
    rows_ts = ",\n    ".join('["' + '", "'.join(esc(c) for c in r) + '"]' for r in rows)
    body = f"""import {{ Divider, H1, H2, Stack, Table, Text }} from "cursor/canvas";

export default function Phase3Version101Overview() {{
  return (
    <Stack gap={{16}}>
      <H1>{esc(VERSION_TITLE)}</H1>
      <Text tone="secondary" size="small">
        {esc(VERSION_NOTE)} Multi-output RF. CM @ threshold 0.5; EER from ROC (metrics_multioutput). Compare with Vector Balanced Version 1 canvases (vector-balanced-v1-*.canvas.tsx).
      </Text>
      <H2>AUROC by label (Val / Test)</H2>
      <Table
        headers={{["Label", "Name", "Val AUROC", "Test AUROC"]}}
        rows={{[
    {rows_ts}
        ]}}
      />
      <Divider />
      <H2>Related canvases</H2>
      <Text size="small">Same folder: phase3-version-101-val.canvas.tsx, phase3-version-101-test.canvas.tsx</Text>
      <Text tone="secondary" size="small">
        Metrics JSON: <REDACTED_PATH> Phase III\\\\artifacts\\\\metrics_phase3_version_1_01.json — Model: rf_phase3_active_vs_inactive.joblib — Features: phase3_features_val.npz, phase3_features_test.npz
      </Text>
    </Stack>
  );
}}
"""
    (OUT_DIR / "phase3-version-101-overview.canvas.tsx").write_text(body, encoding="utf-8")


def emit_split(name: str, title: str, n: int, per_label: list) -> None:
    headers = [
        "Key",
        "Name",
        "n0",
        "n1",
        "EER",
        "thr@EER",
        "FPR@EER",
        "FNR@EER",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    rows = table_rows_from_metrics(per_label)
    rows_ts = ",\n    ".join('["' + '", "'.join(esc(str(c)) for c in r) + '"]' for r in rows)
    headers_ts = '["' + '", "'.join(esc(h) for h in headers) + '"]'
    comp = "Validation" if name == "val" else "Test"
    body = f"""import {{ H1, H2, Stack, Table, Text }} from "cursor/canvas";

export default function Phase3Version101{comp}Detail() {{
  return (
    <Stack gap={{12}}>
      <H1>{esc(VERSION_TITLE)} — {esc(title)}</H1>
      <Text tone="secondary" size="small">n={{{n}}}. CM @ 0.5 (TN, FP / FN, TP). EER from metrics JSON.</Text>
      <H2>Per-label</H2>
      <Table
        headers={{{headers_ts}}}
        rows={{[
    {rows_ts}
        ]}}
      />
    </Stack>
  );
}}
"""
    (OUT_DIR / f"phase3-version-101-{name}.canvas.tsx").write_text(body, encoding="utf-8")


def copy_archives() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for fn in [
        "phase3-version-101-overview.canvas.tsx",
        "phase3-version-101-val.canvas.tsx",
        "phase3-version-101-test.canvas.tsx",
    ]:
        src = OUT_DIR / fn
        if src.exists():
            shutil.copy2(src, SAVE_DIR / fn)


def main() -> None:
    annotate_metrics_json()
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    rf = joblib.load(RF)
    auroc_val: list[float] = []
    auroc_test: list[float] = []
    for key in ["val", "test"]:
        d = np.load(NPZ[key], allow_pickle=True)
        Y = d["Y"]
        probas = rf.predict_proba(d["X"])
        aucs = []
        for j in range(Y.shape[1]):
            y = Y[:, j].astype(int)
            s = probas[j][:, 1].astype(float)
            aucs.append(float(roc_auc_score(y, s)))
        if key == "val":
            auroc_val = aucs
        else:
            auroc_test = aucs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emit_overview(auroc_val, auroc_test)

    sets = data["sets"]
    val_set = next(s for s in sets if "_val." in s["name"] or s["name"].endswith("_val.npz"))
    test_set = next(s for s in sets if "_test." in s["name"] or s["name"].endswith("_test.npz"))
    emit_split("val", "Validation", int(val_set["n"]), val_set["per_label"])
    emit_split("test", "Test", int(test_set["n"]), test_set["per_label"])

    copy_archives()
    print("wrote:", OUT_DIR / "phase3-version-101-overview.canvas.tsx")
    print("archived:", SAVE_DIR)


if __name__ == "__main__":
    main()
