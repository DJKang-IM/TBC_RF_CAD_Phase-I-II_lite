"""Emit three tiny Cursor canvases (overview, val, test) — no large inline JSON."""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "artifacts" / "metrics_phase3_vector_balanced_version_1.json"
RF = ROOT / "rf_phase3_vector_balanced_version_1.joblib"
NPZ = {
    "val": ROOT / "phase3_vector_balanced_version_1_features_val.npz",
    "test": ROOT / "phase3_vector_balanced_version_1_features_test.npz",
}
OUT_DIR = Path(r"C:\Users\SEJONG_ENDO_3\.cursor\projects\d\canvases")
SAVE_DIR = ROOT / "saved_canvases" / "vector_balanced_version_1"
OLD_HUGE = OUT_DIR / "vector-balanced-version-1.canvas.tsx"


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def table_rows_from_metrics(per_label: list, include_cm: bool) -> list[list[str]]:
    rows: list[list[str]] = []
    for p in per_label:
        key = p["key"]
        name = p["name"]
        n0 = str(p["support"]["n0"])
        n1 = str(p["support"]["n1"])
        eer = p["eer"]
        row = [
            key,
            name,
            n0,
            n1,
            fmt4(float(eer["eer"])),
            fmt4(float(eer["threshold"])),
            fmt4(float(eer["fpr"])),
            fmt4(float(eer["fnr"])),
        ]
        if include_cm:
            m = p["confusion_matrix"]["matrix"]
            tn, fp = str(m[0][0]), str(m[0][1])
            fn, tp = str(m[1][0]), str(m[1][1])
            row += [tn, fp, fn, tp]
        rows.append(row)
    return rows


def emit_overview(auroc_val: list[float], auroc_test: list[float]) -> None:
    keys = ["D1", "D2", "D3", "D4", "D6"]
    names = ["AFB", "TB PCR", "Solid culture", "Liquid culture", "NTM"]
    rows = []
    for i, k in enumerate(keys):
        rows.append([k, names[i], fmt4(auroc_val[i]), fmt4(auroc_test[i])])
    rows_ts = ",\n    ".join('["' + '", "'.join(esc(c) for c in r) + '"]' for r in rows)
    body = f"""import {{ Divider, H1, H2, Stack, Table, Text }} from "cursor/canvas";

export default function VectorBalancedV1Overview() {{
  return (
    <Stack gap={{16}}>
      <H1>Vector Balanced Version 1</H1>
      <Text tone="secondary" size="small">
        Phase III multi-output RF. Confusion matrices and EER in metrics JSON use decision threshold 0.5 for CM; EER is from ROC geometry.
      </Text>
      <H2>AUROC by label</H2>
      <Table
        headers={{["Label", "Name", "Val AUROC", "Test AUROC"]}}
        rows={{[
    {rows_ts}
        ]}}
      />
      <Divider />
      <H2>Related canvases</H2>
      <Text size="small">Same folder: vector-balanced-v1-val.canvas.tsx, vector-balanced-v1-test.canvas.tsx. Baseline (pre-balance): phase3-version-101-overview.canvas.tsx (Phase III Version 1.01).</Text>
      <Text tone="secondary" size="small">
        Disk: <REDACTED_PATH> Phase III\\\\ — rf_phase3_vector_balanced_version_1.joblib, phase3_vector_balanced_version_1_features_*.npz, artifacts\\\\metrics_phase3_vector_balanced_version_1.json, artifacts\\\\vector_balanced_version_1\\\\plots\\\\*.png
      </Text>
    </Stack>
  );
}}
"""
    (OUT_DIR / "vector-balanced-v1-overview.canvas.tsx").write_text(body, encoding="utf-8")


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
    rows = table_rows_from_metrics(per_label, include_cm=True)
    rows_ts = ",\n    ".join('["' + '", "'.join(esc(c) for c in r) + '"]' for r in rows)
    headers_ts = '["' + '", "'.join(esc(h) for h in headers) + '"]'
    fname = f"vector-balanced-v1-{name}.canvas.tsx"
    body = f"""import {{ H1, H2, Stack, Table, Text }} from "cursor/canvas";

export default function VectorBalancedV1{title.replace(" ", "")}Detail() {{
  return (
    <Stack gap={{12}}>
      <H1>Vector Balanced Version 1 — {esc(title)}</H1>
      <Text tone="secondary" size="small">n={{{n}}}. CM @ threshold 0.5 (TN, FP / FN, TP). EER row from metrics_multioutput.</Text>
      <H2>Per-label table</H2>
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
    (OUT_DIR / fname).write_text(body, encoding="utf-8")


def main() -> None:
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

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for fn in [
        "vector-balanced-v1-overview.canvas.tsx",
        "vector-balanced-v1-val.canvas.tsx",
        "vector-balanced-v1-test.canvas.tsx",
    ]:
        p = OUT_DIR / fn
        if p.exists():
            shutil.copy2(p, SAVE_DIR / fn)

    if OLD_HUGE.exists():
        OLD_HUGE.unlink()
        print("removed:", OLD_HUGE)
    print("wrote:", OUT_DIR / "vector-balanced-v1-overview.canvas.tsx")
    print("wrote:", OUT_DIR / "vector-balanced-v1-val.canvas.tsx")
    print("wrote:", OUT_DIR / "vector-balanced-v1-test.canvas.tsx")
    print("archived:", SAVE_DIR)


if __name__ == "__main__":
    main()
