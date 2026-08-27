from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def cm(y: np.ndarray, s: np.ndarray, thr: float = 0.5) -> np.ndarray:
    yhat = (s >= float(thr)).astype(int)
    return confusion_matrix(y.astype(int), yhat, labels=[0, 1]).astype(float)


def mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def stack_cm_stats(cms: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.stack(cms, axis=0)
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(m)
    return m, s


def build_rf(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=int(seed),
        class_weight="balanced",
    )


def build_xgb(seed: int, y_train: np.ndarray) -> XGBClassifier:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg / max(pos, 1))
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        random_state=int(seed),
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )


def const_proba(n: int, cls: int) -> np.ndarray:
    return np.full((n,), 1.0 if int(cls) == 1 else 0.0, dtype=float)


def safe_predict_positive_proba(clf, x: np.ndarray) -> np.ndarray:
    classes = np.asarray(getattr(clf, "classes_", []))
    if classes.size == 0:
        return np.zeros((x.shape[0],), dtype=float)
    if classes.size == 1:
        return const_proba(x.shape[0], int(classes[0]))
    pos = np.where(classes == 1)[0]
    if pos.size == 0:
        return np.zeros((x.shape[0],), dtype=float)
    return np.asarray(clf.predict_proba(x)[:, int(pos[0])], dtype=float)


def make_splits(
    strat_labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(strat_labels)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Invalid split counts: {n_train}/{n_val}/{n_test}, n={n}")

    all_idx = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(all_idx, strat_labels))

    val_ratio2 = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio2, random_state=int(seed) + 17)
    tr_sub, va_sub = next(s2.split(trva_idx, strat_labels[trva_idx]))
    tr_idx = trva_idx[tr_sub]
    va_idx = trva_idx[va_sub]
    return tr_idx, va_idx, te_idx


def make_phase3_strata(Y6: np.ndarray) -> np.ndarray:
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in Y6], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    if counts.min() >= 2:
        return bits
    return Y6[:, 5].astype(int)


def norm_sid(v) -> str:
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def load_embed_module(path: Path):
    spec = importlib.util.spec_from_file_location("embed_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def build_missing_sid_sets(meta_csv: Path) -> dict[str, set[str]]:
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    sid_col = "Study No." if "Study No." in df.columns else df.columns[0]
    col_map = {
        "D1": "도말검사",
        "D2": "TB-PCR검사",
        "D3": "배양검사(고체)",
        "D4": "배양검사(액체)",
    }
    out: dict[str, set[str]] = {}
    for lab, c in col_map.items():
        miss = df[c].fillna("").astype(str).str.contains("미검", na=False)
        out[lab] = set(norm_sid(v) for v in df.loc[miss, sid_col].tolist())
    return out


def path_site(path_str: str) -> str:
    parts = Path(path_str).parts
    if "NE" in parts:
        return "ne"
    return "kn"


def build_valid_masks(paths: np.ndarray, embed_mod, missing_sid: dict[str, set[str]]) -> dict[str, np.ndarray]:
    masks = {k: np.ones((len(paths),), dtype=bool) for k in ["D1", "D2", "D3", "D4", "D5", "D6"]}
    for i, p in enumerate(paths.tolist()):
        pp = Path(str(p))
        site = path_site(str(p))
        sid, _ = (embed_mod._parse_ne_filename(pp.name) if site == "ne" else embed_mod._parse_study_id_from_filename(pp.name))
        if not sid:
            for k in ["D1", "D2", "D3", "D4"]:
                masks[k][i] = False
            continue
        candidates = embed_mod._study_key_candidates(sid)
        for k in ["D1", "D2", "D3", "D4"]:
            bad = any(c in missing_sid[k] for c in candidates)
            masks[k][i] = not bad
    return masks


def main() -> None:
    ap = argparse.ArgumentParser(description="v1.05: active-only Phase III CV with per-label 미검 exclusion")
    ap.add_argument("--phase3_all_npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_active_all_260428_d5kw_fix.npz"))
    ap.add_argument("--meta_csv", type=Path, default=Path(r"D:\260428_META_ALL_CSV.csv"))
    ap.add_argument("--embed_script", type=Path, default=Path(r"D:\TB Phase III\embed_tb_labels_into_dicom.py"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path(r"D:\TB Phase III\artifacts\phase3_active_cv_v105_missing_exclude"))
    args = ap.parse_args()

    d = np.load(args.phase3_all_npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)  # D1,D2,D3,D4,D6
    D5 = np.asarray(d["D5"], dtype=int).ravel()
    paths = np.asarray(d["paths"], dtype=object)
    Y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], D5, Y[:, 4]]).astype(int)

    close_count = 0
    if args.force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        close_count = int(close_mask.sum())
        Y6 = Y6.copy()
        Y6[close_mask, :] = 0

    embed_mod = load_embed_module(args.embed_script)
    missing_sid = build_missing_sid_sets(args.meta_csv)
    valid_masks = build_valid_masks(paths, embed_mod, missing_sid)

    label_names = ["D1", "D2", "D3", "D4", "D5", "D6"]
    per_run: list[dict] = []
    by_label_auc: dict[str, list[float]] = {k: [] for k in label_names}
    d6_eers: list[float] = []
    d6_cms: list[np.ndarray] = []
    macro_aurocs: list[float] = []

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        run_aucs: dict[str, float] = {}
        run_n_eval: dict[str, int] = {}

        for j, name in enumerate(label_names):
            mask = valid_masks[name]
            idx = np.where(mask)[0]
            xv = X[idx]
            yv = Y6[idx, j].astype(int)
            tr, va, te = make_splits(yv, args.train_ratio, args.val_ratio, rs + (j + 1) * 7)
            _ = va

            y_tr = yv[tr]
            y_te = yv[te]

            if name == "D6":
                uniq = np.unique(y_tr)
                if uniq.size < 2:
                    p = const_proba(len(te), int(uniq[0]))
                else:
                    xgb = build_xgb(seed=rs + 777, y_train=y_tr)
                    xgb.fit(xv[tr], y_tr)
                    p = safe_predict_positive_proba(xgb, xv[te])
                a = safe_auc(y_te, p)
                e = eer(y_te, p)
                c = cm(y_te, p)
                d6_eers.append(e)
                d6_cms.append(c)
            else:
                uniq = np.unique(y_tr)
                if uniq.size < 2:
                    p = const_proba(len(te), int(uniq[0]))
                else:
                    rf = build_rf(seed=rs + (j + 1) * 13, n_estimators=args.rf_estimators)
                    rf.fit(xv[tr], y_tr)
                    p = safe_predict_positive_proba(rf, xv[te])
                a = safe_auc(y_te, p)
                e = None
                c = None

            run_aucs[name] = a
            run_n_eval[name] = int(len(te))
            by_label_auc[name].append(a)

        macro_vals = np.array([run_aucs[k] for k in label_names], dtype=float)
        macro_vals = macro_vals[np.isfinite(macro_vals)]
        macro_aurocs.append(float(np.mean(macro_vals)) if macro_vals.size else float("nan"))
        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "test_auroc_by_label": run_aucs,
                "test_eval_n_by_label": run_n_eval,
                "test_d6_eer": d6_eers[-1],
                "test_d6_cm": d6_cms[-1].tolist(),
            }
        )

    d6_cm_mean, d6_cm_std = stack_cm_stats(d6_cms)

    out = {
        "version": "1.05",
        "policy": "Per-label 미검 exclusion for D1..D4; D5/D6 unchanged",
        "config": {
            "phase3_all_npz": str(args.phase3_all_npz),
            "meta_csv": str(args.meta_csv),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "force_close_zero": bool(args.force_close_zero),
        },
        "cohort": {
            "n_total": int(len(X)),
            "close_count": int(close_count),
            "label_positive_counts": {k: int(Y6[:, i].sum()) for i, k in enumerate(label_names)},
            "label_valid_counts": {k: int(valid_masks[k].sum()) for k in label_names},
            "label_dropped_counts": {k: int((~valid_masks[k]).sum()) for k in label_names},
        },
        "phase3": {
            "test_auroc_by_label": {k: mean_std(v) for k, v in by_label_auc.items()},
            "test_macro_auroc_d1_d6": mean_std(macro_aurocs),
            "test_d6_eer": mean_std(d6_eers),
            "test_d6_cm_mean": d6_cm_mean.tolist(),
            "test_d6_cm_std": d6_cm_std.tolist(),
            "per_run": per_run,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "phase3_active_cv_v105_missing_exclude_test_metrics.json"
    txt_path = args.out_dir / "phase3_active_cv_v105_missing_exclude_test_summary.txt"
    csv_path = args.out_dir / "phase3_active_cv_v105_missing_exclude_per_run.csv"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p3 = out["phase3"]
    lines: list[str] = []
    lines.append("v1.05 Phase III active-only CV — TEST (per-label 미검 exclusion for D1..D4)")
    lines.append(f"N total: {len(X)}")
    lines.append(f"force_close_zero: {bool(args.force_close_zero)}")
    lines.append("Label valid/drop counts:")
    for k in label_names:
        lines.append(f"  {k}: valid={out['cohort']['label_valid_counts'][k]} drop={out['cohort']['label_dropped_counts'][k]}")
    lines.append(f"Macro AUROC (D1..D6): {p3['test_macro_auroc_d1_d6']['mean']:.4f} ± {p3['test_macro_auroc_d1_d6']['std']:.4f}")
    for lab in label_names:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    lines.append(f"D6 EER (XGB): {p3['test_d6_eer']['mean']:.4f} ± {p3['test_d6_eer']['std']:.4f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "D1", "D2", "D3", "D4", "D5", "D6", "d6_eer"])
        for r in p3["per_run"]:
            w.writerow(
                [
                    r["run"],
                    r["seed"],
                    r["test_auroc_by_label"]["D1"],
                    r["test_auroc_by_label"]["D2"],
                    r["test_auroc_by_label"]["D3"],
                    r["test_auroc_by_label"]["D4"],
                    r["test_auroc_by_label"]["D5"],
                    r["test_auroc_by_label"]["D6"],
                    r["test_d6_eer"],
                ]
            )

    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
