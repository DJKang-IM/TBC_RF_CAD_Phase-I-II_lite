from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
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
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(strat_labels)
    if n_train + n_val + n_test != n:
        raise ValueError(f"Split count mismatch: {n_train}+{n_val}+{n_test}!={n}")

    all_idx = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(all_idx, strat_labels))

    val_ratio = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(seed) + 17)
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


def load_phase3_y6(npz_path: Path, force_close_zero: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)  # D1,D2,D3,D4,D6
    D5 = np.asarray(d["D5"], dtype=int).ravel()
    paths = np.asarray(d["paths"], dtype=object)
    if Y.shape[1] != 5:
        raise ValueError(f"Expected phase3 Y shape (N,5), got {Y.shape}")
    if D5.shape[0] != Y.shape[0]:
        raise ValueError("D5 length mismatch")
    if paths.shape[0] != Y.shape[0]:
        raise ValueError("paths length mismatch")
    Y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], D5, Y[:, 4]]).astype(int)

    close_count = 0
    if force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        close_count = int(close_mask.sum())
        Y6 = Y6.copy()
        Y6[close_mask, :] = 0

    return X, Y6, paths, close_count


def main() -> None:
    ap = argparse.ArgumentParser(description="v1.03 Phase III active-only D1~D6 CV")
    ap.add_argument("--phase3_all_npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_active_all_260428.npz"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path(r"D:\TB Phase III\artifacts\phase3_active_cv_v103"))
    args = ap.parse_args()

    X, Y6, _, close_count = load_phase3_y6(args.phase3_all_npz, force_close_zero=bool(args.force_close_zero))
    n = len(X)
    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid split counts")

    label_names = ["D1", "D2", "D3", "D4", "D5", "D6"]
    strata = make_phase3_strata(Y6)
    per_run: list[dict] = []
    by_label_auc: dict[str, list[float]] = {k: [] for k in label_names}
    d6_eers: list[float] = []
    d6_cms: list[np.ndarray] = []
    macro_aurocs: list[float] = []

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        tr, va, te = make_splits(strata, n_train, n_val, n_test, rs)
        _ = va

        run_aucs: dict[str, float] = {}
        for j, name in enumerate(label_names):
            y_tr = Y6[tr, j].astype(int)
            y_te = Y6[te, j].astype(int)
            uniq = np.unique(y_tr)
            if uniq.size < 2:
                p = const_proba(len(te), int(uniq[0]))
            else:
                rf = build_rf(seed=rs + (j + 1) * 13, n_estimators=args.rf_estimators)
                rf.fit(X[tr], y_tr)
                p = safe_predict_positive_proba(rf, X[te])
            a = safe_auc(y_te, p)
            run_aucs[name] = a
            by_label_auc[name].append(a)

        y6_tr = Y6[tr, 5].astype(int)
        y6_te = Y6[te, 5].astype(int)
        uniq6 = np.unique(y6_tr)
        if uniq6.size < 2:
            p6 = const_proba(len(te), int(uniq6[0]))
        else:
            xgb = build_xgb(seed=rs + 777, y_train=y6_tr)
            xgb.fit(X[tr], y6_tr)
            p6 = safe_predict_positive_proba(xgb, X[te])
        run_aucs["D6"] = safe_auc(y6_te, p6)
        by_label_auc["D6"][-1] = run_aucs["D6"]

        d6_e = eer(y6_te, p6)
        d6_cm = cm(y6_te, p6)
        d6_eers.append(d6_e)
        d6_cms.append(d6_cm)

        macro_vals = np.array([run_aucs[k] for k in label_names], dtype=float)
        macro_vals = macro_vals[np.isfinite(macro_vals)]
        macro_aurocs.append(float(np.mean(macro_vals)) if macro_vals.size else float("nan"))
        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "test_auroc_by_label": run_aucs,
                "test_d6_eer": d6_e,
                "test_d6_cm": d6_cm.tolist(),
            }
        )

    d6_cm_mean, d6_cm_std = stack_cm_stats(d6_cms)

    out = {
        "version": "1.03",
        "config": {
            "phase3_all_npz": str(args.phase3_all_npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "split_counts_train_val_test": [int(n_train), int(n_val), int(n_test)],
            "force_close_zero": bool(args.force_close_zero),
        },
        "cohort": {
            "n_total": int(n),
            "close_count": int(close_count),
            "label_positive_counts": {k: int(Y6[:, i].sum()) for i, k in enumerate(label_names)},
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
    json_path = args.out_dir / "phase3_active_cv_v103_test_metrics.json"
    txt_path = args.out_dir / "phase3_active_cv_v103_test_summary.txt"
    csv_path = args.out_dir / "phase3_active_cv_v103_per_run.csv"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p3 = out["phase3"]
    lines: list[str] = []
    lines.append("v1.03 Phase III active-only CV (variable split per run, fixed sample list) — TEST")
    lines.append(f"N total: {n}")
    lines.append(f"force_close_zero: {bool(args.force_close_zero)}")
    if args.force_close_zero:
        lines.append(f"close_count: {close_count}")
    lines.append(f"Split counts (train/val/test): {n_train}/{n_val}/{n_test}")
    lines.append(f"Macro AUROC (D1..D6): {p3['test_macro_auroc_d1_d6']['mean']:.4f} ± {p3['test_macro_auroc_d1_d6']['std']:.4f}")
    for lab in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    lines.append(f"D6 EER (XGB): {p3['test_d6_eer']['mean']:.4f} ± {p3['test_d6_eer']['std']:.4f}")
    lines.append("D6 CM mean [[TN,FP],[FN,TP]]:")
    lines.append(np.array2string(np.asarray(p3["test_d6_cm_mean"]), precision=2))
    lines.append("D6 CM std [[TN,FP],[FN,TP]]:")
    lines.append(np.array2string(np.asarray(p3["test_d6_cm_std"]), precision=2))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "D1", "D2", "D3", "D4", "D5", "D6", "d6_eer", "d6_tn", "d6_fp", "d6_fn", "d6_tp"])
        for r in p3["per_run"]:
            d6cm = np.asarray(r["test_d6_cm"], dtype=float)
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
                    d6cm[0, 0],
                    d6cm[0, 1],
                    d6cm[1, 0],
                    d6cm[1, 1],
                ]
            )

    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
