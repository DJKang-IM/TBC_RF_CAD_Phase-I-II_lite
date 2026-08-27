from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit


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


@dataclass
class RunOut:
    run: int
    seed: int
    test_auroc: float
    test_eer: float
    test_cm: np.ndarray


def main() -> None:
    ap = argparse.ArgumentParser(description="v1.04 D1-only CV on active-only Phase III features")
    ap.add_argument(
        "--npz",
        type=Path,
        default=Path(r"D:\TB Phase III\phase3_features_active_all_260428.npz"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"D:\TB Phase III\artifacts\d1_active_cv_v104"),
    )
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260428)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument(
        "--force_close_zero",
        action="store_true",
        help="Force D1=0 when filename stem contains 'close' (v1.04 policy check).",
    )
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)
    paths = np.asarray(d["paths"], dtype=object)
    if Y.ndim != 2 or Y.shape[1] < 1:
        raise ValueError(f"Unexpected Y shape: {Y.shape}")
    if paths.shape[0] != Y.shape[0]:
        raise ValueError("paths length mismatch")

    y = Y[:, 0].astype(int)  # D1
    close_count = 0
    if args.force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        close_count = int(close_mask.sum())
        y = y.copy()
        y[close_mask] = 0

    n = int(X.shape[0])
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Invalid split counts: {n_train}/{n_val}/{n_test} for n={n}")

    runs: list[RunOut] = []
    for r in range(args.cv_runs):
        rs = int(args.seed + r * 1009)
        tr, va, te = make_splits(y, n_train, n_val, n_test, rs)
        _ = va
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=rs,
            class_weight="balanced",
        )
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1].astype(float)
        runs.append(
            RunOut(
                run=r + 1,
                seed=rs,
                test_auroc=safe_auc(y[te], p),
                test_eer=eer(y[te], p),
                test_cm=cm(y[te], p),
            )
        )

    aucs = [x.test_auroc for x in runs]
    eers = [x.test_eer for x in runs]
    cm_mean, cm_std = stack_cm_stats([x.test_cm for x in runs])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "d1_active_cv_v104_test_metrics.json"
    out_txt = args.out_dir / "d1_active_cv_v104_test_summary.txt"
    out_csv = args.out_dir / "d1_active_cv_v104_per_run.csv"

    payload = {
        "version": "1.04",
        "config": {
            "npz": str(args.npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "n_estimators": int(args.n_estimators),
            "split_counts_train_val_test": [int(n_train), int(n_val), int(n_test)],
            "force_close_zero": bool(args.force_close_zero),
        },
        "cohort": {
            "n_total": int(n),
            "close_count": int(close_count),
            "d1_counts": {
                "0": int((y == 0).sum()),
                "1": int((y == 1).sum()),
            },
        },
        "test_auroc": mean_std(aucs),
        "test_eer": mean_std(eers),
        "test_cm_mean": cm_mean.tolist(),
        "test_cm_std": cm_std.tolist(),
        "per_run": [
            {
                "run": x.run,
                "seed": x.seed,
                "test_auroc": x.test_auroc,
                "test_eer": x.test_eer,
                "test_cm": x.test_cm.tolist(),
            }
            for x in runs
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("v1.04 D1-only CV (variable split per run, fixed active sample list) — TEST summary")
    lines.append(f"N total: {n}")
    lines.append(f"force_close_zero: {bool(args.force_close_zero)}")
    if args.force_close_zero:
        lines.append(f"close_count: {close_count}")
    lines.append(f"D1 counts: 0={(y == 0).sum()}, 1={(y == 1).sum()}")
    lines.append(f"Split counts (train/val/test): {n_train}/{n_val}/{n_test}")
    lines.append(f"AUROC: {payload['test_auroc']['mean']:.4f} ± {payload['test_auroc']['std']:.4f}")
    lines.append(f"EER:   {payload['test_eer']['mean']:.4f} ± {payload['test_eer']['std']:.4f}")
    lines.append("CM mean [[TN,FP],[FN,TP]]:")
    lines.append(np.array2string(cm_mean, precision=2))
    lines.append("CM std [[TN,FP],[FN,TP]]:")
    lines.append(np.array2string(cm_std, precision=2))
    lines.append("")
    lines.append("Per-run:")
    for x in runs:
        lines.append(
            f"run{x.run} seed={x.seed} AUROC={x.test_auroc:.4f} EER={x.test_eer:.4f} "
            f"CM={np.array2string(x.test_cm, precision=0)}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "test_auroc", "test_eer", "tn", "fp", "fn", "tp"])
        for x in runs:
            tn, fp = x.test_cm[0, 0], x.test_cm[0, 1]
            fn, tp = x.test_cm[1, 0], x.test_cm[1, 1]
            w.writerow([x.run, x.seed, x.test_auroc, x.test_eer, tn, fp, fn, tp])

    print(f"saved: {out_json}")
    print(f"saved: {out_txt}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
