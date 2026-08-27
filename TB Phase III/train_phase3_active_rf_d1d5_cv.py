"""
Phase III — RandomForest per label on frozen DenseNet+CLAHE features, D1–D5 only (D6 not trained / not reported).

- 2.7: class_weight="balanced" (same as train_phase3_active_cv_v103 RF heads).
- 2.8: per-label class_weight = {0: 1.0, 1: n_neg/n_pos} on the train fold
      (same weighting idea as BCE pos_weight; no MLP, no XGB).

Stratification still uses full D1–D6 bit pattern (Y6) for comparable splits to v103.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


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


def build_rf_balanced(seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=int(seed),
        class_weight="balanced",
    )


def build_rf_bce_pos_weight(seed: int, n_estimators: int, y_train: np.ndarray) -> RandomForestClassifier:
    y = y_train.astype(int).ravel()
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    w1 = float(neg / max(pos, 1))
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=int(seed),
        class_weight={0: 1.0, 1: w1},
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
    Y = np.asarray(d["Y"], dtype=int)
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
    ap = argparse.ArgumentParser(description="RF CV D1–D5 only (DenseNet+CLAHE features); 2.7 balanced vs 2.8 BCE-style weights")
    ap.add_argument("--phase3_all_npz", type=Path, required=True)
    ap.add_argument(
        "--rf_class_weight",
        type=str,
        choices=("balanced", "bce_pos_weight"),
        required=True,
        help='balanced = sklearn default (2.7); bce_pos_weight = {0:1, 1:neg/pos} per label (2.8)',
    )
    ap.add_argument("--pipeline_version", type=str, default="", help='Tag in JSON only, e.g. "2.7" or "2.8"')
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    X, Y6, _, close_count = load_phase3_y6(args.phase3_all_npz, force_close_zero=bool(args.force_close_zero))
    n = len(X)
    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    n_test = int(n - n_train - n_val)
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Invalid split counts")

    label_names = ["D1", "D2", "D3", "D4", "D5"]
    label_idx = [0, 1, 2, 3, 4]
    strata = make_phase3_strata(Y6)

    by_label_auc: dict[str, list[float]] = {k: [] for k in label_names}
    macro_aurocs: list[float] = []
    per_run: list[dict] = []

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        tr, va, te = make_splits(strata, n_train, n_val, n_test, rs)
        _ = va

        run_aucs: dict[str, float] = {}
        run_weights: dict[str, float | str] = {}

        for k, j in zip(label_names, label_idx):
            y_tr = Y6[tr, j].astype(int)
            y_te = Y6[te, j].astype(int)
            uniq = np.unique(y_tr)
            if uniq.size < 2:
                p = const_proba(len(te), int(uniq[0]))
            else:
                if args.rf_class_weight == "balanced":
                    rf = build_rf_balanced(seed=rs + (j + 1) * 13, n_estimators=args.rf_estimators)
                    run_weights[k] = "balanced"
                else:
                    rf = build_rf_bce_pos_weight(seed=rs + (j + 1) * 13, n_estimators=args.rf_estimators, y_train=y_tr)
                    neg = int((y_tr == 0).sum())
                    pos = int((y_tr == 1).sum())
                    run_weights[k] = float(neg / max(pos, 1))
                rf.fit(X[tr], y_tr)
                p = safe_predict_positive_proba(rf, X[te])
            a = safe_auc(y_te, p)
            run_aucs[k] = a
            by_label_auc[k].append(a)

        macro_vals = np.array([run_aucs[k] for k in label_names], dtype=float)
        macro_vals = macro_vals[np.isfinite(macro_vals)]
        macro_aurocs.append(float(np.mean(macro_vals)) if macro_vals.size else float("nan"))
        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "rf_class_weight_mode": args.rf_class_weight,
                "per_label_train_weight_note": run_weights,
                "test_auroc_by_label": run_aucs,
            }
        )

    pv = (args.pipeline_version or "").strip()
    weight_desc = (
        'sklearn RandomForest class_weight="balanced" per label'
        if args.rf_class_weight == "balanced"
        else "sklearn RandomForest class_weight={0:1.0, 1:n_neg/n_pos} per label (train fold), analogous to BCE pos_weight"
    )

    out = {
        "version": "1.03_rf_d1d5",
        "pipeline_train": f"{pv}_rf_clahe_densenet121_d1_d5" if pv else "rf_clahe_densenet121_d1_d5",
        "rf_class_weight": args.rf_class_weight,
        "weight_description": weight_desc,
        "d6_policy": "No D6 model; D6 excluded from AUROC and macro. Stratification still uses Y6 pattern.",
        "config": {
            "phase3_all_npz": str(args.phase3_all_npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "split_counts_train_val_test": [int(n_train), int(n_val), int(n_test)],
            "force_close_zero": bool(args.force_close_zero),
            "pipeline_version_tag": pv or None,
        },
        "cohort": {
            "n_total": int(n),
            "close_count": int(close_count),
            "label_positive_counts_d1_d6_full_cohort": {
                "D1": int(Y6[:, 0].sum()),
                "D2": int(Y6[:, 1].sum()),
                "D3": int(Y6[:, 2].sum()),
                "D4": int(Y6[:, 3].sum()),
                "D5": int(Y6[:, 4].sum()),
                "D6": int(Y6[:, 5].sum()),
            },
            "trained_heads": label_names,
        },
        "phase3": {
            "test_auroc_by_label": {k: mean_std(v) for k, v in by_label_auc.items()},
            "test_macro_auroc_d1_d5": mean_std(macro_aurocs),
            "per_run": per_run,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "phase3_active_rf_d1d5_test_metrics.json"
    txt_path = args.out_dir / "phase3_active_rf_d1d5_test_summary.txt"
    csv_path = args.out_dir / "phase3_active_rf_d1d5_per_run.csv"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    p3 = out["phase3"]
    lines: list[str] = []
    lines.append(
        f"Phase III RF CV — D1..D5 only — {args.rf_class_weight} — TEST"
    )
    if pv:
        lines.append(f"pipeline_version tag: {pv}")
    lines.append(out["weight_description"])
    lines.append(f"N total: {n}")
    lines.append(f"force_close_zero: {bool(args.force_close_zero)}")
    if args.force_close_zero:
        lines.append(f"close_count: {close_count}")
    lines.append(f"Split counts (train/val/test): {n_train}/{n_val}/{n_test}")
    mm = p3["test_macro_auroc_d1_d5"]
    lines.append(f"Macro AUROC (D1..D5): {mm['mean']:.4f} ± {mm['std']:.4f}")
    for lab in label_names:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "D1", "D2", "D3", "D4", "D5", "macro_d1_d5"])
        for r in p3["per_run"]:
            aucs = r["test_auroc_by_label"]
            mv = np.array([aucs[k] for k in label_names], dtype=float)
            mv = mv[np.isfinite(mv)]
            macro_r = float(np.mean(mv)) if mv.size else float("nan")
            w.writerow([r["run"], r["seed"], aucs["D1"], aucs["D2"], aucs["D3"], aucs["D4"], aucs["D5"], macro_r])

    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
