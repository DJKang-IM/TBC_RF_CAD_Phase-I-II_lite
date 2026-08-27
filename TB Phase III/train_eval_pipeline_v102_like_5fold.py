from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def _eer(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def _cm(y: np.ndarray, s: np.ndarray, thr: float = 0.5) -> np.ndarray:
    yhat = (s >= float(thr)).astype(int)
    return confusion_matrix(y.astype(int), yhat, labels=[0, 1]).astype(float)


def _mean_std(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def _split_indices(
    y_strat: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y_strat)
    if n_train + n_val + n_test != n:
        raise ValueError(f"split counts mismatch: {n_train}+{n_val}+{n_test} != {n}")

    idx_all = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(idx_all, y_strat))
    y_trva = y_strat[trva_idx]

    val_ratio = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(seed) + 17)
    tr_sub, va_sub = next(s2.split(trva_idx, y_trva))
    tr_idx = trva_idx[tr_sub]
    va_idx = trva_idx[va_sub]
    return tr_idx, va_idx, te_idx


def _build_xgb_d6(X_tr: np.ndarray, y6_tr: np.ndarray, seed: int) -> XGBClassifier:
    neg = int((y6_tr == 0).sum())
    pos = int((y6_tr == 1).sum())
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


@dataclass
class BinaryFoldResult:
    run: int
    seed: int
    test_auroc: float
    test_eer: float
    test_cm: np.ndarray


def run_binary_phase(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    cv_runs: int,
    seed: int,
    n_estimators: int,
) -> list[BinaryFoldResult]:
    out: list[BinaryFoldResult] = []
    for r in range(cv_runs):
        rs = int(seed + r * 1009)
        tr, va, te = _split_indices(y, n_train=n_train, n_val=n_val, n_test=n_test, seed=rs)
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=rs,
            class_weight="balanced",
        )
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1].astype(float)
        out.append(
            BinaryFoldResult(
                run=r + 1,
                seed=rs,
                test_auroc=_safe_auc(y[te], p),
                test_eer=_eer(y[te], p),
                test_cm=_cm(y[te], p, thr=0.5),
            )
        )
    return out


def summarize_binary(folds: list[BinaryFoldResult]) -> dict:
    aucs = [f.test_auroc for f in folds]
    eers = [f.test_eer for f in folds]
    cms = np.stack([f.test_cm for f in folds], axis=0)
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0, ddof=1) if len(folds) > 1 else np.zeros_like(cm_mean)
    return {
        "test_auroc": _mean_std(aucs),
        "test_eer": _mean_std(eers),
        "test_cm_mean": cm_mean.tolist(),
        "test_cm_std": cm_std.tolist(),
        "per_run": [
            {
                "run": f.run,
                "seed": f.seed,
                "test_auroc": f.test_auroc,
                "test_eer": f.test_eer,
                "test_cm": f.test_cm.tolist(),
            }
            for f in folds
        ],
    }


def run_phase3_d1_d6_rf_and_d6_xgb(
    X: np.ndarray,
    Y6: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    cv_runs: int,
    seed: int,
    n_estimators: int,
) -> dict:
    per_run: list[dict] = []
    label_names = ["D1", "D2", "D3", "D4", "D5", "D6"]
    # collect
    aurocs_by_label: dict[str, list[float]] = {k: [] for k in label_names}
    d6_eers: list[float] = []
    d6_cms: list[np.ndarray] = []
    macro_aurocs: list[float] = []

    for r in range(cv_runs):
        rs = int(seed + r * 1009)
        strat = Y6[:, 5].astype(int)  # D6
        tr, va, te = _split_indices(strat, n_train=n_train, n_val=n_val, n_test=n_test, seed=rs)
        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y6[tr], Y6[te]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=-1,
            random_state=rs,
            class_weight="balanced",
        )
        rf.fit(X_tr, Y_tr)
        rf_probs = rf.predict_proba(X_te)  # list length 6

        xgb = _build_xgb_d6(X_tr, Y_tr[:, 5].astype(int), seed=rs)
        xgb.fit(X_tr, Y_tr[:, 5].astype(int))
        p6 = xgb.predict_proba(X_te)[:, 1].astype(float)

        run_aurocs: dict[str, float] = {}
        for i, name in enumerate(label_names):
            if i == 5:
                p = p6
            else:
                p = np.asarray(rf_probs[i], dtype=float)[:, 1]
            a = _safe_auc(Y_te[:, i], p)
            run_aurocs[name] = a
            aurocs_by_label[name].append(a)

        m = np.array([run_aurocs[k] for k in label_names], dtype=float)
        m = m[np.isfinite(m)]
        macro_aurocs.append(float(np.mean(m)) if m.size else float("nan"))

        d6_e = _eer(Y_te[:, 5], p6)
        d6_cm = _cm(Y_te[:, 5], p6, thr=0.5)
        d6_eers.append(d6_e)
        d6_cms.append(d6_cm)
        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "test_auroc": run_aurocs,
                "test_d6_eer": d6_e,
                "test_d6_cm": d6_cm.tolist(),
            }
        )

    d6_cms_arr = np.stack(d6_cms, axis=0)
    d6_cm_mean = np.mean(d6_cms_arr, axis=0)
    d6_cm_std = np.std(d6_cms_arr, axis=0, ddof=1) if len(d6_cms) > 1 else np.zeros_like(d6_cm_mean)

    return {
        "test_auroc_by_label": {k: _mean_std(v) for k, v in aurocs_by_label.items()},
        "test_macro_auroc_d1_d6": _mean_std(macro_aurocs),
        "test_d6_eer": _mean_std(d6_eers),
        "test_d6_cm_mean": d6_cm_mean.tolist(),
        "test_d6_cm_std": d6_cm_std.tolist(),
        "per_run": per_run,
    }


def _load_binary_split_npz(train_p: Path, val_p: Path, test_p: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    tr = np.load(train_p, allow_pickle=True)
    va = np.load(val_p, allow_pickle=True)
    te = np.load(test_p, allow_pickle=True)
    X = np.concatenate([tr["X"], va["X"], te["X"]], axis=0).astype(np.float32)
    y = np.concatenate([tr["y"], va["y"], te["y"]], axis=0).astype(int).ravel()
    return X, y, (int(tr["X"].shape[0]), int(va["X"].shape[0]), int(te["X"].shape[0]))


def _load_phase3_all(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)
    D5 = np.asarray(d["D5"], dtype=int).ravel()
    if Y.ndim != 2 or Y.shape[1] != 5:
        raise ValueError(f"Phase III Y must be (N,5) [D1,D2,D3,D4,D6], got {Y.shape}")
    if D5.shape[0] != Y.shape[0]:
        raise ValueError("Phase III D5 length mismatch")
    Y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], D5, Y[:, 4]]).astype(int)
    return X, Y6


def build_text_report(out: dict, out_path: Path) -> None:
    p1 = out["phase1"]
    p2 = out["phase2"]
    p3 = out["phase3"]
    lines: list[str] = []
    lines.append("TBC_CAD-like pipeline (v1.02 style): 5 runs, TEST-only summary")
    lines.append("")
    lines.append("=== Phase I (RF) TEST ===")
    lines.append(f"AUROC: {p1['test_auroc']['mean']:.4f} ± {p1['test_auroc']['std']:.4f}")
    lines.append(f"EER:   {p1['test_eer']['mean']:.4f} ± {p1['test_eer']['std']:.4f}")
    lines.append("")
    lines.append("=== Phase II (RF) TEST ===")
    lines.append(f"AUROC: {p2['test_auroc']['mean']:.4f} ± {p2['test_auroc']['std']:.4f}")
    lines.append(f"EER:   {p2['test_eer']['mean']:.4f} ± {p2['test_eer']['std']:.4f}")
    lines.append("")
    lines.append("=== Phase III (RF for D1-D6, D6 score from XGB) TEST ===")
    lines.append(f"Macro AUROC (D1..D6): {p3['test_macro_auroc_d1_d6']['mean']:.4f} ± {p3['test_macro_auroc_d1_d6']['std']:.4f}")
    for lab in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    lines.append(f"D6 EER (XGB score): {p3['test_d6_eer']['mean']:.4f} ± {p3['test_d6_eer']['std']:.4f}")
    lines.append("")
    lines.append("Note: train/val/test are re-split per run with stratification (phase1/2 by y, phase3 by D6).")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train/eval pipeline with 5 repeated splits; report TEST mean±std")
    ap.add_argument("--tb_root", type=Path, default=Path(r"D:\TB Test DB\artifacts"))
    ap.add_argument("--phase3_all_npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_embed_d5_all.npz"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--phase3_train_ratio", type=float, default=0.70)
    ap.add_argument("--phase3_val_ratio", type=float, default=0.15)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"D:\TB Phase III\artifacts\pipeline_v102_like_5runs"),
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase I/II from existing artifacts (same as legacy split sizes)
    p1_tr = args.tb_root / "phase1_features_train.npz"
    p1_va = args.tb_root / "phase1_features_val.npz"
    p1_te = args.tb_root / "phase1_features_test.npz"
    p2_tr = args.tb_root / "phase2_features_train.npz"
    p2_va = args.tb_root / "phase2_features_val.npz"
    p2_te = args.tb_root / "phase2_features_test.npz"

    X1, y1, n1 = _load_binary_split_npz(p1_tr, p1_va, p1_te)
    X2, y2, n2 = _load_binary_split_npz(p2_tr, p2_va, p2_te)
    X3, Y6 = _load_phase3_all(args.phase3_all_npz)
    n3 = len(X3)
    n3_train = int(round(n3 * float(args.phase3_train_ratio)))
    n3_val = int(round(n3 * float(args.phase3_val_ratio)))
    n3_test = int(n3 - n3_train - n3_val)
    if min(n3_train, n3_val, n3_test) <= 0:
        raise ValueError("Phase III split counts invalid; adjust ratios")

    folds_p1 = run_binary_phase(
        X1, y1, n_train=n1[0], n_val=n1[1], n_test=n1[2],
        cv_runs=args.cv_runs, seed=args.seed, n_estimators=args.rf_estimators,
    )
    folds_p2 = run_binary_phase(
        X2, y2, n_train=n2[0], n_val=n2[1], n_test=n2[2],
        cv_runs=args.cv_runs, seed=args.seed + 111, n_estimators=args.rf_estimators,
    )
    sum_p3 = run_phase3_d1_d6_rf_and_d6_xgb(
        X3, Y6,
        n_train=n3_train, n_val=n3_val, n_test=n3_test,
        cv_runs=args.cv_runs, seed=args.seed + 222, n_estimators=args.rf_estimators,
    )

    out = {
        "config": {
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "phase1_counts_train_val_test": [int(n1[0]), int(n1[1]), int(n1[2])],
            "phase2_counts_train_val_test": [int(n2[0]), int(n2[1]), int(n2[2])],
            "phase3_counts_train_val_test": [int(n3_train), int(n3_val), int(n3_test)],
            "phase3_all_npz": str(args.phase3_all_npz),
        },
        "phase1": summarize_binary(folds_p1),
        "phase2": summarize_binary(folds_p2),
        "phase3": sum_p3,
    }

    json_path = out_dir / "pipeline_v102_like_5runs_test_metrics.json"
    txt_path = out_dir / "pipeline_v102_like_5runs_test_summary.txt"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    build_text_report(out, txt_path)
    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
