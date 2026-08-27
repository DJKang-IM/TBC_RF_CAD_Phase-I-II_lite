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


@dataclass
class BinRun:
    run: int
    seed: int
    auroc: float
    eer: float
    cm: np.ndarray


def run_binary_phase(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    cv_runs: int,
    seed: int,
    n_estimators: int,
) -> list[BinRun]:
    out: list[BinRun] = []
    for r in range(cv_runs):
        rs = int(seed + r * 1009)
        tr, va, te = make_splits(y, n_train, n_val, n_test, rs)
        rf = build_rf(seed=rs, n_estimators=n_estimators)
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1].astype(float)
        out.append(BinRun(run=r + 1, seed=rs, auroc=safe_auc(y[te], p), eer=eer(y[te], p), cm=cm(y[te], p)))
    return out


def summarize_binary(runs: list[BinRun]) -> dict:
    aucs = [x.auroc for x in runs]
    eers = [x.eer for x in runs]
    cm_mean, cm_std = stack_cm_stats([x.cm for x in runs])
    return {
        "test_auroc": mean_std(aucs),
        "test_eer": mean_std(eers),
        "test_cm_mean": cm_mean.tolist(),
        "test_cm_std": cm_std.tolist(),
        "per_run": [
            {"run": x.run, "seed": x.seed, "test_auroc": x.auroc, "test_eer": x.eer, "test_cm": x.cm.tolist()}
            for x in runs
        ],
    }


def load_phase12(tb_artifacts: Path, phase: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    tr = np.load(tb_artifacts / f"phase{phase}_features_train.npz", allow_pickle=True)
    va = np.load(tb_artifacts / f"phase{phase}_features_val.npz", allow_pickle=True)
    te = np.load(tb_artifacts / f"phase{phase}_features_test.npz", allow_pickle=True)
    X = np.concatenate([tr["X"], va["X"], te["X"]], axis=0).astype(np.float32)
    y = np.concatenate([tr["y"], va["y"], te["y"]], axis=0).astype(int).ravel()
    return X, y, (int(tr["X"].shape[0]), int(va["X"].shape[0]), int(te["X"].shape[0]))


def load_phase3_y6(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)  # D1,D2,D3,D4,D6
    D5 = np.asarray(d["D5"], dtype=int).ravel()
    if Y.shape[1] != 5:
        raise ValueError(f"Expected phase3 Y shape (N,5), got {Y.shape}")
    if D5.shape[0] != Y.shape[0]:
        raise ValueError("D5 length mismatch")
    Y6 = np.column_stack([Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], D5, Y[:, 4]]).astype(int)
    return X, Y6


def make_phase3_strata(Y6: np.ndarray) -> np.ndarray:
    """
    Prefer multi-label strata to preserve per-dimension distribution.
    Fallback to D6-only when a stratum is too sparse for stratified split.
    """
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in Y6], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    if counts.min() >= 2:
        return bits
    return Y6[:, 5].astype(int)


def run_phase3(
    X: np.ndarray,
    Y6: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    cv_runs: int,
    seed: int,
    n_estimators: int,
) -> dict:
    label_names = ["D1", "D2", "D3", "D4", "D5", "D6"]
    strata = make_phase3_strata(Y6)
    per_run: list[dict] = []

    by_label_auc: dict[str, list[float]] = {k: [] for k in label_names}
    d6_eers: list[float] = []
    d6_cms: list[np.ndarray] = []
    macro_aurocs: list[float] = []

    for r in range(cv_runs):
        rs = int(seed + r * 1009)
        tr, va, te = make_splits(strata, n_train, n_val, n_test, rs)

        run_aucs: dict[str, float] = {}
        # Phase III: per-dimension independent RF (parallel heads)
        for j, name in enumerate(label_names):
            y_tr = Y6[tr, j].astype(int)
            y_te = Y6[te, j].astype(int)
            rf = build_rf(seed=rs + (j + 1) * 13, n_estimators=n_estimators)
            rf.fit(X[tr], y_tr)
            p = rf.predict_proba(X[te])[:, 1].astype(float)
            a = safe_auc(y_te, p)
            run_aucs[name] = a
            by_label_auc[name].append(a)

        # D6 final score comes from XGB, not RF
        y6_tr = Y6[tr, 5].astype(int)
        y6_te = Y6[te, 5].astype(int)
        xgb = build_xgb(seed=rs + 777, y_train=y6_tr)
        xgb.fit(X[tr], y6_tr)
        p6 = xgb.predict_proba(X[te])[:, 1].astype(float)
        run_aucs["D6"] = safe_auc(y6_te, p6)
        by_label_auc["D6"][-1] = run_aucs["D6"]  # replace RF D6 with XGB D6

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

    cm_mean, cm_std = stack_cm_stats(d6_cms)
    return {
        "test_auroc_by_label": {k: mean_std(v) for k, v in by_label_auc.items()},
        "test_macro_auroc_d1_d6": mean_std(macro_aurocs),
        "test_d6_eer": mean_std(d6_eers),
        "test_d6_cm_mean": cm_mean.tolist(),
        "test_d6_cm_std": cm_std.tolist(),
        "per_run": per_run,
    }


def write_summary_text(out: dict, path: Path) -> None:
    p1 = out["phase1"]
    p2 = out["phase2"]
    p3 = out["phase3"]
    lines: list[str] = []
    lines.append("CV (variable split per run, fixed sample list) — TEST summary only")
    lines.append("")
    lines.append("=== Phase I (RF) TEST ===")
    lines.append(f"AUROC: {p1['test_auroc']['mean']:.4f} ± {p1['test_auroc']['std']:.4f}")
    lines.append(f"EER:   {p1['test_eer']['mean']:.4f} ± {p1['test_eer']['std']:.4f}")
    lines.append("")
    lines.append("=== Phase II (RF) TEST ===")
    lines.append(f"AUROC: {p2['test_auroc']['mean']:.4f} ± {p2['test_auroc']['std']:.4f}")
    lines.append(f"EER:   {p2['test_eer']['mean']:.4f} ± {p2['test_eer']['std']:.4f}")
    lines.append("")
    lines.append("=== Phase III (D1..D6 separate RF, D6 final=XGB) TEST ===")
    lines.append(f"Macro AUROC (D1..D6): {p3['test_macro_auroc_d1_d6']['mean']:.4f} ± {p3['test_macro_auroc_d1_d6']['std']:.4f}")
    for lab in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        d = p3["test_auroc_by_label"][lab]
        lines.append(f"{lab} AUROC: {d['mean']:.4f} ± {d['std']:.4f}")
    lines.append(f"D6 EER (XGB): {p3['test_d6_eer']['mean']:.4f} ± {p3['test_d6_eer']['std']:.4f}")
    lines.append("")
    lines.append("Split policy: sample list fixed, train/val/test re-split each run (stratified).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="CV with variable split per run; Phase III D6 uses XGB")
    ap.add_argument("--tb_artifacts", type=Path, default=Path(r"D:\TB Test DB\artifacts"))
    ap.add_argument("--phase3_all_npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_embed_d5_all.npz"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--phase3_train_ratio", type=float, default=0.70)
    ap.add_argument("--phase3_val_ratio", type=float, default=0.15)
    ap.add_argument("--out_dir", type=Path, default=Path(r"D:\TB Phase III\artifacts\pipeline_v103_cv_variable_split"))
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X1, y1, n1 = load_phase12(args.tb_artifacts, phase=1)
    X2, y2, n2 = load_phase12(args.tb_artifacts, phase=2)
    X3, Y6 = load_phase3_y6(args.phase3_all_npz)
    n3 = len(X3)
    n3_train = int(round(n3 * float(args.phase3_train_ratio)))
    n3_val = int(round(n3 * float(args.phase3_val_ratio)))
    n3_test = int(n3 - n3_train - n3_val)
    if min(n3_train, n3_val, n3_test) <= 0:
        raise ValueError("Invalid phase3 split counts")

    p1_runs = run_binary_phase(
        X1, y1, n_train=n1[0], n_val=n1[1], n_test=n1[2],
        cv_runs=args.cv_runs, seed=args.seed, n_estimators=args.rf_estimators,
    )
    p2_runs = run_binary_phase(
        X2, y2, n_train=n2[0], n_val=n2[1], n_test=n2[2],
        cv_runs=args.cv_runs, seed=args.seed + 111, n_estimators=args.rf_estimators,
    )
    p3 = run_phase3(
        X3, Y6, n_train=n3_train, n_val=n3_val, n_test=n3_test,
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
        "phase1": summarize_binary(p1_runs),
        "phase2": summarize_binary(p2_runs),
        "phase3": p3,
    }

    json_path = out_dir / "pipeline_v103_cv_variable_split_test_metrics.json"
    txt_path = out_dir / "pipeline_v103_cv_variable_split_test_summary.txt"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_text(out, txt_path)
    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
