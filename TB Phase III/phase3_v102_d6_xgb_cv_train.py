"""
Phase III v1.02 (D6_XGBoost): keep v1.01 multi-output RF for D1–D5, replace ONLY D6 with XGBoost (v1.023-style).

Also runs 5 repeated random train/val/test splits (same pool sizes as the original npz splits)
to report mean/std for D6 AUROC + EER on val/test, and plots summary figures.

Outputs (default under <root>/artifacts/phase3_v1_02_d6_xgb/):
  - cv_fold_metrics.json  (NTM: RF vs XGB per fold + mean ± SD over runs)
  - ntm_rf_vs_xgb_cv_summary.txt  (plain-text mean ± SD + per-fold table)
  - cv_d6_auroc_eer.png  (RF vs XGB AUROC/EER across runs; boxes show mean ± SD)
  - phase3_rf_v1_02_d6_xgb_hybrid.joblib
  - version_1_02_d6_xgb_meta.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve
from xgboost import XGBClassifier


D6_INDEX = 4


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


@dataclass
class FoldNTMMetrics:
    """Per-fold NTM (non-NTM=0 vs NTM=1) scores: multi-output RF channel vs XGBoost head."""

    fold: int
    seed: int
    rf_auroc_val: float
    rf_auroc_test: float
    rf_eer_val: float
    rf_eer_test: float
    xgb_auroc_val: float
    xgb_auroc_test: float
    xgb_eer_val: float
    xgb_eer_test: float


class Phase3HybridV102D6XGB:
    """
    Multi-output RF (trained on original v1.01 training matrix Y) for D1–D5,
    plus a binary XGBoost head for D6.

    predict_proba matches sklearn multioutput RF convention: list length 5,
    each element shape (n, 2) for classes [0, 1].
    """

    kind = "phase3_v1_02_d6_xgb_hybrid"

    def __init__(self, rf: RandomForestClassifier, xgb_d6: XGBClassifier, d6_threshold: float = 0.5) -> None:
        self.rf = rf
        self.xgb_d6 = xgb_d6
        self.d6_threshold = float(d6_threshold)

    def predict_proba(self, X: np.ndarray):  # noqa: ANN001
        X = np.asarray(X, dtype=np.float32)
        base = self.rf.predict_proba(X)
        p6 = self.xgb_d6.predict_proba(X)
        out = list(base)
        out[D6_INDEX] = np.asarray(p6, dtype=float)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        yhat = np.asarray(self.rf.predict(X), dtype=int).copy()
        p1 = self.xgb_d6.predict_proba(X)[:, 1].astype(float)
        yhat[:, D6_INDEX] = (p1 >= self.d6_threshold).astype(int)
        return yhat


def build_xgb_d6(X_tr: np.ndarray, y6_tr: np.ndarray, seed: int) -> XGBClassifier:
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


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "n": int(a.size)}


def _ms_line(name: str, d: dict) -> str:
    return f"{name}: {d['mean']:.3f} ± {d['std']:.3f} (n={d['n']})"


def plot_cv_ntm_rf_vs_xgb(folds: list[FoldNTMMetrics], summaries: dict, out_png: Path) -> None:
    xs = np.arange(len(folds))
    labels = [f"run{f.fold}\n{f.seed}" for f in folds]

    rf_v = [f.rf_auroc_val for f in folds]
    rf_t = [f.rf_auroc_test for f in folds]
    xg_v = [f.xgb_auroc_val for f in folds]
    xg_t = [f.xgb_auroc_test for f in folds]
    e_rf_v = [f.rf_eer_val for f in folds]
    e_rf_t = [f.rf_eer_test for f in folds]
    e_xg_v = [f.xgb_eer_val for f in folds]
    e_xg_t = [f.xgb_eer_test for f in folds]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(xs, rf_v, "s-", color="#1f77b4", lw=2, ms=7, label="RF · val")
    ax.plot(xs, rf_t, "o-", color="#1f77b4", lw=2, ms=6, alpha=0.85, label="RF · test")
    ax.plot(xs, xg_v, "s-", color="#ff7f0e", lw=2, ms=7, label="XGB · val")
    ax.plot(xs, xg_t, "o-", color="#ff7f0e", lw=2, ms=6, alpha=0.85, label="XGB · test")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("NTM (non-NTM vs NTM) · repeated random splits — AUROC", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    au_txt = (
        _ms_line("RF val AUROC", summaries["random_forest"]["auroc_val"])
        + "\n"
        + _ms_line("RF test AUROC", summaries["random_forest"]["auroc_test"])
        + "\n"
        + _ms_line("XGB val AUROC", summaries["xgboost"]["auroc_val"])
        + "\n"
        + _ms_line("XGB test AUROC", summaries["xgboost"]["auroc_test"])
    )
    ax.text(
        0.02,
        0.98,
        au_txt,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.88),
        family="monospace",
    )

    ax = axes[1]
    ax.plot(xs, e_rf_v, "s-", color="#1f77b4", lw=2, ms=7, label="RF · val")
    ax.plot(xs, e_rf_t, "o-", color="#1f77b4", lw=2, ms=6, alpha=0.85, label="RF · test")
    ax.plot(xs, e_xg_v, "s-", color="#ff7f0e", lw=2, ms=7, label="XGB · val")
    ax.plot(xs, e_xg_t, "o-", color="#ff7f0e", lw=2, ms=6, alpha=0.85, label="XGB · test")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("EER", fontsize=12)
    ax.set_title("NTM · EER", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    ee_txt = (
        _ms_line("RF val EER", summaries["random_forest"]["eer_val"])
        + "\n"
        + _ms_line("RF test EER", summaries["random_forest"]["eer_test"])
        + "\n"
        + _ms_line("XGB val EER", summaries["xgboost"]["eer_val"])
        + "\n"
        + _ms_line("XGB test EER", summaries["xgboost"]["eer_test"])
        + "\n"
        + _ms_line("XGB Δ(test−val) EER", summaries["xgboost"]["eer_test_minus_val"])
    )
    ax.text(
        0.02,
        0.02,
        ee_txt,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.88),
        family="monospace",
    )

    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_ntm_cv_summary_text(summaries: dict, folds: list[FoldNTMMetrics], out_txt: Path) -> None:
    lines = [
        "Phase III — NTM binary (non-NTM=0, NTM=1), same repeated train/val/test splits.",
        "Random forest: multi-output RF refit per fold; scores = NTM channel (index 4) P(NTM|X).",
        "XGBoost: binary head refit per fold on the same training slice.",
        "",
        "=== Mean ± SD (internal validation) ===",
        _ms_line("RF  · validation AUROC", summaries["random_forest"]["auroc_val"]),
        _ms_line("RF  · test AUROC", summaries["random_forest"]["auroc_test"]),
        _ms_line("RF  · validation EER", summaries["random_forest"]["eer_val"]),
        _ms_line("RF  · test EER", summaries["random_forest"]["eer_test"]),
        _ms_line("RF  · paired Δ(test−val) EER", summaries["random_forest"]["eer_test_minus_val"]),
        "",
        _ms_line("XGB · validation AUROC", summaries["xgboost"]["auroc_val"]),
        _ms_line("XGB · test AUROC", summaries["xgboost"]["auroc_test"]),
        _ms_line("XGB · validation EER", summaries["xgboost"]["eer_val"]),
        _ms_line("XGB · test EER", summaries["xgboost"]["eer_test"]),
        _ms_line("XGB · paired Δ(test−val) EER", summaries["xgboost"]["eer_test_minus_val"]),
        "",
        "=== Per-fold values ===",
    ]
    for f in folds:
        lines.append(
            f"fold {f.fold} seed={f.seed} | "
            f"RF val/test AUROC {f.rf_auroc_val:.4f}/{f.rf_auroc_test:.4f} EER {f.rf_eer_val:.4f}/{f.rf_eer_test:.4f} | "
            f"XGB val/test AUROC {f.xgb_auroc_val:.4f}/{f.xgb_auroc_test:.4f} EER {f.xgb_eer_val:.4f}/{f.xgb_eer_test:.4f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--rf_v101", type=Path, default=None, help="Default: <root>/rf_phase3_active_vs_inactive.joblib")
    ap.add_argument("--train_npz", type=Path, default=None)
    ap.add_argument("--val_npz", type=Path, default=None)
    ap.add_argument("--test_npz", type=Path, default=None)
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    root: Path = args.root
    rf_path = args.rf_v101 or (root / "rf_phase3_active_vs_inactive.joblib")
    train_p = args.train_npz or (root / "phase3_features_train.npz")
    val_p = args.val_npz or (root / "phase3_features_val.npz")
    test_p = args.test_npz or (root / "phase3_features_test.npz")
    out_dir = args.out_dir or (root / "artifacts" / "phase3_v1_02_d6_xgb")
    out_dir.mkdir(parents=True, exist_ok=True)

    d_tr = np.load(train_p, allow_pickle=True)
    d_va = np.load(val_p, allow_pickle=True)
    d_te = np.load(test_p, allow_pickle=True)
    X_tr0, Y_tr0 = d_tr["X"].astype(np.float32), d_tr["Y"].astype(int)
    X_va0, Y_va0 = d_va["X"].astype(np.float32), d_va["Y"].astype(int)
    X_te0, Y_te0 = d_te["X"].astype(np.float32), d_te["Y"].astype(int)

    n_tr, n_va, n_te = int(X_tr0.shape[0]), int(X_va0.shape[0]), int(X_te0.shape[0])
    X_pool = np.concatenate([X_tr0, X_va0, X_te0], axis=0)
    Y_pool = np.concatenate([Y_tr0, Y_va0, Y_te0], axis=0)
    n_total = int(X_pool.shape[0])
    if n_total != n_tr + n_va + n_te:
        raise RuntimeError("concat size mismatch")

    rf_base: RandomForestClassifier = joblib.load(rf_path)

    folds: list[FoldNTMMetrics] = []
    rng_master = np.random.default_rng(int(args.seed))

    for fold in range(int(args.cv_runs)):
        run_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(run_seed)
        perm = rng.permutation(n_total)
        tr_idx = perm[:n_tr]
        va_idx = perm[n_tr : n_tr + n_va]
        te_idx = perm[n_tr + n_va :]

        X_tr = X_pool[tr_idx]
        Y_tr = Y_pool[tr_idx]
        X_va = X_pool[va_idx]
        Y_va = Y_pool[va_idx]
        X_te = X_pool[te_idx]
        Y_te = Y_pool[te_idx]

        y6_tr = Y_tr[:, D6_INDEX]
        y6_va = Y_va[:, D6_INDEX]
        y6_te = Y_te[:, D6_INDEX]

        rf_fold = clone(rf_base)
        rf_fold.set_params(random_state=run_seed)
        rf_fold.fit(X_tr, Y_tr)
        p_va = rf_fold.predict_proba(X_va)
        p_te = rf_fold.predict_proba(X_te)
        s_va_rf = p_va[D6_INDEX][:, 1].astype(float)
        s_te_rf = p_te[D6_INDEX][:, 1].astype(float)

        rf_auroc_val = float(roc_auc_score(y6_va, s_va_rf)) if y6_va.min() != y6_va.max() else float("nan")
        rf_auroc_test = float(roc_auc_score(y6_te, s_te_rf)) if y6_te.min() != y6_te.max() else float("nan")
        rf_eer_val = float(eer_from_scores(y6_va, s_va_rf))
        rf_eer_test = float(eer_from_scores(y6_te, s_te_rf))

        xgb = build_xgb_d6(X_tr, y6_tr, seed=run_seed)
        xgb.fit(X_tr, y6_tr)

        s_va = xgb.predict_proba(X_va)[:, 1].astype(float)
        s_te = xgb.predict_proba(X_te)[:, 1].astype(float)

        xgb_auroc_val = float(roc_auc_score(y6_va, s_va)) if y6_va.min() != y6_va.max() else float("nan")
        xgb_auroc_test = float(roc_auc_score(y6_te, s_te)) if y6_te.min() != y6_te.max() else float("nan")
        xgb_eer_val = float(eer_from_scores(y6_va, s_va))
        xgb_eer_test = float(eer_from_scores(y6_te, s_te))

        folds.append(
            FoldNTMMetrics(
                fold=fold,
                seed=run_seed,
                rf_auroc_val=rf_auroc_val,
                rf_auroc_test=rf_auroc_test,
                rf_eer_val=rf_eer_val,
                rf_eer_test=rf_eer_test,
                xgb_auroc_val=xgb_auroc_val,
                xgb_auroc_test=xgb_auroc_test,
                xgb_eer_val=xgb_eer_val,
                xgb_eer_test=xgb_eer_test,
            )
        )

    rf_au_v = [f.rf_auroc_val for f in folds]
    rf_au_t = [f.rf_auroc_test for f in folds]
    rf_ee_v = [f.rf_eer_val for f in folds]
    rf_ee_t = [f.rf_eer_test for f in folds]
    rf_gaps = [float(f.rf_eer_test - f.rf_eer_val) for f in folds]

    xg_au_v = [f.xgb_auroc_val for f in folds]
    xg_au_t = [f.xgb_auroc_test for f in folds]
    xg_ee_v = [f.xgb_eer_val for f in folds]
    xg_ee_t = [f.xgb_eer_test for f in folds]
    xg_gaps = [float(f.xgb_eer_test - f.xgb_eer_val) for f in folds]

    rf_summary = {
        "auroc_val": summarize(rf_au_v),
        "auroc_test": summarize(rf_au_t),
        "eer_val": summarize(rf_ee_v),
        "eer_test": summarize(rf_ee_t),
        "eer_test_minus_val": summarize(rf_gaps),
    }
    xgb_summary = {
        "auroc_val": summarize(xg_au_v),
        "auroc_test": summarize(xg_au_t),
        "eer_val": summarize(xg_ee_v),
        "eer_test": summarize(xg_ee_t),
        "eer_test_minus_val": summarize(xg_gaps),
    }

    fold_rows = [
        {
            "fold": f.fold,
            "seed": f.seed,
            "random_forest_ntm": {
                "auroc_val": f.rf_auroc_val,
                "auroc_test": f.rf_auroc_test,
                "eer_val": f.rf_eer_val,
                "eer_test": f.rf_eer_test,
            },
            "xgboost_ntm": {
                "auroc_val": f.xgb_auroc_val,
                "auroc_test": f.xgb_auroc_test,
                "eer_val": f.xgb_eer_val,
                "eer_test": f.xgb_eer_test,
            },
        }
        for f in folds
    ]

    summary = {
        "model_name": "Phase III v1.02 (NTM: RF vs XGBoost, same repeated splits)",
        "task": "NTM binary: non-NTM (0) vs NTM (1); scores compared under identical partitions.",
        "rf_template_path": str(rf_path),
        "note_rf": "Per fold: clone(template RF), refit on train slice, NTM score = predict_proba[..][D6_INDEX][:,1].",
        "note_xgb": "Per fold: XGBoost binary classifier on same train slice (v1.023-style hyperparameters).",
        "pool": {"train_npz": str(train_p), "val_npz": str(val_p), "test_npz": str(test_p), "n_total": n_total},
        "cv": {"runs": int(args.cv_runs), "scheme": "repeated_random_splits_same_sizes", "sizes": {"train": n_tr, "val": n_va, "test": n_te}},
        "random_forest_ntm": rf_summary,
        "xgboost_ntm": xgb_summary,
        "d6": {
            "description": "Backward-compatible alias: same as xgboost_ntm (historical key name).",
            "auroc_val": xgb_summary["auroc_val"],
            "auroc_test": xgb_summary["auroc_test"],
            "eer_val": xgb_summary["eer_val"],
            "eer_test": xgb_summary["eer_test"],
            "eer_test_minus_val": xgb_summary["eer_test_minus_val"],
        },
        "folds": fold_rows,
    }
    summaries_for_plot = {"random_forest": rf_summary, "xgboost": xgb_summary}
    (out_dir / "cv_fold_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_cv_ntm_rf_vs_xgb(folds, summaries_for_plot, out_dir / "cv_d6_auroc_eer.png")
    write_ntm_cv_summary_text(summaries_for_plot, folds, out_dir / "ntm_rf_vs_xgb_cv_summary.txt")

    # Final model: XGB trained on ORIGINAL train split; RF stays v1.01 artifact
    xgb_final = build_xgb_d6(X_tr0, Y_tr0[:, D6_INDEX], seed=int(args.seed))
    xgb_final.fit(X_tr0, Y_tr0[:, D6_INDEX])
    hybrid = Phase3HybridV102D6XGB(rf=rf_base, xgb_d6=xgb_final, d6_threshold=0.5)
    model_path = out_dir / "phase3_rf_v1_02_d6_xgb_hybrid.joblib"
    joblib.dump(hybrid, model_path)

    meta = {
        "version": "1.02",
        "display_name": "1.02 (D6_XGBoost)",
        "description": "v1.01 multi-output RF frozen for D1–D5; D6 replaced by XGBoost head trained on original v1.01 train split.",
        "rf_v101": str(rf_path),
        "xgb_head": "XGBClassifier (same hyperparameters as v1.023 eval script)",
        "hybrid_model": str(model_path),
        "cv_artifact": str(out_dir / "cv_fold_metrics.json"),
        "cv_summary_text": str(out_dir / "ntm_rf_vs_xgb_cv_summary.txt"),
        "plot": str(out_dir / "cv_d6_auroc_eer.png"),
    }
    (out_dir / "version_1_02_d6_xgb_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Quick sanity on original splits (D6 AUROC) using hybrid predict_proba
    def quick_auc(name: str, X: np.ndarray, y6: np.ndarray) -> float:
        p1 = hybrid.predict_proba(X)[D6_INDEX][:, 1].astype(float)
        if y6.min() == y6.max():
            return float("nan")
        return float(roc_auc_score(y6.astype(int), p1))

    print("saved:", model_path)
    print("saved:", out_dir / "cv_fold_metrics.json")
    print("saved:", out_dir / "ntm_rf_vs_xgb_cv_summary.txt")
    print("saved:", out_dir / "cv_d6_auroc_eer.png")
    print("saved:", out_dir / "version_1_02_d6_xgb_meta.json")
    print(
        "sanity D6 AUROC (original splits):",
        f"train={quick_auc('train', X_tr0, Y_tr0[:, D6_INDEX]):.4f}",
        f"val={quick_auc('val', X_va0, Y_va0[:, D6_INDEX]):.4f}",
        f"test={quick_auc('test', X_te0, Y_te0[:, D6_INDEX]):.4f}",
    )


if __name__ == "__main__":
    main()
