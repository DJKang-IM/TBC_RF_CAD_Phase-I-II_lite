"""
Phase III Version 1.02 — legacy single-folder D6 eval (RF + LR + IF + OCSVM) on INITIAL cohort.

Uses phase3_features_*.npz + rf_phase3_active_vs_inactive (same as v1.01).
For the supported pipeline (Phase I/II v1.01 RF + Phase III D1–D4 RF + D6 v1.023 XGB), see
  plot_pipeline_v101_d6_xgb1023_bootstrap5.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.svm import OneClassSVM


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return {
        "eer": eer,
        "threshold": float(thr[i]),
        "fpr": float(fpr[i]),
        "fnr": float(fnr[i]),
    }


def minmax_calibrate(train_s: np.ndarray, s: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(train_s)), float(np.max(train_s))
    if hi - lo < 1e-12:
        return np.full_like(s, 0.5)
    out = (s - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def cm_at_half(y: np.ndarray, scores_train: np.ndarray, scores_eval: np.ndarray) -> np.ndarray:
    cal = minmax_calibrate(scores_train, scores_eval)
    yhat = (cal >= 0.5).astype(int)
    return confusion_matrix(y.astype(int), yhat, labels=[0, 1]).astype(int)


def draw_cm(ax, cm: np.ndarray, title: str) -> None:
    im = ax.imshow(cm, cmap="Blues", vmin=0, interpolation="nearest")
    vmax = float(cm.max()) if cm.size else 1.0
    thr = vmax / 2.0
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > thr else "black",
                fontsize=14,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"], fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    root: Path = args.root
    out_dir = root / "artifacts" / "version_1_02"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_p = root / "phase3_features_train.npz"
    val_p = root / "phase3_features_val.npz"
    test_p = root / "phase3_features_test.npz"
    rf_path = root / "rf_phase3_active_vs_inactive.joblib"

    d_tr = np.load(train_p, allow_pickle=True)
    d_va = np.load(val_p, allow_pickle=True)
    d_te = np.load(test_p, allow_pickle=True)
    X_tr, Y_tr = d_tr["X"].astype(np.float32), d_tr["Y"].astype(int)
    X_va, Y_va = d_va["X"].astype(np.float32), d_va["Y"].astype(int)
    X_te, Y_te = d_te["X"].astype(np.float32), d_te["Y"].astype(int)
    j = 4  # D6
    y_tr, y_va, y_te = Y_tr[:, j], Y_va[:, j], Y_te[:, j]

    rf = joblib.load(rf_path)
    score_rf_tr = rf.predict_proba(X_tr)[j][:, 1].astype(float)
    score_rf_va = rf.predict_proba(X_va)[j][:, 1].astype(float)
    score_rf_te = rf.predict_proba(X_te)[j][:, 1].astype(float)

    lr = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    score_lr_tr = lr.predict_proba(X_tr)[:, 1].astype(float)
    score_lr_va = lr.predict_proba(X_va)[:, 1].astype(float)
    score_lr_te = lr.predict_proba(X_te)[:, 1].astype(float)

    X_neg = X_tr[y_tr == 0]
    pos_rate = float(np.mean(y_tr))
    contam = float(max(pos_rate, 0.02))
    nu = float(min(0.99, max(pos_rate * 2, 0.05)))

    iforest = IsolationForest(
        n_estimators=300,
        contamination=contam,
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_neg)
    # Higher score_samples = more normal; flip so higher = more like positive (minority / abnormal)
    score_if_tr = -iforest.score_samples(X_tr).astype(float)
    score_if_va = -iforest.score_samples(X_va).astype(float)
    score_if_te = -iforest.score_samples(X_te).astype(float)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    ocsvm.fit(X_neg)
    score_oc_tr = -ocsvm.decision_function(X_tr).ravel().astype(float)
    score_oc_va = -ocsvm.decision_function(X_va).ravel().astype(float)
    score_oc_te = -ocsvm.decision_function(X_te).ravel().astype(float)

    report: dict = {
        "version": "1.02",
        "label": "D6",
        "split_files": {"train": str(train_p), "val": str(val_p), "test": str(test_p)},
        "reference_rf": str(rf_path),
        "cm_threshold": 0.5,
        "cm_calibration": "minmax on train scores per method",
        "splits": [],
    }

    # Build report per split
    def eval_split(name: str, y: np.ndarray, sts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> dict:
        block = {"name": name, "n": int(len(y)), "methods": []}
        for label, s_tr, s_va, s_te in sts:
            if name == "val":
                s_eval, y_eval = s_va, y
                s_train = s_tr
            else:
                s_eval, y_eval = s_te, y
                s_train = s_tr
            fpr, tpr, _ = roc_curve(y_eval, s_eval)
            block["methods"].append(
                {
                    "name": label,
                    "auroc": float(auc(fpr, tpr)),
                    "eer": eer_from_scores(y_eval, s_eval),
                    "confusion_matrix_0_5": cm_at_half(y_eval, s_train, s_eval).tolist(),
                }
            )
        return block

    sts = [
        ("RF (ref, D6 head)", score_rf_tr, score_rf_va, score_rf_te),
        ("Logistic regression", score_lr_tr, score_lr_va, score_lr_te),
        ("IsolationForest", score_if_tr, score_if_va, score_if_te),
        ("OneClassSVM", score_oc_tr, score_oc_va, score_oc_te),
    ]
    report["splits"] = [eval_split("val", y_va, sts), eval_split("test", y_te, sts)]

    (out_dir / "metrics_d6_v102.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ROC plots
    for split_name, y_eval, scores_list in (
        ("val", y_va, [(m[0], m[2]) for m in sts]),
        ("test", y_te, [(m[0], m[3]) for m in sts]),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
        for label, s_eval in scores_list:
            fpr, tpr, _ = roc_curve(y_eval, s_eval)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={a:.4f})")
        ax.plot([0, 1], [0, 1], ":", color="gray", lw=1, label="chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"Version 1.02 — D6 ROC ({split_name}, n={len(y_eval)})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)
        p = out_dir / f"v102_d6_roc_{split_name}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("saved:", p)

    # CM heatmaps 2x2 per split
    for split_name, y_eval in (("val", y_va), ("test", y_te)):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), constrained_layout=True)
        fig.suptitle(
            f"Version 1.02 — D6 confusion @0.5 (train min–max) · {split_name} (n={len(y_eval)})",
            fontsize=11,
        )
        for ax, (label, s_tr, s_va, s_te) in zip(
            axes.ravel(),
            sts,
        ):
            s_eval = s_va if split_name == "val" else s_te
            cm = cm_at_half(y_eval, s_tr, s_eval)
            draw_cm(ax, cm, label)
        p = out_dir / f"v102_d6_cm_heatmap_{split_name}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("saved:", p)

    print("saved:", out_dir / "metrics_d6_v102.json")


if __name__ == "__main__":
    main()
