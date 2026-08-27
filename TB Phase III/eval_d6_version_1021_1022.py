"""
Phase III D6-only, parallel tracks (INITIAL cohort — same as Version 1.01, NOT Vector Balanced):
  Version 1.021 — RF (ref) + LR + IsolationForest (neg-only)
  Version 1.022 — RF (ref) + LR + OneClassSVM (neg-only)

Features: phase3_features_{train,val,test}.npz · Reference RF: rf_phase3_active_vs_inactive.joblib (D6 = index 4).
AUROC/EER on raw scores; CM @0.5 after train min–max per method.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
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


def eval_split(name: str, y: np.ndarray, sts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    block: dict = {"name": name, "n": int(len(y)), "methods": []}
    for label, s_tr, s_va, s_te in sts:
        if name == "val":
            s_eval, y_eval, s_train = s_va, y, s_tr
        else:
            s_eval, y_eval, s_train = s_te, y, s_tr
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


def run_track(
    *,
    root: Path,
    out_dir: Path,
    version_label: str,
    file_tag: str,
    display_title: str,
    sts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    y_va: np.ndarray,
    y_te: np.ndarray,
    metrics_json_name: str = "metrics_d6.json",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "version": version_label,
        "label": "D6",
        "track": display_title,
        "cm_threshold": 0.5,
        "cm_calibration": "minmax on train scores per method",
        "splits": [eval_split("val", y_va, sts), eval_split("test", y_te, sts)],
    }
    (out_dir / metrics_json_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

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
        ax.set_title(f"{display_title} — D6 ROC ({split_name}, n={len(y_eval)})")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)
        p = out_dir / f"{file_tag}_d6_roc_{split_name}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("saved:", p)

    for split_name, y_eval in (("val", y_va), ("test", y_te)):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
        fig.suptitle(
            f"{display_title} — D6 CM @0.5 (train min–max) · {split_name} (n={len(y_eval)})",
            fontsize=11,
        )
        for ax, (label, s_tr, s_va, s_te) in zip(axes, sts):
            s_eval = s_va if split_name == "val" else s_te
            draw_cm(ax, cm_at_half(y_eval, s_tr, s_eval), label)
        p = out_dir / f"{file_tag}_d6_cm_{split_name}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print("saved:", p)

    print("saved:", out_dir / metrics_json_name)


def fit_if(X_neg: np.ndarray, X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray, pos_rate: float):
    contam = float(max(pos_rate, 0.02))
    iforest = IsolationForest(n_estimators=300, contamination=contam, random_state=42, n_jobs=-1)
    iforest.fit(X_neg)
    return (
        -iforest.score_samples(X_tr).astype(float),
        -iforest.score_samples(X_va).astype(float),
        -iforest.score_samples(X_te).astype(float),
    )


def fit_ocsvm(X_neg: np.ndarray, X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray, pos_rate: float):
    nu = float(min(0.99, max(pos_rate * 2, 0.05)))
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    ocsvm.fit(X_neg)
    return (
        -ocsvm.decision_function(X_tr).ravel().astype(float),
        -ocsvm.decision_function(X_va).ravel().astype(float),
        -ocsvm.decision_function(X_te).ravel().astype(float),
    )


def write_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--only", choices=("021", "022", "both"), default="both")
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Fit IsolationForest and OneClassSVM in parallel threads before writing outputs",
    )
    args = ap.parse_args()
    root: Path = args.root
    art = root / "artifacts"

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
    j = 4
    y_tr, y_va, y_te = Y_tr[:, j], Y_va[:, j], Y_te[:, j]
    X_neg = X_tr[y_tr == 0]
    pos_rate = float(np.mean(y_tr))

    rf = joblib.load(rf_path)
    score_rf_tr = rf.predict_proba(X_tr)[j][:, 1].astype(float)
    score_rf_va = rf.predict_proba(X_va)[j][:, 1].astype(float)
    score_rf_te = rf.predict_proba(X_te)[j][:, 1].astype(float)

    lr = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42, solver="lbfgs")
    lr.fit(X_tr, y_tr)
    score_lr_tr = lr.predict_proba(X_tr)[:, 1].astype(float)
    score_lr_va = lr.predict_proba(X_va)[:, 1].astype(float)
    score_lr_te = lr.predict_proba(X_te)[:, 1].astype(float)

    score_if_tr = score_if_va = score_if_te = None
    score_oc_tr = score_oc_va = score_oc_te = None

    if args.parallel and args.only == "both":
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_if = ex.submit(fit_if, X_neg, X_tr, X_va, X_te, pos_rate)
            fut_oc = ex.submit(fit_ocsvm, X_neg, X_tr, X_va, X_te, pos_rate)
            score_if_tr, score_if_va, score_if_te = fut_if.result()
            score_oc_tr, score_oc_va, score_oc_te = fut_oc.result()
    else:
        if args.only in ("021", "both"):
            score_if_tr, score_if_va, score_if_te = fit_if(X_neg, X_tr, X_va, X_te, pos_rate)
        if args.only in ("022", "both"):
            score_oc_tr, score_oc_va, score_oc_te = fit_ocsvm(X_neg, X_tr, X_va, X_te, pos_rate)

    parent_dir = art / "version_1_02"
    parent_dir.mkdir(parents=True, exist_ok=True)
    parent_meta = {
        "version_family": "1.02",
        "cohort": "Phase III initial (Version 1.01 lineage): phase3_features_*.npz + rf_phase3_active_vs_inactive.joblib",
        "not_vector_balanced": True,
        "parallel_tracks": ["1.021", "1.022", "1.023"],
        "description": "021=IF, 022=OCSVM (this script). 1.023 XGBoost — eval_d6_version_1023_1024.py (flat outputs under artifacts/).",
        "scripts": ["eval_d6_version_1021_1022.py", "eval_d6_version_1023_1024.py"],
    }
    write_meta(parent_dir / "version_1_02_parallel_index.json", parent_meta)

    if args.only in ("021", "both") and score_if_te is not None:
        out021 = art / "version_1_021"
        out021.mkdir(parents=True, exist_ok=True)
        write_meta(
            out021 / "version_1_021_meta.json",
            {
                "version": "1.021",
                "d6_classifier": "IsolationForest (trained on D6-negative only; score = −score_samples)",
                "shared": "Initial Phase III features (v1.01); RF = multi-output active_vs_inactive D6 head + LR.",
            },
        )
        sts021 = [
            ("RF (ref, D6 head)", score_rf_tr, score_rf_va, score_rf_te),
            ("Logistic regression", score_lr_tr, score_lr_va, score_lr_te),
            ("IsolationForest", score_if_tr, score_if_va, score_if_te),
        ]
        run_track(
            root=root,
            out_dir=out021,
            version_label="1.021",
            file_tag="v1021",
            display_title="Version 1.021",
            sts=sts021,
            y_va=y_va,
            y_te=y_te,
        )

    if args.only in ("022", "both") and score_oc_te is not None:
        out022 = art / "version_1_022"
        out022.mkdir(parents=True, exist_ok=True)
        write_meta(
            out022 / "version_1_022_meta.json",
            {
                "version": "1.022",
                "d6_classifier": "OneClassSVM RBF (trained on D6-negative only; score = −decision_function)",
                "shared": "Initial Phase III features (v1.01); RF = multi-output active_vs_inactive D6 head + LR.",
            },
        )
        sts022 = [
            ("RF (ref, D6 head)", score_rf_tr, score_rf_va, score_rf_te),
            ("Logistic regression", score_lr_tr, score_lr_va, score_lr_te),
            ("OneClassSVM", score_oc_tr, score_oc_va, score_oc_te),
        ]
        run_track(
            root=root,
            out_dir=out022,
            version_label="1.022",
            file_tag="v1022",
            display_title="Version 1.022",
            sts=sts022,
            y_va=y_va,
            y_te=y_te,
        )


if __name__ == "__main__":
    main()
