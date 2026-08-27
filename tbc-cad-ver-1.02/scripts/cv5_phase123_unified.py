"""
Five repeated train/val/test splits (same sizes as original npz) for tbc-cad-ver.1.02:
  Phase I RF, Phase II RF, Phase III D1-D4 multi-output RF.

Each run: sklearn.base.clone(template joblib RF), refit on run train slice, evaluate on run test.

Writes: cv5_metrics.json, ROC PNGs, confusion heatmaps @ threshold 0.5 (default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

PHASE3_LABELS = ("D1 AFB", "D2 TB-PCR", "D3 solid culture", "D4 liquid culture")


def _annotate_text_color_for_cm_cell(
    *,
    rgba: tuple[float, float, float, float],
    dark_text: str = "black",
    light_text: str = "white",
) -> str:
    """Readable annotation color from heatmap cell RGBA (relative luminance)."""
    r, g, b, _a = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return light_text if lum < 0.55 else dark_text


def load_xy_vector(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    if "y" in d.files:
        y = np.asarray(d["y"])
    elif "Y" in d.files:
        y = np.asarray(d["Y"])
    else:
        raise KeyError(f"No y/Y in {npz_path}")
    if y.ndim == 2:
        y = y.astype(int)
    else:
        y = y.astype(int).ravel()
    return X, y


def load_phase3_xy(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    Y = np.asarray(d["Y"], dtype=int)
    if Y.ndim != 2 or Y.shape[1] < 4:
        raise ValueError(f"Expected Y (N,>=4) in {npz_path}, got {Y.shape}")
    return X, Y[:, :4]


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def cm_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    y_true = y_true.astype(int)
    y_hat = (np.asarray(scores, dtype=float) >= thr).astype(int)
    return confusion_matrix(y_true, y_hat, labels=[0, 1])


def summarize(vals: list[float]) -> dict[str, Any]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
    }


def roc_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> tuple[float, float]:
    y_true = y_true.astype(int)
    y_hat = (np.asarray(scores, dtype=float) >= thr).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    if cm.size != 4:
        return float("nan"), float("nan")
    tn, fp, fn, tp = cm
    denom_p = fp + tn
    denom_t = tp + fn
    fpr = float(fp / denom_p) if denom_p else float("nan")
    tpr = float(tp / denom_t) if denom_t else float("nan")
    return fpr, tpr


def plot_binary_roc_cv(
    folds_scores: list[np.ndarray],
    folds_y: list[np.ndarray],
    out_path: Path,
    title: str,
    thr: float,
) -> None:
    base_fpr = np.linspace(0, 1, 101)
    tprs: list[np.ndarray] = []
    thr_fprs: list[float] = []
    thr_tprs: list[float] = []
    for s, y in zip(folds_scores, folds_y):
        if y.min() == y.max():
            tprs.append(np.full_like(base_fpr, float("nan")))
            thr_fprs.append(float("nan"))
            thr_tprs.append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(y, s)
        tprs.append(np.interp(base_fpr, fpr, tpr))
        fp, tp = roc_at_threshold(y, s, thr)
        thr_fprs.append(fp)
        thr_tprs.append(tp)

    tprs_arr = np.vstack(tprs)
    mean_tpr = np.nanmean(tprs_arr, axis=0)
    std_tpr = np.nanstd(tprs_arr, axis=0)
    mean_auc = float(auc(base_fpr, mean_tpr))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.plot(base_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUROC≈{mean_auc:.3f})")
    ax.fill_between(
        base_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        alpha=0.25,
        label="±1 SD over folds",
    )
    mfp = float(np.nanmean(thr_fprs))
    mtp = float(np.nanmean(thr_tprs))
    ax.scatter(
        [mfp],
        [mtp],
        s=80,
        zorder=5,
        marker="o",
        color="C1",
        label=f"Mean (FPR,TPR) @ thr={thr:g}",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title + f"\nTest folds · marker @ threshold = {thr:g}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cm_mean_std(cm_list: list[np.ndarray], out_path: Path, title: str) -> None:
    stack = np.stack(cm_list, axis=0).astype(float)
    mean_m = stack.mean(axis=0)
    std_m = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean_m)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    vmax = float(mean_m.max()) if mean_m.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    im = ax.imshow(mean_m, cmap="Blues", norm=norm, interpolation="nearest")
    cmap = im.cmap
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for (i, j), v in np.ndenumerate(mean_m):
        s = std_m[i, j]
        face = cmap(norm(float(v)))
        ax.text(
            j,
            i,
            f"{v:.1f}\n±{s:.1f}",
            ha="center",
            va="center",
            color=_annotate_text_color_for_cm_cell(rgba=face),
            fontsize=11,
        )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title + "\nTest @ threshold 0.5 · mean ± SD (5 runs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_phase3_roc_grid(
    per_label_folds: list[list[tuple[np.ndarray, np.ndarray]]],
    out_path: Path,
    thr: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    base_fpr = np.linspace(0, 1, 101)
    for j, ax in enumerate(axes.ravel()):
        folds = per_label_folds[j]
        tprs = []
        thr_fprs, thr_tprs = [], []
        for s, y in folds:
            if y.min() == y.max():
                tprs.append(np.full_like(base_fpr, float("nan")))
                thr_fprs.append(float("nan"))
                thr_tprs.append(float("nan"))
                continue
            fpr, tpr, _ = roc_curve(y, s)
            tprs.append(np.interp(base_fpr, fpr, tpr))
            fp, tp = roc_at_threshold(y, s, thr)
            thr_fprs.append(fp)
            thr_tprs.append(tp)
        tprs_arr = np.vstack(tprs)
        mean_tpr = np.nanmean(tprs_arr, axis=0)
        std_tpr = np.nanstd(tprs_arr, axis=0)
        mean_auc = float(auc(base_fpr, mean_tpr))
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        ax.plot(base_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUROC≈{mean_auc:.3f})")
        ax.fill_between(
            base_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            alpha=0.25,
        )
        ax.scatter(
            [float(np.nanmean(thr_fprs))],
            [float(np.nanmean(thr_tprs))],
            s=70,
            zorder=5,
            color="C1",
            label=f"Mean @ thr={thr:g}",
        )
        ax.set_title(PHASE3_LABELS[j], fontsize=24)
        ax.set_xlabel("FPR", fontsize=22)
        ax.set_ylabel("TPR", fontsize=22)
        ax.tick_params(axis="both", labelsize=18)
        ax.legend(loc="lower right", fontsize=14)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Phase III D1–D4 · test ROC (5 repeated splits)", fontsize=24)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_phase3_cm_grid(cm_per_label: list[list[np.ndarray]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for j, ax in enumerate(axes.ravel()):
        cms = cm_per_label[j]
        stack = np.stack(cms, axis=0).astype(float)
        mean_m = stack.mean(axis=0)
        std_m = stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(mean_m)
        vmax = float(mean_m.max()) if mean_m.size else 1.0
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
        im = ax.imshow(mean_m, cmap="Blues", norm=norm, interpolation="nearest")
        cmap = im.cmap
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)
        for (i, ii), v in np.ndenumerate(mean_m):
            s = std_m[i, ii]
            face = cmap(norm(float(v)))
            ax.text(
                ii,
                i,
                f"{v:.1f}\n±{s:.1f}",
                ha="center",
                va="center",
                fontsize=20,
                color=_annotate_text_color_for_cm_cell(rgba=face),
            )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=18)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=18)
        ax.set_title(PHASE3_LABELS[j], fontsize=24)
        ax.tick_params(axis="both", labelsize=18)
    fig.suptitle("Phase III D1–D4 · test CM @ 0.5 · mean ± SD (5 runs)", fontsize=24)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_phase_binary(
    template_path: Path,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    cv_runs: int,
    master_seed: int,
    thr: float,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    tmpl: RandomForestClassifier = joblib.load(template_path)
    X_tr0, y_tr0 = load_xy_vector(train_npz)
    X_va0, y_va0 = load_xy_vector(val_npz)
    X_te0, y_te0 = load_xy_vector(test_npz)
    n_tr, n_va, n_te = X_tr0.shape[0], X_va0.shape[0], X_te0.shape[0]
    X_pool = np.concatenate([X_tr0, X_va0, X_te0], axis=0)
    y_pool = np.concatenate([y_tr0, y_va0, y_te0], axis=0)
    n_total = X_pool.shape[0]
    if n_total != n_tr + n_va + n_te:
        raise RuntimeError("phase binary: pool size mismatch")

    rng_master = np.random.default_rng(master_seed)
    fold_rows: list[dict[str, Any]] = []
    fold_scores: list[np.ndarray] = []
    fold_y: list[np.ndarray] = []
    aucs: list[float] = []
    eers: list[float] = []

    for fold in range(cv_runs):
        run_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(run_seed)
        perm = rng.permutation(n_total)
        te_idx = perm[n_tr + n_va :]
        X_te = X_pool[te_idx]
        y_te = y_pool[te_idx]
        tr_idx = perm[:n_tr]
        X_tr = X_pool[tr_idx]
        y_tr = y_pool[tr_idx]

        est = clone(tmpl)
        est.set_params(random_state=run_seed)
        est.fit(X_tr, y_tr)
        scores = est.predict_proba(X_te)[:, 1].astype(float)
        if y_te.min() == y_te.max():
            au = float("nan")
        else:
            au = float(roc_auc_score(y_te, scores))
        ee = float(eer_from_scores(y_te, scores))
        cm = cm_at_threshold(y_te, scores, thr)
        fold_rows.append(
            {
                "fold": fold,
                "seed": run_seed,
                "test_auroc": au,
                "test_eer": ee,
                "test_cm_threshold": thr,
                "test_confusion_matrix": cm.tolist(),
            }
        )
        aucs.append(au)
        eers.append(ee)
        fold_scores.append(scores)
        fold_y.append(y_te.astype(int))

    mats = [np.array(f["test_confusion_matrix"]) for f in fold_rows]
    block: dict[str, Any] = {
        "template_rf": str(template_path),
        "pool": {
            "train_npz": str(train_npz),
            "val_npz": str(val_npz),
            "test_npz": str(test_npz),
            "n_total": n_total,
            "sizes": {"train": n_tr, "val": n_va, "test": n_te},
        },
        "cv": {"runs": cv_runs, "scheme": "repeated_random_splits_same_sizes"},
        "test_summary": {
            "auroc": summarize(aucs),
            "eer": summarize(eers),
            "confusion_at_threshold": {
                "threshold": thr,
                "mean_matrix": np.mean(mats, axis=0).tolist(),
                "std_matrix": np.std(mats, axis=0, ddof=1).tolist() if cv_runs > 1 else [[0.0, 0.0], [0.0, 0.0]],
            },
        },
        "folds": fold_rows,
    }
    return block, fold_scores, fold_y


def run_phase3_d14(
    template_path: Path,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    cv_runs: int,
    master_seed: int,
    thr: float,
) -> tuple[dict[str, Any], list[list[tuple[np.ndarray, np.ndarray]]], list[list[np.ndarray]]]:
    tmpl: RandomForestClassifier = joblib.load(template_path)
    X_tr0, Y_tr0 = load_phase3_xy(train_npz)
    X_va0, Y_va0 = load_phase3_xy(val_npz)
    X_te0, Y_te0 = load_phase3_xy(test_npz)
    n_tr, n_va, n_te = X_tr0.shape[0], X_va0.shape[0], X_te0.shape[0]
    X_pool = np.concatenate([X_tr0, X_va0, X_te0], axis=0)
    Y_pool = np.concatenate([Y_tr0, Y_va0, Y_te0], axis=0)
    n_total = X_pool.shape[0]
    if n_total != n_tr + n_va + n_te:
        raise RuntimeError("phase3: pool size mismatch")

    rng_master = np.random.default_rng(master_seed)
    n_labels = Y_pool.shape[1]
    fold_records: list[dict[str, Any]] = []
    per_label_folds: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(n_labels)]
    per_label_cms: list[list[np.ndarray]] = [[] for _ in range(n_labels)]

    label_aucs: list[list[float]] = [[] for _ in range(n_labels)]
    label_eers: list[list[float]] = [[] for _ in range(n_labels)]

    for fold in range(cv_runs):
        run_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(run_seed)
        perm = rng.permutation(n_total)
        tr_idx = perm[:n_tr]
        te_idx = perm[n_tr + n_va :]
        X_tr = X_pool[tr_idx]
        Y_tr = Y_pool[tr_idx]
        X_te = X_pool[te_idx]
        Y_te = Y_pool[te_idx]

        est = clone(tmpl)
        est.set_params(random_state=run_seed)
        est.fit(X_tr, Y_tr)
        probas = est.predict_proba(X_te)
        fold_entry: dict[str, Any] = {"fold": fold, "seed": run_seed, "labels": []}
        for j in range(n_labels):
            pj = probas[j][:, 1].astype(float)
            yj = Y_te[:, j].astype(int)
            if yj.min() == yj.max():
                au = float("nan")
            else:
                au = float(roc_auc_score(yj, pj))
            ee = float(eer_from_scores(yj, pj))
            cm = cm_at_threshold(yj, pj, thr)
            label_aucs[j].append(au)
            label_eers[j].append(ee)
            per_label_folds[j].append((pj, yj))
            per_label_cms[j].append(cm)
            fold_entry["labels"].append(
                {
                    "key": PHASE3_LABELS[j].split()[0],
                    "test_auroc": au,
                    "test_eer": ee,
                    "test_confusion_matrix": cm.tolist(),
                }
            )
        fold_records.append(fold_entry)

    summary_labels = []
    for j in range(n_labels):
        summary_labels.append(
            {
                "name": PHASE3_LABELS[j],
                "test_auroc": summarize(label_aucs[j]),
                "test_eer": summarize(label_eers[j]),
                "confusion_at_threshold": {
                    "threshold": thr,
                    "mean_matrix": np.mean(per_label_cms[j], axis=0).tolist(),
                    "std_matrix": np.std(per_label_cms[j], axis=0, ddof=1).tolist()
                    if cv_runs > 1
                    else [[0.0, 0.0], [0.0, 0.0]],
                },
            }
        )

    block: dict[str, Any] = {
        "template_rf": str(template_path),
        "pool": {
            "train_npz": str(train_npz),
            "val_npz": str(val_npz),
            "test_npz": str(test_npz),
            "n_total": n_total,
            "sizes": {"train": n_tr, "val": n_va, "test": n_te},
        },
        "cv": {"runs": cv_runs, "scheme": "repeated_random_splits_same_sizes", "outputs_trained": "D1–D4 only"},
        "test_summary_by_label": summary_labels,
        "folds": fold_records,
    }
    return block, per_label_folds, per_label_cms


def replot_confusion_only_from_metrics(metrics_path: Path, out_dir: Path) -> None:
    """Redraw Phase I/II/III CM PNGs from an existing cv5_metrics.json (no model refit)."""
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    p1_folds = data["phase1"]["folds"]
    plot_cm_mean_std(
        [np.array(f["test_confusion_matrix"]) for f in p1_folds],
        out_dir / "phase1_test_cm_cv5.png",
        "Phase I",
    )
    p2_folds = data["phase2"]["folds"]
    plot_cm_mean_std(
        [np.array(f["test_confusion_matrix"]) for f in p2_folds],
        out_dir / "phase2_test_cm_cv5.png",
        "Phase II",
    )
    p3_folds = data["phase3_d1d4"]["folds"]
    n_labels = len(p3_folds[0]["labels"])
    cms3: list[list[np.ndarray]] = [[] for _ in range(n_labels)]
    for fold in p3_folds:
        for j in range(n_labels):
            cms3[j].append(np.array(fold["labels"][j]["test_confusion_matrix"]))
    plot_phase3_cm_grid(cms3, out_dir / "phase3_d1d4_test_cm_cv5.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="5× repeated-split CV for Phase I, II, III D1–D4")
    ap.add_argument("--tb-artifacts", type=Path, default=Path(r"D:\TB Test DB\artifacts"))
    ap.add_argument("--phase3-root", type=Path, default=Path(r"D:\TB Phase III"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cv-runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--replot-cm-from-json",
        type=Path,
        default=None,
        help="If set, only redraw confusion PNGs from this cv5_metrics.json into --out-dir (no training).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or (Path(__file__).resolve().parents[1] / "artifacts" / "cv5_phase123_unified")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_cm_from_json is not None:
        replot_confusion_only_from_metrics(args.replot_cm_from_json, out_dir)
        print("Redrawn confusion matrices to", out_dir)
        return

    art = args.tb_artifacts
    p3 = args.phase3_root
    thr = float(args.threshold)

    phase1_rf = art / "rf_phase1_huge.joblib"
    phase2_rf = art / "rf_phase2_huge.joblib"
    phase3_rf = p3 / "rf_phase3_active_vs_inactive.joblib"

    p1_train = art / "phase1_features_train.npz"
    p1_val = art / "phase1_features_val.npz"
    p1_test = art / "phase1_features_test.npz"
    p2_train = art / "phase2_features_train.npz"
    p2_val = art / "phase2_features_val.npz"
    p2_test = art / "phase2_features_test.npz"
    p3_train = p3 / "phase3_features_train.npz"
    p3_val = p3 / "phase3_features_val.npz"
    p3_test = p3 / "phase3_features_test.npz"

    for p in [phase1_rf, phase2_rf, phase3_rf, p1_train, p2_train, p3_train]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    report: dict[str, Any] = {
        "model_line": "tbc-cad-ver.1.02",
        "threshold_operating_point": thr,
        "phase1": {},
        "phase2": {},
        "phase3_d1d4": {},
    }

    b1, s1, y1 = run_phase_binary(phase1_rf, p1_train, p1_val, p1_test, args.cv_runs, args.seed, thr)
    report["phase1"] = b1
    plot_binary_roc_cv(s1, y1, out_dir / "phase1_test_roc_cv5.png", "Phase I (normal vs abnormal)", thr)
    plot_cm_mean_std([np.array(f["test_confusion_matrix"]) for f in b1["folds"]], out_dir / "phase1_test_cm_cv5.png", "Phase I")

    b2, s2, y2 = run_phase_binary(phase2_rf, p2_train, p2_val, p2_test, args.cv_runs, args.seed, thr)
    report["phase2"] = b2
    plot_binary_roc_cv(s2, y2, out_dir / "phase2_test_roc_cv5.png", "Phase II (inactive vs active TB)", thr)
    plot_cm_mean_std([np.array(f["test_confusion_matrix"]) for f in b2["folds"]], out_dir / "phase2_test_cm_cv5.png", "Phase II")

    b3, folds3, cms3 = run_phase3_d14(phase3_rf, p3_train, p3_val, p3_test, args.cv_runs, args.seed, thr)
    report["phase3_d1d4"] = b3
    plot_phase3_roc_grid(folds3, out_dir / "phase3_d1d4_test_roc_cv5.png", thr)
    plot_phase3_cm_grid(cms3, out_dir / "phase3_d1d4_test_cm_cv5.png")

    out_json = out_dir / "cv5_metrics.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out_json)
    print("Wrote plots to", out_dir)


if __name__ == "__main__":
    main()
