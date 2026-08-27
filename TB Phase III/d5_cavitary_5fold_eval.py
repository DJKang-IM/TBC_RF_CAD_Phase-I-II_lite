# -*- coding: utf-8 -*-
"""
D5 (Cavitary lesion) binary: 5-fold stratified internal CV on TRAIN features,
with AUROC / CM@0.5 / EER per fold + summary plots.

Use --n_repeats 5 (default) to run 5 independent CVs (different StratifiedKFold+RF seeds), then report
mean +/- std of: mean-fold AUROC, mean-fold EER, and OOF CM cells (0.5) across repeats.

Default feature pools: phase3_features_{train,val,test}.npz with D5 in npz.
Header loading: build_phase3_features.read_vec_from_header (D1-D6 private US 0x1101-0x1106).

Primary output: <REDACTED_PATH> Phase III/artifacts/d5_cavitary_5fold/
  d5_split_summary.txt, d5_5x5_repeats_summary.txt (if n_repeats>1), d5_5fold_metrics.json
  d5_5fold_roc/cm/eer.png (from first repeat)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

DEFAULT_OUT = Path(r"D:\TB Phase III\artifacts\d5_cavitary_5fold")
DEFAULT_ARTIFACT_VER = Path(r"D:\artifact") / "ver. 1.03(D5_cavitary)"
DEFAULT_TB = Path(r"D:\TB Test DB\artifacts")
DEFAULT_P3 = Path(r"D:\TB Phase III")


def _eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    if y_true.min() == y_true.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def _load_p3_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    if "D5" not in d:
        raise KeyError(
            f"{path.name}: missing 'D5' in npz. Re-run build_phase3_features.py on DICOMs with D5 in headers."
        )
    y_d5 = np.asarray(d["D5"]).ravel().astype(int)
    Y = np.asarray(d["Y"], dtype=int) if "Y" in d else None
    return X, y_d5, Y


def _count_npz(path: Path | None) -> str:
    if path is None or not path.is_file():
        return f"(missing) {path}"
    try:
        d = np.load(path, allow_pickle=True)
        n = int(d["X"].shape[0])
        return f"n={n}  {path.name}"
    except Exception as e:
        return f"(error: {e}) {path}"


def build_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        n_jobs=-1,
        random_state=int(seed),
        class_weight="balanced",
    )


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[
    dict,
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[float],
    list[float],
    list[np.ndarray],
    np.ndarray,
]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tprs: list[np.ndarray] = []
    base_fpr = np.linspace(0, 1, 100)
    mean_aurocs: list[float] = []
    eers: list[float] = []
    cms: list[np.ndarray] = []
    oof_s = np.full(len(y), np.nan, dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if y_va.min() == y_va.max():
            auroc = float("nan")
            s_va = np.zeros(len(y_va), dtype=float)
        else:
            rf = build_rf(seed + fold * 13)
            rf.fit(X_tr, y_tr)
            s_va = rf.predict_proba(X_va)[:, 1].astype(float)
            auroc = float(roc_auc_score(y_va, s_va))
        oof_s[va_idx] = s_va
        mean_aurocs.append(auroc)
        eers.append(_eer(y_va, s_va))

        if y_va.size and y_va.min() < y_va.max():
            fpr, tpr, _ = roc_curve(y_va, s_va)
            tpr_i = np.interp(base_fpr, fpr, tpr)
            tprs.append(tpr_i)
        else:
            tprs.append(np.zeros_like(base_fpr))

        y_hat = (s_va >= 0.5).astype(int)
        cm = confusion_matrix(y_va, y_hat, labels=[0, 1])
        cms.append(cm.astype(float))

    mean_tpr = np.nanmean(tprs, axis=0) if tprs else base_fpr * 0.0
    std_tpr = (
        np.nanstd(tprs, axis=0, ddof=1) if len(tprs) > 1 else np.zeros_like(mean_tpr, dtype=float)
    )
    oof_auc = float(np.nanmean(mean_aurocs)) if mean_aurocs else float("nan")
    oof_eer = float(np.nanmean(eers)) if eers else float("nan")

    m = np.isfinite(oof_s)
    if m.all() and m.any():
        y_hat_o = (oof_s >= 0.5).astype(int)
        oof_cm = confusion_matrix(y, y_hat_o, labels=[0, 1]).astype(float)
    else:
        oof_cm = np.zeros((2, 2), dtype=float)
    oof_auc_full: float
    if m.all() and y.min() < y.max():
        try:
            oof_auc_full = float(roc_auc_score(y, oof_s))
        except ValueError:
            oof_auc_full = float("nan")
    else:
        oof_auc_full = float("nan")

    summary: dict = {
        "n_splits": n_splits,
        "cv_seed": int(seed),
        "mean_fold_auroc": oof_auc,
        "std_fold_auroc": float(np.nanstd(mean_aurocs, ddof=1)) if len(mean_aurocs) > 1 else 0.0,
        "mean_fold_eer": oof_eer,
        "std_fold_eer": float(np.nanstd(eers, ddof=1)) if len(eers) > 1 else 0.0,
        "oof_auroc_full": oof_auc_full,
        "oof_cm_0.5": oof_cm.tolist(),
        "per_fold": [
            {
                "fold": i + 1,
                "val_auroc": float(mean_aurocs[i]),
                "val_eer": float(eers[i]),
                "val_cm_0.5": cms[i].tolist(),
            }
            for i in range(n_splits)
        ],
    }
    return summary, tprs, base_fpr, mean_tpr, std_tpr, mean_aurocs, eers, cms, oof_cm


def plot_roc(
    base_fpr: np.ndarray,
    tprs: list[np.ndarray],
    mean_tpr: np.ndarray,
    std_tpr: np.ndarray,
    mean_aurocs: list[float],
    out: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.2))
    for i, tpr in enumerate(tprs, start=1):
        a = float(mean_aurocs[i - 1]) if (i - 1) < len(mean_aurocs) else float("nan")
        lab = f"fold {i} (AUROC={a:.3f})" if np.isfinite(a) else f"fold {i} (AUROC=nan)"
        ax.plot(base_fpr, tpr, lw=1.2, alpha=0.55, label=lab)
    mean_auc = auc(base_fpr, mean_tpr) if mean_tpr.size and np.isfinite(np.nanmax(mean_tpr)) else float("nan")
    ax.plot(
        base_fpr,
        mean_tpr,
        color="k",
        lw=2.4,
        label=f"mean TPR  macro-AUC={mean_auc:.3f}",
    )
    ax.fill_between(
        base_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color="gray",
        alpha=0.2,
        label="mean TPR plus/minus 1 SD",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("D5 Cavitary lesion  internal 5-fold CV  ROC (train pool)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_cm_grid(cms: list[np.ndarray], out: Path) -> None:
    n = len(cms)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    for k, cm in enumerate(cms):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        ax.imshow(cm, cmap="Blues", vmin=0, interpolation="nearest")
        for i in range(2):
            for j in range(2):
                v = float(cm[i, j])
                txt = str(int(v)) if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="white" if v > (float(cm.max()) or 0) / 2 else "black",
                    fontsize=10,
                )
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title(f"Fold {k + 1} CM@0.5 (val fold)")
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("D5  internal 5-fold  confusion at threshold 0.5", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_eer_bars(folds: list[int], eers: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xs = np.arange(len(folds))
    eers2 = [float(x) for x in eers if np.isfinite(x)]
    mean_eer = float(np.mean(eers2)) if eers2 else float("nan")
    ax.bar(xs, eers, color="#2c7fb8", edgecolor="k", alpha=0.85)
    if np.isfinite(mean_eer):
        ax.axhline(mean_eer, color="r", ls="--", label=f"mean EER = {mean_eer:.4f}")
    ax.set_xticks(xs, [f"fold {f}" for f in folds])
    ax.set_ylabel("EER")
    ax.set_title("D5  Cavitary  internal 5-fold  EER (val fold)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3_root", type=Path, default=DEFAULT_P3, help="TB Phase III (phase3_features_*.npz)")
    ap.add_argument("--train_npz", type=Path, default=None)
    ap.add_argument("--val_npz", type=Path, default=None)
    ap.add_argument("--test_npz", type=Path, default=None)
    ap.add_argument("--tb_artifacts", type=Path, default=DEFAULT_TB, help="TB Test DB artifacts for P1/P2 size report")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT,
        help="All PNG/JSON/TXT (default: TB Phase III/artifacts/d5_cavitary_5fold).",
    )
    ap.add_argument(
        "--copy_to_artifact",
        type=Path,
        nargs="?",
        const=DEFAULT_ARTIFACT_VER,
        default=None,
        help="Also copy outputs to <REDACTED_PATH> 1.03(D5_cavitary)/ (or pass a path).",
    )
    ap.add_argument("--n_splits", type=int, default=5, help="Stratified K-fold (default 5).")
    ap.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Full CV re-runs with different seeds; report mean+/- std across repeats (default 5).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; repeat r uses seed + r * 7919.",
    )
    args = ap.parse_args()

    root = args.p3_root
    train_p = args.train_npz or (root / "phase3_features_train.npz")
    val_p = args.val_npz or (root / "phase3_features_val.npz")
    test_p = args.test_npz or (root / "phase3_features_test.npz")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("D5 CAVITARY 5-FOLD  OUTPUT FOLDER (absolute):")
    print(out_dir.resolve())
    print("=" * 72)

    try:
        X_tr, y5_tr, Y_tr = _load_p3_npz(train_p)
        X_va, y5_va, Y_va = _load_p3_npz(val_p)
        X_te, y5_te, Y_te = _load_p3_npz(test_p)
    except KeyError as e:
        err_txt = out_dir / "00_D5_prerequisite.txt"
        err_txt.write_text(
            "D5 evaluation did NOT run because your phase3 feature .npz has no D5 array.\n\n"
            f"Error: {e}\n\n"
            "Fix:\n"
            "  1) Embed D5 (cavitary) into DICOM with embed_tb_labels_into_dicom.py --d5-csv (Version 2 meta),\n"
            "  2) Re-run: python build_phase3_features.py --in_dir <DICOMs> --out <npz> (split tr/va/te as you use now),\n"
            f"  3) Expect keys: X, Y, D5, paths\n"
            f"\nChecked files:\n  {train_p}\n  {val_p}\n  {test_p}\n"
            f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        print("ERROR", e, file=sys.stderr)
        print("WROTE (read this):", err_txt.resolve(), file=sys.stderr)
        return 1

    if y5_tr.min() == y5_tr.max():
        bad = out_dir / "d5_error_constant_class_on_train.txt"
        bad.write_text(
            "D5 is all 0 or all 1 on TRAIN - cannot run StratifiedKFold for two classes.\n"
            f"Train D5 sum: {int(y5_tr.sum())} / n={len(y5_tr)}\n",
            encoding="utf-8",
        )
        print("ERROR: D5 is constant on TRAIN; need both 0 and 1 for StratifiedKFold.", file=sys.stderr)
        print("WROTE:", bad.resolve(), file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("=== Pool sizes (Phase III D5 / same feature matrix as D1-D4) ===")
    lines.append(f"TRAIN: {train_p.name}  n={X_tr.shape[0]}  D5 pos={int(y5_tr.sum())} ({100*y5_tr.mean():.2f}%)")
    lines.append(f"VAL:   {val_p.name}  n={X_va.shape[0]}  D5 pos={int(y5_va.sum())} ({100*y5_va.mean():.2f}%)")
    lines.append(f"TEST:  {test_p.name}  n={X_te.shape[0]}  D5 pos={int(y5_te.sum())} ({100*y5_te.mean():.2f}%)")
    if Y_tr is not None:
        lines.append("Y (D1-D4,D6) positive counts on TRAIN (column order):")
        lines.append("  " + " ".join(f"D{j+1}={Y_tr[:, j].sum()}" for j in range(min(4, Y_tr.shape[1]))))
        if Y_tr.shape[1] >= 5:
            lines.append(f"  D6 (index 4 in Y)={Y_tr[:, 4].sum()}")

    lines.append("")
    lines.append("=== Phase I / II feature npz (row counts; models unchanged) ===")
    for ph, name in [("1", "phase1_features"), ("2", "phase2_features")]:
        for sp in ("train", "val", "test"):
            p = args.tb_artifacts / f"{name}_{sp}.npz"
            lines.append(f"Phase {ph}  {sp}:  {_count_npz(p)}")

    n_rep = max(1, int(args.n_repeats))
    repeat_summaries: list[dict] = []
    repeat_oof_cms: list[np.ndarray] = []
    first_plot: tuple | None = None

    for rep in range(n_rep):
        rep_seed = int(args.seed + rep * 7919)
        summary, tprs, base_fpr, mean_tpr, std_tpr, mean_aurocs, eers, cms, oof_cm = run_cv(
            X_tr, y5_tr, n_splits=args.n_splits, seed=rep_seed
        )
        summary["repeat_index"] = rep
        repeat_summaries.append(summary)
        repeat_oof_cms.append(oof_cm.astype(float))
        if rep == 0:
            first_plot = (tprs, base_fpr, mean_tpr, std_tpr, mean_aurocs, eers, cms)

    assert first_plot is not None
    tprs, base_fpr, mean_tpr, std_tpr, mean_aurocs, eers, cms = first_plot

    mfa = np.array([float(s["mean_fold_auroc"]) for s in repeat_summaries], dtype=float)
    mfe = np.array([float(s["mean_fold_eer"]) for s in repeat_summaries], dtype=float)
    oof_au = np.array(
        [float(s.get("oof_auroc_full", float("nan"))) for s in repeat_summaries], dtype=float
    )
    stack_cm = np.stack(repeat_oof_cms, axis=0) if repeat_oof_cms else np.zeros((0, 2, 2))

    def _mean_std(a: np.ndarray) -> tuple[float, float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan"), float("nan")
        m = float(np.mean(a))
        sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        return m, sd

    am, asd = _mean_std(mfa)
    em, esd = _mean_std(mfe)
    oam, oasd = _mean_std(oof_au)
    cm_mean = np.nanmean(stack_cm, axis=0) if stack_cm.size else np.zeros((2, 2))
    cm_std = (
        np.nanstd(stack_cm, axis=0, ddof=1) if stack_cm.shape[0] > 1 else np.zeros((2, 2), dtype=float)
    )

    lines.append("")
    lines.append(
        f"=== D5 internal CV on TRAIN (n_splits={args.n_splits}, n_repeats={n_rep}) ==="
    )
    lines.append(
        f"Repeat 0 (seed {int(args.seed)}): mean-fold AUROC={repeat_summaries[0]['mean_fold_auroc']:.4f} "
        f"+/- {repeat_summaries[0]['std_fold_auroc']:.4f} (across folds within that run)"
    )
    if n_rep > 1:
        lines.append(
            f"Across {n_rep} repeats - mean-fold AUROC: {am:.4f} +/- {asd:.4f} (mean +/- std over repeats)"
        )
        lines.append(
            f"Across {n_rep} repeats - mean-fold EER:  {em:.4f} +/- {esd:.4f}"
        )
        lines.append(
            f"Across {n_rep} repeats - OOF AUROC (full train, one score per sample): {oam:.4f} +/- {oasd:.4f}"
        )
        lines.append("OOF CM@0.5 - mean over repeats (rows=true 0/1, cols=pred 0/1):")
        for i in range(2):
            lines.append(
                "  "
                + "  ".join(f"{float(cm_mean[i, j]):.2f} +/- {float(cm_std[i, j]):.2f}" for j in range(2))
            )
    else:
        s0 = repeat_summaries[0]
        lines.append(
            f"mean AUROC (folds): {s0['mean_fold_auroc']:.4f}  +/- {s0['std_fold_auroc']:.4f}"
        )
        lines.append(
            f"mean EER (folds):  {s0['mean_fold_eer']:.4f}  +/- {s0['std_fold_eer']:.4f}"
        )
        lines.append(
            f"OOF AUROC (full train): {s0.get('oof_auroc_full', float('nan')):.4f}"
        )
        lines.append("OOF CM@0.5 (rows=true, cols=pred):")
        for i in range(2):
            lines.append("  " + "  ".join(f"{float(s0['oof_cm_0.5'][i][j]):.0f}" for j in range(2)))

    split_txt = out_dir / "d5_split_summary.txt"
    split_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if n_rep > 1:
        rep_lines: list[str] = []
        rep_lines.append(f"D5 cavitary - {args.n_splits}-fold x {n_rep} repeats (TRAIN pool)")
        rep_lines.append(f"base_seed={args.seed}  (repeat r uses seed + r*7919)")
        rep_lines.append("")
        rep_lines.append("--- Across repeats (mean +/- std) ---")
        rep_lines.append(f"Mean-fold AUROC: {am:.6f} +/- {asd:.6f}")
        rep_lines.append(f"Mean-fold EER:   {em:.6f} +/- {esd:.6f}")
        rep_lines.append(f"OOF AUROC:       {oam:.6f} +/- {oasd:.6f}")
        rep_lines.append("")
        rep_lines.append("OOF confusion @0.5 - cell mean +/- std (true=row, pred=col):")
        for i in range(2):
            rep_lines.append(
                "  "
                + "  ".join(f"{float(cm_mean[i, j]):.2f} +/- {float(cm_std[i, j]):.2f}" for j in range(2))
            )
        rep_lines.append("")
        rep_lines.append("--- Per repeat (mean of fold metrics) ---")
        for s in repeat_summaries:
            rep_lines.append(
                f"rep {s['repeat_index']}: AUROC={s['mean_fold_auroc']:.6f}  EER={s['mean_fold_eer']:.6f}  "
                f"oof_AUROC={s.get('oof_auroc_full', float('nan')):.6f}"
            )
        (out_dir / "d5_5x5_repeats_summary.txt").write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    json_path = out_dir / "d5_5fold_metrics.json"
    out_json: dict = {
        "p3_train_npz": str(train_p.resolve()),
        "p3_val_npz": str(val_p.resolve()),
        "p3_test_npz": str(test_p.resolve()),
        "n_train_val_test": [int(X_tr.shape[0]), int(X_va.shape[0]), int(X_te.shape[0])],
        "d5_pos_train_val_test": [int(y5_tr.sum()), int(y5_va.sum()), int(y5_te.sum())],
        "n_splits": int(args.n_splits),
        "n_repeats": n_rep,
        "base_seed": int(args.seed),
        "cv": repeat_summaries[0],
        "cv_first_repeat": repeat_summaries[0],
        "cv_repeats": repeat_summaries,
    }
    if n_rep > 1:
        out_json["across_repeats"] = {
            "mean_fold_auroc_mean": am,
            "mean_fold_auroc_std": asd,
            "mean_fold_eer_mean": em,
            "mean_fold_eer_std": esd,
            "oof_auroc_mean": oam,
            "oof_auroc_std": oasd,
            "oof_cm_mean": cm_mean.tolist(),
            "oof_cm_std": cm_std.tolist(),
        }
    json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_roc(
        base_fpr,
        tprs,
        mean_tpr,
        std_tpr,
        mean_aurocs,
        out_dir / "d5_5fold_roc.png",
    )
    plot_cm_grid(cms, out_dir / "d5_5fold_cm.png")
    plot_eer_bars(list(range(1, args.n_splits + 1)), eers, out_dir / "d5_5fold_eer.png")

    print("\n".join(lines))
    print("saved:", split_txt)
    if n_rep > 1:
        print("saved:", out_dir / "d5_5x5_repeats_summary.txt")
    print("saved:", json_path)
    print("saved:", out_dir / "d5_5fold_roc.png")
    print("saved:", out_dir / "d5_5fold_cm.png")
    print("saved:", out_dir / "d5_5fold_eer.png")

    done = out_dir / "d5_run_completed.txt"
    done.write_text(
        f"OK {datetime.now(timezone.utc).isoformat()}\n"
        f"out_dir: {out_dir.resolve()}\n"
        f"train_npz: {train_p}\n",
        encoding="utf-8",
    )
    print("saved:", done.resolve())

    if args.copy_to_artifact is not None:
        md: Path = args.copy_to_artifact
        md.mkdir(parents=True, exist_ok=True)
        names = [
            "d5_split_summary.txt",
            "d5_5fold_metrics.json",
            "d5_5fold_roc.png",
            "d5_5fold_cm.png",
            "d5_5fold_eer.png",
            "d5_run_completed.txt",
        ]
        if n_rep > 1:
            names.append("d5_5x5_repeats_summary.txt")
        for name in names:
            src = out_dir / name
            if src.is_file():
                shutil.copy2(src, md / name)
        print("also copied to:", md.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
