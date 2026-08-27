# -*- coding: utf-8 -*-
"""
Phase III D6 (NTM): RF vs XGBoost figures (no logistic regression).

Writes separate PNGs for readability:
  - phase3_d6_ntm_roc_only.png       ROC only (legend below figure, no box on curve)
  - phase3_d6_ntm_cm_heatmaps_only.png  confusion heatmaps only

Default: **five repeated random splits** (same sizes as phase3_features train/val/test pool),
ROC from **raw** test scores with mean TPR ±1 SD band; legend reports **mean ± SD** AUROC & EER
per method (fold-wise). CM @0.5 with **train min-max per fold** (same as eval_d6_version_1021_1022),
cells show **mean ± SD** counts over folds.

Use --canonical-split for the legacy single fixed split figure (no deviation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import eval_d6_version_1021_1022 as d6base
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from xgboost import XGBClassifier

D6_INDEX = 4


def _annot_color(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _a = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else "black"


def fit_xgb_scores(X_tr, y_tr, X_va, X_te, seed: int):
    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = float(neg / max(pos, 1))
    clf = XGBClassifier(
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
    clf.fit(X_tr, y_tr)
    return (
        clf.predict_proba(X_tr)[:, 1].astype(float),
        clf.predict_proba(X_va)[:, 1].astype(float),
        clf.predict_proba(X_te)[:, 1].astype(float),
    )


def eer_point(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(fpr[i]), float(tpr[i])


def eer_float(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(d6base.eer_from_scores(y_true, scores)["eer"])


def _ms(a: list[float]) -> tuple[float, float]:
    x = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(x.std(ddof=1))


def cm_heatmap(ax, cm, title):
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, interpolation="nearest")
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
                fontsize=16,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred non-NTM", "Pred NTM"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True non-NTM", "True NTM"], fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="semibold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def cm_heatmap_mean_std(ax, mean_m: np.ndarray, std_m: np.ndarray, title: str) -> None:
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
            fontsize=14,
            fontweight="bold",
            color=_annot_color(face),
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred non-NTM", "Pred NTM"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True non-NTM", "True NTM"], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="semibold")


def _plot_roc_canonical(ax_r, y_eval, pairs, colors: dict, title: str) -> None:
    for label, s_tr, s_eval, key in pairs:
        fpr, tpr, _ = roc_curve(y_eval, s_eval)
        a = auc(fpr, tpr)
        eer = d6base.eer_from_scores(y_eval, s_eval)["eer"]
        c = colors[key]
        ax_r.plot(fpr, tpr, lw=2.6, color=c, label=f"{label}: AUROC={a:.3f}, EER={eer:.3f}")
        fx, ty = eer_point(y_eval, s_eval)
        ax_r.scatter([fx], [ty], s=85, color=c, edgecolors="white", linewidths=1.3, zorder=6)

    ax_r.plot([0, 1], [0, 1], ":", color="0.45", lw=1.2, label="Chance")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1.02)
    ax_r.set_xlabel("False positive rate", fontsize=12)
    ax_r.set_ylabel("True positive rate", fontsize=12)
    ax_r.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax_r.grid(True, alpha=0.28)
    ax_r.set_aspect("equal", adjustable="box")

    h_roc, l_roc = ax_r.get_legend_handles_labels()
    extra_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["RF"],
            markeredgecolor="white",
            markersize=10,
            linestyle="None",
            label="EER (RF)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors["XGB"],
            markeredgecolor="white",
            markersize=10,
            linestyle="None",
            label="EER (XGBoost)",
        ),
    ]
    ax_r.legend(
        handles=h_roc + extra_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--canonical-split",
        action="store_true",
        help="Single fixed train/val/test (legacy); no mean±SD on figure.",
    )
    ap.add_argument("--cv-runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "artifacts" / "d6_rf_vs_xgb_figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    d_tr = np.load(root / "phase3_features_train.npz", allow_pickle=True)
    d_va = np.load(root / "phase3_features_val.npz", allow_pickle=True)
    d_te = np.load(root / "phase3_features_test.npz", allow_pickle=True)
    X_tr0 = d_tr["X"].astype(np.float32)
    Y_tr0 = d_tr["Y"].astype(int)
    X_va0 = d_va["X"].astype(np.float32)
    Y_va0 = d_va["Y"].astype(int)
    X_te0 = d_te["X"].astype(np.float32)
    Y_te0 = d_te["Y"].astype(int)
    j = D6_INDEX

    rf_template: RandomForestClassifier = joblib.load(root / "rf_phase3_active_vs_inactive.joblib")
    colors = {"RF": "#1565C0", "XGB": "#C62828"}

    if args.canonical_split:
        y_tr, y_va, y_te = Y_tr0[:, j], Y_va0[:, j], Y_te0[:, j]
        s_rf_tr = rf_template.predict_proba(X_tr0)[j][:, 1].astype(float)
        s_rf_va = rf_template.predict_proba(X_va0)[j][:, 1].astype(float)
        s_rf_te = rf_template.predict_proba(X_te0)[j][:, 1].astype(float)
        s_xgb_tr, s_xgb_va, s_xgb_te = fit_xgb_scores(X_tr0, y_tr, X_va0, X_te0, seed=42)
        y_eval = y_te
        pairs = (
            ("RF (multi-output, NTM head)", s_rf_tr, s_rf_te, "RF"),
            ("XGBoost (binary NTM)", s_xgb_tr, s_xgb_te, "XGB"),
        )
        fig_r, ax_r = plt.subplots(figsize=(7.2, 6.2), dpi=args.dpi, facecolor="white")
        _plot_roc_canonical(
            ax_r,
            y_eval,
            pairs,
            colors,
            f"Phase III: NTM vs non-NTM (test, n={len(y_eval)})",
        )
        fig_r.subplots_adjust(bottom=0.28)
        out_roc = out_dir / "phase3_d6_ntm_roc_only.png"
        fig_r.savefig(out_roc, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig_r)

        fig_h, axes_h = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=args.dpi, facecolor="white")
        cm_rf = d6base.cm_at_half(y_eval, s_rf_tr, s_rf_te)
        cm_xgb = d6base.cm_at_half(y_eval, s_xgb_tr, s_xgb_te)
        cm_heatmap(axes_h[0], cm_rf, "Random Forest\nconfusion @ 0.5 (train min-max)")
        cm_heatmap(axes_h[1], cm_xgb, "XGBoost\nconfusion @ 0.5 (train min-max)")
        fig_h.suptitle(
            f"Phase III: NTM vs non-NTM (test, n={len(y_eval)})",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig_h.tight_layout()
        out_cm = out_dir / "phase3_d6_ntm_cm_heatmaps_only.png"
        fig_h.savefig(out_cm, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig_h)

        auc_rf_te = auc(*roc_curve(y_te, s_rf_te)[:2])
        auc_xgb_te = auc(*roc_curve(y_te, s_xgb_te)[:2])
        eer_rf_te = d6base.eer_from_scores(y_te, s_rf_te)["eer"]
        eer_xgb_te = d6base.eer_from_scores(y_te, s_xgb_te)["eer"]
        auc_rf_va = auc(*roc_curve(y_va, s_rf_va)[:2])
        auc_xgb_va = auc(*roc_curve(y_va, s_xgb_va)[:2])
        eer_rf_va = d6base.eer_from_scores(y_va, s_rf_va)["eer"]
        eer_xgb_va = d6base.eer_from_scores(y_va, s_xgb_va)["eer"]

        legend_txt = (
            "Figure legend (Phase III NTM). CANONICAL single split. "
            "Files: phase3_d6_ntm_roc_only.png; phase3_d6_ntm_cm_heatmaps_only.png. "
            "RF: NTM score from rf_phase3_active_vs_inactive.joblib. "
            "XGBoost: binary head (hist, scale_pos_weight). ROC: raw test scores. "
            "CM: train min-max, threshold 0.5. "
            f"Test: RF AUROC {auc_rf_te:.3f}, EER {eer_rf_te:.3f}; XGB AUROC {auc_xgb_te:.3f}, EER {eer_xgb_te:.3f}. "
            f"Val (n={len(y_va)}): RF AUROC {auc_rf_va:.3f}, EER {eer_rf_va:.3f}; "
            f"XGB AUROC {auc_xgb_va:.3f}, EER {eer_xgb_va:.3f}."
        )
        (out_dir / "phase3_d6_ntm_rf_vs_xgboost_figure_legend.txt").write_text(legend_txt + "\n", encoding="utf-8")
        print("saved:", out_roc.resolve())
        print("saved:", out_cm.resolve())
        return

    # -------- Five repeated splits (default): mean ± SD on test --------
    n_tr, n_va, n_te = int(X_tr0.shape[0]), int(X_va0.shape[0]), int(X_te0.shape[0])
    X_pool = np.concatenate([X_tr0, X_va0, X_te0], axis=0)
    Y_pool = np.concatenate([Y_tr0, Y_va0, Y_te0], axis=0)
    n_total = int(X_pool.shape[0])
    if n_total != n_tr + n_va + n_te:
        raise RuntimeError("pool size mismatch")

    base_fpr = np.linspace(0, 1, 101)
    tprs_rf: list[np.ndarray] = []
    tprs_xgb: list[np.ndarray] = []
    aucs_rf: list[float] = []
    aucs_xgb: list[float] = []
    eers_rf: list[float] = []
    eers_xgb: list[float] = []
    eer_xy_rf: list[tuple[float, float]] = []
    eer_xy_xgb: list[tuple[float, float]] = []
    cms_rf: list[np.ndarray] = []
    cms_xgb: list[np.ndarray] = []
    aucs_rf_va: list[float] = []
    aucs_xgb_va: list[float] = []
    eers_rf_va: list[float] = []
    eers_xgb_va: list[float] = []

    rng_master = np.random.default_rng(int(args.seed))
    for _fold in range(int(args.cv_runs)):
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
        y_tr = Y_tr[:, j]
        y_va = Y_va[:, j]
        y_te = Y_te[:, j]

        rf_fold = clone(rf_template)
        rf_fold.set_params(random_state=run_seed)
        rf_fold.fit(X_tr, Y_tr)
        s_rf_tr = rf_fold.predict_proba(X_tr)[D6_INDEX][:, 1].astype(float)
        s_rf_va = rf_fold.predict_proba(X_va)[D6_INDEX][:, 1].astype(float)
        s_rf_te = rf_fold.predict_proba(X_te)[D6_INDEX][:, 1].astype(float)

        s_xgb_tr, s_xgb_va, s_xgb_te = fit_xgb_scores(X_tr, y_tr, X_va, X_te, seed=run_seed)

        if y_te.min() != y_te.max():
            fpr_r, tpr_r, _ = roc_curve(y_te, s_rf_te)
            tprs_rf.append(np.interp(base_fpr, fpr_r, tpr_r))
            aucs_rf.append(float(auc(fpr_r, tpr_r)))
            eers_rf.append(eer_float(y_te, s_rf_te))
            eer_xy_rf.append(eer_point(y_te, s_rf_te))

            fpr_x, tpr_x, _ = roc_curve(y_te, s_xgb_te)
            tprs_xgb.append(np.interp(base_fpr, fpr_x, tpr_x))
            aucs_xgb.append(float(auc(fpr_x, tpr_x)))
            eers_xgb.append(eer_float(y_te, s_xgb_te))
            eer_xy_xgb.append(eer_point(y_te, s_xgb_te))
        else:
            tprs_rf.append(np.full_like(base_fpr, np.nan))
            tprs_xgb.append(np.full_like(base_fpr, np.nan))
            aucs_rf.append(float("nan"))
            aucs_xgb.append(float("nan"))
            eers_rf.append(float("nan"))
            eers_xgb.append(float("nan"))
            eer_xy_rf.append((float("nan"), float("nan")))
            eer_xy_xgb.append((float("nan"), float("nan")))

        if y_va.min() != y_va.max():
            aucs_rf_va.append(float(auc(*roc_curve(y_va, s_rf_va)[:2])))
            aucs_xgb_va.append(float(auc(*roc_curve(y_va, s_xgb_va)[:2])))
            eers_rf_va.append(eer_float(y_va, s_rf_va))
            eers_xgb_va.append(eer_float(y_va, s_xgb_va))
        else:
            aucs_rf_va.append(float("nan"))
            aucs_xgb_va.append(float("nan"))
            eers_rf_va.append(float("nan"))
            eers_xgb_va.append(float("nan"))

        cms_rf.append(d6base.cm_at_half(y_te, s_rf_tr, s_rf_te))
        cms_xgb.append(d6base.cm_at_half(y_te, s_xgb_tr, s_xgb_te))

    mean_rf = np.nanmean(np.vstack(tprs_rf), axis=0)
    std_rf = np.nanstd(np.vstack(tprs_rf), axis=0)
    mean_xg = np.nanmean(np.vstack(tprs_xgb), axis=0)
    std_xg = np.nanstd(np.vstack(tprs_xgb), axis=0)

    m_rf_au, s_rf_au = _ms(aucs_rf)
    m_xg_au, s_xg_au = _ms(aucs_xgb)
    m_rf_ee, s_rf_ee = _ms(eers_rf)
    m_xg_ee, s_xg_ee = _ms(eers_xgb)
    m_rf_au_v, s_rf_au_v = _ms(aucs_rf_va)
    m_xg_au_v, s_xg_au_v = _ms(aucs_xgb_va)
    m_rf_ee_v, s_rf_ee_v = _ms(eers_rf_va)
    m_xg_ee_v, s_xg_ee_v = _ms(eers_xgb_va)

    mx_rf_f = float(np.nanmean([p[0] for p in eer_xy_rf]))
    mx_rf_t = float(np.nanmean([p[1] for p in eer_xy_rf]))
    mx_xg_f = float(np.nanmean([p[0] for p in eer_xy_xgb]))
    mx_xg_t = float(np.nanmean([p[1] for p in eer_xy_xgb]))

    fig_r, ax_r = plt.subplots(figsize=(7.4, 6.4), dpi=args.dpi, facecolor="white")
    c_rf, c_xg = colors["RF"], colors["XGB"]
    ax_r.plot(base_fpr, mean_rf, lw=2.8, color=c_rf, label=f"RF (NTM head): AUROC={m_rf_au:.3f}±{s_rf_au:.3f}, EER={m_rf_ee:.3f}±{s_rf_ee:.3f}")
    ax_r.fill_between(
        base_fpr,
        np.clip(mean_rf - std_rf, 0, 1),
        np.clip(mean_rf + std_rf, 0, 1),
        color=c_rf,
        alpha=0.22,
    )
    ax_r.plot(
        base_fpr,
        mean_xg,
        lw=2.8,
        color=c_xg,
        label=f"XGBoost (binary NTM): AUROC={m_xg_au:.3f}±{s_xg_au:.3f}, EER={m_xg_ee:.3f}±{s_xg_ee:.3f}",
    )
    ax_r.fill_between(
        base_fpr,
        np.clip(mean_xg - std_xg, 0, 1),
        np.clip(mean_xg + std_xg, 0, 1),
        color=c_xg,
        alpha=0.22,
    )
    ax_r.scatter([mx_rf_f], [mx_rf_t], s=90, color=c_rf, edgecolors="white", linewidths=1.3, zorder=6)
    ax_r.scatter([mx_xg_f], [mx_xg_t], s=90, color=c_xg, edgecolors="white", linewidths=1.3, zorder=6)

    ax_r.plot([0, 1], [0, 1], ":", color="0.45", lw=1.2, label="Chance")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1.02)
    ax_r.set_xlabel("False positive rate", fontsize=12)
    ax_r.set_ylabel("True positive rate", fontsize=12)
    ax_r.set_title(
        f"Phase III: NTM vs non-NTM (test, {args.cv_runs} splits · mean ± SD)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax_r.grid(True, alpha=0.28)
    ax_r.set_aspect("equal", adjustable="box")

    h_roc, _ = ax_r.get_legend_handles_labels()
    extra_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c_rf,
            markeredgecolor="white",
            markersize=10,
            linestyle="None",
            label="Mean EER point (RF)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c_xg,
            markeredgecolor="white",
            markersize=10,
            linestyle="None",
            label="Mean EER point (XGB)",
        ),
        Line2D([0], [0], color=c_rf, lw=6, alpha=0.35, label="±1 SD (TPR over splits)"),
    ]
    ax_r.legend(
        handles=h_roc + extra_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
    )
    fig_r.subplots_adjust(bottom=0.32)
    out_roc = out_dir / "phase3_d6_ntm_roc_only.png"
    fig_r.savefig(out_roc, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig_r)

    stack_rf = np.stack(cms_rf, axis=0).astype(float)
    stack_xg = np.stack(cms_xgb, axis=0).astype(float)
    mean_rf_cm = stack_rf.mean(axis=0)
    std_rf_cm = stack_rf.std(axis=0, ddof=1) if stack_rf.shape[0] > 1 else np.zeros_like(mean_rf_cm)
    mean_xg_cm = stack_xg.mean(axis=0)
    std_xg_cm = stack_xg.std(axis=0, ddof=1) if stack_xg.shape[0] > 1 else np.zeros_like(mean_xg_cm)

    fig_h, axes_h = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=args.dpi, facecolor="white")
    cm_heatmap_mean_std(
        axes_h[0],
        mean_rf_cm,
        std_rf_cm,
        f"Random Forest · test CM @ 0.5\n(train min-max) · mean ± SD ({args.cv_runs} runs)",
    )
    cm_heatmap_mean_std(
        axes_h[1],
        mean_xg_cm,
        std_xg_cm,
        f"XGBoost · test CM @ 0.5\n(train min-max) · mean ± SD ({args.cv_runs} runs)",
    )
    fig_h.suptitle(
        f"Phase III: NTM vs non-NTM (test, {args.cv_runs} repeated splits)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig_h.tight_layout()
    out_cm = out_dir / "phase3_d6_ntm_cm_heatmaps_only.png"
    fig_h.savefig(out_cm, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig_h)

    legend_txt = (
        "Figure legend (Phase III NTM). "
        f"{args.cv_runs} repeated random splits; same pool sizes as phase3_features train/val/test. "
        "RF: multi-output RF refit each run; NTM score = RF NTM channel (raw ROC). "
        "XGBoost: binary NTM head refit each run (hist, scale_pos_weight). "
        "ROC: mean TPR(FPR) ±1 SD over runs; legend = fold-wise mean ± SD AUROC & EER on test. "
        "EER markers = mean (FPR,TPR) at fold-wise EER. "
        "CM: min-max on that run’s train scores, threshold 0.5; cells = mean ± SD counts over runs. "
        f"Test AUROC mean±SD: RF {m_rf_au:.3f}±{s_rf_au:.3f}, XGB {m_xg_au:.3f}±{s_xg_au:.3f}. "
        f"Test EER mean±SD: RF {m_rf_ee:.3f}±{s_rf_ee:.3f}, XGB {m_xg_ee:.3f}±{s_xg_ee:.3f}. "
        f"Val AUROC mean±SD: RF {m_rf_au_v:.3f}±{s_rf_au_v:.3f}, XGB {m_xg_au_v:.3f}±{s_xg_au_v:.3f}. "
        f"Val EER mean±SD: RF {m_rf_ee_v:.3f}±{s_rf_ee_v:.3f}, XGB {m_xg_ee_v:.3f}±{s_xg_ee_v:.3f}."
    )
    (out_dir / "phase3_d6_ntm_rf_vs_xgboost_figure_legend.txt").write_text(legend_txt + "\n", encoding="utf-8")

    print("saved:", out_roc.resolve())
    print("saved:", out_cm.resolve())


if __name__ == "__main__":
    main()
