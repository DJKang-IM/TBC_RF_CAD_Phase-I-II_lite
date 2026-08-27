"""
End-to-end pipeline plots (no 1.021 / 1.022 / 1.024):

  Phase I  — v1.01 RF (TB Test DB artifacts)
  Phase II — v1.01 RF
  Phase III — v1.01 multi-output RF for D1–D4 + v1.023-style XGBoost for D6 only

Evaluation: concatenate val + test for each phase; 5× bootstrap resampling of the pool
to overlay mean ±1σ (TPR vs FPR) on top of the full-pool ROC (fixed trained models).

Outputs (single folder — Phase I / II / III PNGs together, no sub-split):
  <REDACTED_PATH> 1.023(D6_XGboost)/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, roc_curve
from xgboost import XGBClassifier

D6_INDEX = 4

# All PNGs for this pipeline + v1.023 D6 track go here (flat; no phase subfolders).
DEFAULT_ARTIFACT_VER = Path(r"D:\artifact") / "ver. 1.023(D6_XGboost)"


def ensure_output_dir(out_dir: Path) -> Path:
    """Create <REDACTED_PATH> and the version folder if missing; return resolved path."""
    p = out_dir.expanduser()
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        raise RuntimeError(f"Cannot create or access output directory: {p}")
    return p


class Phase3HybridV101D6XGB1023:
    """RF v1.01 heads for all five labels; replace D6 probabilities with XGBoost."""

    def __init__(self, rf: RandomForestClassifier, xgb_d6: XGBClassifier, d6_threshold: float = 0.5) -> None:
        self.rf = rf
        self.xgb_d6 = xgb_d6
        self.d6_threshold = float(d6_threshold)

    def predict_proba(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        base = self.rf.predict_proba(X)
        p6 = self.xgb_d6.predict_proba(X)
        out = list(base)
        out[D6_INDEX] = np.asarray(p6, dtype=float)
        return out


def build_xgb_d6_v1023(X_tr: np.ndarray, y6_tr: np.ndarray, seed: int) -> XGBClassifier:
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


def load_xy_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"] if "y" in d.files else d["Y"]
    y = np.asarray(y).astype(int).ravel()
    return X, y


def load_xy_multi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(np.float32), d["Y"].astype(int)


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    if y_true.max() == y_true.min():
        return {"eer": float("nan"), "threshold": float("nan"), "fpr": float("nan"), "fnr": float("nan")}
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


def tpr_on_grid(y: np.ndarray, s: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    if y.max() == y.min():
        return np.clip(fpr_grid, 0, 1)
    fpr, tpr, _ = roc_curve(y, s)
    return np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)


def auc_safe(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.max() == y.min():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    return float(auc(fpr, tpr))


def op_point_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float]:
    y = y.astype(int)
    yhat = (s >= thr).astype(int)
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    tp = int(np.sum((y == 1) & (yhat == 1)))
    n0, n1 = tn + fp, fn + tp
    return (float(fp / n0) if n0 else 0.0, float(tp / n1) if n1 else 0.0)


def cm_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> np.ndarray:
    y = y.astype(int)
    yhat = (s >= thr).astype(int)
    return confusion_matrix(y, yhat, labels=[0, 1]).astype(float)


def cm_at_eer_threshold(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float, float]:
    info = eer_from_scores(y, s)
    t = info["threshold"]
    if not np.isfinite(t):
        z = np.zeros((2, 2))
        return z, float("nan"), float("nan")
    return cm_at_threshold(y, s, t), t, info["eer"]


def plot_roc_reference_and_bootstrap(
    out_png: Path,
    title: str,
    y_pool: np.ndarray,
    s_pool: np.ndarray,
    boot_tpr: np.ndarray,
    boot_aucs: np.ndarray,
    fpr_grid: np.ndarray,
    thr_mark: float,
) -> None:
    mean_b = boot_tpr.mean(axis=0)
    std_b = boot_tpr.std(axis=0)
    fpr_f, tpr_f, _ = roc_curve(y_pool.astype(int), s_pool.astype(float))
    auc_full = float(auc(fpr_f, tpr_f))

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ax.fill_between(
        fpr_grid,
        np.clip(mean_b - std_b, 0, 1),
        np.clip(mean_b + std_b, 0, 1),
        alpha=0.28,
        color="C0",
        label="Bootstrap: mean TPR ±1σ (5× pool resample)",
    )
    ax.plot(fpr_grid, mean_b, color="C0", lw=1.8, ls="--", label="Bootstrap mean ROC")
    ax.plot(fpr_f, tpr_f, color="k", lw=2.3, label=f"Full pool ROC (AUROC={auc_full:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.85, ls=":", label="chance")

    fpr_op, tpr_op = op_point_at_threshold(y_pool, s_pool, thr_mark)
    ax.scatter(
        [fpr_op],
        [tpr_op],
        s=80,
        c="C1",
        zorder=6,
        edgecolors="k",
        linewidths=0.5,
        label=f"Full pool @ thr={thr_mark}",
    )

    m_auc = float(np.nanmean(boot_aucs))
    s_auc = float(np.nanstd(boot_aucs))
    ax.text(
        0.03,
        0.97,
        f"Bootstrap AUROC: {m_auc:.4f} ± {s_auc:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.28)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_cm_mean_std(out_png: Path, title: str, mean_cm: np.ndarray, std_cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 4.5), constrained_layout=True)
    im = ax.imshow(mean_cm, cmap="Blues", vmin=0, interpolation="nearest")
    vmax = float(mean_cm.max()) if mean_cm.size else 1.0
    mid = vmax / 2.0 if vmax > 0 else 0.5
    for i in range(2):
        for j in range(2):
            m, s = mean_cm[i, j], std_cm[i, j]
            ax.text(
                j,
                i,
                f"{m:.1f}\n±{s:.1f}",
                ha="center",
                va="center",
                color="white" if m > mid else "black",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title + "\n@ thr=0.5 · mean ± std (5× pool bootstrap)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_eer_table_mean_std(
    out_png: Path,
    title: str,
    mean_cm: np.ndarray,
    std_cm: np.ndarray,
    mean_thr: float,
    std_thr: float,
    mean_eer: float,
    std_eer: float,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.7), constrained_layout=True)
    im = ax.imshow(mean_cm, cmap="Greens", vmin=0, interpolation="nearest")
    labels = [["TN", "FP"], ["FN", "TP"]]
    vmax = float(mean_cm.max()) if mean_cm.size else 1.0
    mid = vmax / 2.0 if vmax > 0 else 0.5
    for i in range(2):
        for j in range(2):
            m, s = mean_cm[i, j], std_cm[i, j]
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{m:.1f}\n±{s:.1f}",
                ha="center",
                va="center",
                color="white" if m > mid else "black",
                fontsize=10,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(
        title + f"\n@ EER thr · mean ± std\nEER={mean_eer:.4f} ± {std_eer:.4f} · thr={mean_thr:.4f} ± {std_thr:.4f}",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def run_binary_phase(
    phase_title: str,
    tag: str,
    rf,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    out_dir: Path,
    n_boot: int,
    seed: int,
) -> None:
    _, _ = load_xy_binary(train_npz)
    X_va, y_va = load_xy_binary(val_npz)
    X_te, y_te = load_xy_binary(test_npz)
    X_pool = np.vstack([X_va, X_te])
    y_pool = np.concatenate([y_va, y_te])
    n = len(y_pool)
    rng = np.random.default_rng(seed)
    s_pool = rf.predict_proba(X_pool)[:, 1].astype(float)

    fpr_grid = np.linspace(0, 1, 101)
    boot_tpr = np.zeros((n_boot, len(fpr_grid)))
    boot_aucs = np.zeros(n_boot)
    cms_05: list[np.ndarray] = []
    cms_eer: list[np.ndarray] = []
    thrs_eer: list[float] = []
    eers: list[float] = []

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b, s_b = y_pool[idx], s_pool[idx]
        boot_tpr[b] = tpr_on_grid(y_b, s_b, fpr_grid)
        boot_aucs[b] = auc_safe(y_b, s_b)
        cms_05.append(cm_at_threshold(y_b, s_b, 0.5))
        cm_e, t_e, eer_v = cm_at_eer_threshold(y_b, s_b)
        cms_eer.append(cm_e)
        thrs_eer.append(t_e)
        eers.append(eer_v)

    plot_roc_reference_and_bootstrap(
        out_dir / f"{tag}_binary_auroc_roc_bootstrap5.png",
        f"{phase_title} · v1.01 RF · val+test pooled (n={n})",
        y_pool,
        s_pool,
        boot_tpr,
        boot_aucs,
        fpr_grid,
        0.5,
    )
    m05 = np.mean(np.stack(cms_05, axis=0), axis=0)
    s05 = np.std(np.stack(cms_05, axis=0), axis=0)
    plot_cm_mean_std(out_dir / f"{tag}_binary_confusion_thr05_meanstd_bootstrap5.png", phase_title, m05, s05)
    me = np.mean(np.stack(cms_eer, axis=0), axis=0)
    se = np.std(np.stack(cms_eer, axis=0), axis=0)
    plot_eer_table_mean_std(
        out_dir / f"{tag}_binary_eer_2x2_meanstd_bootstrap5.png",
        phase_title,
        me,
        se,
        float(np.nanmean(thrs_eer)),
        float(np.nanstd(thrs_eer)),
        float(np.nanmean(eers)),
        float(np.nanstd(eers)),
    )
    print("saved:", tag)


def run_phase3_hybrid(
    hybrid: Phase3HybridV101D6XGB1023,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    out_dir: Path,
    n_boot: int,
    seed: int,
) -> None:
    _, _ = load_xy_multi(train_npz)
    X_va, Y_va = load_xy_multi(val_npz)
    X_te, Y_te = load_xy_multi(test_npz)
    X_pool = np.vstack([X_va, X_te])
    Y_pool = np.vstack([Y_va, Y_te])
    n = Y_pool.shape[0]
    rng = np.random.default_rng(seed)
    probas = hybrid.predict_proba(X_pool)
    keys = ["D1", "D2", "D3", "D4", "D6"]
    names = ["AFB(D1)", "TB PCR(D2)", "Solid culture(D3)", "Liquid culture(D4)", "NTM(D6)"]
    fpr_grid = np.linspace(0, 1, 101)

    for j, (key, human) in enumerate(zip(keys, names)):
        y_pool = Y_pool[:, j]
        if y_pool.max() == y_pool.min():
            print("skip single-class", key)
            continue
        s_pool = probas[j][:, 1].astype(float)
        boot_tpr = np.zeros((n_boot, len(fpr_grid)))
        boot_aucs = np.zeros(n_boot)
        cms_05: list[np.ndarray] = []
        cms_eer: list[np.ndarray] = []
        thrs_eer: list[float] = []
        eers: list[float] = []
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            y_b, s_b = y_pool[idx], s_pool[idx]
            boot_tpr[b] = tpr_on_grid(y_b, s_b, fpr_grid)
            boot_aucs[b] = auc_safe(y_b, s_b)
            cms_05.append(cm_at_threshold(y_b, s_b, 0.5))
            cm_e, t_e, eer_v = cm_at_eer_threshold(y_b, s_b)
            cms_eer.append(cm_e)
            thrs_eer.append(t_e)
            eers.append(eer_v)

        d6_note = " · D6 = v1.023 XGBoost" if j == D6_INDEX else " · D1–D4 = v1.01 RF"
        phase_title = f"Phase III · {key} ({human}){d6_note}"
        tag = f"phase3_{key}_{human.replace(' ', '_').replace('/', '-')}"
        plot_roc_reference_and_bootstrap(
            out_dir / f"{tag}_auroc_roc_bootstrap5.png",
            f"{phase_title}\nval+test pooled (n={n})",
            y_pool,
            s_pool,
            boot_tpr,
            boot_aucs,
            fpr_grid,
            0.5,
        )
        m05 = np.mean(np.stack(cms_05, axis=0), axis=0)
        s05 = np.std(np.stack(cms_05, axis=0), axis=0)
        plot_cm_mean_std(out_dir / f"{tag}_confusion_thr05_meanstd_bootstrap5.png", phase_title, m05, s05)
        me = np.mean(np.stack(cms_eer, axis=0), axis=0)
        se = np.std(np.stack(cms_eer, axis=0), axis=0)
        plot_eer_table_mean_std(
            out_dir / f"{tag}_eer_2x2_meanstd_bootstrap5.png",
            phase_title,
            me,
            se,
            float(np.nanmean(thrs_eer)),
            float(np.nanstd(thrs_eer)),
            float(np.nanmean(eers)),
            float(np.nanstd(eers)),
        )
        print("saved:", tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_root", type=Path, default=Path(r"D:\TB Test DB"))
    ap.add_argument("--phase3_root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=r"Default: D:\artifact\ver. 1.023(D6_XGboost)",
    )
    ap.add_argument("--n_boot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260419)
    args = ap.parse_args()

    tb = args.tb_root
    p3 = args.phase3_root
    art_tb = tb / "artifacts"
    out_dir = ensure_output_dir(args.out_dir or DEFAULT_ARTIFACT_VER)
    print("output_dir (created if needed):", out_dir.resolve())

    rf1 = joblib.load(art_tb / "rf_phase1_huge.joblib")
    run_binary_phase(
        "Phase I (v1.01 RF)",
        "phase1",
        rf1,
        art_tb / "phase1_features_train.npz",
        art_tb / "phase1_features_val.npz",
        art_tb / "phase1_features_test.npz",
        out_dir,
        args.n_boot,
        args.seed,
    )

    rf2 = joblib.load(art_tb / "rf_phase2_huge.joblib")
    run_binary_phase(
        "Phase II (v1.01 RF)",
        "phase2",
        rf2,
        art_tb / "phase2_features_train.npz",
        art_tb / "phase2_features_val.npz",
        art_tb / "phase2_features_test.npz",
        out_dir,
        args.n_boot,
        args.seed + 1000,
    )

    rf3 = joblib.load(p3 / "rf_phase3_active_vs_inactive.joblib")
    X_tr, Y_tr = load_xy_multi(p3 / "phase3_features_train.npz")
    xgb = build_xgb_d6_v1023(X_tr, Y_tr[:, D6_INDEX], seed=42)
    xgb.fit(X_tr, Y_tr[:, D6_INDEX])
    hybrid = Phase3HybridV101D6XGB1023(rf3, xgb)

    run_phase3_hybrid(
        hybrid,
        p3 / "phase3_features_train.npz",
        p3 / "phase3_features_val.npz",
        p3 / "phase3_features_test.npz",
        out_dir,
        args.n_boot,
        args.seed + 2000,
    )
    print("done →", out_dir)


if __name__ == "__main__":
    main()
