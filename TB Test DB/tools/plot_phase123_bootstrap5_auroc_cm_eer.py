"""
Phase I, II (binary) and Phase III (5 labels): for each task draw PNGs:

1) ROC / AUROC: model score vs logistic regression (LR refit on 5 bootstrap samples
   of *train*, evaluated on fixed val+test pool). Shaded band = ±1 std of TPR over
   FPR grid for each method. Threshold 0.5 operating point marked on full pool.

2) Confusion matrix @ threshold 0.5: mean ± std of cell counts over 5 bootstrap
   resamples of the val+test pool.

3) EER: 2×2 table (TN, FP, FN, TP) at EER-derived threshold per bootstrap on pool;
   cells show mean ± std.

Outputs (default): <tb_root>/artifacts/plots_phase123_bootstrap5/{phase1,phase2,phase3}/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve


def _annotate_text_color_for_heatmap(
    *,
    rgba: tuple[float, float, float, float],
    dark_text: str = "black",
    light_text: str = "white",
) -> str:
    """Pick black vs white annotation text based on the cell facecolor luminance."""
    r, g, b, _a = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return light_text if lum < 0.55 else dark_text


def eer_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_true = y_true.astype(int)
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


def load_xy_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    if "y" in d.files:
        y = np.asarray(d["y"])
    elif "Y" in d.files:
        y = np.asarray(d["Y"])
    else:
        raise KeyError(f"No y/Y in {path}")
    y = y.astype(int).ravel()
    return X, y


def load_xy_multi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return d["X"].astype(np.float32), d["Y"].astype(int)


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
    n0 = tn + fp
    n1 = fn + tp
    fpr = fp / n0 if n0 else 0.0
    tpr = tp / n1 if n1 else 0.0
    return float(fpr), float(tpr)


def cm_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> np.ndarray:
    y = y.astype(int)
    yhat = (s >= thr).astype(int)
    return confusion_matrix(y, yhat, labels=[0, 1]).astype(float)


def cm_at_eer_threshold(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float, float]:
    info = eer_from_scores(y, s)
    t = info["threshold"]
    return cm_at_threshold(y, s, t), t, info["eer"]


def plot_roc_bundle(
    out_png: Path,
    title: str,
    y_pool: np.ndarray,
    rf_pool: np.ndarray,
    rf_boot_tpr: np.ndarray,
    rf_aucs: np.ndarray,
    lr_pool: np.ndarray,
    lr_boot_tpr: np.ndarray,
    lr_aucs: np.ndarray,
    fpr_grid: np.ndarray,
    thr_mark: float,
) -> None:
    mean_rf = rf_boot_tpr.mean(axis=0)
    std_rf = rf_boot_tpr.std(axis=0)
    mean_lr = lr_boot_tpr.mean(axis=0)
    std_lr = lr_boot_tpr.std(axis=0)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    ax.fill_between(
        fpr_grid,
        np.clip(mean_rf - std_rf, 0, 1),
        np.clip(mean_rf + std_rf, 0, 1),
        alpha=0.25,
        color="C0",
        label="Model: mean TPR ±1σ (pool bootstrap)",
    )
    ax.plot(fpr_grid, mean_rf, color="C0", lw=2.2, label="Model: mean ROC")

    ax.fill_between(
        fpr_grid,
        np.clip(mean_lr - std_lr, 0, 1),
        np.clip(mean_lr + std_lr, 0, 1),
        alpha=0.22,
        color="C1",
        label="LR: mean TPR ±1σ (train-refit bootstrap)",
    )
    ax.plot(fpr_grid, mean_lr, color="C1", lw=2.0, ls="--", label="LR: mean ROC")

    ax.plot([0, 1], [0, 1], color="gray", lw=0.9, ls=":", label="chance")

    fpr_op, tpr_op = op_point_at_threshold(y_pool, rf_pool, thr_mark)
    ax.scatter([fpr_op], [tpr_op], s=85, c="C2", zorder=6, edgecolors="k", linewidths=0.6, label=f"Model @ thr={thr_mark}")

    def auc_txt(name: str, xs: np.ndarray) -> str:
        m = float(np.nanmean(xs))
        sd = float(np.nanstd(xs))
        return f"{name} AUROC: {m:.4f} ± {sd:.4f}"

    ax.text(
        0.03,
        0.97,
        auc_txt("Model", rf_aucs) + "\n" + auc_txt("LR", lr_aucs),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.28)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_cm_mean_std(out_png: Path, title: str, mean_cm: np.ndarray, std_cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    vmax = float(mean_cm.max()) if mean_cm.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    im = ax.imshow(mean_cm, cmap="Blues", norm=norm, interpolation="nearest")
    cmap = im.cmap
    for i in range(2):
        for j in range(2):
            m = mean_cm[i, j]
            s = std_cm[i, j]
            txt = f"{m:.1f}\n±{s:.1f}"
            face = cmap(norm(float(m)))
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=_annotate_text_color_for_heatmap(rgba=face),
                fontsize=13,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + "\nConfusion @ thr=0.5 · mean ± std (5× pool bootstrap)", fontsize=10)
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
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    labels = [["TN", "FP"], ["FN", "TP"]]
    vmax = float(mean_cm.max()) if mean_cm.size else 1.0
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    im = ax.imshow(mean_cm, cmap="Greens", norm=norm, interpolation="nearest")
    cmap = im.cmap
    for i in range(2):
        for j in range(2):
            m = mean_cm[i, j]
            s = std_cm[i, j]
            lab = labels[i][j]
            txt = f"{lab}\n{m:.1f}\n±{s:.1f}"
            face = cmap(norm(float(m)))
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=_annotate_text_color_for_heatmap(rgba=face),
                fontsize=11,
                fontweight="bold",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(
        title + f"\n@ EER threshold · mean ± std (5× pool bootstrap)\n"
        f"EER = {mean_eer:.4f} ± {std_eer:.4f} · thr = {mean_thr:.4f} ± {std_thr:.4f}",
        fontsize=9.5,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def run_binary_phase(
    phase_name: str,
    out_sub: Path,
    rf_path: Path,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    dim_slug: str,
    n_boot: int,
    base_seed: int,
) -> None:
    rf = joblib.load(rf_path)
    X_tr, y_tr = load_xy_binary(train_npz)
    X_va, y_va = load_xy_binary(val_npz)
    X_te, y_te = load_xy_binary(test_npz)
    X_pool = np.vstack([X_va, X_te])
    y_pool = np.concatenate([y_va, y_te])
    n = len(y_pool)
    rng_pool = np.random.default_rng(base_seed)
    rng_tr = np.random.default_rng(base_seed + 17)

    rf_pool = rf.predict_proba(X_pool)[:, 1].astype(float)

    fpr_grid = np.linspace(0, 1, 101)
    rf_boot_tpr = np.zeros((n_boot, len(fpr_grid)))
    rf_aucs = np.zeros(n_boot)
    cms_05 = []
    cms_eer = []
    thrs_eer = []
    eers = []

    for b in range(n_boot):
        idx = rng_pool.choice(n, size=n, replace=True)
        y_b, s_b = y_pool[idx], rf_pool[idx]
        rf_boot_tpr[b] = tpr_on_grid(y_b, s_b, fpr_grid)
        rf_aucs[b] = auc_safe(y_b, s_b)
        cms_05.append(cm_at_threshold(y_b, s_b, 0.5))
        cm_e, t_e, eer_v = cm_at_eer_threshold(y_b, s_b)
        cms_eer.append(cm_e)
        thrs_eer.append(t_e)
        eers.append(eer_v)

    lr_boot_tpr = np.zeros((n_boot, len(fpr_grid)))
    lr_aucs = np.zeros(n_boot)
    n_tr = len(y_tr)
    for b in range(n_boot):
        idx_tr = rng_tr.choice(n_tr, size=n_tr, replace=True)
        lr = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=base_seed + b,
            solver="lbfgs",
        )
        lr.fit(X_tr[idx_tr], y_tr[idx_tr])
        lr_pool_b = lr.predict_proba(X_pool)[:, 1].astype(float)
        lr_boot_tpr[b] = tpr_on_grid(y_pool, lr_pool_b, fpr_grid)
        lr_aucs[b] = auc_safe(y_pool, lr_pool_b)

    lr_ref = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr_ref.fit(X_tr, y_tr)
    lr_pool = lr_ref.predict_proba(X_pool)[:, 1].astype(float)

    tag = f"{phase_name}_{dim_slug}"
    plot_roc_bundle(
        out_sub / f"{tag}_auroc_roc_lr_bootstrap{n_boot}.png",
        f"{phase_name} · {dim_slug}\nVal+Test pooled (n={n})",
        y_pool,
        rf_pool,
        rf_boot_tpr,
        rf_aucs,
        lr_pool,
        lr_boot_tpr,
        lr_aucs,
        fpr_grid,
        0.5,
    )
    m05 = np.mean(np.stack(cms_05, axis=0), axis=0)
    s05 = np.std(np.stack(cms_05, axis=0), axis=0)
    plot_cm_mean_std(
        out_sub / f"{tag}_confusion_thr05_meanstd_bootstrap{n_boot}.png",
        f"{phase_name} · {dim_slug}",
        m05,
        s05,
    )
    me = np.mean(np.stack(cms_eer, axis=0), axis=0)
    se = np.std(np.stack(cms_eer, axis=0), axis=0)
    plot_eer_table_mean_std(
        out_sub / f"{tag}_eer_2x2_meanstd_bootstrap{n_boot}.png",
        f"{phase_name} · {dim_slug}",
        me,
        se,
        float(np.mean(thrs_eer)),
        float(np.std(thrs_eer)),
        float(np.mean(eers)),
        float(np.std(eers)),
    )
    print("saved:", out_sub / f"{tag}_auroc_roc_lr_bootstrap{n_boot}.png")


def run_phase3(
    out_sub: Path,
    rf_path: Path,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    n_boot: int,
    base_seed: int,
    keys: list[str],
    names: list[str],
) -> None:
    rf = joblib.load(rf_path)
    X_tr, Y_tr = load_xy_multi(train_npz)
    X_va, Y_va = load_xy_multi(val_npz)
    X_te, Y_te = load_xy_multi(test_npz)
    X_pool = np.vstack([X_va, X_te])
    Y_pool = np.vstack([Y_va, Y_te])
    n = Y_pool.shape[0]
    rng_pool = np.random.default_rng(base_seed)
    rng_tr = np.random.default_rng(base_seed + 99)

    probas = rf.predict_proba(X_pool)

    fpr_grid = np.linspace(0, 1, 101)

    for j, (key, human) in enumerate(zip(keys, names)):
        y_tr = Y_tr[:, j]
        y_pool = Y_pool[:, j]
        if y_pool.max() == y_pool.min():
            print("skip single-class:", key)
            continue

        rf_pool = probas[j][:, 1].astype(float)

        rf_boot_tpr = np.zeros((n_boot, len(fpr_grid)))
        rf_aucs = np.zeros(n_boot)
        cms_05 = []
        cms_eer = []
        thrs_eer = []
        eers = []

        for b in range(n_boot):
            idx = rng_pool.choice(n, size=n, replace=True)
            y_b, s_b = y_pool[idx], rf_pool[idx]
            rf_boot_tpr[b] = tpr_on_grid(y_b, s_b, fpr_grid)
            rf_aucs[b] = auc_safe(y_b, s_b)
            cms_05.append(cm_at_threshold(y_b, s_b, 0.5))
            cm_e, t_e, eer_v = cm_at_eer_threshold(y_b, s_b)
            cms_eer.append(cm_e)
            thrs_eer.append(t_e)
            eers.append(eer_v)

        lr_boot_tpr = np.zeros((n_boot, len(fpr_grid)))
        lr_aucs = np.zeros(n_boot)
        n_tr = Y_tr.shape[0]
        for b in range(n_boot):
            idx_tr = rng_tr.choice(n_tr, size=n_tr, replace=True)
            lr = LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=base_seed + b + j * 31,
                solver="lbfgs",
            )
            lr.fit(X_tr[idx_tr], y_tr[idx_tr])
            lr_pool_b = lr.predict_proba(X_pool)[:, 1].astype(float)
            lr_boot_tpr[b] = tpr_on_grid(y_pool, lr_pool_b, fpr_grid)
            lr_aucs[b] = auc_safe(y_pool, lr_pool_b)

        lr_ref = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        lr_ref.fit(X_tr, y_tr)
        lr_pool = lr_ref.predict_proba(X_pool)[:, 1].astype(float)

        dim_slug = f"{key}_{human.replace(' ', '_').replace('/', '-')}"
        tag = f"Phase_III_{dim_slug}"
        plot_roc_bundle(
            out_sub / f"{tag}_auroc_roc_lr_bootstrap{n_boot}.png",
            f"Phase III · {key} ({human})\nVal+Test pooled (n={n}) · model: {rf_path.name}",
            y_pool,
            rf_pool,
            rf_boot_tpr,
            rf_aucs,
            lr_pool,
            lr_boot_tpr,
            lr_aucs,
            fpr_grid,
            0.5,
        )
        m05 = np.mean(np.stack(cms_05, axis=0), axis=0)
        s05 = np.std(np.stack(cms_05, axis=0), axis=0)
        plot_cm_mean_std(
            out_sub / f"{tag}_confusion_thr05_meanstd_bootstrap{n_boot}.png",
            f"Phase III · {key} ({human})",
            m05,
            s05,
        )
        me = np.mean(np.stack(cms_eer, axis=0), axis=0)
        se = np.std(np.stack(cms_eer, axis=0), axis=0)
        plot_eer_table_mean_std(
            out_sub / f"{tag}_eer_2x2_meanstd_bootstrap{n_boot}.png",
            f"Phase III · {key} ({human})",
            me,
            se,
            float(np.mean(thrs_eer)),
            float(np.std(thrs_eer)),
            float(np.mean(eers)),
            float(np.std(eers)),
        )
        print("saved:", out_sub / f"{tag}_auroc_roc_lr_bootstrap{n_boot}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--phase3_root", type=Path, default=Path(r"D:\TB Phase III"))
    ap.add_argument(
        "--phase3_model",
        type=Path,
        default=None,
        help="Default: <phase3_root>/rf_phase3_active_vs_inactive.joblib",
    )
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--n_boot", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260418)
    args = ap.parse_args()

    tb = args.tb_root
    art = tb / "artifacts"
    p3 = args.phase3_root
    p3_model = args.phase3_model or (p3 / "rf_phase3_active_vs_inactive.joblib")
    out_root = args.out_dir or (art / "plots_phase123_bootstrap5")

    keys = ["D1", "D2", "D3", "D4", "D6"]
    names = ["AFB(D1)", "TB PCR(D2)", "Solid culture(D3)", "Liquid culture(D4)", "NTM(D6)"]

    run_binary_phase(
        "Phase_I",
        out_root / "phase1",
        art / "rf_phase1_huge.joblib",
        art / "phase1_features_train.npz",
        art / "phase1_features_val.npz",
        art / "phase1_features_test.npz",
        "binary",
        args.n_boot,
        args.seed,
    )
    run_binary_phase(
        "Phase_II",
        out_root / "phase2",
        art / "rf_phase2_huge.joblib",
        art / "phase2_features_train.npz",
        art / "phase2_features_val.npz",
        art / "phase2_features_test.npz",
        "binary",
        args.n_boot,
        args.seed + 1000,
    )
    run_phase3(
        out_root / "phase3",
        p3_model,
        p3 / "phase3_features_train.npz",
        p3 / "phase3_features_val.npz",
        p3 / "phase3_features_test.npz",
        args.n_boot,
        args.seed + 2000,
        keys,
        names,
    )
    print("done. Root:", out_root)


if __name__ == "__main__":
    main()
