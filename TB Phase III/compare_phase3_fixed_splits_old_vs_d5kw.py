from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


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


def make_key(p: str) -> str:
    parts = Path(p).parts
    for i, x in enumerate(parts):
        up = x.upper()
        if up in {"KN", "NE"}:
            return "/".join(parts[i:]).lower()
    return Path(p).name.lower()


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


def proba_pos_or_const(clf, x: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    uniq = np.unique(y_train.astype(int))
    if uniq.size == 1:
        return np.full((x.shape[0],), float(uniq[0]), dtype=float)
    classes = np.asarray(getattr(clf, "classes_", []))
    pos = np.where(classes == 1)[0]
    if pos.size == 0:
        return np.zeros((x.shape[0],), dtype=float)
    return np.asarray(clf.predict_proba(x)[:, int(pos[0])], dtype=float)


def load_y6(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    x = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["Y"], dtype=int)  # D1,D2,D3,D4,D6
    d5 = np.asarray(d["D5"], dtype=int).ravel()
    paths = np.asarray(d["paths"], dtype=object)
    y6 = np.column_stack([y[:, 0], y[:, 1], y[:, 2], y[:, 3], d5, y[:, 4]]).astype(int)
    return x, y6, paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare old vs new(D5-keyword) with fixed split indices")
    ap.add_argument("--old_npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_active_all_260428.npz"))
    ap.add_argument("--new_npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_active_all_260428_d5kw.npz"))
    ap.add_argument("--out_dir", type=Path, default=Path(r"D:\TB Phase III\artifacts\phase3_fixed_split_compare_old_vs_d5kw"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--rf_estimators", type=int, default=600)
    args = ap.parse_args()

    x_old, y6_old, p_old = load_y6(args.old_npz)
    x_new, y6_new, p_new = load_y6(args.new_npz)

    idx_old = {make_key(str(p)): i for i, p in enumerate(p_old.tolist())}
    idx_new = {make_key(str(p)): i for i, p in enumerate(p_new.tolist())}
    common = sorted(set(idx_old.keys()) & set(idx_new.keys()))
    if not common:
        raise RuntimeError("No common samples between old/new npz")

    io = np.array([idx_old[k] for k in common], dtype=int)
    inew = np.array([idx_new[k] for k in common], dtype=int)

    xo = x_old[io]
    xn = x_new[inew]
    yo = y6_old[io]
    yn = y6_new[inew]

    # Fixed split indices are created ONCE from old strata and reused for both datasets.
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in yo], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    strata = bits if counts.min() >= 2 else yo[:, 5].astype(int)

    n = len(common)
    n_train = int(round(n * float(args.train_ratio)))
    n_val = int(round(n * float(args.val_ratio)))
    n_test = int(n - n_train - n_val)

    labs = ["D1", "D2", "D3", "D4", "D5", "D6"]
    old_auc = {k: [] for k in labs}
    new_auc = {k: [] for k in labs}

    for r in range(args.cv_runs):
        rs = int(args.seed + 222 + r * 1009)
        all_idx = np.arange(n)
        s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=rs)
        trva, te = next(s1.split(all_idx, strata))
        val_ratio = n_val / (n_train + n_val)
        s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=rs + 17)
        tr_sub, _ = next(s2.split(trva, strata[trva]))
        tr = trva[tr_sub]

        for j, lab in enumerate(labs):
            ytr_old = yo[tr, j].astype(int)
            yte_old = yo[te, j].astype(int)
            ytr_new = yn[tr, j].astype(int)
            yte_new = yn[te, j].astype(int)

            if lab == "D6":
                xgb_o = build_xgb(seed=rs + 777, y_train=ytr_old)
                xgb_o.fit(xo[tr], ytr_old)
                po = proba_pos_or_const(xgb_o, xo[te], ytr_old)
                xgb_n = build_xgb(seed=rs + 777, y_train=ytr_new)
                xgb_n.fit(xn[tr], ytr_new)
                pn = proba_pos_or_const(xgb_n, xn[te], ytr_new)
            else:
                rf_o = RandomForestClassifier(
                    n_estimators=args.rf_estimators,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=rs + (j + 1) * 13,
                    class_weight="balanced",
                )
                rf_o.fit(xo[tr], ytr_old)
                po = proba_pos_or_const(rf_o, xo[te], ytr_old)

                rf_n = RandomForestClassifier(
                    n_estimators=args.rf_estimators,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=rs + (j + 1) * 13,
                    class_weight="balanced",
                )
                rf_n.fit(xn[tr], ytr_new)
                pn = proba_pos_or_const(rf_n, xn[te], ytr_new)

            old_auc[lab].append(safe_auc(yte_old, po))
            new_auc[lab].append(safe_auc(yte_new, pn))

    summary = {
        "config": {
            "old_npz": str(args.old_npz),
            "new_npz": str(args.new_npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "split_counts_train_val_test": [int(n_train), int(n_val), int(n_test)],
            "n_common_samples": int(n),
        },
        "label_positive_counts_old": {k: int(yo[:, i].sum()) for i, k in enumerate(labs)},
        "label_positive_counts_new": {k: int(yn[:, i].sum()) for i, k in enumerate(labs)},
        "auroc_old": {k: mean_std(v) for k, v in old_auc.items()},
        "auroc_new": {k: mean_std(v) for k, v in new_auc.items()},
        "delta_new_minus_old": {
            k: float(np.nanmean(np.asarray(new_auc[k], dtype=float) - np.asarray(old_auc[k], dtype=float)))
            for k in labs
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "fixed_split_compare_old_vs_d5kw.json"
    out_txt = args.out_dir / "fixed_split_compare_old_vs_d5kw.txt"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("Fixed-split comparison (old vs d5-keyword)")
    lines.append(f"n_common={n} split={n_train}/{n_val}/{n_test}")
    lines.append("")
    for lab in labs:
        o = summary["auroc_old"][lab]
        nn = summary["auroc_new"][lab]
        d = summary["delta_new_minus_old"][lab]
        lines.append(
            f"{lab}: old={o['mean']:.4f}±{o['std']:.4f}  "
            f"new={nn['mean']:.4f}±{nn['std']:.4f}  delta={d:+.6f}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
