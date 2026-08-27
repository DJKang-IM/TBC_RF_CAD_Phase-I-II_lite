from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit


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


def make_phase3_strata(y6: np.ndarray) -> np.ndarray:
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in y6], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    if counts.min() >= 2:
        return bits
    return y6[:, 5].astype(int)


def make_splits(
    strat_labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(strat_labels)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = int(n - n_train - n_val)
    all_idx = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(all_idx, strat_labels))
    val_ratio2 = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio2, random_state=int(seed) + 17)
    tr_sub, va_sub = next(s2.split(trva_idx, strat_labels[trva_idx]))
    tr_idx = trva_idx[tr_sub]
    va_idx = trva_idx[va_sub]
    return tr_idx, va_idx, te_idx


def load_embed_module(path: Path):
    spec = importlib.util.spec_from_file_location("embed_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def norm_sid(v) -> str:
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def build_missing_sid_sets(meta_csv: Path) -> dict[str, set[str]]:
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    sid_col = "Study No." if "Study No." in df.columns else df.columns[0]
    col_map = {
        "D1": "도말검사",
        "D2": "TB-PCR검사",
        "D3": "배양검사(고체)",
        "D4": "배양검사(액체)",
    }
    out: dict[str, set[str]] = {}
    for lab, c in col_map.items():
        miss = df[c].fillna("").astype(str).str.contains("미검", na=False)
        out[lab] = set(norm_sid(v) for v in df.loc[miss, sid_col].tolist())
    return out


def path_site(path_str: str) -> str:
    parts = Path(path_str).parts
    if "NE" in parts:
        return "ne"
    return "kn"


def build_valid_masks(paths: np.ndarray, embed_mod, missing_sid: dict[str, set[str]]) -> dict[str, np.ndarray]:
    masks = {k: np.ones((len(paths),), dtype=bool) for k in ["D1", "D2", "D3", "D4"]}
    for i, p in enumerate(paths.tolist()):
        pp = Path(str(p))
        site = path_site(str(p))
        sid, _ = (embed_mod._parse_ne_filename(pp.name) if site == "ne" else embed_mod._parse_study_id_from_filename(pp.name))
        if not sid:
            for k in masks:
                masks[k][i] = False
            continue
        candidates = embed_mod._study_key_candidates(sid)
        for k in masks:
            bad = any(c in missing_sid[k] for c in candidates)
            masks[k][i] = not bad
    return masks


def fit_predict_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    y_train = y_train.astype(int)
    uniq = np.unique(y_train)
    if uniq.size < 2:
        return np.full((x_test.shape[0],), int(uniq[0]), dtype=int)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=int(seed),
        class_weight="balanced",
    )
    rf.fit(x_train, y_train)
    p = rf.predict_proba(x_test)[:, 1].astype(float)
    return (p >= 0.5).astype(int)


def eval_mode(
    mode: str,
    x: np.ndarray,
    y6: np.ndarray,
    paths: np.ndarray,
    valid_masks: dict[str, np.ndarray],
    cv_runs: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    rf_estimators: int,
) -> dict:
    strata = make_phase3_strata(y6)
    qwk_runs: list[float] = []
    mae_runs: list[float] = []
    per_run: list[dict] = []
    heads = ["D1", "D2", "D3", "D4", "D5"]

    for r in range(cv_runs):
        rs = int(seed + 222 + r * 1009)
        tr, va, te = make_splits(strata, train_ratio, val_ratio, rs)
        _ = va

        gt_sum = y6[te, :5].sum(axis=1).astype(int)
        pred_parts: list[np.ndarray] = []
        n_train_by_head: dict[str, int] = {}

        for j, h in enumerate(heads):
            if mode == "v105" and h in {"D1", "D2", "D3", "D4"}:
                tr_h = tr[valid_masks[h][tr]]
            else:
                tr_h = tr

            x_tr = x[tr_h]
            y_tr = y6[tr_h, j].astype(int)
            x_te = x[te]
            yhat = fit_predict_binary(
                x_train=x_tr,
                y_train=y_tr,
                x_test=x_te,
                seed=rs + (j + 1) * 13,
                n_estimators=rf_estimators,
            )
            pred_parts.append(yhat)
            n_train_by_head[h] = int(len(tr_h))

        pred_sum = np.column_stack(pred_parts).sum(axis=1).astype(int)
        qwk = float(cohen_kappa_score(gt_sum, pred_sum, weights="quadratic"))
        mae = float(mean_absolute_error(gt_sum, pred_sum))
        qwk_runs.append(qwk)
        mae_runs.append(mae)
        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "test_n": int(len(te)),
                "qwk": qwk,
                "mae": mae,
                "train_n_by_head": n_train_by_head,
            }
        )

    return {
        "mode": mode,
        "test_qwk": mean_std(qwk_runs),
        "test_mae": mean_std(mae_runs),
        "per_run": per_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare sum(D1..D5) metrics: v1.03 vs v1.05 policy")
    ap.add_argument("--npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_active_all_260428_d5kw_fix.npz"))
    ap.add_argument("--meta_csv", type=Path, default=Path(r"D:\260428_META_ALL_CSV.csv"))
    ap.add_argument("--embed_script", type=Path, default=Path(r"D:\TB Phase III\embed_tb_labels_into_dicom.py"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path(r"D:\TB Phase III\artifacts\sum_qwk_mae_v103_vs_v105"))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["Y"], dtype=int)  # D1,D2,D3,D4,D6
    d5 = np.asarray(d["D5"], dtype=int).ravel()
    paths = np.asarray(d["paths"], dtype=object)
    y6 = np.column_stack([y[:, 0], y[:, 1], y[:, 2], y[:, 3], d5, y[:, 4]]).astype(int)

    close_count = 0
    if args.force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        close_count = int(close_mask.sum())
        y6 = y6.copy()
        y6[close_mask, :] = 0

    embed_mod = load_embed_module(args.embed_script)
    missing_sid = build_missing_sid_sets(args.meta_csv)
    valid_masks = build_valid_masks(paths, embed_mod, missing_sid)

    out = {
        "config": {
            "npz": str(args.npz),
            "meta_csv": str(args.meta_csv),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "force_close_zero": bool(args.force_close_zero),
        },
        "cohort": {
            "n_total": int(len(x)),
            "close_count": int(close_count),
            "d1_d5_gt_sum_mean": float(np.mean(y6[:, :5].sum(axis=1))),
            "label_valid_counts_for_v105": {
                "D1": int(valid_masks["D1"].sum()),
                "D2": int(valid_masks["D2"].sum()),
                "D3": int(valid_masks["D3"].sum()),
                "D4": int(valid_masks["D4"].sum()),
            },
        },
        "v103": eval_mode(
            mode="v103",
            x=x,
            y6=y6,
            paths=paths,
            valid_masks=valid_masks,
            cv_runs=args.cv_runs,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rf_estimators=args.rf_estimators,
        ),
        "v105": eval_mode(
            mode="v105",
            x=x,
            y6=y6,
            paths=paths,
            valid_masks=valid_masks,
            cv_runs=args.cv_runs,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rf_estimators=args.rf_estimators,
        ),
    }

    out["delta_v105_minus_v103"] = {
        "qwk_mean": float(out["v105"]["test_qwk"]["mean"] - out["v103"]["test_qwk"]["mean"]),
        "mae_mean": float(out["v105"]["test_mae"]["mean"] - out["v103"]["test_mae"]["mean"]),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "sum_qwk_mae_v103_vs_v105.json"
    txt_path = args.out_dir / "sum_qwk_mae_v103_vs_v105.txt"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("GT sum(D1..D5) vs Pred sum(D1..D5) on TEST")
    lines.append(f"N={out['cohort']['n_total']} close={out['cohort']['close_count']}")
    lines.append("")
    lines.append(f"v1.03 QWK: {out['v103']['test_qwk']['mean']:.4f} ± {out['v103']['test_qwk']['std']:.4f}")
    lines.append(f"v1.03 MAE: {out['v103']['test_mae']['mean']:.4f} ± {out['v103']['test_mae']['std']:.4f}")
    lines.append("")
    lines.append(f"v1.05 QWK: {out['v105']['test_qwk']['mean']:.4f} ± {out['v105']['test_qwk']['std']:.4f}")
    lines.append(f"v1.05 MAE: {out['v105']['test_mae']['mean']:.4f} ± {out['v105']['test_mae']['std']:.4f}")
    lines.append("")
    lines.append(f"Delta (v1.05 - v1.03) QWK: {out['delta_v105_minus_v103']['qwk_mean']:+.4f}")
    lines.append(f"Delta (v1.05 - v1.03) MAE: {out['delta_v105_minus_v103']['mae_mean']:+.4f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")


if __name__ == "__main__":
    main()
