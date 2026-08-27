from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit


def make_phase3_strata(y6: np.ndarray) -> np.ndarray:
    bits = np.array(["".join(str(int(v)) for v in row.tolist()) for row in y6], dtype=object)
    _, counts = np.unique(bits, return_counts=True)
    if counts.min() >= 2:
        return bits
    return y6[:, 5].astype(int)


def make_splits(strat_labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int):
    n = len(strat_labels)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = int(n - n_train - n_val)
    all_idx = np.arange(n)
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=int(seed))
    trva_idx, te_idx = next(s1.split(all_idx, strat_labels))
    val_ratio2 = n_val / (n_train + n_val)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio2, random_state=int(seed) + 17)
    tr_sub, _ = next(s2.split(trva_idx, strat_labels[trva_idx]))
    tr_idx = trva_idx[tr_sub]
    return tr_idx, te_idx


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
    col_map = {"D1": "도말검사", "D2": "TB-PCR검사", "D3": "배양검사(고체)", "D4": "배양검사(액체)"}
    out: dict[str, set[str]] = {}
    for lab, c in col_map.items():
        miss = df[c].fillna("").astype(str).str.contains("미검", na=False)
        out[lab] = set(norm_sid(v) for v in df.loc[miss, sid_col].tolist())
    return out


def path_site(path_str: str) -> str:
    parts = Path(path_str).parts
    return "ne" if "NE" in parts else "kn"


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
        keys = embed_mod._study_key_candidates(sid)
        for k in masks:
            masks[k][i] = not any(x in missing_sid[k] for x in keys)
    return masks


def fit_predict_binary(x_train, y_train, x_test, seed: int, n_estimators: int) -> np.ndarray:
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


def count_bins(arr: np.ndarray) -> dict[str, int]:
    out = {str(i): 0 for i in range(6)}
    vals, cts = np.unique(arr.astype(int), return_counts=True)
    for v, c in zip(vals.tolist(), cts.tolist()):
        if 0 <= int(v) <= 5:
            out[str(int(v))] = int(c)
    return out


def run_mode(mode: str, x, y6, strata, valid_masks, cv_runs, seed, train_ratio, val_ratio, rf_estimators):
    pooled_pred: list[np.ndarray] = []
    pooled_gt: list[np.ndarray] = []
    per_run: list[dict] = []
    heads = ["D1", "D2", "D3", "D4", "D5"]

    for r in range(cv_runs):
        rs = int(seed + 222 + r * 1009)
        tr, te = make_splits(strata, train_ratio, val_ratio, rs)
        pred_parts = []

        for j, h in enumerate(heads):
            tr_h = tr
            if mode == "v105" and h in {"D1", "D2", "D3", "D4"}:
                tr_h = tr[valid_masks[h][tr]]
            yhat = fit_predict_binary(
                x_train=x[tr_h],
                y_train=y6[tr_h, j],
                x_test=x[te],
                seed=rs + (j + 1) * 13,
                n_estimators=rf_estimators,
            )
            pred_parts.append(yhat)

        pred_sum = np.column_stack(pred_parts).sum(axis=1).astype(int)
        gt_sum = y6[te, :5].sum(axis=1).astype(int)
        per_run.append(
            {
                "run": r + 1,
                "seed": rs,
                "test_n": int(len(te)),
                "pred_sum_dist": count_bins(pred_sum),
                "gt_sum_dist": count_bins(gt_sum),
            }
        )
        pooled_pred.append(pred_sum)
        pooled_gt.append(gt_sum)

    pred_all = np.concatenate(pooled_pred)
    gt_all = np.concatenate(pooled_gt)
    return {
        "mode": mode,
        "pooled_pred_sum_dist": count_bins(pred_all),
        "pooled_gt_sum_dist": count_bins(gt_all),
        "pooled_n": int(pred_all.shape[0]),
        "per_run": per_run,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pred sum(D1..D5) distribution: v1.03 vs v1.05")
    ap.add_argument("--npz", type=Path, default=Path(r"D:\TB Phase III\phase3_features_active_all_260428_d5kw_fix.npz"))
    ap.add_argument("--meta_csv", type=Path, default=Path(r"D:\260428_META_ALL_CSV.csv"))
    ap.add_argument("--embed_script", type=Path, default=Path(r"D:\TB Phase III\embed_tb_labels_into_dicom.py"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path(r"D:\TB Phase III\artifacts\sum_pred_distribution_v103_vs_v105"))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    x = np.asarray(d["X"], dtype=np.float32)
    y = np.asarray(d["Y"], dtype=int)  # D1,D2,D3,D4,D6
    d5 = np.asarray(d["D5"], dtype=int).ravel()
    paths = np.asarray(d["paths"], dtype=object)
    y6 = np.column_stack([y[:, 0], y[:, 1], y[:, 2], y[:, 3], d5, y[:, 4]]).astype(int)
    if args.force_close_zero:
        close_mask = np.array([("close" in Path(str(p)).stem.lower()) for p in paths], dtype=bool)
        y6[close_mask, :] = 0

    strata = make_phase3_strata(y6)
    embed_mod = load_embed_module(args.embed_script)
    missing_sid = build_missing_sid_sets(args.meta_csv)
    valid_masks = build_valid_masks(paths, embed_mod, missing_sid)

    out = {
        "config": {
            "npz": str(args.npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "force_close_zero": bool(args.force_close_zero),
        },
        "v103": run_mode("v103", x, y6, strata, valid_masks, args.cv_runs, args.seed, args.train_ratio, args.val_ratio, args.rf_estimators),
        "v105": run_mode("v105", x, y6, strata, valid_masks, args.cv_runs, args.seed, args.train_ratio, args.val_ratio, args.rf_estimators),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "sum_pred_distribution_v103_vs_v105.json"
    txt_path = args.out_dir / "sum_pred_distribution_v103_vs_v105.txt"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt_dist(dct: dict[str, int], n: int) -> list[str]:
        rows = []
        for i in range(6):
            c = int(dct.get(str(i), 0))
            rows.append(f"  sum={i}: {c} ({(100.0*c/max(n,1)):.2f}%)")
        return rows

    lines = []
    for m in ["v103", "v105"]:
        lines.append(f"[{m}] pooled TEST pred sum distribution (N={out[m]['pooled_n']})")
        lines.extend(fmt_dist(out[m]["pooled_pred_sum_dist"], out[m]["pooled_n"]))
        lines.append("")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")


if __name__ == "__main__":
    main()
