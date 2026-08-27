from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("weighted_eval_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-model weighted score eval (v1.03 policy)")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--version", type=str, required=True)
    ap.add_argument("--feature_extractor", type=str, required=True)
    ap.add_argument("--meta_csv", type=Path, default=Path(r"D:\260428_META_ALL_CSV.csv"))
    ap.add_argument("--embed_script", type=Path, default=Path(r"D:\TB Phase III\embed_tb_labels_into_dicom.py"))
    ap.add_argument("--eval_module", type=Path, default=Path(r"D:\TB Phase III\eval_weighted_score_qwk_mae_v103_v105.py"))
    ap.add_argument("--cv_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--rf_estimators", type=int, default=600)
    ap.add_argument("--force_close_zero", action="store_true")
    ap.add_argument("--out_dir", type=Path, required=True)
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

    m = load_module(args.eval_module)
    strata = m.make_phase3_strata(y6)
    embed_mod = m.load_embed_module(args.embed_script)
    missing_sid = m.build_missing_sid_sets(args.meta_csv)
    valid_masks = m.build_valid_masks(paths, embed_mod, missing_sid)
    res = m.eval_mode(
        mode="v103",
        x=x,
        y6=y6,
        strata=strata,
        valid_masks=valid_masks,
        cv_runs=args.cv_runs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        rf_estimators=args.rf_estimators,
    )

    out = {
        "version": args.version,
        "feature_extractor": args.feature_extractor,
        "weights": {"D1": 2, "D2": 2, "D3": 1, "D4": 1, "D5": 3},
        "max_score": 9,
        "config": {
            "npz": str(args.npz),
            "cv_runs": int(args.cv_runs),
            "seed": int(args.seed),
            "rf_estimators": int(args.rf_estimators),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "force_close_zero": bool(args.force_close_zero),
        },
        "cohort": {"n_total": int(len(x)), "close_count": int(close_count)},
        "result": res,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"weighted_score_qwk_mae_{args.version}.json"
    txt_path = args.out_dir / f"weighted_score_qwk_mae_{args.version}.txt"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = (
        f"Weighted score (D1-D5) TEST, {args.version} {args.feature_extractor}\n"
        f"QWK: {res['test_qwk']['mean']:.4f} +- {res['test_qwk']['std']:.4f}\n"
        f"MAE: {res['test_mae']['mean']:.4f} +- {res['test_mae']['std']:.4f}\n"
    )
    txt_path.write_text(txt, encoding="utf-8")
    print(f"saved: {json_path}")
    print(f"saved: {txt_path}")


if __name__ == "__main__":
    main()
