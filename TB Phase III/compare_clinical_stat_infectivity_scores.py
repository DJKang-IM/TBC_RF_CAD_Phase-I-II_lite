"""
Compare clinical weighted score vs CFA Relative Influence weighted score (binary indicators).

Clinical default weights [3,2,2,1,1] apply to COLS order:
  Cavity, AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture
(as in the original Phase III variable list).

Statistical weights: recomputed CFA relative influence (|Est.Std| normalized to sum 1),
same pipeline as analyze_infectivity_latent.py (StandardScaler then semopy CFA).

Outputs correlation (Pearson/Spearman), discordant patients CSV, pattern summaries.

Usage:
  python compare_clinical_stat_infectivity_scores.py --npz <.npz> --meta-main <meta.csv> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler

# Reuse data + CFA helpers
from analyze_infectivity_latent import (
    COLS,
    _parse_sid_from_path,
    extract_cfa_standardized_loadings,
    fit_cfa_sem,
    influence_share_0_1,
    load_from_csv,
    load_from_npz,
    meta_complete_case_mask,
)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def parse_weight_list(s: str) -> np.ndarray:
    parts = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(parts) != len(COLS):
        raise ValueError(f"Expected {len(COLS)} weights for COLS, got {len(parts)}: {s}")
    return np.asarray(parts, dtype=float)


def stat_weights_from_cfa(Z_df: pd.DataFrame) -> dict[str, float]:
    _model, _stats, _raw, ins_std = fit_cfa_sem(Z_df)
    if ins_std is None:
        return {}
    std_ld = extract_cfa_standardized_loadings(ins_std)
    return influence_share_0_1(std_ld)


def label_phenotype(row: pd.Series) -> str:
    """Short pattern tag for interpretation."""
    c, a, t, s, l = (int(row[k]) for k in COLS)
    cul_any = bool(s or l)
    smear_or_pcr = bool(a or t)

    if c and s == 0 and l == 0:
        if not smear_or_pcr:
            return "Cavity_pos_both_cultures_neg_no_afb_pcr"
        return "Cavity_pos_both_cultures_neg_afb_or_pcr_pos"
    if c and cul_any:
        return "Cavity_pos_any_culture_pos"
    if not c and cul_any:
        return "Culture_pos_no_cavity"
    if not c and smear_or_pcr and not cul_any:
        return "AFB_or_PCR_only_no_cavity_no_culture"
    if not (c or a or t or s or l):
        return "All_negative"
    return "Other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--meta-main", type=Path, default=None)
    ap.add_argument("--meta-d5", type=Path, default=None)
    ap.add_argument(
        "--clinical-weights",
        type=str,
        default="3,2,2,1,1",
        help=f"Comma weights for {COLS}",
    )
    ap.add_argument(
        "--weight-order",
        choices=("cols", "dicom"),
        default="cols",
        help="cols = Cavity,AFB,TB_PCR,Solid,Liquid; dicom = D1..D5 as AFB,TB_PCR,Solid,Liquid,Cavity",
    )
    ap.add_argument(
        "--discordant-quantile",
        type=float,
        default=0.90,
        help="Flag rows in top (1-q) of |z_clinical - z_statistical| (default 0.90 = upper 10%).",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if (args.npz is None) == (args.csv is None):
        print("Provide exactly one of --npz or --csv", file=sys.stderr)
        return 2

    raw_w = parse_weight_list(args.clinical_weights)
    if args.weight_order == "dicom":
        # D1,D2,D3,D4,D5 -> AFB, TB_PCR, Solid, Liquid, Cavity
        order_dicom = ["AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture", "Cavity"]
        w_map = dict(zip(order_dicom, raw_w))
        w_clin = np.array([w_map[c] for c in COLS], dtype=float)
    else:
        w_clin = raw_w

    meta_info = None
    if args.npz is not None:
        df, paths = load_from_npz(args.npz)
        if args.meta_main is not None:
            if paths is None:
                print("NPZ missing paths; cannot apply meta filter.", file=sys.stderr)
                return 2
            mask, meta_info = meta_complete_case_mask(paths, args.meta_main, args.meta_d5)
            df = df.loc[mask].reset_index(drop=True)
            paths = paths[mask]
    else:
        df, paths = load_from_csv(args.csv)
        paths = None

    df = df.apply(pd.to_numeric, errors="coerce")
    if paths is not None:
        df = df.assign(_path=np.asarray(paths, dtype=object))
    df = df.dropna()
    if paths is not None:
        paths = df["_path"].values
        df = df.drop(columns=["_path"])
    if len(df) < 10:
        print(f"n={len(df)} too small.", file=sys.stderr)
        return 2

    X = df[COLS].values.astype(float)
    # Clinical score on binary manifest scale
    s_clin = X @ w_clin

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    Z_df = pd.DataFrame(Z, columns=COLS)
    w_stat_dict = stat_weights_from_cfa(Z_df)
    if not w_stat_dict:
        print("[error] Could not obtain CFA relative influence.", file=sys.stderr)
        return 1
    w_stat = np.array([w_stat_dict[c] for c in COLS], dtype=float)
    s_stat = X @ w_stat

    r_pearson, p_pearson = scipy_stats.pearsonr(s_clin, s_stat)
    r_spear, p_spear = scipy_stats.spearmanr(s_clin, s_stat)

    z_clin = (s_clin - np.mean(s_clin)) / (np.std(s_clin, ddof=1) + 1e-12)
    z_stat = (s_stat - np.mean(s_stat)) / (np.std(s_stat, ddof=1) + 1e-12)
    disc = z_clin - z_stat
    thr = np.quantile(np.abs(disc), float(args.discordant_quantile))
    discord_mask = np.abs(disc) >= thr

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sid_list: list[str | None]
    if paths is not None:
        sid_list = [_parse_sid_from_path(str(paths[i])) for i in range(len(df))]
    else:
        sid_list = [""] * len(df)

    out_df = df.copy()
    out_df["s_clinical"] = s_clin
    out_df["s_statistical"] = s_stat
    out_df["z_clinical"] = z_clin
    out_df["z_statistical"] = z_stat
    out_df["z_discordance"] = disc
    out_df["discordant"] = discord_mask
    if paths is not None:
        pa = np.asarray(paths, dtype=object)
        if len(pa) != len(out_df):
            pa = pa[: len(out_df)]
        out_df["abs_path"] = [str(pa[i]) for i in range(len(out_df))]
    else:
        out_df["abs_path"] = ""
    out_df["study_id"] = sid_list
    out_df["phenotype"] = [label_phenotype(out_df.iloc[i]) for i in range(len(out_df))]

    discord_df = out_df[out_df["discordant"]].copy()
    discord_df["_abs_disc"] = discord_df["z_discordance"].abs()
    discord_df = discord_df.sort_values("_abs_disc", ascending=False).drop(columns=["_abs_disc"])
    discord_df.to_csv(out / "discordant_patients.csv", index=False, encoding="utf-8-sig")

    # One row per parsed study_id (max |discordance|), when multiple DICOMs per study
    if discord_df["study_id"].astype(str).str.len().gt(0).any():
        ds = discord_df.copy()
        ds["_abs_disc"] = ds["z_discordance"].abs()
        ds = ds.sort_values("_abs_disc", ascending=False).drop_duplicates(subset=["study_id"], keep="first")
        ds = ds.drop(columns=["_abs_disc"])
        ds.to_csv(out / "discordant_studies_unique.csv", index=False, encoding="utf-8-sig")

    # Example subgroup: Cavity=1 & Solid=0 & Liquid=0
    cav_cul_neg = out_df[(out_df["Cavity"] == 1) & (out_df["Solid_Culture"] == 0) & (out_df["Liquid_Culture"] == 0)]

    # Pattern counts (full vs discordant)
    def phen_counts(d: pd.DataFrame) -> dict[str, int]:
        return d["phenotype"].value_counts().to_dict()

    report = [
        "=== Clinical vs statistical infectivity scores ===",
        f"n = {len(df)}",
        f"Clinical weights ({args.weight_order}): {dict(zip(COLS, w_clin.tolist()))}",
        f"CFA relative influence: {json.dumps(w_stat_dict, ensure_ascii=False)}",
        "",
        f"Pearson r (s_clin, s_stat) = {r_pearson:.4f} (p={p_pearson:.4g})",
        f"Spearman rho = {r_spear:.4f} (p={float(p_spear):.2e})",
        "",
        f"Discordance = z(s_clinical) - z(s_statistical); flag top {100*(1-args.discordant_quantile):.1f}% |discordance| (threshold |disc| >= {thr:.3f})",
        f"n_discordant_flagged = {int(discord_mask.sum())}",
        "",
        "Phenotype counts (all patients):",
        json.dumps(phen_counts(out_df), indent=2, ensure_ascii=False),
        "",
        "Phenotype counts (discordant flagged):",
        json.dumps(phen_counts(out_df[out_df["discordant"]]), indent=2, ensure_ascii=False),
        "",
        f"Cavity=1 & Solid=0 & Liquid=0: n={len(cav_cul_neg)}",
        f"  mean s_clinical={cav_cul_neg['s_clinical'].mean():.3f} mean s_stat={cav_cul_neg['s_statistical'].mean():.3f}",
        f"  mean z_discordance={cav_cul_neg['z_discordance'].mean():.3f}",
        "",
        "Discordant split (z_discordance = z_clinical - z_stat):",
        f"  clinical score higher than statistical: n={int((out_df['discordant'] & (out_df['z_discordance'] > 0)).sum())}",
        f"  statistical score higher than clinical: n={int((out_df['discordant'] & (out_df['z_discordance'] < 0)).sum())}",
        "",
        "Use discordant_patients.csv for full list (sorted by |z_discordance|).",
    ]
    text = "\n".join(report)
    (out / "correlation_report.txt").write_text(text, encoding="utf-8")
    if meta_info is not None:
        (out / "meta_exclusion.json").write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    summary = {
        "n": len(df),
        "clinical_weights": {c: float(w) for c, w in zip(COLS, w_clin)},
        "cfa_relative_influence": w_stat_dict,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(r_spear),
        "spearman_p": float(p_spear),
        "discordant_quantile": float(args.discordant_quantile),
        "discordance_abs_threshold": float(thr),
        "n_discordant": int(discord_mask.sum()),
        "cavity_pos_both_cultures_neg_n": int(len(cav_cul_neg)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_df.to_csv(out / "all_patient_scores.csv", index=False, encoding="utf-8-sig")

    print(text)
    print(f"\nSaved: {out / 'correlation_report.txt'}")
    print(f"Saved: {out / 'discordant_patients.csv'}")
    if (out / "discordant_studies_unique.csv").exists():
        print(f"Saved: {out / 'discordant_studies_unique.csv'}")
    print(f"Saved: {out / 'all_patient_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
