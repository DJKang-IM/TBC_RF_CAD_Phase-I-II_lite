"""
Infectivity latent construct — PCA, CFA/SEM (semopy), PLS-PM-style outer weights, plots.

Label mapping (Phase III DICOM private tags / NPZ):
  Cavity         ↔ D5  (cavitary, separate D5 array in .npz)
  AFB_Smear      ↔ D1
  TB_PCR         ↔ D2
  Solid_Culture  ↔ D3
  Liquid_Culture ↔ D4

All indicators are z-scored (StandardScaler) before PCA / SEM / PLS.

Inputs:
  --npz   Phase III feature .npz with keys Y (N,5) [D1..D4,D6] and D5 (N,)
  --csv   Optional CSV with columns: Cavity, AFB_Smear, TB_PCR, Solid_Culture, Liquid_Culture

Metadata exclusion (optional, NPZ only):
  --meta-main   UTF-8 meta CSV (Study No. + 도말검사, TB-PCR검사, 배양검사(고체/액체))
  --meta-d5     Optional Version-2 style CSV with 판독/Reading for cavity (미검 exclusion)
  Rows are kept only if all five indicators are not '미검' in meta (same rules as v1.05 CV script).

Output (--out):
  pca_biplot.png, sem_path_diagram.png, report.txt, summary.json
  - CFA: semopy Est. Std (standardized loadings) and relative influence on [0,1] (|Est.Std| / sum)

Requires: pip install semopy networkx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

COLS = ["Cavity", "AFB_Smear", "TB_PCR", "Solid_Culture", "Liquid_Culture"]

# ---------------------------------------------------------------------------
# Meta: 미검 exclusion (align with train_phase3_active_cv_v105_missing_exclude.py)
# ---------------------------------------------------------------------------


def _norm_sid(v) -> str:
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _parse_study_id_from_filename(name: str) -> tuple[str | None, bool]:
    base = Path(name).stem.strip()
    m = re.match(r"^(?P<id>\d+)(?P<rest>.*)$", base)
    if not m:
        return None, False
    study_id = m.group("id")
    rest = (m.group("rest") or "").lower()
    is_close = "close" in rest
    return study_id, is_close


def _parse_ne_filename(name: str) -> tuple[str | None, bool]:
    stem = Path(name).stem
    low = stem.lower()
    is_close = "close" in low
    m = re.match(r"^(\d+)", stem)
    if not m:
        return None, is_close
    digits = m.group(1)
    study_id = digits if len(digits) < 5 else digits[:5]
    return study_id, is_close


def _study_key_candidates(study_id: str) -> list[str]:
    sid = str(study_id).strip()
    if not sid:
        return []
    out: list[str] = [sid]
    if sid.isdigit() and len(sid) > 5:
        pref5 = sid[:5]
        if pref5 not in out:
            out.append(pref5)
        t = sid
        while len(t) > 5:
            t = t[:-1]
            if t not in out:
                out.append(t)
    return out


def _path_site(path_str: str) -> str:
    parts = Path(path_str).parts
    return "ne" if "NE" in parts else "kn"


def _parse_sid_from_path(path_str: str) -> str | None:
    name = Path(str(path_str)).name
    if _path_site(path_str) == "ne":
        return _parse_ne_filename(name)[0]
    return _parse_study_id_from_filename(name)[0]


def build_missing_sid_sets(meta_csv: Path) -> dict[str, set[str]]:
    """Per study: label column contains '미검' -> excluded for that lab (D1..D4)."""
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
        if c not in df.columns:
            raise ValueError(f"Meta CSV missing column {c!r}: {meta_csv}")
        miss = df[c].fillna("").astype(str).str.contains("미검", na=False)
        out[lab] = {_norm_sid(v) for v in df.loc[miss, sid_col].tolist()}
    return out


def _find_reading_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        s = str(c).strip()
        sl = s.lower()
        if "reading" in sl or "판독" in s or "소견" in s or "impression" in sl:
            return c
    return None


def build_d5_migeom_sid_set(meta_csv: Path, reading_col: str | None = None) -> set[str]:
    """Studies whose cavity-related text column is marked 미검 (substring rule)."""
    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    sid_col = "Study No." if "Study No." in df.columns else df.columns[0]
    rc = reading_col or _find_reading_column(df)
    if rc is None:
        return set()
    miss = df[rc].fillna("").astype(str).str.contains("미검", na=False)
    return {_norm_sid(v) for v in df.loc[miss, sid_col].tolist()}


def meta_complete_case_mask(
    paths: np.ndarray,
    meta_main: Path,
    meta_d5: Path | None,
) -> tuple[np.ndarray, dict]:
    """
    True = keep row: D1..D4 not 미검 in meta_main; D5 text not 미검
    (main Reading or meta_d5 판독 column).
    """
    missing_sid = build_missing_sid_sets(meta_main)
    if meta_d5 is not None:
        d5_bad = build_d5_migeom_sid_set(meta_d5)
        d5_source = str(meta_d5)
    else:
        d5_bad = build_d5_migeom_sid_set(meta_main)
        d5_source = str(meta_main)

    mask = np.zeros(len(paths), dtype=bool)
    dropped_sid = 0
    for i, p in enumerate(paths.tolist()):
        sid = _parse_sid_from_path(str(p))
        if not sid:
            dropped_sid += 1
            continue
        cand = _study_key_candidates(sid)
        ok_d1 = not any(c in missing_sid["D1"] for c in cand)
        ok_d2 = not any(c in missing_sid["D2"] for c in cand)
        ok_d3 = not any(c in missing_sid["D3"] for c in cand)
        ok_d4 = not any(c in missing_sid["D4"] for c in cand)
        ok_d5 = not any(c in d5_bad for c in cand)
        mask[i] = ok_d1 and ok_d2 and ok_d3 and ok_d4 and ok_d5

    info = {
        "meta_main": str(meta_main),
        "meta_d5_source": d5_source,
        "n_paths": int(len(paths)),
        "n_kept": int(mask.sum()),
        "n_dropped": int((~mask).sum()),
        "n_parse_fail_paths": int(dropped_sid),
        "d5_migeom_studies_n": len(d5_bad),
    }
    return mask, info


def load_from_npz(path: Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    d = np.load(path, allow_pickle=True)
    Y = np.asarray(d["Y"], dtype=int)
    d5 = np.asarray(d["D5"], dtype=int).ravel()
    if Y.shape[1] != 5:
        raise ValueError("NPZ Y must be (N,5) for D1..D4,D6")
    if len(d5) != len(Y):
        raise ValueError("D5 length must match Y")
    paths = np.asarray(d["paths"], dtype=object) if "paths" in d.files else None
    df = pd.DataFrame(
        {
            "AFB_Smear": Y[:, 0],
            "TB_PCR": Y[:, 1],
            "Solid_Culture": Y[:, 2],
            "Liquid_Culture": Y[:, 3],
            "Cavity": d5,
        }
    )[COLS]
    return df, paths


def load_from_csv(path: Path) -> tuple[pd.DataFrame, None]:
    df = pd.read_csv(path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Need: {COLS}")
    return df[COLS].copy(), None


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def pca_pc1_loadings_and_weights(Z: np.ndarray, col_names: list[str]) -> dict:
    """Correlation loadings on PC1; nonnegative weights = squared loadings normalized."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Z already standardized; PCA on Z is correlation PCA
    pca = PCA(n_components=min(5, Z.shape[1]))
    scores = pca.fit_transform(Z)
    pc1 = scores[:, 0]
    # Correlation loadings = corr(z_j, PC1)  (for standardized z, same as covariance)
    loadings = np.array(
        [np.corrcoef(Z[:, j], pc1)[0, 1] for j in range(Z.shape[1])],
        dtype=float,
    )
    # Nonnegative weights (same direction as squared contribution share)
    w_raw = loadings**2
    w_sum = w_raw.sum()
    weights = (w_raw / w_sum) if w_sum > 0 else np.ones_like(w_raw) / len(w_raw)

    return {
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "loadings_pc1_correlation": {col_names[j]: float(loadings[j]) for j in range(len(col_names))},
        "normalized_weights_sq_loading": {col_names[j]: float(weights[j]) for j in range(len(col_names))},
        "pca_model": pca,
        "scores": scores,
    }


def plot_pca_biplot(Z: np.ndarray, col_names: list[str], pca_res: dict, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca: PCA = pca_res["pca_model"]
    scores = pca_res["scores"]
    xs, ys = scores[:, 0], scores[:, 1]
    load = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=8, alpha=0.35, c="steelblue", label="samples")
    scale = 3.0
    for j, name in enumerate(col_names):
        ax.arrow(
            0,
            0,
            load[j, 0] * scale,
            load[j, 1] * scale,
            head_width=0.05,
            length_includes_head=True,
            color="crimson",
        )
        ax.text(load[j, 0] * scale * 1.15, load[j, 1] * scale * 1.15, name, fontsize=9)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA biplot (standardized indicators)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SEM / CFA
# ---------------------------------------------------------------------------

def fit_cfa_sem(Z_df: pd.DataFrame) -> tuple[object, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    from semopy import Model
    from semopy.stats import calc_stats

    desc = """
Infectivity =~ Cavity + AFB_Smear + TB_PCR + Solid_Culture + Liquid_Culture
"""
    model = Model(desc)
    model.fit(Z_df)
    stats = calc_stats(model)
    ins_raw = model.inspect()
    ins_std = model.inspect(std_est=True)
    return model, stats, ins_raw, ins_std


def extract_cfa_standardized_loadings(sem_ins_std: pd.DataFrame) -> dict[str, float]:
    """Reflective paths: indicator ~ Infectivity -> Est. Std (standardized loading)."""
    rows = sem_ins_std[(sem_ins_std["op"] == "~") & (sem_ins_std["rval"] == "Infectivity")]
    col = "Est. Std" if "Est. Std" in sem_ins_std.columns else None
    if col is None:
        return {}
    out: dict[str, float] = {}
    for _, r in rows.iterrows():
        v = r[col]
        if pd.notna(v):
            out[str(r["lval"])] = float(v)
    return out


def influence_share_0_1(std_loadings: dict[str, float]) -> dict[str, float]:
    """Nonnegative shares summing to 1 from absolute standardized loadings (relative impact)."""
    a = {k: abs(float(v)) for k, v in std_loadings.items() if np.isfinite(v)}
    s = float(sum(a.values()))
    if s <= 0:
        n = max(len(a), 1)
        return {k: 1.0 / n for k in a}
    return {k: float(v / s) for k, v in a.items()}


def srmr_from_sigma(Z_df: pd.DataFrame, model) -> float:
    """SRMR on correlation residuals (semopy implied vs sample)."""
    out = model.calc_sigma()
    Sigma_hat = np.asarray(out[0], dtype=float)
    X = Z_df.values
    R = np.corrcoef(X.T)
    # implied correlation
    d = np.sqrt(np.clip(np.diag(Sigma_hat), 1e-12, None))
    R_hat = Sigma_hat / np.outer(d, d)
    p = R.shape[0]
    triu = np.triu_indices(p, k=1)
    return float(np.sqrt(np.mean((R[triu] - R_hat[triu]) ** 2)))


def sem_path_diagram(
    ins: pd.DataFrame,
    out_png: Path,
    fit_row: pd.DataFrame | None,
    use_standardized_on_edges: bool = False,
    ins_std: pd.DataFrame | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import patheffects as pe

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Latent node
    latent_xy = (5.0, 3.0)
    circ = plt.Circle(latent_xy, 0.55, color="lightyellow", ec="black", lw=2, zorder=3)
    ax.add_patch(circ)
    ax.text(latent_xy[0], latent_xy[1], "Infectivity", ha="center", va="center", fontsize=11, weight="bold")

    rows = ins[(ins["op"] == "~") & (ins["rval"] == "Infectivity") & (ins["lval"] != "Infectivity")]
    std_col = "Est. Std" if ins_std is not None and "Est. Std" in ins_std.columns else None
    std_lookup: dict[str, float] = {}
    if use_standardized_on_edges and ins_std is not None and std_col:
        for _, sr in ins_std[
            (ins_std["op"] == "~") & (ins_std["rval"] == "Infectivity") & (ins_std["lval"] != "Infectivity")
        ].iterrows():
            if pd.notna(sr[std_col]):
                std_lookup[str(sr["lval"])] = float(sr[std_col])
    y_positions = np.linspace(1.0, 5.0, len(rows))
    for i, (_, r) in enumerate(rows.iterrows()):
        ind = str(r["lval"])
        if ind in std_lookup:
            est = std_lookup[ind]
        else:
            est = float(r["Estimate"])
        bx, by = 1.2, float(y_positions[i])
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx - 0.55, by - 0.22), 1.1, 0.44, boxstyle="round,pad=0.02", ec="black", fc="white"
            )
        )
        ax.text(bx, by, ind.replace("_", "\n"), ha="center", va="center", fontsize=8)
        ax.annotate(
            "",
            xy=latent_xy,
            xytext=(bx + 0.55, by),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="navy"),
        )
        mid_x = (bx + 0.55 + latent_xy[0]) / 2
        mid_y = (by + latent_xy[1]) / 2
        t = ax.text(mid_x, mid_y + 0.15, f"{est:.3f}", fontsize=9, color="navy")
        t.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    title = "CFA: Infectivity (reflective)" + (" [Std. Est. on paths]" if std_lookup else "")
    if fit_row is not None and not fit_row.empty:
        fr = fit_row.iloc[0]
        bits = []
        for k in ("CFI", "TLI", "RMSEA", "chi2"):
            if k in fr.index and pd.notna(fr[k]):
                bits.append(f"{k}={float(fr[k]):.4g}")
        title += "\n" + "  ".join(bits)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PLS-PM (reflective outer scheme — Mode A style iterative correlations)
# ---------------------------------------------------------------------------


def plspm_reflective_outer(Z: np.ndarray, col_names: list[str], max_iter: int = 200, tol: float = 1e-8) -> dict:
    """Single-block reflective PLS: outer weights via iterative corr(Z_j, F)."""
    n, p = Z.shape
    w = np.ones(p) / np.sqrt(p)
    F = Z @ w
    for _ in range(max_iter):
        w_new = np.array([np.corrcoef(Z[:, j], F)[0, 1] for j in range(p)])
        if not np.all(np.isfinite(w_new)):
            w_new = np.nan_to_num(w_new, nan=0.0)
        nw = np.linalg.norm(w_new)
        if nw < 1e-12:
            break
        w_new /= nw
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new
        F = Z @ w
    loadings = np.array([np.corrcoef(Z[:, j], F)[0, 1] for j in range(p)])
    return {
        "outer_weights": {col_names[j]: float(w[j]) for j in range(p)},
        "outer_loadings_corr_F": {col_names[j]: float(loadings[j]) for j in range(p)},
    }


# ---------------------------------------------------------------------------
# Priority ranking
# ---------------------------------------------------------------------------


def rank_priority(names: list[str], scores: dict[str, float]) -> list[tuple[str, float]]:
    """Higher score = higher priority."""
    items = sorted(scores.items(), key=lambda x: -abs(x[1]))
    return items


def mean_rank_table(names: list[str], rank_lists: list[list[str]]) -> dict[str, float]:
    """Average rank (1=best) across criteria."""
    ranks = {n: [] for n in names}
    for ordering in rank_lists:
        for r, name in enumerate(ordering):
            ranks[name].append(r + 1)
    return {n: float(np.mean(ranks[n])) for n in names}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Infectivity latent: PCA, CFA, PLS-PM, plots")
    ap.add_argument("--npz", type=Path, default=None, help="Phase III NPZ (Y + D5)")
    ap.add_argument("--csv", type=Path, default=None, help="CSV with 5 columns")
    ap.add_argument(
        "--meta-main",
        type=Path,
        default=None,
        help="Meta CSV (UTF-8) for 미검 exclusion: D1..D4 columns + join by filename Study ID",
    )
    ap.add_argument(
        "--meta-d5",
        type=Path,
        default=None,
        help="Optional separate meta CSV for D5 cavity text (미검); else use Reading/판독 from --meta-main",
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/infectivity_latent_analysis"))
    args = ap.parse_args()

    if (args.npz is None) == (args.csv is None):
        print("Provide exactly one of --npz or --csv", file=sys.stderr)
        sys.exit(2)

    if args.meta_main is not None and args.csv is not None:
        print("--meta-main is only supported with --npz (need paths for study ID).", file=sys.stderr)
        sys.exit(2)

    meta_info: dict | None = None
    if args.npz is not None:
        df, paths = load_from_npz(args.npz)
        if args.meta_main is not None:
            if paths is None:
                print("NPZ has no 'paths' key; cannot apply meta 미검 filter.", file=sys.stderr)
                sys.exit(2)
            mask, meta_info = meta_complete_case_mask(paths, args.meta_main, args.meta_d5)
            df = df.loc[mask].reset_index(drop=True)
            if meta_info.get("d5_migeom_studies_n") == 0 and args.meta_d5 is None:
                mdf = pd.read_csv(args.meta_main, encoding="utf-8-sig")
                if _find_reading_column(mdf) is None:
                    print(
                        "[warn] No Reading/판독 column in meta-main; D5 미검 not filtered. "
                        "Pass --meta-d5 with a Version-2 style CSV to exclude cavity 미검.",
                        file=sys.stderr,
                    )
    else:
        df, _paths = load_from_csv(args.csv)

    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 30:
        print(f"Warning: only n={len(df)} rows after NA drop — SEM fit indices are indicative only.", file=sys.stderr)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Z = scaler.fit_transform(df.values)
    Z_df = pd.DataFrame(Z, columns=COLS)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- PCA ---
    pca_res = pca_pc1_loadings_and_weights(Z, COLS)
    plot_pca_biplot(Z, COLS, pca_res, out_dir / "pca_biplot.png")

    # --- CFA / SEM ---
    sem_ins = None
    sem_ins_std = None
    sem_stats = None
    sem_model = None
    std_loadings: dict[str, float] = {}
    influence_01: dict[str, float] = {}
    try:
        sem_model, sem_stats, sem_ins, sem_ins_std = fit_cfa_sem(Z_df)
        srmr = srmr_from_sigma(Z_df, sem_model)
        sem_stats = sem_stats.copy()
        sem_stats["SRMR"] = np.nan
        sem_stats.loc[sem_stats.index[0], "SRMR"] = srmr
        if sem_ins_std is not None:
            std_loadings = extract_cfa_standardized_loadings(sem_ins_std)
            influence_01 = influence_share_0_1(std_loadings)
    except Exception as e:
        sem_stats = pd.DataFrame({"Error": [str(e)]})
        print(f"[CFA] Failed: {e}", file=sys.stderr)

    if sem_ins is not None:
        sem_path_diagram(
            sem_ins,
            out_dir / "sem_path_diagram.png",
            sem_stats,
            use_standardized_on_edges=sem_ins_std is not None,
            ins_std=sem_ins_std,
        )

    # --- PLS-PM style ---
    pls = plspm_reflective_outer(Z, COLS)

    # --- Extract SEM path coefs (indicator ~ Infectivity) ---
    path_coefs: dict[str, float] = {}
    if sem_ins is not None:
        rows = sem_ins[(sem_ins["op"] == "~") & (sem_ins["rval"] == "Infectivity")]
        for _, r in rows.iterrows():
            path_coefs[str(r["lval"])] = float(r["Estimate"])

    # --- Priority: combine ranks ---
    pca_w = pca_res["normalized_weights_sq_loading"]
    pls_ld = pls["outer_loadings_corr_F"]
    sem_ld = {k: abs(path_coefs[k]) for k in path_coefs}

    order_pca = [x[0] for x in sorted(pca_w.items(), key=lambda x: -x[1])]
    order_pls = [x[0] for x in sorted(pls_ld.items(), key=lambda x: -abs(x[1]))]
    order_sem = [x[0] for x in sorted(sem_ld.items(), key=lambda x: -x[1])] if sem_ld else order_pca

    avg_rank = mean_rank_table(COLS, [order_pca, order_pls, order_sem])
    priority_order = sorted(avg_rank.items(), key=lambda x: x[1])

    # --- Interpret fit ---
    fit_notes = []
    if sem_stats is not None and "CFI" in sem_stats.columns:
        row = sem_stats.iloc[0]
        cfi = row.get("CFI", np.nan)
        tli = row.get("TLI", np.nan)
        rmsea = row.get("RMSEA", np.nan)
        srmr_v = row.get("SRMR", np.nan)
        fit_notes.append(
            "Model fit (rules of thumb; ML on standardized scores -- binary 0/1 indicators are approximate): "
            "CFI/TLI >= 0.95 good; RMSEA <= 0.06 reasonable; SRMR <= 0.08 good."
        )
        fit_notes.append(
            f"Observed: CFI={cfi:.4f}, TLI={tli:.4f}, RMSEA={rmsea}, SRMR={srmr_v}"
            if pd.notna(cfi)
            else "Fit indices incomplete."
        )

    report_lines = [
        "=== Infectivity latent analysis ===",
        f"n = {len(df)}",
    ]
    if meta_info is not None:
        report_lines.extend(
            [
                "",
                "Meta filter (exclude any row with 미검 in D1..D4 lab columns, or in D5 reading text):",
                json.dumps(meta_info, indent=2, ensure_ascii=False),
            ]
        )
    report_lines.extend(
        [
            "",
            "1) Standardization: sklearn StandardScaler on all 5 indicators.",
            "",
            "2) PCA (PC1 correlation loadings & squared-loading normalized weights)",
            json.dumps(pca_res["loadings_pc1_correlation"], indent=2, ensure_ascii=False),
            json.dumps(pca_res["normalized_weights_sq_loading"], indent=2, ensure_ascii=False),
            f"PC1 variance ratio: {pca_res['pc1_explained_variance_ratio']:.4f}",
            "",
            "3) CFA / SEM (semopy, MLW) - Infectivity =~ 5 indicators",
            sem_ins.to_string() if sem_ins is not None else "(not available)",
            "",
            "Fit indices:",
            sem_stats.to_string() if sem_stats is not None else "",
            "",
            "CFA standardized coefficients (semopy Est. Std = standardized reflective loadings, indicator ~ Infectivity):",
            json.dumps(std_loadings, indent=2, ensure_ascii=False) if std_loadings else "(not available)",
            "",
            "Relative influence on [0, 1] (|Est. Std| scaled to sum = 1; compare magnitude across indicators):",
            json.dumps(influence_01, indent=2, ensure_ascii=False) if influence_01 else "(not available)",
            "",
            "Rank by CFA relative influence (1 = largest share):",
        ]
    )
    if influence_01:
        for rank, name in enumerate(sorted(influence_01.keys(), key=lambda k: -influence_01[k]), 1):
            report_lines.append(f"  {rank}. {name}: {influence_01[name]:.4f}")
    else:
        report_lines.append("  (n/a)")
    report_lines.extend(
        [
            "",
            "4) PLS-PM (reflective outer, iterative corr weights)",
            json.dumps(pls, indent=2, ensure_ascii=False),
            "",
            "5) Priority (lower mean rank = higher priority across PCA weight order, PLS |loading|, SEM |path|):",
        ]
    )
    for name, mr in priority_order:
        report_lines.append(f"  {name}: mean rank = {mr:.3f}")
    report_lines.append("")
    report_lines.append("Suggested overall priority (1 = most influential):")
    for i, (name, _) in enumerate(priority_order, 1):
        report_lines.append(f"  {i}. {name}")
    report_lines.extend(["", *fit_notes])

    report_text = "\n".join(report_lines)
    (out_dir / "report.txt").write_text(report_text, encoding="utf-8")

    summary = {
        "n": len(df),
        "meta_exclusion": meta_info,
        "pca": {
            "pc1_variance_ratio": pca_res["pc1_explained_variance_ratio"],
            "loadings_pc1": pca_res["loadings_pc1_correlation"],
            "weights_normalized_sq": pca_res["normalized_weights_sq_loading"],
        },
        "cfa_path_coefficients": path_coefs,
        "cfa_standardized_loadings": std_loadings,
        "cfa_relative_influence_0_1": influence_01,
        "cfa_fit": sem_stats.to_dict() if sem_stats is not None and "CFI" in sem_stats.columns else {},
        "pls_pm": pls,
        "priority_mean_rank": avg_rank,
        "priority_ordered": [p[0] for p in priority_order],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.write(report_text + "\n")
    except UnicodeEncodeError:
        sys.stdout.write(report_text.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")
    print(f"\nSaved: {out_dir / 'report.txt'}")
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'pca_biplot.png'}")
    if sem_ins is not None:
        print(f"Saved: {out_dir / 'sem_path_diagram.png'}")


if __name__ == "__main__":
    main()
