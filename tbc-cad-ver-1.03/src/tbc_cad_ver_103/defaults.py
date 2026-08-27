"""Artifact search for Phase III v1.03 hybrid and D6 XGB (same v1.023 JSON as 1.02)."""

from __future__ import annotations

from pathlib import Path


def default_hybrid_candidates(phase3_root: Path) -> list[Path]:
    root = phase3_root.expanduser()
    return [
        root / "artifacts" / "phase3_v1_03_d5_d6_xgb" / "phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib",
        root / "phase3_rf_v1_03_d5_d6_xgb_hybrid.joblib",
    ]


def default_xgb_d6_candidates(phase3_root: Path) -> list[Path]:
    root = phase3_root.expanduser()
    return [
        root / "artifacts" / "version_1_023" / "d6_ntm_xgb.json",
        root / "d6_ntm_xgb.json",
    ]


def resolve_hybrid_joblib(phase3_root: Path) -> Path | None:
    for p in default_hybrid_candidates(phase3_root):
        if p.is_file():
            return p
    return None


def resolve_xgb_d6_json(phase3_root: Path) -> Path | None:
    for p in default_xgb_d6_candidates(phase3_root):
        if p.is_file():
            return p
    return None
