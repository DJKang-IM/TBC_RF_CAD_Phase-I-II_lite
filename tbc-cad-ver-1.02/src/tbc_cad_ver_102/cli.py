"""CLI: ``tbc-cad-infer``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tbc_cad_ver_102.bundle import IntegratedCad, IntegratedCadConfig
from tbc_cad_ver_102.defaults import resolve_hybrid_joblib, resolve_xgb_d6_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tbc-cad-infer",
        description="tbc-cad-ver.1.02 — DICOM → Phase I / II / III (+ D6 XGBoost) JSON",
    )
    ap.add_argument("--dicom", required=True, type=Path, help="Path to a single .dcm file")
    ap.add_argument(
        "--tb-artifacts",
        type=Path,
        default=Path(r"D:\TB Test DB\artifacts"),
        help="Folder with rf_phase1_huge.joblib and rf_phase2_huge.joblib",
    )
    ap.add_argument(
        "--phase3-root",
        type=Path,
        default=Path(r"D:\TB Phase III"),
        help="TB Phase III root",
    )
    ap.add_argument(
        "--xgb-d6-json",
        type=Path,
        default=None,
        help="XGBoost D6 model (.json). If omitted, search under --phase3-root.",
    )
    ap.add_argument(
        "--hybrid-joblib",
        type=Path,
        default=None,
        help="Hybrid joblib. If omitted, search under --phase3-root.",
    )
    ap.add_argument("--backbone-weights", type=Path, default=None, help="Optional ResNet state_dict .pt")
    ap.add_argument("--no-pretrained", action="store_true", help="Random-init backbone (not recommended)")
    ap.add_argument("--device", default=None, help="Torch device: cuda | cpu | (empty=auto)")
    args = ap.parse_args(argv)

    hybrid = args.hybrid_joblib
    xgb_path = args.xgb_d6_json
    if hybrid is None and xgb_path is None:
        hybrid = resolve_hybrid_joblib(args.phase3_root)
        if hybrid is None:
            xgb_path = resolve_xgb_d6_json(args.phase3_root)

    if hybrid is not None and hybrid.is_file():
        cfg = IntegratedCadConfig(
            tb_artifacts_dir=args.tb_artifacts,
            phase3_root=args.phase3_root,
            hybrid_joblib_path=hybrid,
            backbone_weights=args.backbone_weights,
            pretrained_backbone=not args.no_pretrained,
            torch_device=args.device,
        )
    elif xgb_path is not None and xgb_path.is_file():
        cfg = IntegratedCadConfig(
            tb_artifacts_dir=args.tb_artifacts,
            phase3_root=args.phase3_root,
            xgb_d6_model_path=xgb_path,
            backbone_weights=args.backbone_weights,
            pretrained_backbone=not args.no_pretrained,
            torch_device=args.device,
        )
    else:
        ap.error(
            "Could not find Phase III+D6 model. Pass --hybrid-joblib or --xgb-d6-json, or place one of:\n"
            f"  {Path(args.phase3_root) / 'artifacts' / 'phase3_v1_02_d6_xgb' / 'phase3_rf_v1_02_d6_xgb_hybrid.joblib'}\n"
            f"  {Path(args.phase3_root) / 'artifacts' / 'version_1_023' / 'd6_ntm_xgb.json'}"
        )

    cad = IntegratedCad(cfg)
    out = cad.predict(args.dicom)
    text = json.dumps(out.as_dict(), ensure_ascii=False, indent=2)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
