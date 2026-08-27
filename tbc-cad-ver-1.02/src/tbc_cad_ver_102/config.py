from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IntegratedCadConfig:
    """Disk layout for v1.01 + v1.023-style D6 integration."""

    tb_artifacts_dir: Path
    """Directory containing ``rf_phase1_huge.joblib``, ``rf_phase2_huge.joblib``."""

    phase3_root: Path
    """TB Phase III project root (``rf_phase3_active_vs_inactive.joblib``)."""

    phase1_rf_name: str = "rf_phase1_huge.joblib"
    phase2_rf_name: str = "rf_phase2_huge.joblib"
    phase3_rf_name: str = "rf_phase3_active_vs_inactive.joblib"

    xgb_d6_model_path: Path | None = None
    """XGBoost model saved with ``Booster.save_model`` / sklearn ``XGBClassifier.load_model`` (``.json``)."""

    hybrid_joblib_path: Path | None = None
    """Optional single file: ``phase3_rf_v1_02_d6_xgb_hybrid.joblib`` (overrides RF + xgb_d6)."""

    backbone: str = "resnet18"
    image_size: int = 224
    backbone_weights: Path | None = None
    pretrained_backbone: bool = True
    torch_device: str | None = None
    """``cuda`` / ``cpu`` / ``None`` (auto)."""

    d6_threshold: float = 0.5
    phase12_threshold: float = 0.5
    phase3_threshold: float = 0.5

    def validate(self) -> None:
        ta = self.tb_artifacts_dir
        if not ta.is_dir():
            raise FileNotFoundError(f"tb_artifacts_dir not found: {ta}")
        if not (ta / self.phase1_rf_name).is_file():
            raise FileNotFoundError(f"Missing Phase I RF: {ta / self.phase1_rf_name}")
        if not (ta / self.phase2_rf_name).is_file():
            raise FileNotFoundError(f"Missing Phase II RF: {ta / self.phase2_rf_name}")
        if self.hybrid_joblib_path:
            if not self.hybrid_joblib_path.is_file():
                raise FileNotFoundError(f"hybrid_joblib_path not found: {self.hybrid_joblib_path}")
            return
        p3 = self.phase3_root
        if not p3.is_dir():
            raise FileNotFoundError(f"phase3_root not found: {p3}")
        if not (p3 / self.phase3_rf_name).is_file():
            raise FileNotFoundError(f"Missing Phase III RF: {p3 / self.phase3_rf_name}")
        if self.xgb_d6_model_path is None:
            raise ValueError("Provide xgb_d6_model_path or hybrid_joblib_path for Phase III + D6.")
        if not self.xgb_d6_model_path.is_file():
            raise FileNotFoundError(f"xgb_d6_model_path not found: {self.xgb_d6_model_path}")
