"""Load all models and run end-to-end inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from xgboost import XGBClassifier

from tbc_cad_ver_102.config import IntegratedCadConfig
from tbc_cad_ver_102.dicom_featurizer import DicomFeaturizer
from tbc_cad_ver_102.hybrid_phase3 import D6_INDEX, LABEL_KEYS, Phase3HybridD6XGB, register_pickle_aliases


@dataclass
class InferenceResult:
    model_line: str
    dicom_path: str
    backbone: str
    feature_dim: int

    phase1_prob_class1: float
    phase1_pred: int

    phase2_prob_class1: float
    phase2_pred: int

    phase3: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntegratedCad:
    """
    **tbc-cad-ver.1.02** — DICOM in → Phase I / II / III (+ D6 XGB) scores out.

    Underlying artifacts: v1.01 RF (Phase I & II), v1.01 Phase III RF + v1.023-style D6 XGBoost.
    """

    model_line = "tbc-cad-ver.1.02"

    def __init__(self, config: IntegratedCadConfig) -> None:
        config.validate()
        self.cfg = config
        self._fe = DicomFeaturizer(
            config.backbone,
            image_size=config.image_size,
            pretrained=config.pretrained_backbone,
            weights=config.backbone_weights,
            device=config.torch_device,
        )

        ta = config.tb_artifacts_dir
        p3 = config.phase3_root

        self._rf1 = joblib.load(ta / config.phase1_rf_name)
        self._rf2 = joblib.load(ta / config.phase2_rf_name)

        if config.hybrid_joblib_path:
            register_pickle_aliases()
            self._p3 = joblib.load(config.hybrid_joblib_path)
        else:
            rf3 = joblib.load(p3 / config.phase3_rf_name)
            xgb = XGBClassifier()
            xgb.load_model(str(config.xgb_d6_model_path))
            self._p3 = Phase3HybridD6XGB(rf3, xgb, d6_threshold=config.d6_threshold)

    def predict(self, dicom_path: str | Path) -> InferenceResult:
        p = Path(dicom_path)
        X = self._fe.embed_path(p)

        p1 = float(self._rf1.predict_proba(X)[0, 1])
        p2 = float(self._rf2.predict_proba(X)[0, 1])
        y1 = int(p1 >= self.cfg.phase12_threshold)
        y2 = int(p2 >= self.cfg.phase12_threshold)

        probas = self._p3.predict_proba(X)
        p3_block: dict[str, dict[str, Any]] = {}
        rf_row: np.ndarray | None = None
        if hasattr(self._p3, "rf"):
            rf_arr = np.asarray(self._p3.rf.predict(X), dtype=int)
            if rf_arr.ndim == 2:
                rf_row = rf_arr[0]
            elif rf_arr.ndim == 1:
                rf_row = rf_arr
            else:
                rf_row = None

        for j, key in enumerate(LABEL_KEYS):
            pr = probas[j][0]
            p_pos = float(pr[1]) if pr.shape[0] > 1 else float(pr[0])
            thr = self.cfg.d6_threshold if j == D6_INDEX else self.cfg.phase3_threshold
            pred = int(p_pos >= thr)
            entry: dict[str, Any] = {
                "prob_class_1": p_pos,
                "pred_at_threshold": pred,
                "threshold": thr,
            }
            if rf_row is not None:
                entry["rf_head_pred"] = int(rf_row[j])
            p3_block[key] = entry

        return InferenceResult(
            model_line=self.model_line,
            dicom_path=str(p.resolve()),
            backbone=self._fe.backbone_name,
            feature_dim=int(self._fe.feat_dim),
            phase1_prob_class1=p1,
            phase1_pred=y1,
            phase2_prob_class1=p2,
            phase2_pred=y2,
            phase3=p3_block,
        )

    def predict_json(self, dicom_path: str | Path, *, indent: int | None = 2) -> str:
        return json.dumps(self.predict(dicom_path).as_dict(), ensure_ascii=False, indent=indent)
