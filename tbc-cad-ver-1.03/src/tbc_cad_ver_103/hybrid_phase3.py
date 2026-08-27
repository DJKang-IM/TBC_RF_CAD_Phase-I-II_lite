"""Phase III: multi-output RF for D1 through D5, XGBoost for D6 (tbc-cad-ver.1.03)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Order: D1, D2, D3, D4, D5 (Cavitary lesion), D6 (NTM)
D6_INDEX = 5
LABEL_KEYS = ("D1", "D2", "D3", "D4", "D5", "D6")


class Phase3HybridD6XGB:
    """
    ``predict_proba`` returns six matrices ``(n,2)`` like sklearn multi-output RF.

    D1 through D5 come from ``rf``; D6 is replaced with ``xgb_d6`` probabilities.
    """

    kind = "tbc_cad_ver_103_phase3_hybrid_d6_xgb"

    def __init__(
        self,
        rf: RandomForestClassifier,
        xgb_d6: XGBClassifier,
        d6_threshold: float = 0.5,
    ) -> None:
        self.rf = rf
        self.xgb_d6 = xgb_d6
        self.d6_threshold = float(d6_threshold)

    def predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        base = self.rf.predict_proba(X)
        p6 = self.xgb_d6.predict_proba(X)
        out = list(base)
        out[D6_INDEX] = np.asarray(p6, dtype=float)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        yhat = np.asarray(self.rf.predict(X), dtype=int).copy()
        p1 = self.xgb_d6.predict_proba(X)[:, 1].astype(float)
        yhat[:, D6_INDEX] = (p1 >= self.d6_threshold).astype(int)
        return yhat


def register_pickle_aliases() -> None:
    """Map training-script pickle class names to ``Phase3HybridD6XGB``."""
    import sys
    import types

    import __main__ as main_mod

    main_mod.Phase3HybridV103D6XGB = Phase3HybridD6XGB  # type: ignore[attr-defined]

    m = sys.modules.get("phase3_v103_d5_d6_hybrid_train")
    if m is None:
        m = types.ModuleType("phase3_v103_d5_d6_hybrid_train")
        sys.modules["phase3_v103_d5_d6_hybrid_train"] = m
    m.Phase3HybridV103D6XGB = Phase3HybridD6XGB  # type: ignore[attr-defined]
