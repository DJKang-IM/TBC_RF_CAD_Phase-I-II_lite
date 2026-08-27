"""Phase III: RF v1.01 multi-output + XGBoost D6 head (v1.023-style)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

D6_INDEX = 4
LABEL_KEYS = ("D1", "D2", "D3", "D4", "D6")


class Phase3HybridD6XGB:
    """
    ``predict_proba`` returns five matrices (n,2) like sklearn multi-output RF.

    D1–D4 (and unused RF slot semantics) come from ``rf``; D6 probabilities are
    replaced with ``xgb_d6.predict_proba``.
    """

    kind = "tbc_cad_ver_102_phase3_hybrid_d6_xgb"

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
    """
    Allow ``joblib.load`` of hybrid objects saved from
    ``TB Phase III/phase3_v102_d6_xgb_cv_train.py``.

    Training is often run as ``python phase3_v102_d6_xgb_cv_train.py``, so the
    class is pickled as ``__main__.Phase3HybridV102D6XGB``; map that (and the
    real module name) to this package's compatible implementation.
    """
    import sys
    import types

    import __main__ as main_mod

    main_mod.Phase3HybridV102D6XGB = Phase3HybridD6XGB  # type: ignore[attr-defined]

    mod_name = "phase3_v102_d6_xgb_cv_train"
    m = sys.modules.get(mod_name)
    if m is None:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m
    m.Phase3HybridV102D6XGB = Phase3HybridD6XGB  # type: ignore[attr-defined]
