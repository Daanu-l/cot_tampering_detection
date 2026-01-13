from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FitResult:
    model: Pipeline
    metrics: dict[str, float]


def fit_logreg_classifier(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    clf = Pipeline(
        steps=[("scaler", StandardScaler(with_mean=True, with_std=True)), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    )
    clf.fit(X_train, y_train)
    return clf


def eval_classifier(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    out = {"balanced_acc": float(balanced_accuracy_score(y, pred))}
    try:
        out["auroc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auroc"] = float("nan")

    out["pos_rate_pred"] = float(pred.mean())
    out["pos_rate_true"] = float(y.mean())
    return out
