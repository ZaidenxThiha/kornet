from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, labels, scores):
    try:
        return float(fn(labels, scores))
    except ValueError:
        return None


def calibrate_threshold(normal_scores, percentile: float = 99.0) -> float:
    scores = np.asarray(normal_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot calibrate a threshold without validation-normal scores")
    return float(np.percentile(scores, percentile))


def classification_metrics(labels, scores, threshold) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    result = {
        "auroc": _safe_metric(roc_auc_score, labels, scores),
        "average_precision": _safe_metric(average_precision_score, labels, scores),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
    }
    if len(np.unique(labels)) == 2:
        tn, fp, _, _ = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        result["false_positive_rate"] = float(fp / max(tn + fp, 1))
    else:
        result["false_positive_rate"] = None
    return result


def pixel_metrics(masks, maps, threshold=None) -> dict:
    labels = np.asarray(masks).reshape(-1).astype(int)
    scores = np.asarray(maps).reshape(-1)
    result = {
        "pixel_auroc": _safe_metric(roc_auc_score, labels, scores),
        "pixel_average_precision": _safe_metric(average_precision_score, labels, scores),
        "aupro": region_pro_auc(np.asarray(masks), np.asarray(maps)),
    }
    if threshold is not None:
        predictions = scores >= threshold
        result["pixel_f1"] = float(f1_score(labels, predictions, zero_division=0))
    return result


def region_pro_auc(masks, maps, max_fpr: float = 0.3, steps: int = 100) -> float | None:
    """Area under per-region overlap versus FPR, normalized to [0, 1]."""
    masks = np.asarray(masks).astype(bool)
    maps = np.asarray(maps, dtype=float)
    if not masks.any() or masks.all():
        return None
    thresholds = np.linspace(float(maps.max()), float(maps.min()), steps)
    normal = ~masks
    fprs, pros = [], []
    for threshold in thresholds:
        prediction = maps >= threshold
        fpr = float((prediction & normal).sum() / max(normal.sum(), 1))
        if fpr > max_fpr:
            continue
        overlaps = []
        for mask, pred in zip(masks, prediction):
            regions, count = ndimage.label(mask)
            for region_id in range(1, count + 1):
                region = regions == region_id
                overlaps.append(float((pred & region).sum() / region.sum()))
        if overlaps:
            fprs.append(fpr)
            pros.append(float(np.mean(overlaps)))
    if len(fprs) < 2:
        return None
    order = np.argsort(fprs)
    return float(np.trapezoid(np.asarray(pros)[order], np.asarray(fprs)[order]) / max_fpr)


def summarize_iterations(iterations) -> dict:
    values = np.asarray(iterations, dtype=float)
    return {
        "avg_iterations": float(values.mean()),
        "median_iterations": float(np.median(values)),
        "p95_iterations": float(np.percentile(values, 95)),
    }


def accuracy_efficiency_score(auroc, latency_ms, parameters_millions) -> float | None:
    if auroc is None:
        return None
    return float(auroc / (1.0 + math.log1p(latency_ms) + math.log1p(parameters_millions)))
