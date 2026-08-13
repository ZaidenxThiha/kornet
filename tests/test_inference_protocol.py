import json
import os

import numpy as np
import pytest
import torch

from utils.checkpoint import confined_checkpoint, load_checkpoint_payload
from utils.inference import (
    InferenceProtocol,
    calibrated_image_threshold,
    gaussian_smooth,
    resolve_inference_protocol,
)
from utils.metrics import calibrate_threshold, pixel_metrics, region_pro_auc


class _Malicious:
    def __init__(self, target):
        self.target = target

    def __reduce__(self):
        return os.system, (f"touch {self.target}",)


def test_protocol_defaults_are_deterministic_and_threshold_requires_exact_match():
    config = {"evaluation": {"gaussian_sigma": 4.0}}
    protocol = resolve_inference_protocol(config)
    assert protocol == InferenceProtocol(sort_mode="hard", adaptive=False, gaussian_sigma=4.0)
    metrics = {
        "image": {"threshold": 0.25},
        "protocol": {"inference": protocol.as_dict()},
    }
    assert calibrated_image_threshold(metrics, protocol) == 0.25
    changed = resolve_inference_protocol(config, sort_mode="soft")
    assert calibrated_image_threshold(metrics, changed) is None
    assert calibrated_image_threshold(json.loads(json.dumps(metrics)), protocol) == 0.25


def test_gaussian_smoothing_is_shape_stable_and_changes_map_distribution():
    maps = torch.zeros(2, 1, 33, 33)
    maps[:, :, 16, 16] = 1
    smoothed = gaussian_smooth(maps, 2.0)
    assert smoothed.shape == maps.shape
    assert 0 < smoothed.max() < 1
    assert torch.allclose(smoothed[0], smoothed[1])
    assert calibrate_threshold(maps.numpy().reshape(-1)) != calibrate_threshold(
        smoothed.numpy().reshape(-1)
    )


def test_threshold_rejects_empty_and_nonfinite_scores():
    with pytest.raises(ValueError, match="without validation-normal"):
        calibrate_threshold([])
    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold([0.0, np.nan])


def test_aupro_works_without_numpy_trapezoid(monkeypatch):
    monkeypatch.delattr(np, "trapezoid", raising=False)
    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    masks[:, 2:5, 2:5] = 1
    assert region_pro_auc(masks, masks.astype(float)) is not None


def test_pixel_metrics_reports_unavailable_when_no_valid_masks():
    result = pixel_metrics(np.empty((0, 8, 8)), np.empty((0, 8, 8)), threshold=0.5)
    assert result == {
        "pixel_auroc": None,
        "pixel_average_precision": None,
        "aupro": None,
        "pixel_f1": None,
    }


def test_checkpoint_loader_uses_safe_payloads_and_confines_paths(tmp_path):
    allowed = tmp_path / "runs"
    allowed.mkdir()
    valid = allowed / "model.pt"
    torch.save({"model": {"weight": torch.ones(1)}, "config": {}}, valid)
    assert load_checkpoint_payload(valid)["model"]["weight"].item() == 1
    assert confined_checkpoint(valid, [allowed]) == valid.resolve()
    outside = tmp_path / "outside.pt"
    torch.save({"model": {}}, outside)
    with pytest.raises(ValueError, match="outside"):
        confined_checkpoint(outside, [allowed])
    marker = tmp_path / "executed"
    malicious = allowed / "malicious.pt"
    torch.save({"model": _Malicious(marker)}, malicious)
    with pytest.raises(Exception, match="Weights only load failed"):
        load_checkpoint_payload(malicious)
    assert not marker.exists()


def test_evaluation_collect_applies_the_declared_protocol_to_maps():
    from evaluate import collect

    class Dummy(torch.nn.Module):
        def forward(self, images, adaptive, sort_mode):
            assert adaptive is False
            assert sort_mode == "hard"
            anomaly_map = torch.zeros(images.shape[0], 1, 33, 33)
            anomaly_map[:, :, 16, 16] = 1
            return {
                "image_score": torch.ones(images.shape[0]),
                "anomaly_map": anomaly_map,
                "iterations": torch.ones(images.shape[0], dtype=torch.long),
            }

    batch = {
        "image": torch.zeros(1, 3, 33, 33),
        "label": torch.zeros(1, dtype=torch.long),
        "mask": torch.zeros(1, 1, 33, 33),
        "mask_valid": torch.ones(1, dtype=torch.bool),
    }
    result = collect(Dummy(), [batch], torch.device("cpu"), InferenceProtocol(gaussian_sigma=2))
    assert 0 < np.max(result["maps"]) < 1


def test_export_wrapper_uses_protocol_and_smooths_maps():
    from export import ExportWrapper

    class Dummy(torch.nn.Module):
        def forward(self, images, adaptive, sort_mode):
            assert adaptive is False
            assert sort_mode == "hard"
            anomaly_map = torch.zeros(images.shape[0], 1, 33, 33)
            anomaly_map[:, :, 16, 16] = 1
            return {
                "image_score": torch.ones(images.shape[0]),
                "anomaly_map": anomaly_map,
                "deltas": torch.ones(images.shape[0], 1),
            }

    wrapper = ExportWrapper(Dummy(), InferenceProtocol(gaussian_sigma=2))
    _, anomaly_map, _ = wrapper(torch.zeros(1, 3, 33, 33))
    assert 0 < anomaly_map.max() < 1
