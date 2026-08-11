import pytest
import torch

from losses import KORNetLoss
from models.baseline_models import build_model
from models.kornet import KORNet


def small_model(**kwargs):
    return KORNet(
        backbone="resnet18",
        pretrained=False,
        feature_dim=16,
        rank_heads=2,
        max_tokens=24,
        max_iterations=3,
        train_iterations=2,
        dropout=0.0,
        **kwargs,
    )


def test_model_outputs_and_shared_parameters():
    model = small_model()
    model.train()
    output = model(torch.randn(2, 3, 64, 64))
    assert output["image_score"].shape == (2,)
    assert output["anomaly_map"].shape == (2, 1, 64, 64)
    assert output["deltas"].shape == (2, 2)
    # Recursion is represented by one module instance, not copied blocks.
    assert len([name for name, _ in model.named_modules() if name == "kor"]) == 1


def test_adaptive_stopping_and_cpu_inference():
    model = small_model(convergence_threshold=1e9, convergence_patience=1, min_iterations=1)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64), adaptive=True)
    assert output["iterations"].item() == 1
    assert torch.isfinite(output["anomaly_map"]).all()


def test_late_training_adaptive_stopping_keeps_gradients():
    model = small_model(convergence_threshold=1e9, convergence_patience=1, min_iterations=1)
    model.train()
    output = model(torch.randn(2, 3, 64, 64), adaptive=True, iterations=3)
    output["tokens"].square().mean().backward()
    assert output["iterations"].tolist() == [1, 1]
    assert model.kor.phi[0].weight.grad is not None


def test_prototype_update_and_state_round_trip(tmp_path):
    model = small_model()
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 64))
        model.anomaly_head.update_prototype(output["tokens"])
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    restored = small_model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.anomaly_head.prototype_initialized
    assert torch.allclose(model.anomaly_head.prototype, restored.anomaly_head.prototype)


def test_spatial_residual_head_updates_and_round_trips(tmp_path):
    options = {
        "spatial_weight": 0.8,
        "initial_residual_weight": 0.6,
        "topk_fraction": 0.25,
    }
    model = small_model(**options).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
        model.anomaly_head.update_prototype(output["tokens"], output["initial_tokens"])
        scored = model(torch.randn(1, 3, 64, 64))
    assert torch.isfinite(scored["image_score"]).all()
    assert torch.isfinite(scored["anomaly_map"]).all()
    path = tmp_path / "spatial_state.pt"
    torch.save(model.state_dict(), path)
    restored = small_model(**options)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.allclose(
        model.anomaly_head.spatial_prototype,
        restored.anomaly_head.spatial_prototype,
    )


def test_cnn_baseline_has_no_kor_parameters_and_loss_executes():
    config = {
        "variant": "cnn",
        "backbone": "resnet18",
        "pretrained": False,
        "feature_dim": 16,
        "rank_heads": 2,
        "max_tokens": 24,
        "max_iterations": 2,
        "train_iterations": 2,
    }
    model = build_model(config)
    assert not any(name.startswith("kor.") for name, _ in model.named_parameters())
    output = model(torch.randn(2, 3, 64, 64))
    losses = KORNetLoss()(output, model.anomaly_head.prototype)
    assert torch.isfinite(losses["total"])


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0", "convnext_tiny"])
def test_supported_backbone_shapes(backbone):
    model = KORNet(
        backbone=backbone,
        pretrained=False,
        feature_dim=8,
        rank_heads=1,
        max_tokens=12,
        max_iterations=1,
        train_iterations=1,
    ).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64), adaptive=False, sort_mode="hard")
    assert output["anomaly_map"].shape == (1, 1, 64, 64)
    assert output["tokens"].shape[-1] == 8
