import pytest

from train import train


def test_hard_sort_training_is_rejected_before_dataset_access():
    config = {
        "seed": 1,
        "dataset": {"name": "mvtec", "root": "missing", "category": "bottle"},
        "model": {"sort_mode": "hard"},
    }
    with pytest.raises(ValueError, match="differentiable soft sorting"):
        train(config)
