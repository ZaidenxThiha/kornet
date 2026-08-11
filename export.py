from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from models import build_model
from utils.checkpoint import load_checkpoint


class ExportWrapper(nn.Module):
    def __init__(self, model, sort_mode="hard"):
        super().__init__()
        self.model = model
        self.sort_mode = sort_mode

    def forward(self, image):
        output = self.model(image, adaptive=False, sort_mode=self.sort_mode)
        return output["image_score"], output["anomaly_map"], output["deltas"]


def export_model(checkpoint, output, fmt="torchscript", sort_mode="hard"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_model(config["model"]).eval()
    load_checkpoint(checkpoint, model)
    wrapper = ExportWrapper(model, sort_mode).eval()
    size = config["dataset"].get("image_size", 256)
    sample = torch.randn(1, 3, size, size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "torchscript":
        traced = torch.jit.trace(wrapper, sample, strict=False)
        traced.save(str(output))
    else:
        torch.onnx.export(
            wrapper,
            sample,
            output,
            input_names=["image"],
            output_names=["image_score", "anomaly_map", "deltas"],
            dynamic_axes={
                "image": {0: "batch"},
                "image_score": {0: "batch"},
                "anomaly_map": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    print(f"Exported {fmt} model to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--sort-mode", choices=["soft", "hard"], default="hard")
    args = parser.parse_args()
    export_model(args.checkpoint, args.output, args.format, args.sort_mode)


if __name__ == "__main__":
    main()
