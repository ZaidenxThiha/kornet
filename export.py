from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from models import build_model
from utils.checkpoint import load_checkpoint, load_checkpoint_payload
from utils.inference import gaussian_smooth, resolve_inference_protocol


class ExportWrapper(nn.Module):
    def __init__(self, model, protocol):
        super().__init__()
        self.model = model
        self.protocol = protocol

    def forward(self, image):
        output = self.model(
            image,
            adaptive=self.protocol.adaptive,
            sort_mode=self.protocol.sort_mode,
        )
        anomaly_map = gaussian_smooth(output["anomaly_map"], self.protocol.gaussian_sigma)
        return output["image_score"], anomaly_map, output["deltas"]


def export_model(
    checkpoint, output, fmt="torchscript", sort_mode=None, adaptive=None, gaussian_sigma=None
):
    state = load_checkpoint_payload(checkpoint)
    config = state["config"]
    protocol = resolve_inference_protocol(
        config,
        sort_mode=sort_mode,
        adaptive=adaptive,
        gaussian_sigma=gaussian_sigma,
    )
    if protocol.adaptive:
        raise ValueError(
            "Adaptive stopping cannot be faithfully represented by trace-based export; "
            "use the default fixed protocol or pass --no-adaptive"
        )
    model = build_model(config["model"]).eval()
    load_checkpoint(checkpoint, model)
    wrapper = ExportWrapper(model, protocol).eval()
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
    parser.add_argument("--sort-mode", choices=["soft", "hard"])
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gaussian-sigma", type=float)
    args = parser.parse_args()
    export_model(
        args.checkpoint,
        args.output,
        args.format,
        args.sort_mode,
        args.adaptive,
        args.gaussian_sigma,
    )


if __name__ == "__main__":
    main()
