from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def nested(row, *keys):
    for key in keys:
        row = row.get(key, {}) if isinstance(row, dict) else {}
    return row if row != {} else None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate only metrics actually produced by evaluate.py"
    )
    parser.add_argument(
        "--results", nargs="+", required=True, help="metrics.json files or directories"
    )
    parser.add_argument("--output", default="runs/benchmark.csv")
    args = parser.parse_args()
    files = []
    for item in args.results:
        path = Path(item)
        files.extend(path.rglob("metrics.json") if path.is_dir() else [path])
    rows = []
    for path in files:
        data = json.loads(path.read_text())
        run = path.parent.parent.name if path.parent.name.startswith("seed_") else path.parent.name
        seed = (
            path.parent.name.removeprefix("seed_") if path.parent.name.startswith("seed_") else None
        )
        rows.append(
            {
                "Run": run,
                "Seed": seed,
                "Model": data.get("model"),
                "Dataset": data.get("dataset"),
                "Category": data.get("category"),
                "Image AUROC": nested(data, "image", "auroc"),
                "Pixel AUROC": nested(data, "pixel", "pixel_auroc"),
                "AUPRO": nested(data, "pixel", "aupro"),
                "F1": nested(data, "image", "f1"),
                "Params": nested(data, "efficiency", "parameters"),
                "Model MB": nested(data, "efficiency", "model_size_mb"),
                "Latency ms": nested(data, "efficiency", "latency_ms_mean"),
                "Latency device": nested(data, "efficiency", "latency_device"),
                "Avg Iterations": nested(data, "efficiency", "avg_iterations"),
                "Source": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".md").write_text(
        frame.to_markdown(index=False) if not frame.empty else "No measured results.\n"
    )
    if not frame.empty:
        measures = [
            "Image AUROC",
            "Pixel AUROC",
            "AUPRO",
            "F1",
            "Params",
            "Latency ms",
            "Avg Iterations",
        ]
        summary = frame.groupby(["Run", "Model", "Dataset", "Category"], dropna=False)[
            measures
        ].agg(["count", "mean", "std"])
        summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
        summary = summary.reset_index()
        summary_output = output.with_name(f"{output.stem}_summary{output.suffix}")
        summary.to_csv(summary_output, index=False)
        summary_output.with_suffix(".md").write_text(summary.to_markdown(index=False))
    print(frame.to_string(index=False) if not frame.empty else "No metrics files found.")


if __name__ == "__main__":
    main()
