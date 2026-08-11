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
        rows.append(
            {
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
    print(frame.to_string(index=False) if not frame.empty else "No metrics files found.")


if __name__ == "__main__":
    main()
