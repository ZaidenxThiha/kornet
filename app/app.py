from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transforms import build_transform
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.visualization import heatmap_overlay, normalize_map

st.set_page_config(page_title="KORNet Inspector", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1450px;}
    [data-testid="stMetric"] {background:#111827; border:1px solid #283548; padding:16px; border-radius:12px;}
    .status-normal {padding:16px;border-radius:12px;background:#073b2c;color:#a7f3d0;font-size:24px;font-weight:700;}
    .status-defect {padding:16px;border-radius:12px;background:#4c151a;color:#fecaca;font-size:24px;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def checkpoints():
    return sorted(PROJECT_ROOT.glob("runs/**/best.pt")) + sorted(
        PROJECT_ROOT.glob("checkpoints/*.pt")
    )


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state.get("config")
    if not config:
        raise ValueError("Checkpoint has no embedded configuration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]).to(device).eval()
    load_checkpoint(path, model, map_location=device)
    metrics_path = Path(path).with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    threshold = metrics.get("image", {}).get("threshold")
    return model, config, device, threshold, metrics


def box_image(image: Image.Image, anomaly_map: np.ndarray):
    normalized = normalize_map(anomaly_map)
    region = normalized >= max(0.65, float(np.quantile(normalized, 0.95)))
    ys, xs = np.where(region)
    result = image.copy().resize((anomaly_map.shape[1], anomaly_map.shape[0]))
    if len(xs):
        draw = ImageDraw.Draw(result)
        draw.rectangle(
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline="#ff334f", width=3
        )
    return result


st.title("KORNet Visual Quality Inspector")
st.caption("Kaprekar-Inspired Recursive Opposing-Ranking Network · experimental research interface")

available = checkpoints()
with st.sidebar:
    st.header("Model")
    if available:
        selected = st.selectbox(
            "Checkpoint", available, format_func=lambda p: str(p.relative_to(PROJECT_ROOT))
        )
    else:
        selected_text = st.text_input("Checkpoint path", placeholder="runs/.../best.pt")
        selected = Path(selected_text) if selected_text else None
    sorting = st.selectbox(
        "Inference ranking",
        ["hard", "soft"],
        help="Hard ranking is exact and optimized; soft ranking matches training relaxation.",
    )
    uploaded = st.file_uploader("Inspection image", type=["jpg", "jpeg", "png"])
    st.divider()
    st.caption(
        "A calibrated NORMAL/DEFECT decision requires metrics.json produced by evaluate.py. An uncalibrated score is never presented as a probability."
    )

if not selected or not Path(selected).exists():
    st.info("Train a model or enter a valid checkpoint path to begin.")
    st.stop()
if uploaded is None:
    st.info("Upload a JPG or PNG image for inspection.")
    st.stop()

try:
    model, config, device, threshold, metrics = load_model(str(selected))
    image = Image.open(uploaded).convert("RGB")
    size = config["dataset"].get("image_size", 256)
    tensor, _ = build_transform(size)(image)
    with st.spinner("Running recursive opposing-ranking analysis…"), torch.inference_mode():
        start = time.perf_counter()
        output = model(tensor[None].to(device), sort_mode=sorting)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency = (time.perf_counter() - start) * 1000
except Exception as exc:  # noqa: BLE001 - the UI must render model/load failures cleanly
    st.exception(exc)
    st.stop()

score = float(output["image_score"][0])
iterations = int(output["iterations"][0])
deltas = output["deltas"][0, :iterations].cpu().numpy()
anomaly_map = output["anomaly_map"][0, 0].cpu().numpy()
display_image = np.asarray(image.resize((size, size)))
overlay = heatmap_overlay(display_image, anomaly_map)
status = "UNCALIBRATED" if threshold is None else ("DEFECT" if score >= threshold else "NORMAL")
status_class = "status-defect" if status == "DEFECT" else "status-normal"

st.subheader("KORNet Analysis")
left, right = st.columns([1, 3])
with left:
    st.markdown(f'<div class="{status_class}">{status}</div>', unsafe_allow_html=True)
with right:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Anomaly score", f"{score:.4f}")
    c2.metric("Iterations", f"{iterations} / {model.max_iterations}")
    c3.metric("Inference", f"{latency:.1f} ms")
    c4.metric("Device", str(device).upper())

if threshold is None:
    st.warning(
        "No validation-calibrated threshold is available. Run evaluate.py; this score is not a probability or class decision."
    )
else:
    margin = abs(score - threshold) / (abs(threshold) + 1e-8)
    st.caption(
        f"Validation threshold: {threshold:.4f} · relative decision margin: {margin:.1%} (not a calibrated probability)"
    )

tab_visual, tab_convergence, tab_model = st.tabs(
    ["Defect localization", "Convergence", "Model provenance"]
)
with tab_visual:
    a, b, c, d = st.columns(4)
    a.image(display_image, caption="Original", use_container_width=True)
    b.image(
        normalize_map(anomaly_map),
        caption="Pixel anomaly map",
        clamp=True,
        use_container_width=True,
    )
    c.image(overlay, caption="Heatmap overlay", use_container_width=True)
    d.image(
        box_image(image, anomaly_map), caption="Estimated defect region", use_container_width=True
    )
with tab_convergence:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.arange(1, len(deltas) + 1),
            y=deltas,
            mode="lines+markers",
            name="representation delta",
        )
    )
    figure.add_hline(
        y=model.convergence_threshold, line_dash="dash", annotation_text="stopping threshold"
    )
    figure.update_layout(
        xaxis_title="Recursive iteration", yaxis_title="Relative representation delta", height=420
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        {"Iteration": np.arange(1, len(deltas) + 1), "Delta": deltas},
        hide_index=True,
        use_container_width=True,
    )
with tab_model:
    st.json(
        {
            "model_version": "KORNet 0.1.0",
            "checkpoint": str(Path(selected).name),
            "dataset": config["dataset"].get("name"),
            "category": config["dataset"].get("category"),
            "backbone": config["model"].get("backbone"),
            "ranking": sorting,
            "measured_test_metrics": metrics or "Not evaluated",
        }
    )
