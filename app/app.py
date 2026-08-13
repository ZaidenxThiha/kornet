from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
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
from utils.checkpoint import confined_checkpoint, load_checkpoint, load_checkpoint_payload
from utils.device import default_device, synchronize
from utils.inference import (
    calibrated_image_threshold,
    gaussian_smooth,
    protocol_matches,
    resolve_inference_protocol,
)
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

DEMO_ROOT = Path("/tmp/kornet_bottle_demo")
DEMO_ASSETS = {
    "best.pt": (
        (
            "https://github.com/ZaidenxThiha/kornet/releases/download/"
            "demo-model-v1/kornet_bottle_demo.pt"
        ),
        "9b130ff573db482e7724949f4d61607197f588231792ba1cc3d30ecfbf29559f",
    ),
    "metrics.json": (
        (
            "https://github.com/ZaidenxThiha/kornet/releases/download/"
            "demo-model-v1/kornet_bottle_metrics.json"
        ),
        "f938422e6548313e99dc119ada8235e9a07684c3ada5f00311368a34517a79fe",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_resource(show_spinner="Downloading verified Bottle demo model…")
def demo_checkpoint() -> Path:
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected_hash) in DEMO_ASSETS.items():
        destination = DEMO_ROOT / filename
        if destination.exists() and _sha256(destination) == expected_hash:
            continue
        temporary = destination.with_suffix(f"{destination.suffix}.download")
        urllib.request.urlretrieve(url, temporary)
        if _sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Integrity verification failed for {filename}")
        temporary.replace(destination)
    return DEMO_ROOT / "best.pt"


def checkpoints():
    local = sorted(PROJECT_ROOT.glob("runs/**/best.pt")) + sorted(
        PROJECT_ROOT.glob("checkpoints/*.pt")
    )
    return local or [demo_checkpoint()]


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str):
    path = confined_checkpoint(
        path,
        [PROJECT_ROOT / "runs", PROJECT_ROOT / "checkpoints", DEMO_ROOT],
    )
    state = load_checkpoint_payload(path)
    config = state.get("config")
    if not config:
        raise ValueError("Checkpoint has no embedded configuration")
    device = default_device()
    model = build_model(config["model"]).to(device).eval()
    load_checkpoint(path, model, map_location=device)
    metrics_path = Path(path).with_name("metrics.json")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    protocol = resolve_inference_protocol(config)
    threshold = calibrated_image_threshold(metrics, protocol)
    return model, config, device, threshold, metrics, protocol


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
    selected = (
        st.selectbox(
            "Checkpoint",
            available,
            format_func=lambda p: (
                str(p.relative_to(PROJECT_ROOT))
                if PROJECT_ROOT in p.resolve().parents
                else "Optimized KORNet · MVTec Bottle demo"
            ),
        )
        if available
        else None
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
    model, config, device, threshold, metrics, protocol = load_model(str(selected))
    image = Image.open(uploaded).convert("RGB")
    size = config["dataset"].get("image_size", 256)
    tensor, _ = build_transform(size)(image)
    with st.spinner("Running recursive opposing-ranking analysis…"), torch.inference_mode():
        start = time.perf_counter()
        output = model(
            tensor[None].to(device),
            adaptive=protocol.adaptive,
            sort_mode=protocol.sort_mode,
        )
        synchronize(device)
        latency = (time.perf_counter() - start) * 1000
except Exception as exc:  # noqa: BLE001 - the UI must render model/load failures cleanly
    st.error(f"Unable to run this trusted checkpoint: {type(exc).__name__}")
    st.stop()

score = float(output["image_score"][0])
iterations = int(output["iterations"][0])
deltas = output["deltas"][0, :iterations].cpu().numpy()
anomaly_map = gaussian_smooth(output["anomaly_map"], protocol.gaussian_sigma)
anomaly_map = anomaly_map[0, 0].cpu().numpy()
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
            "inference_protocol": protocol.as_dict(),
            "metrics_protocol_match": protocol_matches(metrics, protocol),
            "measured_test_metrics": metrics or "Not evaluated",
        }
    )
