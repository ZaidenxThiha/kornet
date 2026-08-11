# KORNet

**Kaprekar-Inspired Recursive Opposing-Ranking Network for Automated Visual Quality Inspection**

KORNet is an experimental one-class visual-anomaly architecture. It asks whether a shared recursive operator, inspired by the opposing orderings in Kaprekar's 6174 routine, can improve the accuracy–efficiency tradeoff of industrial defect detection. It does **not** push representations toward the number 6174, and this repository makes no state-of-the-art claim.

> **KORNet is an experimental research architecture. Performance claims must be supported by reproducible benchmark results.**

## Method

For token features \(X_t\in\mathbb{R}^{B\times N\times D}\), each ranking head learns scores, creates differentiable descending and ascending permutation relaxations, and forms an opposing difference:

\[
D_t^{(h)} = P_{\downarrow}^{(h)}X_t-P_{\uparrow}^{(h)}X_t,
\qquad
X_{t+1}=\mathrm{LayerNorm}\left(X_t+g_t\Phi(D_t)\right).
\]

The same `KORBlock` parameters are reused at every iteration. At inference, a sample may stop when

\[
\delta_t=\frac{\lVert X_{t+1}-X_t\rVert_2}{\lVert X_t\rVert_2+\epsilon}
\]

remains below a configurable threshold. Training uses a fixed unroll by default for stable gradients. The score exposes three independently ablatable signals: distance from the normal prototype, recursive dynamics, and opposing-ranking magnitude.

```mermaid
flowchart LR
    A["Image"] --> B["CNN backbone"]
    B --> C["Multi-scale projected tokens"]
    C --> D["Shared KORBlock"]
    D --> E{"Converged or max iterations?"}
    E -- "No" --> D
    E -- "Yes" --> F["Normal-attractor distance"]
    F --> G["Image score + pixel map"]
```

The soft training path uses NeuralSort and costs \(O(BHN^2)\). `max_tokens` makes that expense explicit and bounded. The hard deployment path uses `torch.argsort`. Multi-scale maps are reconstructed independently and averaged at input resolution.

## Install

Python 3.10+ is required. CUDA is used automatically when available.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[app,research,export,dev]'
pytest
```

For a minimal training environment, use `pip install -r requirements.txt`.

## Datasets and leakage policy

KORNet trains on **normal training images only**. The loader creates a deterministic held-out normal validation split. This validation split calibrates anomaly thresholds and hyperparameters. Test anomalies are read only by final evaluation.

Dataset downloads are not automated because users must review the current license/terms at each official source.

```bash
python scripts/prepare_mvtec.py
python scripts/prepare_visa.py
python scripts/prepare_mvtec_ad2.py
```

Expected MVTec AD structure:

```text
data/mvtec/bottle/
├── train/good/*.png
├── test/good/*.png
├── test/<defect>/*.png
└── ground_truth/<defect>/*_mask.png
```

VisA accepts the official `split_csv/1cls.csv` layout. MVTec AD 2 accepts its official `train/good`, `test` or `test_public` hierarchy. Private MVTec AD 2 test labels must not be used for calibration or model selection.

## Commands

Train one category:

```bash
python train.py --config configs/kornet_mvtec.yaml
python train.py --config configs/kornet_mvtec.yaml --category cable --seed 123
```

Train all categories with the default three seeds (42, 123, 456):

```bash
python scripts/train_all_categories.py --config configs/kornet_mvtec.yaml
```

Evaluate. The first command matches soft training; the second benchmarks exact hard ranking. Both calibrate thresholds using validation normals before touching the test loader.

```bash
python evaluate.py --checkpoint runs/kornet_mvtec_bottle/seed_42/best.pt --sort-mode soft
python evaluate.py --checkpoint runs/kornet_mvtec_bottle/seed_42/best.pt --sort-mode hard --output runs/kornet_mvtec_bottle/seed_42/metrics_hard.json
```

Run fair internal baselines with the same configuration/backbone:

```bash
python train.py --config configs/kornet_mvtec.yaml --variant cnn
python train.py --config configs/kornet_mvtec.yaml --variant attention
python train.py --config configs/kornet_mvtec.yaml --variant kor_single
python train.py --config configs/kornet_mvtec.yaml --variant kornet_fixed
python train.py --config configs/kornet_mvtec.yaml --variant kornet_adaptive
```

Run the mandatory ablations, or select a subset:

```bash
python scripts/ablation.py --config configs/kornet_mvtec.yaml
python scripts/ablation.py --config configs/kornet_mvtec.yaml --experiments no_opposing heads_1 heads_4 iterations_1 iterations_7 fixed magnitude_ranking single_scale no_convergence_loss
python scripts/ablation.py --config configs/kornet_mvtec.yaml --dry-run
```

The ablation registry covers opposing subtraction, recursion, 1/2/4/8 heads, 1/2/3/5/7/10 iterations, fixed/adaptive stopping, soft/hard inference (hard is evaluated from the same checkpoint), single/multi-scale features, magnitude/learned ranking, and convergence loss.

Validation-only Optuna search:

```bash
python scripts/optuna_search.py --config configs/kornet_mvtec.yaml --trials 20
```

Aggregate real evaluation files—missing baselines stay missing rather than receiving invented values:

```bash
python scripts/benchmark.py --results runs --output runs/benchmark.csv
```

Inspect one image and launch the application:

```bash
python predict.py --checkpoint runs/kornet_mvtec_bottle/seed_42/best.pt --image sample.png --threshold 0.42
streamlit run app/app.py
```

Export and benchmark the deployment operator separately:

```bash
python export.py --checkpoint runs/kornet_mvtec_bottle/seed_42/best.pt --format torchscript --sort-mode hard --output outputs/kornet.ts
python export.py --checkpoint runs/kornet_mvtec_bottle/seed_42/best.pt --format onnx --sort-mode soft --output outputs/kornet.onnx
```

## Configuration

The YAML files control the backbone (`resnet18`, `efficientnet_b0`, `convnext_tiny`), feature dimension, ranking heads and temperature, token budget, recursive limits, stopping policy, score weights, optimizer, and every loss coefficient. Pretrained ImageNet weights are preferred; if they cannot be downloaded, the backbone emits a warning and remains executable with random initialization.

The normal prototype is an exponential moving average. The objective combines compactness, late-iteration convergence, fixed-point stability, and a variance hinge. Feature variance is logged each epoch and values below `1e-3` trigger an anti-collapse warning.

## Metrics and research protocol

`evaluate.py` writes JSON containing image AUROC, average precision, F1, precision, recall, FPR, pixel AUROC, pixel average precision, AUPRO, parameter count, disk footprint, latency, and mean/median/p95 iterations. A custom accuracy-efficiency score exists in `utils/metrics.py`, but standard metrics must always be reported first.

For paper tables, report mean ± standard deviation over at least three predetermined seeds. Do not select the best seed. Preserve every run directory and configuration, including negative results. Compare KOR variants using exactly the same backbone, image resolution, data split, and training budget.

| Model | Image AUROC | Pixel AUROC | AUPRO | F1 | Params | FLOPs | GPU ms | CPU ms | Avg Iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | — | — | — | — | — | — | — | — | 0 |
| CNN + attention | — | — | — | — | — | — | — | — | 1 |
| CNN + KOR-1 | — | — | — | — | — | — | — | — | 1 |
| KORNet fixed | — | — | — | — | — | — | — | — | — |
| KORNet adaptive | — | — | — | — | — | — | — | — | — |
| PatchCore | — | — | — | — | — | — | — | — | — |
| PaDiM | — | — | — | — | — | — | — | — | — |
| EfficientAD | — | — | — | — | — | — | — | — | — |

“—” means **not measured**, not zero. Cells are populated only by reproducible runs.

For established baselines, use their official repositories or a pinned release of a maintained framework such as Anomalib. Record repository URL, commit hash/package version, preprocessing, backbone, hardware, and command beside the resulting JSON. Do not compare KORNet to an improvised reimplementation. PatchCore, PaDiM, EfficientAD, Reverse Distillation, and FastFlow may have distinct backbone or licensing constraints; disclose these rather than implying a controlled KOR ablation.

## Visual outputs

The Streamlit app shows the input, raw heatmap, overlay, estimated region, score, threshold margin, device latency, recursive iterations, and convergence curve. It deliberately labels a result `UNCALIBRATED` when no validation-derived `metrics.json` is present. A raw anomaly score or threshold margin is not described as a probability.

Ranking scores are returned per head and iteration by the model for interpretability experiments. Training curves are saved in every run. `utils/visualization.py` provides research-ready raster heatmaps, while experiment JSON/CSV remains the source of record for plots and paper tables.

## Project structure

```text
configs/                  Reproducible experiment YAML
datasets/                 MVTec AD, VisA, MVTec AD 2 loaders
models/                   Backbones, NeuralSort, KOR, anomaly head, baselines
losses/                   Modular normal-only objective
utils/                    Metrics, profiling, plots, seeds, checkpoints
scripts/                  Preparation, multi-seed, ablation, search, benchmark
app/app.py                Streamlit inspection application
tests/                    Gradient, shape, recursion, stopping, data tests
train.py                  Training and validation
evaluate.py               Leakage-safe test evaluation
predict.py                Single-image CLI
export.py                 TorchScript and ONNX export
```

## Limitations and research status

- No benchmark metrics ship with the repository; large licensed datasets and trained checkpoints are not included.
- NeuralSort is quadratic in token count. Token reduction makes this honest and configurable but may discard small defects.
- The simple global normal prototype can under-represent multimodal normal data. A category-specific memory bank is a justified future comparison.
- AUPRO protocol details differ between libraries. This repository integrates PRO up to FPR 0.3; paper comparisons must use a consistent implementation.
- Gaussian smoothing is evaluation-only and recorded in the output protocol.
- Per-sample stopping still executes batched recursive calls until all samples finish, though completed samples are frozen. Deployment throughput therefore depends on batch composition.
- ONNX operator support varies by runtime, particularly for exact sorting; the soft and hard exports must be validated on the intended runtime.
- The primary hypothesis remains unconfirmed until controlled, repeated experiments beat or improve the efficiency frontier of strong reproduced baselines.

## Research integrity

The code never trains from the test loader, never optimizes Optuna trials on final-test scores, saves the complete configuration in every checkpoint, and does not fabricate missing results. If KORNet loses in accuracy, efficiency, or both, retain that outcome and use the ablations to diagnose it.
