# MASTER CURSOR PROMPT — KORNet Research Project

You are a senior machine-learning researcher, computer-vision engineer, PyTorch engineer, and MLOps engineer.

Build a complete research-quality project called:

# KORNet

## Kaprekar-Inspired Recursive Opposing-Ranking Network for Automated Visual Quality Inspection

The purpose is to investigate whether a neural operator inspired by Kaprekar's 6174 routine can improve the accuracy/efficiency tradeoff of visual anomaly and defect detection.

Do NOT claim that KORNet is state-of-the-art or "best in the world" unless experimental results actually demonstrate it against strong baselines.

Do NOT fabricate metrics.

The project must be reproducible, modular, GPU-ready, and suitable for a university research paper.

---

# 1. RESEARCH IDEA

Kaprekar's four-digit process repeatedly performs:

1. Arrange values high → low.
2. Arrange the same values low → high.
3. Calculate their difference.
4. Repeat until reaching a stable attractor.

Generalize this principle into a differentiable neural operator.

For latent visual features X_t:

D_t = S_down(X_t) - S_up(X_t)

Then:

X_(t+1) = LayerNorm(
X_t + g_t * Phi(D_t)
)

where:

* S_down = learned differentiable descending ordering
* S_up = learned differentiable ascending ordering
* D_t = opposing-ranking difference
* Phi = learned nonlinear transformation
* g_t = adaptive update gate

Apply the SAME KOR block recursively:

X_0 → K(X_0) → X_1 → K(X_1) → ... → X_T

Use shared weights across iterations.

Maximum default iterations:

T_max = 7

The number 7 is inspired by the convergence bound of the classic four-digit Kaprekar 6174 routine, but this must remain configurable and must be tested experimentally.

Do not numerically force representations toward the number 6174.

Instead, learn useful stable latent attractors.

---

# 2. KOR OPERATOR

Implement a module:

KORBlock

Input:

[B, N, D]

where:

B = batch size
N = number of visual tokens
D = feature dimension

First calculate a learned ranking score for every token:

r_i = RankNet(x_i)

Support multiple ranking heads:

H = configurable, default 4

For every head h:

create a differentiable approximation of a descending permutation:

P_down_h

and ascending permutation:

P_up_h

Then:

X_down_h = P_down_h @ X

X_up_h = P_up_h @ X

D_h = X_down_h - X_up_h

Combine heads:

D = projection(concat(D_1, ..., D_H))

Map the ranked result back into the original/token coordinate representation where appropriate.

Apply:

U = Phi(D)

where Phi initially contains lightweight operations such as:

Linear
→ GELU
→ Dropout
→ Linear

or efficient 1D/depthwise token mixing when experimentally beneficial.

Calculate adaptive gate:

g_t = sigmoid(
GateNet(
concat(
pooled(X_t),
pooled(abs(D_t))
)
)
)

Update:

X_(t+1) =
LayerNorm(
X_t + g_t * U
)

Implement stable residual connections.

Avoid numerical explosion.

---

# 3. DIFFERENTIABLE SORTING

Sorting is critical to the research contribution.

Implement at least two modes.

MODE A — differentiable soft sorting for training.

Use a mathematically valid differentiable sorting/ranking method such as NeuralSort, Sinkhorn-based soft permutation, SoftSort, or a compatible differentiable sorting library.

Do not silently convert the entire training operator into non-differentiable argsort.

MODE B — hard ranking for inference.

Use efficient torch.argsort/top-k when appropriate.

Benchmark:

soft sorting inference

vs

hard sorting inference.

Because full differentiable sorting may become expensive when N is large, implement configurable token reduction:

adaptive pooling

or

top-k salient tokens

or

spatial patch pooling.

Keep the design scientifically fair.

Do not hide excessive O(N²) complexity.

Record actual FLOPs and latency.

---

# 4. CNN FEATURE EXTRACTOR

Use a strong but reasonably efficient CNN backbone.

Support:

ResNet18
EfficientNet-B0
ConvNeXt-Tiny

Prefer pretrained ImageNet weights when available.

Allow:

--backbone resnet18
--backbone efficientnet_b0
--backbone convnext_tiny

Extract multi-scale intermediate features.

Example:

F1
F2
F3

Project each feature level into common dimension D.

Convert spatial features into tokens.

Preserve spatial coordinates so the model can later generate anomaly maps.

Important:

The CNN and KOR operator must remain separate modules so we can perform fair ablation experiments.

---

# 5. FULL KORNET ARCHITECTURE

Implement:

Image
↓
CNN Backbone
↓
Multi-scale Feature Extraction
↓
Token Projection
↓
KORBlock
↓
KORBlock
↓
...
shared recursively
↓
Adaptive Convergence
↓
Anomaly Representation
↓
Image Anomaly Score + Pixel Anomaly Map

Use shared KOR parameters across recursive iterations.

Implement adaptive stopping.

Calculate:

delta_t =
||X_(t+1) - X_t|| /
(||X_t|| + epsilon)

Stop inference early if:

delta_t < convergence_threshold

for a configurable number of consecutive iterations.

Default:

max_iterations = 7

min_iterations = 1

Do NOT use dynamic stopping during early training if it causes unstable gradients.

Initially train using a fixed number of unrolled iterations.

Enable adaptive stopping during later training/inference.

---

# 6. ANOMALY DETECTION STRATEGY

The main problem is unsupervised / one-class visual anomaly detection.

Training must primarily use NORMAL images, because benchmarks such as MVTec AD follow this setting.

Do not leak test defect labels into training.

Learn compact normal attractor representations.

Implement a normal prototype / attractor memory:

C_normal

Encourage normal representations to approach stable normal feature regions.

Possible anomaly score:

A_attractor =
distance(X_final, C_normal)

Also calculate:

A_dynamic =
mean_t(
||X_(t+1)-X_t||
)

and:

A_rank =
statistics of opposing-ranking differences

Combine using learned/calibrated weights:

A =
w1*A_attractor

* w2*A_dynamic
* w3*A_rank
* optional reconstruction/localization signal

Do not assume this combination is automatically optimal.

Make components individually switchable for ablation studies.

Generate pixel-level anomaly maps from spatial token anomaly scores.

Upsample anomaly maps to original image resolution.

Optionally apply light Gaussian smoothing only during evaluation if standard benchmark protocols justify it.

---

# 7. TRAINING LOSSES

Create modular losses.

Main normal compactness / representation loss.

Add convergence loss:

L_convergence =
mean(
||X_(t+1)-X_t||²
)

for later recursive iterations.

Add stability loss:

L_stability =
||K(X_final)-X_final||²

Add representation regularization.

If augmented pseudo-anomalies are used, implement them carefully and separately so performance can be measured with and without synthetic anomalies.

Total:

L =
L_main

* lambda_c * L_convergence
* lambda_s * L_stability
* lambda_r * L_regularization

All lambda values configurable through YAML.

Do not minimize convergence so aggressively that all inputs collapse into identical representations.

Implement anti-collapse checks.

Monitor feature variance during training.

---

# 8. DATASETS

Support these datasets:

MVTec AD
VisA
MVTec AD 2

Optional later support:

MVTec LOCO AD

Create:

datasets/
mvtec.py
visa.py
mvtec_ad2.py

Provide download/setup scripts only where licensing and hosting rules legally allow automated downloading.

Where automated downloading is not allowed, print clear instructions explaining where the user must manually place the dataset.

Expected structure should be documented in README.md.

Implement:

train
validation
test

splits correctly.

Never use test anomalies for training or hyperparameter optimization.

Support category-specific and all-category experiments.

Data augmentation for normal training:

Resize
RandomCrop
HorizontalFlip where semantically appropriate
ColorJitter
small rotations
normalization

Do not use transformations that create unrealistic industrial defects unless explicitly running the synthetic-anomaly experiment.

---

# 9. BASELINES

This is extremely important.

Implement fair internal baselines:

A. CNN only

B. CNN + lightweight attention

C. CNN + single non-recursive KOR block

D. CNN + recursive KOR

E. CNN + recursive KOR + adaptive stopping

Use EXACTLY the same CNN backbone whenever comparing the contribution of KOR.

Also provide integration/evaluation instructions for strong established anomaly-detection baselines such as:

PatchCore
PaDiM
EfficientAD
Reverse Distillation
FastFlow if compatible

Prefer official implementations or reliable established libraries.

Do not rewrite every baseline badly from scratch and then claim KORNet beats them.

Record version/commit information where possible.

---

# 10. ABLATION EXPERIMENTS

Create scripts for:

KORNet without opposing subtraction

KORNet without recursion

KORNet with one ranking head

KORNet with 2 / 4 / 8 ranking heads

KORNet with max iterations:

1
2
3
5
7
10

KORNet with fixed iterations

vs

adaptive stopping

soft sorting

vs

hard sorting inference

single-scale CNN features

vs

multi-scale CNN features

ranking by magnitude

vs

learned ranking

with convergence loss

vs

without convergence loss

These experiments are mandatory because we need to prove whether the Kaprekar mechanism contributes to performance.

---

# 11. PERFORMANCE METRICS

Measure anomaly detection quality using appropriate benchmark metrics:

Image-level AUROC

Pixel-level AUROC

AUPRO / PRO where applicable

Average Precision

F1

Precision

Recall

False Positive Rate

Also measure efficiency:

Number of parameters

Model disk size

FLOPs / MACs

Peak GPU memory

CPU inference latency

GPU inference latency

Throughput images/sec

Average recursive iterations

Median recursive iterations

95th percentile recursive iterations

Energy measurement if practical

Create an important metric:

Accuracy-Efficiency Score

but do not replace standard metrics with a custom metric.

Report standard metrics first.

---

# 12. STATISTICAL RELIABILITY

For important experiments:

run at least 3 random seeds.

Report:

mean
standard deviation

Save every experiment configuration.

Use deterministic seeds where technically possible.

Do not cherry-pick the best run.

---

# 13. TRAINING OPTIMIZATION

Implement:

PyTorch

CUDA support

automatic mixed precision

gradient clipping

AdamW

cosine learning-rate scheduler

warmup

early stopping

checkpoint saving

resume training

TensorBoard logging

optional Weights & Biases support

Use DataLoader optimizations:

pin_memory
persistent_workers
configurable num_workers
prefetch_factor

Support:

python train.py --config configs/kornet_mvtec.yaml

Example configuration:

dataset: mvtec
category: bottle

backbone: efficientnet_b0

image_size: 256

feature_dim: 256

rank_heads: 4

max_iterations: 7

train_iterations: 4

adaptive_stop: true

convergence_threshold: 0.001

batch_size: 16

epochs: 100

optimizer: adamw

learning_rate: 0.0001

weight_decay: 0.0001

mixed_precision: true

---

# 14. AUTOMATIC HYPERPARAMETER SEARCH

Add optional Optuna support.

Search:

learning rate

feature dimension

ranking heads

temperature for differentiable sorting

number of recursive iterations

convergence threshold

loss weights

dropout

token count

Do NOT optimize using the final test set.

Use validation data only.

Save:

best_params.json

---

# 15. PROJECT STRUCTURE

Create a clean structure similar to:

kornet/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   ├── kornet_mvtec.yaml
│   ├── kornet_visa.yaml
│   └── kornet_mvtec_ad2.yaml
│
├── datasets/
│   ├── mvtec.py
│   ├── visa.py
│   ├── mvtec_ad2.py
│   └── transforms.py
│
├── models/
│   ├── backbone.py
│   ├── differentiable_sort.py
│   ├── kor_operator.py
│   ├── kornet.py
│   ├── anomaly_head.py
│   └── baseline_models.py
│
├── losses/
│   └── losses.py
│
├── utils/
│   ├── metrics.py
│   ├── profiling.py
│   ├── visualization.py
│   ├── seed.py
│   └── checkpoint.py
│
├── scripts/
│   ├── prepare_mvtec.py
│   ├── prepare_visa.py
│   ├── prepare_mvtec_ad2.py
│   ├── train_all_categories.py
│   ├── benchmark.py
│   └── ablation.py
│
├── app/
│   └── app.py
│
├── tests/
│   ├── test_sort.py
│   ├── test_kor_operator.py
│   ├── test_model.py
│   └── test_dataset.py
│
├── train.py
├── evaluate.py
├── predict.py
└── export.py

---

# 16. TEST APPLICATION

Build a professional Streamlit application.

Run:

streamlit run app/app.py

The application must allow the user to:

Upload JPG/PNG image.

Select trained dataset/category model.

Run KORNet inference.

Display:

original image

NORMAL / DEFECT result

anomaly probability / anomaly score

pixel-level heatmap

heatmap overlay

estimated defect region

number of KOR iterations used

convergence curve

CPU/GPU inference latency

confidence information

model version

checkpoint name

Create a visual panel:

KORNet Analysis

Status:
NORMAL / DEFECT

Anomaly Score:
0.xx

Iterations:
x / 7

Inference:
xx ms

Also display a convergence graph:

iteration
vs
representation delta

Example:

Iteration 1: 0.43
Iteration 2: 0.17
Iteration 3: 0.052
Iteration 4: 0.008
Iteration 5: 0.0009 → converged

This is important because it demonstrates the Kaprekar-inspired recursive behavior.

---

# 17. VISUALIZATION

Generate research-quality visualizations:

training loss

validation metrics

ROC curve

precision-recall curve

normal vs anomaly score distribution

anomaly heatmaps

convergence curves

iterations distribution

accuracy vs latency

AUROC vs FLOPs

AUROC vs parameter count

baseline comparison charts

KOR ranking visualizations

Show which visual tokens/features receive highest and lowest ranking scores.

This will help interpret whether the opposing-ranking mechanism actually identifies meaningful defect regions.

---

# 18. MODEL EXPORT

Support:

PyTorch checkpoint

TorchScript if compatible

ONNX

Optional later:

TensorRT

Benchmark exported models separately.

Make hard-ranking inference an optional optimized deployment mode.

---

# 19. TESTING

Write unit tests.

Verify:

tensor shapes

gradient flow through soft ranking

no NaNs

recursive parameter sharing

adaptive stopping

CPU inference

GPU inference if available

dataset loader correctness

model save/load

heatmap generation

Test permutation behavior of the KOR operator.

Test that gradients reach RankNet.

---

# 20. SAFETY AGAINST BAD RESEARCH

The project must automatically protect against common research mistakes.

Never:

train using test labels

optimize thresholds on the final test set

report fabricated results

compare different backbones unfairly

hide failed experiments

claim SOTA without verified comparisons

If KORNet performs worse than a baseline, report it honestly.

Save all experiment results as CSV/JSON.

---

# 21. RESULTS TABLE

Automatically generate a table:

Model | Image AUROC | Pixel AUROC | AUPRO | F1 | Params | FLOPs | GPU ms | CPU ms | Avg Iterations

Include:

CNN

CNN+Attention

CNN+KOR-1

KORNet Fixed

KORNet Adaptive

PatchCore

PaDiM

EfficientAD

other successfully reproduced baselines

Highlight the best metric only after actual evaluation.

---

# 22. RESEARCH GOAL

The primary hypothesis is:

"Can a Kaprekar-inspired recursive opposing-ranking neural operator improve the accuracy-efficiency tradeoff of visual anomaly detection?"

Secondary hypotheses:

1. Opposing ranking exposes anomalous feature relationships.

2. Shared recursive processing reduces parameter requirements.

3. Adaptive stopping reduces average inference computation.

4. Convergence behavior itself provides useful anomaly information.

5. Learned ranking performs better than simple magnitude ranking.

Every hypothesis must be testable through ablation experiments.

---

# 23. IMPORTANT: DO NOT OVERENGINEER VERSION 1

First create a working MVP:

CNN backbone
+
single KOR block
+
shared recursion
+
normal-only training
+
anomaly score
+
MVTec AD
+
evaluation
+
Streamlit app

Make that work end-to-end first.

Then add:

multi-head ranking

adaptive stopping

multi-scale features

VisA

MVTec AD 2

Optuna

additional baselines

ONNX

advanced optimization

Do not build 30 unfinished modules before confirming that the fundamental model can train.

---

# 24. DEVELOPMENT PROCESS

Work autonomously.

Do not ask unnecessary questions.

When something is ambiguous, choose a reasonable research-standard default and document the decision.

After each major implementation stage:

run tests

run a tiny synthetic training test

fix all errors

continue only when the current stage works.

Use small synthetic data when necessary to verify the pipeline before downloading/training large datasets.

---

# 25. README

Create a detailed README explaining:

What KORNet is

Kaprekar/6174 inspiration

Mathematical equations

Architecture diagram using Mermaid

Installation

Dataset preparation

Training commands

Evaluation commands

App commands

Benchmark commands

Ablation commands

Project structure

Results table

Limitations

Research status

Clearly state:

"KORNet is an experimental research architecture. Performance claims must be supported by reproducible benchmark results."

---

# 26. FINAL TARGET

The objective is NOT simply:

highest accuracy.

Optimize the Pareto frontier between:

Accuracy ↑

AUROC ↑

AUPRO ↑

F1 ↑

while:

Parameters ↓

FLOPs ↓

Latency ↓

Memory ↓

Average iterations ↓

The ideal result is:

comparable or better anomaly-detection quality than strong existing models

while requiring significantly less computation or memory.

If KORNet loses in accuracy but wins strongly in efficiency, preserve and analyze that result.

If KORNet wins in accuracy but is inefficient, optimize the operator.

If KORNet loses in both accuracy and efficiency, diagnose why and redesign the operator based on experimental evidence rather than hiding the result.

---

# 27. START NOW

Begin by creating the repository structure.

Then implement in this order:

environment and configuration
→ dataset loader
→ CNN feature extractor
→ differentiable ranking
→ KORBlock
→ KORNet
→ losses
→ training loop
→ evaluation
→ MVTec AD experiment
→ baseline CNN
→ ablations
→ Streamlit app
→ additional datasets
→ optimization
→ final benchmark report

After implementation, provide exact terminal commands for:

installation

dataset setup

training one category

training all categories

evaluation

running baselines

running ablations

launching the Streamlit app

exporting the trained model

Do not leave placeholder functions or TODO implementations in the core pipeline.

Build an executable research project, not pseudocode.
