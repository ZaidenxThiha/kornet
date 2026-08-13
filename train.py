from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_datasets
from losses import KORNetLoss
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config, save_config
from utils.device import default_device
from utils.inference import resolve_inference_protocol
from utils.seed import seed_everything, seed_worker
from utils.visualization import plot_training


def make_loader(dataset, batch_size, shuffle, workers, device, seed):
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**options)


def scheduler_for(optimizer, epochs, warmup):
    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.inference_mode()
def validate(model, loader, criterion, device, protocol):
    """Validate normals and deterministic patch-flip proxies with the deployment scorer."""
    model.eval()
    losses, scores, proxy_scores, stds = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        output = model(
            images,
            adaptive=protocol.adaptive,
            sort_mode=protocol.sort_mode,
        )
        items = criterion(
            output,
            model.anomaly_head.prototype,
            bool(model.anomaly_head.prototype_initialized),
        )
        losses.append(float(items["total"]))
        stds.append(float(items["feature_std"]))
        scores.extend(output["image_score"].detach().cpu().tolist())
        proxy_images = images.clone()
        height, width = proxy_images.shape[-2:]
        y0, y1 = height // 4, 3 * height // 4
        x0, x1 = width // 4, 3 * width // 4
        proxy_images[..., y0:y1, x0:x1] = torch.flip(proxy_images[..., y0:y1, x0:x1], dims=(-1,))
        proxy_output = model(
            proxy_images,
            adaptive=protocol.adaptive,
            sort_mode=protocol.sort_mode,
        )
        proxy_scores.extend(proxy_output["image_score"].detach().cpu().tolist())
    proxy_labels = np.r_[np.zeros(len(scores)), np.ones(len(proxy_scores))]
    combined_scores = np.r_[scores, proxy_scores]
    proxy_auroc = float(roc_auc_score(proxy_labels, combined_scores))
    proxy_separation = float(
        (np.mean(proxy_scores) - np.mean(scores)) / (np.std(combined_scores) + 1e-8)
    )
    return {
        "val_loss": float(np.mean(losses)),
        "val_proxy_auroc": proxy_auroc,
        "val_proxy_separation": proxy_separation,
        "val_proxy_score": proxy_auroc + 1e-3 * float(np.tanh(proxy_separation)),
        "val_score_mean": float(np.mean(scores)),
        "val_score_p99": float(np.percentile(scores, 99)),
        "val_feature_std": float(np.mean(stds)),
    }


def train(config: dict) -> Path:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    if config["model"].get("sort_mode", "soft") != "soft":
        raise ValueError(
            "Training requires differentiable soft sorting; hard sorting is inference-only"
        )
    protocol = resolve_inference_protocol(config)
    device = default_device()
    datasets = build_datasets(config["dataset"])
    training = config["training"]
    selection_metric = training.get("checkpoint_metric", "val_proxy_score")
    if selection_metric not in {"val_proxy_score", "val_proxy_auroc", "val_loss"}:
        raise ValueError(
            "checkpoint_metric must be 'val_proxy_score', 'val_proxy_auroc', or 'val_loss'"
        )
    maximize_selection = selection_metric != "val_loss"
    workers = int(os.environ.get("KORNET_NUM_WORKERS", config["dataset"].get("num_workers", 4)))
    train_loader = make_loader(
        datasets["train"],
        training["batch_size"],
        True,
        workers,
        device,
        seed,
    )
    val_loader = make_loader(
        datasets["val"],
        training["batch_size"],
        False,
        workers,
        device,
        seed + 1,
    )
    model = build_model(config["model"]).to(device)
    criterion = KORNetLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = scheduler_for(optimizer, training["epochs"], training.get("warmup_epochs", 0))
    amp_enabled = bool(training.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run_dir = (
        Path(config.get("output_dir", "runs"))
        / config["experiment_name"]
        / f"seed_{config.get('seed', 42)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(run_dir / "tensorboard")
    except ImportError:
        writer = None
    wandb_run = None
    if training.get("wandb", False):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Set training.wandb=false or install the 'research' extra") from exc
        wandb_run = wandb.init(project="kornet", name=config["experiment_name"], config=config)
    start_epoch = 0
    best_value = -float("inf") if maximize_selection else float("inf")
    resume = training.get("resume")
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = int(state["epoch"]) + 1
        if state.get("metric_name") == selection_metric and state.get("best_metric") is not None:
            best_value = float(state["best_metric"])
    history, stale_epochs = [], 0
    history_path = run_dir / "history.csv"
    if start_epoch and history_path.exists():
        with history_path.open(newline="") as handle:
            for saved_row in csv.DictReader(handle):
                history.append(
                    {
                        key: int(value) if key == "epoch" else float(value)
                        for key, value in saved_row.items()
                    }
                )
        validation_values = [row[selection_metric] for row in history if selection_metric in row]
        if validation_values:
            selected = max(validation_values) if maximize_selection else min(validation_values)
            best_value = (
                max(best_value, selected) if maximize_selection else min(best_value, selected)
            )
            last_best_index = max(
                index for index, value in enumerate(validation_values) if value == selected
            )
            stale_epochs = len(validation_values) - last_best_index - 1
    for epoch in range(start_epoch, training["epochs"]):
        model.train()
        totals, feature_stds = [], []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{training['epochs']}",
            leave=False,
            disable=os.environ.get("KORNET_QUIET") == "1",
        )
        adaptive_training = bool(
            getattr(model, "adaptive_stop", False)
            and epoch >= training.get("fixed_iterations_until_epoch", training["epochs"])
        )
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(
                    images,
                    adaptive=adaptive_training,
                    iterations=model.max_iterations if adaptive_training else None,
                )
                items = criterion(
                    output,
                    model.anomaly_head.prototype,
                    bool(model.anomaly_head.prototype_initialized),
                )
            scaler.scale(items["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("gradient_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            model.anomaly_head.update_prototype(output["tokens"], output.get("initial_tokens"))
            totals.append(float(items["total"].detach()))
            feature_stds.append(float(items["feature_std"]))
            progress.set_postfix(loss=f"{totals[-1]:.4f}")
        scheduler.step()
        validation = validate(model, val_loader, criterion, device, protocol)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(totals)),
            "feature_std": float(np.mean(feature_stds)),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **validation,
        }
        history.append(row)
        with history_path.open("w", newline="") as handle:
            output_csv = csv.DictWriter(handle, fieldnames=row.keys())
            output_csv.writeheader()
            output_csv.writerows(history)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch + 1)
        if wandb_run:
            wandb_run.log(row, step=epoch + 1)
        current_value = validation[selection_metric]
        improved = current_value > best_value if maximize_selection else current_value < best_value
        if improved:
            best_value, stale_epochs = current_value, 0
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_value,
                config,
                selection_metric,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_value,
            config,
            selection_metric,
        )
        if row["feature_std"] < 1e-3:
            print("WARNING: feature variance is near zero; possible representation collapse.")
        if stale_epochs >= training.get("early_stopping_patience", 15):
            break
    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()
    plot_training(history, run_dir / "training.png")
    return run_dir / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train KORNet using normal images only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--category")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant")
    parser.add_argument("--resume", help="Resume from a last.pt checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.category:
        config["dataset"]["category"] = args.category
        config["experiment_name"] = (
            f"{config['model']['variant']}_{config['dataset']['name']}_{args.category}"
        )
    if args.seed is not None:
        config["seed"] = args.seed
        config["dataset"]["seed"] = args.seed
    if args.variant:
        config["model"]["variant"] = args.variant
        config["experiment_name"] = (
            f"{args.variant}_{config['dataset']['name']}_{config['dataset']['category']}"
        )
    if args.resume:
        config["training"]["resume"] = args.resume
    checkpoint = train(config)
    print(f"Best checkpoint: {checkpoint.resolve()}")


if __name__ == "__main__":
    main()
