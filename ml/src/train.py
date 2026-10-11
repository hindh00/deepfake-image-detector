"""Training loop: DataLoader -> DualBranchDetector -> multi-task loss -> AdamW + cosine
warmup, with checkpoint save/resume, mixed precision, gradient accumulation, early
stopping, per-epoch diagnostics (train/val curves, per-branch losses, gate-weight
tracking, per-generator-family val accuracy), and per-epoch canary-batch explainability
snapshots - all written after every epoch so a Colab/Kaggle session disconnect doesn't
lose the signal.

Deliberately NOT implemented yet: W&B logging (configs/train.yaml anticipates it via
logging.backend, but nothing in this project needs it yet).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.datasets.dataset import DeepfakeDataset
from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.explainability import explain
from src.models.full_model import DualBranchDetector

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_configs() -> tuple[dict, dict]:
    with open(CONFIG_DIR / "model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "train.yaml") as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def multi_task_loss(outputs: dict, labels: torch.Tensor, loss_cfg: dict) -> tuple[torch.Tensor, dict]:
    fused = F.cross_entropy(outputs["fused_logits"], labels)
    spatial = F.cross_entropy(outputs["spatial_logits"], labels)
    freq = F.cross_entropy(outputs["freq_logits"], labels)
    total = (
        loss_cfg["fused_weight"] * fused
        + loss_cfg["spatial_aux_weight"] * spatial
        + loss_cfg["freq_aux_weight"] * freq
    )
    return total, {"fused": fused.item(), "spatial": spatial.item(), "freq": freq.item()}


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == labels).float().mean().item()


class _EpochStats:
    """Accumulates per-batch scalars (loss components, gate weights, accuracy) and
    per-generator-family correct/total counts over one epoch, then averages them."""

    def __init__(self) -> None:
        self.totals = {"loss": 0.0, "fused": 0.0, "spatial": 0.0, "freq": 0.0, "acc": 0.0, "gate_spatial": 0.0, "gate_freq": 0.0}
        self.n_batches = 0
        self.family_correct: dict[str, int] = {}
        self.family_total: dict[str, int] = {}

    def update(self, loss: float, components: dict, outputs: dict, labels: torch.Tensor, families: tuple[str, ...]) -> None:
        self.totals["loss"] += loss
        self.totals["fused"] += components["fused"]
        self.totals["spatial"] += components["spatial"]
        self.totals["freq"] += components["freq"]
        self.totals["acc"] += _accuracy(outputs["fused_logits"], labels)
        self.totals["gate_spatial"] += outputs["gate_weights"][:, 0].mean().item()
        self.totals["gate_freq"] += outputs["gate_weights"][:, 1].mean().item()
        self.n_batches += 1

        preds = outputs["fused_logits"].argmax(dim=-1)
        for is_correct, family in zip((preds == labels).tolist(), families):
            self.family_total[family] = self.family_total.get(family, 0) + 1
            self.family_correct[family] = self.family_correct.get(family, 0) + int(is_correct)

    def averaged(self) -> dict:
        n = max(self.n_batches, 1)
        return {k: v / n for k, v in self.totals.items()}

    def family_accuracy(self) -> dict[str, float]:
        return {family: self.family_correct[family] / self.family_total[family] for family in self.family_total}


@torch.no_grad()
def evaluate(model: DualBranchDetector, loader: DataLoader, loss_cfg: dict, device: torch.device) -> tuple[dict, dict]:
    model.eval()
    stats = _EpochStats()
    for rgb, freq, label, families in loader:
        rgb, freq, label = rgb.to(device), freq.to(device), label.to(device)
        outputs = model(rgb, freq)
        loss, components = multi_task_loss(outputs, label, loss_cfg)
        stats.update(loss.item(), components, outputs, label, families)
    model.train()
    return stats.averaged(), stats.family_accuracy()


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["best_val_loss"]


def _save_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def _plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(epochs, [h["train"]["loss"] for h in history], label="train")
    ax.plot(epochs, [h["val"]["loss"] for h in history], label="val")
    ax.set_title("Total loss")
    ax.set_xlabel("epoch")
    ax.legend()

    ax = axes[0, 1]
    for branch in ("fused", "spatial", "freq"):
        ax.plot(epochs, [h["val"][branch] for h in history], label=branch)
    ax.set_title("Val per-branch loss")
    ax.set_xlabel("epoch")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [h["val"]["gate_spatial"] for h in history], label="spatial weight")
    ax.plot(epochs, [h["val"]["gate_freq"] for h in history], label="freq weight")
    ax.set_title("Val fusion gate weights")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, [h["val"]["acc"] for h in history], label="val acc (fused)")
    ax.set_title("Val accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _denormalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device).view(1, 3, 1, 1)
    return (rgb * std + mean).clamp(0, 1)


def log_canary_batch(model: DualBranchDetector, canary_rgb: torch.Tensor, canary_freq: torch.Tensor, path: Path) -> None:
    """Snapshots the current model's explainability output on a fixed batch of images,
    so comparing epoch_000.png -> epoch_010.png shows how the heatmaps/confidence evolve
    over training - not meaningful on its own for any single epoch, only as a timeline."""
    result = explain(model, canary_rgb, canary_freq)

    images = _denormalize_rgb(canary_rgb).permute(0, 2, 3, 1).cpu().numpy()
    spatial_hm = result["spatial_heatmap"].cpu().numpy()
    freq_hm = result["frequency_heatmap"].cpu().numpy()
    fused_probs = result["fused_probs"].detach().cpu().numpy()
    gate = result["gate_weights"].detach().cpu().numpy()

    n = images.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"input (p_fake={fused_probs[i, 1]:.2f})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(images[i])
        axes[i, 1].imshow(spatial_hm[i], cmap="jet", alpha=0.5)
        axes[i, 1].set_title(f"spatial (gate={gate[i, 0]:.2f})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(freq_hm[i], cmap="jet")
        axes[i, 2].set_title(f"frequency (gate={gate[i, 1]:.2f})")
        axes[i, 2].axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def train(
    model_cfg: dict,
    train_cfg: dict,
    checkpoint_dir: Path | None = None,
    data_dir: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """Runs training. `max_steps` caps steps per epoch - for fast sanity checks, not real runs.

    `data_dir` overrides where the manifest/images are read from (e.g. a Drive-mounted
    path in Colab, matching whatever was passed to build_manifest()); defaults to ml/data.
    """
    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = train_cfg["mixed_precision"] and device.type == "cuda"
    grad_accum_steps = train_cfg["grad_accum_steps"]

    model = DualBranchDetector.from_config(model_cfg).to(device)

    dataset_kwargs = {}
    if data_dir is not None:
        dataset_kwargs = {"manifest_path": data_dir / "manifest.csv", "data_dir": data_dir}
    train_ds = DeepfakeDataset(split="train", **dataset_kwargs)
    val_ds = DeepfakeDataset(split="val", **dataset_kwargs)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = max(train_cfg["num_epochs"] * steps_per_epoch, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg["schedule"]["warmup_steps"],
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    canary_batch_size = min(train_cfg["logging"]["canary_batch_size"], len(train_ds))
    canary_loader = DataLoader(train_ds, batch_size=canary_batch_size, shuffle=False)
    canary_rgb, canary_freq, _, _ = next(iter(canary_loader))
    canary_rgb, canary_freq = canary_rgb.to(device), canary_freq.to(device)

    checkpoint_dir = checkpoint_dir or (Path(__file__).resolve().parent.parent / train_cfg["checkpoint"]["dir"])
    checkpoint_path = checkpoint_dir / "last.pt"
    history_path = checkpoint_dir / "training_history.json"
    plot_path = checkpoint_dir / "training_curves.png"
    canary_dir = checkpoint_dir / "canary"

    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict] = []
    if train_cfg["checkpoint"]["resume"] and checkpoint_path.exists():
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)
        start_epoch += 1
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

    patience = train_cfg["early_stopping_patience"]
    epochs_without_improvement = 0

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        train_stats = _EpochStats()

        for step, (rgb, freq, label, families) in enumerate(train_loader):
            if max_steps is not None and step >= max_steps:
                break
            rgb, freq, label = rgb.to(device), freq.to(device), label.to(device)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(rgb, freq)
                loss, components = multi_task_loss(outputs, label, model_cfg["loss"])
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            train_stats.update(loss.item(), components, outputs, label, families)

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % train_cfg["logging"]["log_every_steps"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} {components}")

        train_avg = train_stats.averaged()
        val_avg, val_family_acc = evaluate(model, val_loader, model_cfg["loss"], device)
        print(f"epoch {epoch} val_loss {val_avg['loss']:.4f} val_acc {val_avg['acc']:.4f}")
        print(f"epoch {epoch} val_family_accuracy {val_family_acc}")

        history.append(
            {
                "epoch": epoch,
                "train": train_avg,
                "val": val_avg,
                "val_family_accuracy": val_family_acc,
            }
        )
        _save_history(history, history_path)
        _plot_history(history, plot_path)

        log_canary_batch(model, canary_rgb, canary_freq, canary_dir / f"epoch_{epoch:03d}.png")
        # frequency_saliency's backward() inside log_canary_batch populates .grad on
        # frequency_branch/freq_aux_head params as a side effect (it needs grad w.r.t.
        # the input, but autograd fills in param grads too) - clear it so it can't
        # leak into the next epoch's first optimizer step.
        optimizer.zero_grad(set_to_none=True)

        if val_avg["loss"] < best_val_loss:
            best_val_loss = val_avg["loss"]
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_val_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    return {"history": history, "best_val_loss": best_val_loss}


if __name__ == "__main__":
    _model_cfg, _train_cfg = load_configs()
    train(_model_cfg, _train_cfg)
