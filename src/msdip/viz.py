from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def save_summary(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_curves(history: pd.DataFrame, summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(history["iteration"], history["train_loss"], label="train")
    axes[0].plot(history["iteration"], history["val_loss"], label="held-out")
    axes[0].set_yscale("log")
    axes[0].set_title("Measurement losses")
    axes[0].legend()

    axes[1].plot(history["iteration"], history["psnr"], label="PSNR")
    axes[1].axvline(summary["oracle_iteration"], color="black", linestyle="--", label="oracle")
    axes[1].axvline(summary["smooth_val_iteration"], color="tab:orange", linestyle=":", label="smooth val")
    axes[1].set_title("Ground-truth analysis")
    axes[1].legend()

    axes[2].plot(history["iteration"], history["output_change"])
    axes[2].set_yscale("log")
    axes[2].set_title("Output change")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=180)
    plt.close(fig)


def plot_reconstructions(x_true: torch.Tensor, selected: dict[str, torch.Tensor], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = {"ground truth": x_true.cpu(), **selected}
    fig, axes = plt.subplots(1, len(items), figsize=(3 * len(items), 3))
    if len(items) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, items.items()):
        arr = img.detach().clamp(0, 1).cpu()[0]
        if arr.shape[0] == 1:
            ax.imshow(arr[0], cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(arr.permute(1, 2, 0))
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "reconstructions.png", dpi=180)
    plt.close(fig)

