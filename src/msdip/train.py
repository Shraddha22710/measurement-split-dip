from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange

from .metrics import moving_average, psnr, ssim
from .models import ConvDIP, make_latent
from .operators import SplitOperator


@dataclass
class DIPConfig:
    iterations: int = 800
    lr: float = 1e-2
    latent_channels: int = 64
    hidden_channels: int = 64
    start_size: int = 8
    log_every: int = 10
    smooth_window: int = 7
    output_change_patience: int = 10
    seed: int = 0
    device: str = "cpu"


def run_dip(
    x_true: torch.Tensor,
    op: SplitOperator,
    cfg: DIPConfig,
    out_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str], dict[str, torch.Tensor]]:
    device = torch.device(cfg.device)
    x_true = x_true.to(device)

    torch.manual_seed(cfg.seed)
    _, c, h, _ = x_true.shape
    model = ConvDIP(c, h, cfg.latent_channels, cfg.hidden_channels, cfg.start_size).to(device)
    z = make_latent(1, cfg.latent_channels, cfg.start_size, device, cfg.seed + 101)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rows: list[dict[str, float | int]] = []
    snapshots: dict[int, torch.Tensor] = {}
    previous: torch.Tensor | None = None

    iterator = trange(cfg.iterations + 1, desc=f"DIP-{op.name}", leave=False)
    for it in iterator:
        x_hat = model(z)
        train_loss = op.loss(x_hat, "train")

        if it > 0:
            opt.zero_grad(set_to_none=True)
            train_loss.backward()
            opt.step()

        if it % cfg.log_every == 0 or it == cfg.iterations:
            with torch.no_grad():
                x_eval = model(z).detach()
                val_loss = op.loss(x_eval, "val").item()
                train_eval = op.loss(x_eval, "train").item()
                p = psnr(x_eval, x_true)
                s = ssim(x_eval, x_true)
                change = 0.0 if previous is None else torch.mean((x_eval - previous) ** 2).sqrt().item()
                previous = x_eval.clone()
                rows.append(
                    {
                        "iteration": it,
                        "train_loss": train_eval,
                        "val_loss": val_loss,
                        "psnr": p,
                        "ssim": s,
                        "output_change": change,
                    }
                )
                snapshots[it] = x_eval.cpu()
                iterator.set_postfix(train=f"{train_eval:.4g}", val=f"{val_loss:.4g}", psnr=f"{p:.2f}")

    history = pd.DataFrame(rows)
    summary = summarize_history(history, cfg)
    summary.update({"operator": op.name, "split_domain": op.split_domain, **asdict(cfg)})

    selected = {
        "oracle": snapshots[int(summary["oracle_iteration"])],
        "val": snapshots[int(summary["val_iteration"])],
        "smooth_val": snapshots[int(summary["smooth_val_iteration"])],
        "final": snapshots[int(history.iloc[-1]["iteration"])],
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        history.to_csv(out / "history.csv", index=False)
    return history, summary, selected


def summarize_history(history: pd.DataFrame, cfg: DIPConfig) -> dict[str, float | int]:
    oracle_idx = int(history["psnr"].idxmax())
    val_idx = int(history["val_loss"].idxmin())
    smooth_vals = moving_average(history["val_loss"].tolist(), cfg.smooth_window)
    smooth_idx = int(min(range(len(smooth_vals)), key=lambda i: smooth_vals[i]))

    change_idx = len(history) - 1
    changes = history["output_change"].tolist()
    for i in range(cfg.output_change_patience, len(changes)):
        window = changes[i - cfg.output_change_patience + 1 : i + 1]
        if max(window) < 1e-4:
            change_idx = i
            break

    def row(idx: int) -> pd.Series:
        return history.iloc[idx]

    return {
        "oracle_iteration": int(row(oracle_idx)["iteration"]),
        "oracle_psnr": float(row(oracle_idx)["psnr"]),
        "final_iteration": int(history.iloc[-1]["iteration"]),
        "final_psnr": float(history.iloc[-1]["psnr"]),
        "val_iteration": int(row(val_idx)["iteration"]),
        "val_psnr": float(row(val_idx)["psnr"]),
        "smooth_val_iteration": int(row(smooth_idx)["iteration"]),
        "smooth_val_psnr": float(row(smooth_idx)["psnr"]),
        "change_iteration": int(row(change_idx)["iteration"]),
        "change_psnr": float(row(change_idx)["psnr"]),
        "val_oracle_gap": float(row(oracle_idx)["psnr"] - row(val_idx)["psnr"]),
        "smooth_val_oracle_gap": float(row(oracle_idx)["psnr"] - row(smooth_idx)["psnr"]),
    }

