from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PALETTE = {
    "oracle": "#202124",
    "smooth": "#D55E00",
    "final": "#0072B2",
    "psnr": "#009E73",
    "train": "#CC79A7",
    "val": "#E69F00",
    "gap": "#56B4E9",
    "warn": "#B22222",
}


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_dir = summary_path.parent
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        history = pd.read_csv(history_path)
        runs.append({"name": run_dir.name, "dir": run_dir, "summary": summary, "history": history})
    return runs


def run_label(run: dict) -> str:
    summary = run["summary"]
    op = summary.get("operator", "unknown")
    split = summary.get("split_domain", "split")
    if op == "superres":
        factor = summary.get("sr_factor")
        if factor is None:
            factor = 4 if "stress" in str(run["dir"]) else 2
        return f"{factor}x SR / {split}"
    if op == "compressed_sensing":
        return f"CS / {split}"
    return f"{op} / {split}"


def pretty_tag(tag: str) -> str:
    return tag.replace("_", " ").replace("cpu", "CPU").title().replace("Cpu", "CPU")


def make_summary_figure(runs: list[dict], out_dir: Path, tag: str) -> None:
    rows = []
    for run in runs:
        s = run["summary"]
        rows.append(
            {
                "label": run_label(run),
                "oracle": s["oracle_psnr"],
                "selected": s["smooth_val_psnr"],
                "final": s["final_psnr"],
                "gap": s["smooth_val_oracle_gap"],
            }
        )
    df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    x = np.arange(len(df))
    width = 0.27

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.7, 1.0]})
    axes[0].bar(x - width, df["oracle"], width, label="Oracle", color=PALETTE["oracle"])
    axes[0].bar(x, df["selected"], width, label="Held-out stop", color=PALETTE["smooth"])
    axes[0].bar(x + width, df["final"], width, label="Final", color=PALETTE["final"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"], rotation=25, ha="right")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Stopping quality")
    axes[0].legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    colors = [PALETTE["warn"] if g > 0.5 else PALETTE["gap"] for g in df["gap"]]
    axes[1].barh(df["label"], df["gap"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Oracle - selected PSNR (dB)")
    axes[1].set_title("Validation stopping gap")
    axes[1].axvline(0.5, color=PALETTE["warn"], linestyle="--", linewidth=1)
    for i, gap in enumerate(df["gap"]):
        axes[1].text(gap + 0.02, i, f"{gap:.2f}", va="center", fontsize=9)

    fig.suptitle(f"Measurement-split DIP: {pretty_tag(tag)}", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_summary.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{tag}_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def make_superres_comparison(runs: list[dict], out_dir: Path, tag: str) -> None:
    sr_runs = [r for r in runs if r["summary"].get("operator") == "superres"]
    sr_runs = sorted(sr_runs, key=lambda r: 0 if r["summary"].get("split_domain") == "pixel" else 1)
    if len(sr_runs) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, run in zip(axes, sr_runs):
        h = run["history"]
        s = run["summary"]
        ax.plot(h["iteration"], h["psnr"], color=PALETTE["psnr"], linewidth=2.4, label="PSNR")
        ax2 = ax.twinx()
        ax2.plot(h["iteration"], h["val_loss"], color=PALETTE["val"], linewidth=1.8, alpha=0.85, label="Held-out loss")
        ax.axvline(s["oracle_iteration"], color=PALETTE["oracle"], linestyle="--", linewidth=1.5, label="Oracle")
        ax.axvline(s["smooth_val_iteration"], color=PALETTE["smooth"], linestyle=":", linewidth=2.2, label="Held-out stop")
        ax.set_title(run_label(run))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("PSNR (dB)")
        ax2.set_ylabel("Validation loss")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("4x super-resolution exposes split-dependent stopping behavior", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_superres_case_study.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{tag}_superres_case_study.pdf", bbox_inches="tight")
    plt.close(fig)


def make_interactive_dashboard(runs: list[dict], out_dir: Path, tag: str) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("PSNR curves", "Held-out loss", "Training loss", "Oracle gap summary"),
        specs=[[{}, {}], [{}, {}]],
    )
    for run in runs:
        h = run["history"]
        label = run_label(run)
        fig.add_trace(go.Scatter(x=h["iteration"], y=h["psnr"], mode="lines", name=f"{label} PSNR"), row=1, col=1)
        fig.add_trace(go.Scatter(x=h["iteration"], y=h["val_loss"], mode="lines", name=f"{label} val"), row=1, col=2)
        fig.add_trace(go.Scatter(x=h["iteration"], y=h["train_loss"], mode="lines", name=f"{label} train"), row=2, col=1)
        s = run["summary"]
        fig.add_trace(
            go.Scatter(
                x=[s["oracle_iteration"], s["smooth_val_iteration"]],
                y=[s["oracle_psnr"], s["smooth_val_psnr"]],
                mode="markers+lines",
                marker={"size": 10},
                name=f"{label} stop markers",
                hovertemplate="iter=%{x}<br>PSNR=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    labels = [run_label(r) for r in runs]
    gaps = [r["summary"]["smooth_val_oracle_gap"] for r in runs]
    fig.add_trace(
        go.Bar(
            x=gaps,
            y=labels,
            orientation="h",
            marker_color=[PALETTE["warn"] if g > 0.5 else PALETTE["gap"] for g in gaps],
            name="Gap",
            hovertemplate="%{y}<br>gap=%{x:.3f} dB<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(
        title=f"Interactive measurement-split DIP dashboard: {tag}",
        template="plotly_white",
        height=850,
        width=1250,
        legend={"orientation": "v", "font": {"size": 10}},
        hovermode="x unified",
    )
    fig.write_html(out_dir / f"{tag}_interactive_dashboard.html", include_plotlyjs="cdn")


def make_3d_views(runs: list[dict], out_dir: Path, tag: str) -> None:
    """Create 3D interactive and static views of PSNR/validation dynamics."""
    labels = [run_label(r) for r in runs]

    fig = go.Figure()
    for idx, run in enumerate(runs):
        h = run["history"]
        fig.add_trace(
            go.Scatter3d(
                x=h["iteration"],
                y=np.full(len(h), idx),
                z=h["psnr"],
                mode="lines+markers",
                marker={"size": 3},
                line={"width": 6},
                name=labels[idx],
                text=[labels[idx]] * len(h),
                hovertemplate="run=%{text}<br>iter=%{x}<br>PSNR=%{z:.2f} dB<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"3D PSNR trajectory view: {pretty_tag(tag)}",
        template="plotly_white",
        height=780,
        scene={
            "xaxis_title": "Iteration",
            "yaxis_title": "Operator / split",
            "zaxis_title": "PSNR (dB)",
            "yaxis": {"tickmode": "array", "tickvals": list(range(len(labels))), "ticktext": labels},
        },
        legend={"font": {"size": 10}},
    )
    fig.write_html(out_dir / f"{tag}_3d_psnr_trajectories.html", include_plotlyjs="cdn")

    fig2 = go.Figure()
    for run in runs:
        h = run["history"]
        label = run_label(run)
        fig2.add_trace(
            go.Scatter3d(
                x=h["iteration"],
                y=h["val_loss"],
                z=h["psnr"],
                mode="lines+markers",
                marker={"size": 3},
                line={"width": 5},
                name=label,
                hovertemplate=f"{label}<br>iter=%{{x}}<br>val=%{{y:.4g}}<br>PSNR=%{{z:.2f}}<extra></extra>",
            )
        )
    fig2.update_layout(
        title=f"3D validation-loss versus PSNR dynamics: {pretty_tag(tag)}",
        template="plotly_white",
        height=780,
        scene={
            "xaxis_title": "Iteration",
            "yaxis_title": "Held-out loss",
            "zaxis_title": "PSNR (dB)",
            "yaxis": {"type": "log"},
        },
        legend={"font": {"size": 10}},
    )
    fig2.write_html(out_dir / f"{tag}_3d_validation_psnr.html", include_plotlyjs="cdn")

    mpl_fig = plt.figure(figsize=(11, 7))
    ax = mpl_fig.add_subplot(111, projection="3d")
    for idx, run in enumerate(runs):
        h = run["history"]
        ax.plot(h["iteration"], np.full(len(h), idx), h["psnr"], linewidth=2.2, label=labels[idx])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Operator / split")
    ax.set_zlabel("PSNR (dB)")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.view_init(elev=24, azim=-58)
    ax.set_title(f"3D PSNR Trajectories: {pretty_tag(tag)}")
    mpl_fig.tight_layout()
    mpl_fig.savefig(out_dir / f"{tag}_3d_psnr_trajectories.png", dpi=240, bbox_inches="tight")
    mpl_fig.savefig(out_dir / f"{tag}_3d_psnr_trajectories.pdf", bbox_inches="tight")
    plt.close(mpl_fig)


def make_qualitative_montage(runs: list[dict], out_dir: Path, tag: str) -> None:
    image_paths = [(run_label(r), r["dir"] / "reconstructions.png") for r in runs if (r["dir"] / "reconstructions.png").exists()]
    if not image_paths:
        return
    fig, axes = plt.subplots(len(image_paths), 1, figsize=(13, 3.2 * len(image_paths)))
    if len(image_paths) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, image_paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(label, loc="left", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"Qualitative reconstructions: {pretty_tag(tag)}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_qualitative_montage.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{tag}_qualitative_montage.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown_table(runs: list[dict], out_dir: Path, tag: str) -> None:
    rows = []
    for run in runs:
        s = run["summary"]
        rows.append(
            {
                "Case": run_label(run),
                "Oracle iter": int(s["oracle_iteration"]),
                "Held-out iter": int(s["smooth_val_iteration"]),
                "Oracle PSNR": f"{s['oracle_psnr']:.2f}",
                "Held-out PSNR": f"{s['smooth_val_psnr']:.2f}",
                "Final PSNR": f"{s['final_psnr']:.2f}",
                "Gap": f"{s['smooth_val_oracle_gap']:.2f}",
            }
        )
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    (out_dir / f"{tag}_results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    roots = [Path(p) for p in sys.argv[1:]] or [Path("results_pilot_cpu"), Path("results_stress_cpu")]
    fig_root = Path("figures")
    fig_root.mkdir(exist_ok=True)
    for root in roots:
        runs = load_runs(root)
        if not runs:
            print(f"No runs found in {root}")
            continue
        tag = root.name.replace("results_", "")
        out_dir = fig_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        make_summary_figure(runs, out_dir, tag)
        make_superres_comparison(runs, out_dir, tag)
        make_interactive_dashboard(runs, out_dir, tag)
        make_3d_views(runs, out_dir, tag)
        make_qualitative_montage(runs, out_dir, tag)
        write_markdown_table(runs, out_dir, tag)
        print(f"Wrote visualizations for {root} to {out_dir}")


if __name__ == "__main__":
    main()
