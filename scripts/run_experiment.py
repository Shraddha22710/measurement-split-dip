from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msdip import build_operator, load_demo_image, load_image_path, run_dip
from msdip.train import DIPConfig
from msdip.viz import plot_curves, plot_reconstructions, save_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measurement-split DIP pilot experiments.")
    parser.add_argument("--operator", default="inpainting", choices=["inpainting", "fourier", "compressed_sensing", "deblur", "superres", "all"])
    parser.add_argument("--split-domain", default="pixel", choices=["pixel", "fourier"])
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--start-size", type=int, default=8)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--sampling-ratio", type=float, default=0.5)
    parser.add_argument("--noise-sigma", type=float, default=0.03)
    parser.add_argument("--blur-sigma", type=float, default=1.6)
    parser.add_argument("--sr-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image", default="camera", choices=["camera", "astronaut", "coins"])
    parser.add_argument("--image-path", default=None, help="Optional path to a local image file.")
    parser.add_argument("--crop-size", type=int, default=None, help="Optional native-resolution center crop size for image-path inputs.")
    parser.add_argument("--no-resize", action="store_true", help="Keep the native crop size instead of resizing to --img-size.")
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operators = ["inpainting", "fourier", "compressed_sensing", "deblur", "superres"] if args.operator == "all" else [args.operator]

    if args.image_path:
        x = load_image_path(
            args.image_path,
            args.img_size,
            args.channels,
            crop_size=args.crop_size,
            resize=not args.no_resize,
        ).to(args.device)
    else:
        x = load_demo_image(args.img_size, args.channels, args.image).to(args.device)
    cfg = DIPConfig(
        iterations=args.iterations,
        lr=args.lr,
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        start_size=args.start_size,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
    )

    for op_name in operators:
        split_domain = args.split_domain
        if op_name not in {"deblur", "superres"} and split_domain == "fourier":
            split_domain = "pixel" if op_name != "fourier" else "fourier"
        op = build_operator(
            x,
            operator=op_name,
            split_ratio=args.split_ratio,
            noise_sigma=args.noise_sigma,
            seed=args.seed,
            split_domain=split_domain,
            blur_sigma=args.blur_sigma,
            sr_factor=args.sr_factor,
            sampling_ratio=args.sampling_ratio,
        )
        out_dir = Path(args.out) / f"{op_name}_{op.split_domain}_seed{args.seed}"
        history, summary, selected = run_dip(x, op, cfg, out_dir)
        summary.update(
            {
                "img_size": args.img_size,
                "channels": args.channels,
                "split_ratio": args.split_ratio,
                "sampling_ratio": args.sampling_ratio,
                "noise_sigma": args.noise_sigma,
                "blur_sigma": args.blur_sigma,
                "sr_factor": args.sr_factor,
                "image": args.image,
                "image_path": args.image_path,
                "crop_size": args.crop_size,
                "resize": not args.no_resize,
                "effective_size": int(x.shape[-1]),
            }
        )
        save_summary(summary, out_dir)
        plot_curves(history, summary, out_dir)
        plot_reconstructions(x, selected, out_dir)
        print(f"\n[{op_name}/{op.split_domain}]")
        print(f"  oracle PSNR:      {summary['oracle_psnr']:.2f} dB at iter {summary['oracle_iteration']}")
        print(f"  smooth-val PSNR:  {summary['smooth_val_psnr']:.2f} dB at iter {summary['smooth_val_iteration']}")
        print(f"  smooth-val gap:   {summary['smooth_val_oracle_gap']:.2f} dB")
        print(f"  outputs:          {out_dir}")


if __name__ == "__main__":
    main()
