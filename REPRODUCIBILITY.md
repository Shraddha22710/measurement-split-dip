# Reproducibility

The repository includes deterministic CPU experiment outputs and the commands
used to regenerate them.

## Pilot And Stress Suites

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot_suite.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_stress_suite.ps1
python scripts/make_visualizations.py results_pilot_cpu results_stress_cpu
```

## Set5 Butterfly Native-Resolution Showcase

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_set5.ps1
python scripts/run_experiment.py --operator inpainting --iterations 900 --log-every 30 --img-size 256 --channels 3 --latent-channels 48 --hidden-channels 48 --start-size 16 --noise-sigma 0.0 --sampling-ratio 0.99 --image-path data\set5\Set5_HR\butterfly.png --crop-size 256 --no-resize --out results_butterfly_native_showcase --seed 4
python scripts/run_experiment.py --operator deblur --split-domain pixel --iterations 900 --log-every 30 --img-size 256 --channels 3 --latent-channels 48 --hidden-channels 48 --start-size 16 --noise-sigma 0.0 --blur-sigma 0.45 --image-path data\set5\Set5_HR\butterfly.png --crop-size 256 --no-resize --out results_butterfly_native_showcase --seed 4
python scripts/run_experiment.py --operator superres --split-domain pixel --sr-factor 2 --iterations 900 --log-every 30 --img-size 256 --channels 3 --latent-channels 48 --hidden-channels 48 --start-size 16 --noise-sigma 0.0 --image-path data\set5\Set5_HR\butterfly.png --crop-size 256 --no-resize --out results_butterfly_native_showcase --seed 4
python scripts/make_visualizations.py results_butterfly_native_showcase
```

## Output Files

Each experiment directory contains:

- `history.csv`
- `summary.json`
- `curves.png`
- `reconstructions.png`

Visualization commands create aggregate CSV summaries, Markdown tables, PNG/PDF
figures, qualitative montages, and Plotly HTML dashboards.
