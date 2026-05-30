# Reproducibility

The included CPU pilot and stress results were generated with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot_suite.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_stress_suite.ps1
python scripts/make_visualizations.py results_pilot_cpu results_stress_cpu
```

The Set5 butterfly showcase was generated with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_set5.ps1
python scripts/run_experiment.py --operator inpainting --iterations 220 --log-every 10 --img-size 96 --channels 3 --latent-channels 48 --hidden-channels 48 --noise-sigma 0.005 --sampling-ratio 0.85 --image-path data\set5\Set5_HR\butterfly.png --out results_butterfly_showcase --seed 2
python scripts/run_experiment.py --operator deblur --split-domain pixel --iterations 220 --log-every 10 --img-size 96 --channels 3 --latent-channels 48 --hidden-channels 48 --noise-sigma 0.005 --blur-sigma 1.0 --image-path data\set5\Set5_HR\butterfly.png --out results_butterfly_showcase --seed 2
python scripts/run_experiment.py --operator superres --split-domain pixel --sr-factor 2 --iterations 180 --log-every 10 --img-size 96 --channels 3 --latent-channels 32 --hidden-channels 32 --noise-sigma 0.01 --image-path data\set5\Set5_HR\butterfly.png --out results_butterfly_showcase --seed 2
python scripts/make_visualizations.py results_butterfly_showcase
```

The runs use deterministic seeds and write:

- `history.csv`
- `summary.json`
- `curves.png`
- `reconstructions.png`
- aggregate CSV summaries
- static PNG/PDF figures
- interactive Plotly HTML dashboards

The current results are small CPU pilots. For publication-scale evidence, repeat
the stress suite across several seeds, image classes, noise levels, split ratios,
and operator severities.
