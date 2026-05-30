# Reproducibility

The included CPU pilot and stress results were generated with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot_suite.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_stress_suite.ps1
python scripts/make_visualizations.py results_pilot_cpu results_stress_cpu
```

The Set5 butterfly low-degradation showcase was generated with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_set5.ps1
python scripts/run_experiment.py --operator inpainting --iterations 420 --log-every 20 --img-size 128 --channels 3 --latent-channels 64 --hidden-channels 64 --noise-sigma 0.0 --sampling-ratio 0.97 --image-path data\set5\Set5_HR\butterfly.png --out results_butterfly_lowdeg --seed 3
python scripts/run_experiment.py --operator deblur --split-domain pixel --iterations 420 --log-every 20 --img-size 128 --channels 3 --latent-channels 64 --hidden-channels 64 --noise-sigma 0.0 --blur-sigma 0.6 --image-path data\set5\Set5_HR\butterfly.png --out results_butterfly_lowdeg --seed 3
python scripts/run_experiment.py --operator superres --split-domain pixel --sr-factor 2 --iterations 420 --log-every 20 --img-size 128 --channels 3 --latent-channels 64 --hidden-channels 64 --noise-sigma 0.0 --image-path data\set5\Set5_HR\butterfly.png --out results_butterfly_lowdeg --seed 3
python scripts/make_visualizations.py results_butterfly_lowdeg
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
