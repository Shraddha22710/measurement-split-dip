# Reproducibility

The included CPU pilot and stress results were generated with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot_suite.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_stress_suite.ps1
python scripts/make_visualizations.py results_pilot_cpu results_stress_cpu
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

