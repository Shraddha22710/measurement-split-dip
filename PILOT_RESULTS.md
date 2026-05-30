# Experimental Results

This document summarizes the included CPU experiments for measurement-split Deep
Image Prior. The results provide reproducible examples of held-out
measurement-consistency stopping across several inverse-imaging operators.

## Commands

Run the pilot suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot_suite.ps1
```

Run the stress suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stress_suite.ps1
```

Aggregate summaries:

```powershell
python scripts/summarize_results.py results_pilot_cpu
python scripts/summarize_results.py results_stress_cpu
```

Generate static and interactive figures:

```powershell
python scripts/make_visualizations.py results_pilot_cpu results_stress_cpu
```

## Pilot Suite

Setting:

- image size: 64
- iterations: 120
- seed: 0
- noise sigma: 0.03
- sampling ratio: 0.5

| Operator | Split | Oracle PSNR | Smooth-val PSNR | Gap |
|---|---:|---:|---:|---:|
| inpainting | pixel | 25.06 | 25.01 | 0.05 |
| Fourier subsampling | Fourier | 23.28 | 23.28 | 0.00 |
| compressed sensing | vector | 25.60 | 25.35 | 0.25 |
| deblur | pixel | 23.77 | 23.77 | 0.00 |
| deblur | Fourier | 23.44 | 23.44 | 0.00 |
| super-resolution | pixel | 25.06 | 24.94 | 0.12 |
| super-resolution | Fourier | 23.91 | 23.47 | 0.44 |

The pilot suite verifies the core training, splitting, logging, and visualization
pipeline on compact CPU-friendly experiments.

## Stress Suite

Setting:

- image size: 64
- iterations: 300
- seed: 1
- inpainting noise sigma: 0.08
- compressed sensing sampling ratio: 0.3, noise sigma: 0.08
- deblur blur sigma: 3.0, noise sigma: 0.05
- super-resolution factor: 4, noise sigma: 0.05

| Operator | Split | Oracle Iter | Smooth-val Iter | Oracle PSNR | Smooth-val PSNR | Final PSNR | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| inpainting | pixel | 280 | 300 | 24.77 | 24.59 | 24.59 | 0.17 |
| compressed sensing | vector | 300 | 300 | 24.89 | 24.89 | 24.89 | 0.00 |
| deblur | pixel | 300 | 280 | 23.37 | 23.24 | 23.37 | 0.13 |
| deblur | Fourier | 300 | 300 | 23.13 | 23.13 | 23.13 | 0.00 |
| 4x super-resolution | pixel | 110 | 110 | 21.85 | 21.85 | 19.69 | 0.00 |
| 4x super-resolution | Fourier | 130 | 300 | 22.36 | 21.65 | 21.65 | 0.71 |

The stress suite demonstrates that held-out measurement validation can identify
the oracle stopping point in some settings, while alternative split designs can
select later iterations. The 4x super-resolution runs show this contrast clearly:
pixel-domain validation selects iteration 110, while Fourier-domain validation
selects the final iteration in the included configuration.

## Figure Files

Butterfly native-resolution showcase:

- `figures/butterfly_native_showcase/butterfly_native_showcase_qualitative_montage.png`
- `figures/butterfly_native_showcase/butterfly_native_showcase_summary.png`
- `figures/butterfly_native_showcase/butterfly_native_showcase_interactive_dashboard.html`
- `figures/butterfly_lowdeg/butterfly_lowdeg_qualitative_montage.png`
- `figures/butterfly_lowdeg/butterfly_lowdeg_summary.png`
- `figures/butterfly_lowdeg/butterfly_lowdeg_interactive_dashboard.html`
- `figures/butterfly_showcase/butterfly_showcase_qualitative_montage.png`
- `figures/butterfly_showcase/butterfly_showcase_summary.png`
- `figures/butterfly_showcase/butterfly_showcase_interactive_dashboard.html`

Key static figures:

- `figures/pilot_cpu/pilot_cpu_summary.png`
- `figures/pilot_cpu/pilot_cpu_superres_case_study.png`
- `figures/stress_cpu/stress_cpu_summary.png`
- `figures/stress_cpu/stress_cpu_superres_case_study.png`

Interactive dashboards:

- `figures/pilot_cpu/pilot_cpu_interactive_dashboard.html`
- `figures/stress_cpu/stress_cpu_interactive_dashboard.html`

Figure index:

- `figures/index.html`
