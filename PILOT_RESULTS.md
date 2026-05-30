# Pilot Results

These results were generated on CPU-only PyTorch using a small 64 x 64 grayscale
demo image and a compact DIP decoder with 32 latent/hidden channels. They are
not publication-scale yet, but they are useful because they already show both
positive evidence and a clear failure mode.

## Commands

Fast pilot suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pilot_suite.ps1
```

Stress suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stress_suite.ps1
```

Aggregate summaries:

```powershell
python scripts/summarize_results.py results_pilot_cpu
python scripts/summarize_results.py results_stress_cpu
```

Generate publication-style and interactive figures:

```powershell
python scripts/make_visualizations.py results_pilot_cpu results_stress_cpu
```

Generated figures are saved under:

```text
figures/
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

Interpretation:

The easy-to-split operators behave as expected: held-out measurement loss tracks
the oracle PSNR point closely. The short run is not enough to strongly expose
overfitting for most operators, but it verifies that the framework is working.

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

## Interesting Findings

1. **Strong positive case:** 4x super-resolution with pixel-domain held-out
   validation shows real early stopping behavior. Oracle PSNR peaks at iteration
   110 and then degrades to 19.69 dB by iteration 300. Smoothed validation also
   selects iteration 110, preserving the oracle reconstruction.

2. **Failure mode:** Fourier validation for 4x super-resolution is weaker in
   this setup. It selects iteration 300, missing the oracle by 0.71 dB. This is
   exactly the kind of operator/split-dependence the paper should analyze.

3. **Easy-to-split operators are stable:** inpainting, Fourier subsampling, and
   compressed sensing have small validation-oracle gaps in these pilots.

4. **Deblurring is less decisive here:** both pixel and Fourier validation remain
   close to oracle, but the oracle is still near the end of the run. Longer runs,
   stronger blur, or higher noise may be needed to expose deblurring overfit.

## Research Takeaway

The preliminary evidence supports a publishable direction:

```text
Measurement-split validation can be a strong ground-truth-free stopping signal,
but its reliability is operator- and split-dependent.
```

The best next experiment is a larger sweep over seeds, split ratios, noise
levels, blur severity, and super-resolution factors, reporting the distribution
of:

```text
oracle_psnr - psnr_at_validation_stop
```

## Figure Files

Butterfly low-degradation showcase:

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
