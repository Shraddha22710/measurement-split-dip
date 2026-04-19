# Self-Supervised Image Reconstruction

This repository explores practical and research-oriented approaches for **self-supervised image reconstruction**. The focus is on methods that recover images from degraded measurements **without paired clean targets**.

Self-supervision is especially useful in domains such as MRI, CT, microscopy, and astronomy, where collecting aligned ground-truth images is expensive or impossible.

## Problem Setup

Inverse imaging typically models measurements as:

$$
y = A(x) + \varepsilon
$$

where:
- $x$ is the unknown clean image,
- $A$ is the forward operator (e.g., blur, masking, subsampling),
- $\varepsilon$ is noise.

Instead of training with $(y, x)$ pairs, self-supervised methods learn from measurements and structure in the acquisition process.

## Methods Covered

This project is organized around the main families of self-supervised reconstruction:

1. **Deep Image Prior (DIP)**  
   Uses an untrained convolutional generator as an implicit prior and optimizes it directly for a single measurement.
2. **Measurement Splitting (SSDU-style)**  
   Hides part of the measurements and trains the model to predict the held-out part from the visible subset.
3. **Equivariant Imaging / Equivariant Splitting**  
   Enforces consistency under transformations (e.g., flips, rotations), leveraging invariances of natural images.
4. **Noise-Based Self-Supervision**  
   Uses known noise statistics (e.g., SURE-type objectives) to estimate risk without clean labels.
5. **Multi-Operator Consistency**  
   Learns reconstruction behavior across multiple forward operators through consistency constraints.

## Current Repository Contents

- `dip_experiments.ipynb`  
  Main notebook with an extended Deep Image Prior workflow in DeepInverse, including:
  - manual DIP optimization loop,
  - baseline comparisons,
  - architecture and channel sweeps,
  - optimization-dynamics analysis,
  - metric tracking (PSNR/SSIM and measurement consistency).

## Getting Started

### 1. Create an environment

Use any Python environment manager (venv/conda). Then install core dependencies:

```bash
pip install deepinv jupyter matplotlib numpy scikit-image
```

If needed, install a PyTorch build compatible with your hardware before installing `deepinv`.

### 2. Launch the notebook

```bash
jupyter notebook dip_experiments.ipynb
```

### 3. Run experiments

Inside the notebook, you can:
- choose the forward model (inpainting, blur, etc.),
- tune DIP architecture (`in_size`, channels),
- inspect reconstruction quality over iterations,
- study early stopping behavior.

## Why This Matters

Self-supervised reconstruction reduces dependence on expensive labels while preserving strong performance in ill-posed inverse problems. This makes it a practical direction for scientific and medical imaging, where acquisition physics are known but clean targets are limited.

## Roadmap

- Add dedicated notebooks for:
  - measurement splitting (SSDU),
  - equivariant imaging,
  - equivariant splitting,
  - multi-operator self-supervision.
- Add comparative experiments against supervised and Plug-and-Play baselines.
- Add uncertainty and posterior-sampling experiments for inverse problems.

## References

- Ulyanov, Vedaldi, Lempitsky. *Deep Image Prior* (2018).
- Chen et al. *Self-Supervised Learning for MRI Reconstruction with SSDU* (2021).
- Chen et al. *Equivariant Imaging* (2021).
- DeepInverse documentation and examples: https://deepinv.github.io/
