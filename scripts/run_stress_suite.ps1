$ErrorActionPreference = "Stop"

$common = @(
  "--iterations", "300",
  "--log-every", "10",
  "--img-size", "64",
  "--latent-channels", "32",
  "--hidden-channels", "32",
  "--out", "results_stress_cpu",
  "--seed", "1"
)

python scripts/run_experiment.py --operator inpainting --noise-sigma 0.08 --sampling-ratio 0.5 @common
python scripts/run_experiment.py --operator compressed_sensing --noise-sigma 0.08 --sampling-ratio 0.3 @common
python scripts/run_experiment.py --operator deblur --split-domain pixel --noise-sigma 0.05 --blur-sigma 3.0 @common
python scripts/run_experiment.py --operator deblur --split-domain fourier --noise-sigma 0.05 --blur-sigma 3.0 @common
python scripts/run_experiment.py --operator superres --split-domain pixel --noise-sigma 0.05 --sr-factor 4 @common
python scripts/run_experiment.py --operator superres --split-domain fourier --noise-sigma 0.05 --sr-factor 4 @common
python scripts/summarize_results.py results_stress_cpu

