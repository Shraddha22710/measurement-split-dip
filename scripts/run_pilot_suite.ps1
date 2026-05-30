$ErrorActionPreference = "Stop"

$common = @(
  "--iterations", "120",
  "--log-every", "5",
  "--img-size", "64",
  "--latent-channels", "32",
  "--hidden-channels", "32",
  "--out", "results_pilot_cpu",
  "--seed", "0"
)

python scripts/run_experiment.py --operator inpainting @common
python scripts/run_experiment.py --operator fourier @common
python scripts/run_experiment.py --operator compressed_sensing @common
python scripts/run_experiment.py --operator deblur --split-domain pixel @common
python scripts/run_experiment.py --operator deblur --split-domain fourier @common
python scripts/run_experiment.py --operator superres --split-domain pixel @common
python scripts/run_experiment.py --operator superres --split-domain fourier @common
python scripts/summarize_results.py results_pilot_cpu

