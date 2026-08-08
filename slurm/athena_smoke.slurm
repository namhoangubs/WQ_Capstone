#!/bin/bash
# Athena/PLGrid one-GPU smoke test.
# Replace YOUR_GRANT with the allocation name shown by `hpc-grants`.
#SBATCH --job-name=wgan-smoke
#SBATCH --account=YOUR_GRANT-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/athena-smoke-%j.out
#SBATCH --error=logs/athena-smoke-%j.err

set -euo pipefail

# Submit this file from the repository root on Athena.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

module purge
module load GCCcore/12.3.0
module load Python/3.11.3

source .venv-athena/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_CPP_MIN_LOG_LEVEL=1

echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow version:", tf.__version__)
print("TensorFlow GPUs:", gpus)
if len(gpus) != 1:
    raise SystemExit(f"Expected exactly one allocated GPU, found {len(gpus)}")
PY

python run_pipeline.py --config pipeline_config_smoke.json --dry-run
python run_pipeline.py --config pipeline_config_smoke.json --portfolio-id 1

test -f smoke_outputs/portfolio_01/SUCCESS.json
echo "Smoke test completed successfully."
cat smoke_outputs/portfolio_01/SUCCESS.json
