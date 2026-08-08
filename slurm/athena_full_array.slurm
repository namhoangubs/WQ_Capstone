#!/bin/bash
# Full production run on Athena: one 50-stock portfolio per A100 array task.
# Submit from the repository root after running --prepare-only once.
#SBATCH --job-name=wgan-full
#SBATCH --account=plgmicrogranz-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH --time=48:00:00
#SBATCH --output=logs/full-%A_%a.out
#SBATCH --error=logs/full-%A_%a.err

set -euo pipefail

PROJECT_DIR="/net/pr2/projects/plgrid/plgg15102025/TruongCo_Station/Final_Master/WQ_Capstone-WQfinalproject_Nam_pipeline"
VENV_DIR="$PROJECT_DIR/.venv-capstone"

cd "$PROJECT_DIR"

module purge
module load GCCcore/12.3.0
module load Python/3.11.3

source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MALLOC_ARENA_MAX=2
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Detect missing or mutually incompatible installed packages before training.
python -m pip check

# Expose the CUDA libraries installed by tensorflow[and-cuda].
NVIDIA_LIB_DIRS="$(
python - <<'PY'
import glob
import site

directories = []
for base in site.getsitepackages():
    directories.extend(glob.glob(base + "/nvidia/*/lib"))
print(":".join(directories))
PY
)"

if [[ -z "$NVIDIA_LIB_DIRS" ]]; then
    echo "No pip-installed NVIDIA library directories were found." >&2
    exit 1
fi

export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${LD_LIBRARY_PATH:-}"

PORTFOLIO_ID="$SLURM_ARRAY_TASK_ID"
PORTFOLIO_LABEL="$(printf '%02d' "$PORTFOLIO_ID")"
MANIFEST_FILE="$PROJECT_DIR/run_outputs/portfolio_manifests/portfolio_${PORTFOLIO_LABEL}.csv"
OUTPUT_DIR="$PROJECT_DIR/run_outputs/portfolio_${PORTFOLIO_LABEL}"
SUCCESS_FILE="$OUTPUT_DIR/SUCCESS.json"

heartbeat() {
    while true; do
        echo "--- heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ) portfolio=$PORTFOLIO_ID ---"
        nvidia-smi \
          --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
          --format=csv,noheader || true
        sleep 300
    done
}

HEARTBEAT_PID=""
MODEL_PID=""
cleanup() {
    local status=$?
    trap - EXIT TERM INT
    if [[ -n "$HEARTBEAT_PID" ]]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
    if [[ "$status" -eq 0 && ! -f "$SUCCESS_FILE" ]]; then
        echo "ERROR: command sequence ended without $SUCCESS_FILE" >&2
        status=1
    fi
    echo "Ended UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Portfolio: $PORTFOLIO_ID"
    echo "Exit status: $status"
    exit "$status"
}

terminate() {
    local signal_name="$1"
    local status="$2"
    echo "Received $signal_name; terminating portfolio $PORTFOLIO_ID." >&2
    if [[ -n "$MODEL_PID" ]]; then
        kill -TERM "$MODEL_PID" 2>/dev/null || true
        wait "$MODEL_PID" 2>/dev/null || true
        MODEL_PID=""
    fi
    exit "$status"
}

trap cleanup EXIT
trap 'terminate TERM 143' TERM
trap 'terminate INT 130' INT

echo "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SLURM array job: ${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM array task: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "SLURM job: ${SLURM_JOB_ID:-unset}"
echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Python: $(which python)"
echo "Virtual environment: ${VIRTUAL_ENV:-unset}"
echo "Portfolio: $PORTFOLIO_ID"
echo "Manifest: $MANIFEST_FILE"
echo "Output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

test -f "$PROJECT_DIR/pipeline_config.json"
test -f "$PROJECT_DIR/phase_1/selected_models_for_later_phases_q4.csv"
test -f "$MANIFEST_FILE"

python - <<'PY'
import json

with open("pipeline_config.json", encoding="utf-8") as handle:
    config = json.load(handle)

print("Production configuration:")
print("  run_scope (file):", config["run_scope"])
print("  portfolios:", config["portfolio_sampling"]["number_of_samples"])
print("  stocks per portfolio:", config["portfolio_sampling"]["stocks_per_sample"])
print("  initial WGAN iteration limit:", config["wgan"]["initial_training_iterations"])
print("  rolling WGAN iteration limit:", config["wgan"]["rolling_training_iterations"])
print("  critic steps:", config["wgan"]["critic_steps"])
print("  latent dimension:", config["wgan"]["latent_dimension"])
print("  simulation paths:", config["simulation"]["paths_per_forecast_date"])
print("  forecast start:", config["data"]["forecast_start"])
print("  forecast end:", config["data"]["forecast_end"])

assert config["portfolio_sampling"]["number_of_samples"] == 10
assert config["portfolio_sampling"]["stocks_per_sample"] == 50
assert config["wgan"]["initial_training_iterations"] == 2000
assert config["wgan"]["rolling_training_iterations"] == 10
assert config["simulation"]["paths_per_forecast_date"] == 10000
assert config["execution"]["portfolio_workers"] == 1
PY

python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow version:", tf.__version__)
print("TensorFlow GPUs:", gpus)
if len(gpus) != 1:
    raise SystemExit(f"Expected exactly one allocated GPU, found {len(gpus)}")
PY

# Original-paper diagnostics run after the expensive WGAN stage. Import their
# third-party dependencies now so a missing package fails immediately instead
# of after all rolling dates have completed.
python - <<'PY'
from importlib.metadata import version

import networkx
import powerlaw
import seaborn
from p3_1_stat_prop import p3_1_stat_prop
from p3_2_stat_prop_mv import p3_2_stat_prop_mv

print("Diagnostics dependencies:")
for package in ["networkx", "powerlaw", "seaborn"]:
    print(f"  {package}:", version(package))
print("  original-paper diagnostic modules: import OK")
PY

# Compile and execute the exact optimized WGAN step before committing hours to
# the rolling run. This also verifies parameter-free Maxout and GPU placement.
python -u verify_optimized_wgan_runtime.py

heartbeat &
HEARTBEAT_PID=$!

# The scope override prevents every array task from rerunning Phase 1.
# Portfolio-specific mode also defers shared aggregation until all tasks finish.
python -u run_pipeline.py \
  --config pipeline_config.json \
  --run-scope later_phases \
  --portfolio-id "$PORTFOLIO_ID" &
MODEL_PID=$!

set +e
wait "$MODEL_PID"
MODEL_STATUS=$?
set -e
MODEL_PID=""
if [[ "$MODEL_STATUS" -ne 0 ]]; then
    echo "Portfolio process failed with status $MODEL_STATUS." >&2
    exit "$MODEL_STATUS"
fi

test -f "$SUCCESS_FILE"
python -u validate_optimized_portfolio.py "$OUTPUT_DIR" --production

echo "Finished UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Portfolio $PORTFOLIO_ID completed successfully."
cat "$SUCCESS_FILE"
