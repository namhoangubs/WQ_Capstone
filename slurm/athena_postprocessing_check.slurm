#!/bin/bash
# Verify the complete diagnostics stage using an existing portfolio's CSV files.
# Override the default with: sbatch --array=PORTFOLIO-ID-PORTFOLIO-ID ...
#SBATCH --job-name=wgan-postcheck
#SBATCH --account=plgmicrogranz-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --array=10-10
#SBATCH --time=02:00:00
#SBATCH --output=logs/postcheck-%A_%a.out
#SBATCH --error=logs/postcheck-%A_%a.err

set -euo pipefail

PROJECT_DIR="/net/pr2/projects/plgrid/plgg15102025/TruongCo_Station/Final_Master/WQ_Capstone-WQfinalproject_Nam_pipeline"
VENV_DIR="$PROJECT_DIR/.venv-capstone"

cd "$PROJECT_DIR"
module purge
module load GCCcore/12.3.0
module load Python/3.11.3
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true

portfolio_label=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
portfolio_dir="$PROJECT_DIR/run_outputs/portfolio_${portfolio_label}"

echo "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Portfolio source: $portfolio_dir"

python -m pip check
python -u verify_postprocessing_runtime.py "$portfolio_dir"

echo "Finished UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Post-processing verification completed successfully."
