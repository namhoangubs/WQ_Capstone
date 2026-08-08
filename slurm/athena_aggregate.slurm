#!/bin/bash
# Aggregate the ten completed portfolio results on the known Athena allocation.
# Submit only after the production array finishes successfully.
#SBATCH --job-name=wgan-aggregate
#SBATCH --account=plgmicrogranz-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/aggregate-%j.out
#SBATCH --error=logs/aggregate-%j.err

set -euo pipefail

PROJECT_DIR="/net/pr2/projects/plgrid/plgg15102025/TruongCo_Station/Final_Master/WQ_Capstone-WQfinalproject_Nam_pipeline"
VENV_DIR="$PROJECT_DIR/.venv-capstone"

cd "$PROJECT_DIR"
module purge
module load GCCcore/12.3.0
module load Python/3.11.3
source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

echo "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"

for portfolio_id in $(seq 1 10); do
    portfolio_label=$(printf '%02d' "$portfolio_id")
    output_dir="$PROJECT_DIR/run_outputs/portfolio_${portfolio_label}"
    python -u validate_optimized_portfolio.py "$output_dir" --production
done

python -u run_pipeline.py \
  --config pipeline_config.json \
  --run-scope later_phases \
  --aggregate-only

test -f "$PROJECT_DIR/run_outputs/run_summary.json"
test -f "$PROJECT_DIR/run_outputs/aggregate_comparison/primary_scorecard_summary.csv"

echo "Finished UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "All ten portfolios validated and aggregated successfully."
