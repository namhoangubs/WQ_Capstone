#!/bin/bash
#SBATCH --job-name=hmm-wgan-portfolio
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%10
#SBATCH --time=48:00:00
#SBATCH --output=logs/portfolio-%A_%a.out
#SBATCH --error=logs/portfolio-%A_%a.err

set -euo pipefail

PROJECT_DIR=/path/to/WQ_Capstone-WQfinalproject_Nam_pipeline
cd "$PROJECT_DIR"

module purge
module load cuda
module load python
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "Portfolio: $SLURM_ARRAY_TASK_ID"
echo "Allocated CUDA devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# Each task reads the prepared registry/manifests and writes only portfolio_XX.
# Portfolio-specific runs deliberately do not write aggregate comparison files.
python run_pipeline.py \
  --config pipeline_config.json \
  --run-scope later_phases \
  --portfolio-id "$SLURM_ARRAY_TASK_ID"
