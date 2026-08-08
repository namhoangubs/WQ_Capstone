#!/bin/bash
#SBATCH --job-name=hmm-wgan-prepare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err

set -euo pipefail

# Change this to the repository path on the cluster.
PROJECT_DIR=/path/to/WQ_Capstone-WQfinalproject_Nam_pipeline
cd "$PROJECT_DIR"

# Adapt module names and environment activation to the cluster.
module purge
module load cuda
module load python
source .venv/bin/activate

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -c "import tensorflow as tf; print('TensorFlow GPUs:', tf.config.list_physical_devices('GPU'))"

# Refit/select Phase 1 once and create all ten immutable manifests. No WGANs run.
python run_pipeline.py --config pipeline_config.json --run-scope full --prepare-only
