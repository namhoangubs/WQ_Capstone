#!/bin/bash
#SBATCH --job-name=hmm-wgan-aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aggregate-%j.out
#SBATCH --error=logs/aggregate-%j.err

set -euo pipefail

PROJECT_DIR=/path/to/WQ_Capstone-WQfinalproject_Nam_pipeline
cd "$PROJECT_DIR"

module purge
module load python
source .venv/bin/activate

# Discover SUCCESS.json/risk_scores.csv from all completed portfolios.
python run_pipeline.py --config pipeline_config.json --aggregate-only
