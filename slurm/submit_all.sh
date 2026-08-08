#!/bin/bash
set -euo pipefail

mkdir -p logs

prepare_job=$(sbatch --parsable slurm/prepare.slurm)
array_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" slurm/portfolio_array.slurm)
aggregate_job=$(sbatch --parsable --dependency="afterok:${array_job}" slurm/aggregate.slurm)

echo "Prepare job:   $prepare_job"
echo "Portfolio job: $array_job"
echo "Aggregate job: $aggregate_job"
