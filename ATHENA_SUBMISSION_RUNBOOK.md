# Athena production submission runbook

Project directory:

```bash
/net/pr2/projects/plgrid/plgg15102025/TruongCo_Station/Final_Master/WQ_Capstone-WQfinalproject_Nam_pipeline
```

Use the production configuration: four regimes, 50 stocks per portfolio, 2,000
initial WGAN iterations, at most 10 rolling WGAN iterations, and 10,000
simulation paths per model/date. One SLURM task owns one A100 and one portfolio.

| Step | Purpose | Command |
|---:|---|---|
| 1 | Enter the project | `cd /net/pr2/projects/plgrid/plgg15102025/TruongCo_Station/Final_Master/WQ_Capstone-WQfinalproject_Nam_pipeline` |
| 2 | Create the log directory | `mkdir -p logs` |
| 3 | Check SLURM syntax | `bash -n slurm/athena_full_array.slurm && bash -n slurm/athena_aggregate.slurm` |
| 4 | Load the tested Python module | `module purge && module load GCCcore/12.3.0 && module load Python/3.11.3` |
| 5 | Activate the environment | `source .venv-capstone/bin/activate` |
| 6 | Confirm the interpreter | `which python && python --version` |
| 7 | Check Python files | `python -m py_compile run_pipeline.py pipeline_wgan.py p2_0_utils.py validate_optimized_portfolio.py verify_optimized_wgan_runtime.py` |
| 8 | Run local tests | `python -m unittest -v test_pipeline_wgan_adaptive_memory.py test_pipeline_portfolio_sampling.py test_validate_optimized_portfolio.py` |
| 9 | Validate the production plan | `python run_pipeline.py --config pipeline_config.json --run-scope later_phases --dry-run` |
| 10 | Confirm all manifests exist | `for id in $(seq -w 1 10); do test -f run_outputs/portfolio_manifests/portfolio_${id}.csv || echo "missing portfolio_${id}.csv"; done` |
| 11 | Submit portfolio 1 as the production pilot | `pilot_job=$(sbatch --parsable --array=1-1 slurm/athena_full_array.slurm); echo "$pilot_job"` |
| 12 | Monitor the pilot | `squeue -j "$pilot_job"` |
| 13 | Follow model progress | `tail -f "logs/full-${pilot_job}_1.out"` |
| 14 | Inspect final resource use | `sacct -j "$pilot_job" --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS,AllocTRES%80` |
| 15 | Validate portfolio 1 | `python validate_optimized_portfolio.py run_outputs/portfolio_01 --production` |
| 16 | Submit portfolios 2-10 after the pilot passes | `array_job=$(sbatch --parsable --array=2-10%9 slurm/athena_full_array.slurm); echo "$array_job"` |
| 17 | Submit aggregation with a success dependency | `aggregate_job=$(sbatch --parsable --dependency="afterok:${array_job}" slurm/athena_aggregate.slurm); echo "$aggregate_job"` |
| 18 | Monitor production and aggregation | `squeue -j "$array_job,$aggregate_job"` |
| 19 | Inspect all final states | `sacct -j "$array_job,$aggregate_job" --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS` |
| 20 | Inspect the final summary | `cat run_outputs/run_summary.json` |

## Pilot acceptance gates

Proceed to portfolios 2-10 only when all of these are true:

- The log contains `Optimized WGAN runtime verification passed.`
- TensorFlow reports exactly one GPU and the variables are on `GPU:0`.
- `sacct` reports `COMPLETED` and `ExitCode=0:0`.
- `run_outputs/portfolio_01/SUCCESS.json` exists.
- The production validator reports `"status": "valid"`.
- The completed pilot fits within the desired wall-clock budget.
- `MaxRSS` remains below the requested 120 GB and GPU memory remains below 40 GB.
- Review `updates_reaching_iteration_limit_rate`. A value above 0.10 means the
  10-iteration cap is frequently binding and should be investigated before the
  remaining portfolios are launched.

The first progress ETA is not reliable because graph compilation and startup
dominate it. Judge the live ETA only after at least 50-100 rolling dates.

If a completed training run fails during original-paper diagnostics, install
the declared requirements and verify that stage against its existing CSV files
before retraining. For portfolio 10:

```bash
postcheck_job=$(sbatch --parsable --array=10-10 slurm/athena_postprocessing_check.slurm)
echo "$postcheck_job"
```

Only retrain after the post-processing check exits successfully and reports 48
non-empty diagnostic artifacts.

## Existing incomplete output

Before resubmitting a failed portfolio, inspect its directory. `FAILURE.json`
alone is safe. If other partial files exist and there is no `SUCCESS.json`, move
the entire portfolio directory to a dated backup location, then resubmit. Do not
mix files from two runs.

Example for portfolio 1 after confirming it is incomplete:

```bash
backup_dir="run_outputs/incomplete_backup_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$backup_dir"
mv run_outputs/portfolio_01 "$backup_dir/"
```

## Completion artifacts

Successful portfolio tasks produce:

```text
run_outputs/portfolio_01/SUCCESS.json
...
run_outputs/portfolio_10/SUCCESS.json
```

The aggregation job validates every portfolio before creating:

```text
run_outputs/run_summary.json
run_outputs/aggregate_comparison/primary_scorecard_summary.csv
```
