# WorldQuant Capstone: Integrated HMM-WGAN Risk Pipeline

This repository implements a reproducible stock-market simulation and risk
measurement pipeline inspired by the paper **"Stock market simulator using
hidden Markov generative model and its application in risk measurement."**

The current implementation extends the original research code with:

- honest train/validation/forecast splits;
- fixed-HMM versus dynamic regime-transition comparison;
- production refitting and immutable portfolio manifests;
- regime-conditional rolling WGAN-GP training;
- paired Monte Carlo risk forecasts;
- ten independent 50-stock portfolios;
- optimized TensorFlow execution on NVIDIA A100 GPUs;
- strict output validation and SLURM failure handling;
- an Athena/PLGrid array-job workflow.

Project members:

- Co Nguyen
- Hieu Ha
- Nam Hoang

## Current Production Experiment

The checked-in production configuration is `pipeline_config.json`.

| Setting | Production value |
|---|---:|
| HMM regimes | 4 |
| HMM production refit restarts | 10 |
| Portfolios | 10 |
| Stocks per portfolio | 50 |
| Initial WGAN iteration limit | 2,000 per regime |
| Rolling WGAN iteration limit | 10 per applied regime update |
| WGAN batch size | 32 |
| Critic steps | 5 |
| Latent dimension | 128 |
| Rolling regime-memory base window | 256 trading days |
| Recency half-life | 256 trading days |
| Minimum weighted effective sample size | 64 |
| Forecast start | 2017-01-03 |
| Forecast end | latest common available date |
| Monte Carlo paths | 10,000 per model and forecast date |

The fixed HMM transition is always retained as the baseline. The current
selected dynamic candidate is:

```text
group_weighted_multinomial_logistic_gam
```

## End-to-End Design

```text
market and stock data
        |
        v
four-state production HMM and canonical state labels
        |
        +---------------- fixed HMM transition probabilities
        |
        +---------------- selected dynamic transition probabilities
        |
        v
four shared regime-conditional WGAN generators per portfolio
        |
        v
paired fixed/dynamic simulations using common random numbers
        |
        v
VaR, ES/CVaR, exceedance tests, loss functions and diagnostics
        |
        v
validated per-portfolio outputs and aggregate scorecard
```

### Fixed and dynamic models

"Fixed" and "dynamic" refer to the **regime-transition layer**, not to two
separate WGAN systems.

- The fixed model uses the HMM transition matrix.
- The dynamic model predicts date-specific next-state probabilities from the
  selected transition specification.
- Both models share the same four state-specific WGAN generators.
- Both models use paired uniform and latent random draws, which reduces Monte
  Carlo noise in their comparison.
- A normal return model is retained as an additional risk benchmark.

This isolates the effect of changing transition probabilities from the effect
of changing generated returns within a state.

## Production Pipeline

| Stage | Main implementation | Responsibility |
|---|---|---|
| Configuration and validation | `pipeline_runtime.py` | Loads configuration, validates scientific invariants, and manages portfolio manifests. |
| Phase 1 orchestration | `pipeline_phase1.py` | Selects/refits the HMM and dynamic transition candidate and produces the later-phase registry. |
| Dynamic transition candidates | `p1_4_dynamic_transition.py`, `p1_4_mlg_transition.py`, `p1_10_weighted_mlg_transition.py`, `p1_11_pvalue_vif_mlr_transition.py`, `p1_12_state_interaction_mlg_transition.py` | Fits and evaluates the eligible dynamic transition models. |
| WGAN and simulation | `pipeline_wgan.py` | Trains four rolling regime WGANs and produces paired fixed/dynamic simulations. |
| Risk evaluation | `pipeline_risk.py` | Calculates VaR/ES forecasts, scores, exceedance statistics, and comparison figures. |
| Original-paper diagnostics | `p3_1_stat_prop.py`, `p3_2_stat_prop_mv.py` | Produces univariate and multivariate stylized-fact diagnostics. |
| CLI entry point | `run_pipeline.py` | Supports dry runs, preparation, individual portfolios, aggregation, and full execution. |
| Portfolio validator | `validate_optimized_portfolio.py` | Checks completeness, finite values, model/date coverage, state counts, paths, and iteration settings. |

## Scientific Data Splits

The transition-model experiment uses explicit date-based information sets.

| Split | Purpose |
|---|---|
| Training: through 2009-06-01 | Fits transition candidates and the initial production models. |
| Validation: 2009-06-02 through 2016-12-30 | Selects the dynamic candidate without using forecast data. |
| Forecast: from 2017-01-03 | Final out-of-sample fixed-versus-dynamic comparison. |

The prediction period is excluded from model selection. Production artifacts
are refit according to `pipeline_config.json` after selection.

## Adaptive Regime Memory

Each rolling state update is causal: only observations strictly before the
forecast date are eligible.

The update begins with a 256-row lookback. If a regime is sparse, the memory is
expanded backward and exponentially weighted. Training is applied only when
Kish weighted effective sample size is at least 64. Otherwise, that state keeps
its last valid WGAN weights and the skipped update is recorded.

This prevents unstable fitting on a small number of effective observations.

## WGAN Performance Improvements

The production WGAN retains the original network dimensions and WGAN-GP
hyperparameters while improving execution substantially:

- one compiled `tf.function` covers a complete WGAN iteration;
- critic optimizer variables are built once;
- the parameter-free TensorFlow Addons Maxout behavior is implemented locally;
- generator output for critic updates is created outside the critic gradient
  tape and stopped from retaining the generator graph;
- rolling critic-loss history persists by state, matching the source stopping
  logic;
- weighted bootstrap batches use replacement;
- unused Python objects are collected periodically;
- simulation paths are summarized rather than all written to disk;
- progress logs report elapsed time, ETA, average iterations, and ESS skips.

TensorFlow operation determinism and explicit random seeds are enabled for
reproducibility. Exact reproducibility still requires the same hardware and
software stack.

## Installation

The supported production environment is Linux with Python 3.11 and an NVIDIA
GPU. Athena uses:

```bash
module purge
module load GCCcore/12.3.0
module load Python/3.11.3

python -m venv .venv-capstone
source .venv-capstone/bin/activate
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install -r requirements-athena.txt
python -m pip check
```

The production dependency list includes TensorFlow CUDA support and the final
diagnostic packages `powerlaw`, `seaborn`, and `networkx`.

`requirements.txt` is retained for the legacy scripts. Use
`requirements-athena.txt` for the integrated production pipeline.

## Required Data

The Dynamic Scale branch includes the minimum frozen artifact bundle required
to reproduce the configured later-phase experiment:

```text
Inputs/STOCKS.csv
Inputs/STOCK_STATUS_1999.csv
Inputs/STOCK_STATUS_2023.csv
phase_1/selected_models_for_later_phases_q4.csv
phase_1/production_q4/
baseline_fixed_transition/portfolio_manifests/
baseline_fixed_transition/portfolio_*/daily_risk_forecasts.csv
baseline_fixed_transition/portfolio_*/risk_scores.csv
```

The bundled Phase 1 directory contains only the selected q4 Group-Weighted MLG
handoff and its validation dependencies. The fixed benchmark contains only the
files needed to pair dates and realized returns and to import fixed-model risk
scores. All generated Dynamic Scale outputs, volatility caches, environments,
logs, checkpoints, PDFs, and unrelated source data remain ignored by Git.

This artifact bundle supports the checked-in `later_phases` Dynamic Scale run.
It does not contain every intermediate dataset needed to recalibrate all Phase
1 candidate models from raw inputs.

## Local Validation

Validate syntax:

```bash
python -m py_compile \
  run_pipeline.py pipeline_phase1.py pipeline_runtime.py \
  pipeline_risk.py pipeline_volatility.py pipeline_wgan.py p2_0_utils.py \
  validate_optimized_portfolio.py \
  verify_optimized_wgan_runtime.py \
  verify_postprocessing_runtime.py
```

Run the lightweight tests:

```bash
python -m unittest -v \
  test_pipeline_dynamic_scale.py \
  test_pipeline_wgan_adaptive_memory.py \
  test_pipeline_portfolio_sampling.py \
  test_validate_optimized_portfolio.py
```

The current focused suite contains 16 tests covering Dynamic Scale causality
and reconstruction, fixed-baseline pairing, adaptive memory, ESS gating,
deterministic balanced portfolios, manifest diagnostics, strict CSV boolean
parsing, and production-output validation.

## Pipeline Commands

### Inspect the production plan

```bash
python run_pipeline.py \
  --config pipeline_config.json \
  --run-scope later_phases \
  --dry-run
```

### Run/refit Phase 1 and prepare immutable manifests

```bash
python run_pipeline.py \
  --config pipeline_config.json \
  --run-scope full \
  --prepare-only
```

If production Phase 1 artifacts already exist and are validated, preparation
can use `--run-scope later_phases`.

### Run one portfolio

```bash
python run_pipeline.py \
  --config pipeline_config.json \
  --run-scope later_phases \
  --portfolio-id 1
```

### Aggregate completed portfolios

```bash
python run_pipeline.py \
  --config pipeline_config.json \
  --run-scope later_phases \
  --aggregate-only
```

Aggregation requires all ten configured portfolios and refuses partial sets.

### Run the Dynamic Scale experiment

`pipeline_config_dynamic_scale.json` is a separate later-phase experiment for
the selected Group-Weighted MLG only. It trains the WGAN on volatility-standardized
returns, applies causal daily GJR-GARCH scaling to simulated residuals, and imports
the preserved fixed-transition scores for comparison instead of rerunning that
baseline.

```bash
python run_pipeline.py \
  --config pipeline_config_dynamic_scale.json \
  --dry-run

python run_pipeline.py \
  --config pipeline_config_dynamic_scale.json \
  --prepare-only

python run_pipeline.py \
  --config pipeline_config_dynamic_scale.json \
  --portfolio-id 1
```

The dedicated outputs are written under `run_outputs_dynamic_scale/`. See
`RUN_LATER_PHASE.txt` for the full ten-portfolio command, fixed-baseline pairing
checks, volatility fallbacks, and generated files.

## Athena / PLGrid Execution

The local Athena deployment uses production scripts for one A100 per
portfolio:

```text
slurm/athena_full_array.slurm
slurm/athena_aggregate.slurm
slurm/athena_postprocessing_check.slurm
```

The `slurm/` directory is intentionally excluded from Git because its account,
partition, paths, and allocation settings are deployment-specific. The
commands below assume those local scripts are present in the Athena working
copy.

The production array requests, per task:

```text
1 NVIDIA A100 GPU
16 CPU cores
120 GB RAM
48-hour limit
```

Update `PROJECT_DIR`, `VENV_DIR`, account, and partition directives if the
Athena allocation or project location differs from the checked-in example.

### Pre-submission checks

```bash
mkdir -p logs

bash -n slurm/athena_full_array.slurm
bash -n slurm/athena_aggregate.slurm
bash -n slurm/athena_postprocessing_check.slurm

python -m pip check
```

The array script fails before training unless it can:

- find the configuration, selected-model registry, and portfolio manifest;
- import every final diagnostic dependency;
- detect exactly one TensorFlow GPU;
- compile and run the optimized WGAN step;
- verify the expected parameter-free Maxout model sizes.

### Production pilot

Run one portfolio before committing the full allocation:

```bash
pilot_job=$(sbatch --parsable --array=1-1 slurm/athena_full_array.slurm)
echo "$pilot_job"
```

Monitor it:

```bash
squeue -j "$pilot_job"
tail -f "logs/full-${pilot_job}_1.out"
```

Validate completion:

```bash
python validate_optimized_portfolio.py \
  run_outputs/portfolio_01 \
  --production
```

### Remaining portfolios and aggregation

```bash
array_job=$(sbatch --parsable --array=2-10%9 slurm/athena_full_array.slurm)

aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${array_job}" \
  slurm/athena_aggregate.slurm)

echo "Array: $array_job"
echo "Aggregation: $aggregate_job"
```

For a clean full rerun, submit `--array=1-10%10`.

Detailed instructions are in `ATHENA_SUBMISSION_RUNBOOK.md`.

## Monitoring and Completion

Monitor array state and resource use:

```bash
squeue -j "$array_job"

sacct -j "$array_job" \
  --format=JobID,State,ExitCode,Elapsed,ReqMem,MaxRSS
```

Progress logs distinguish the pre-2017 WGAN warm-up stage from the forecast
stage. The first ETA is dominated by TensorFlow graph compilation; use later
progress observations for runtime estimates.

A portfolio is complete only when:

- SLURM reports `COMPLETED` and `ExitCode=0:0`;
- `run_outputs/portfolio_XX/SUCCESS.json` exists;
- `validate_optimized_portfolio.py --production` reports `status: valid`;
- final generator checkpoints exist.

The SLURM cleanup trap returns a nonzero exit code when a command fails, the job
is interrupted, or the success marker is missing.

## Post-Processing Verification

Original-paper diagnostics are deliberately tested separately because they run
after the expensive training/simulation stage.

If an incomplete run already contains full forecast CSVs, verify the complete
diagnostic path without retraining. For portfolio 10:

```bash
postcheck_job=$(sbatch --parsable --array=10-10 \
  slurm/athena_postprocessing_check.slurm)
```

The check runs both univariate and multivariate diagnostics for the fixed and
dynamic models and requires 48 non-empty artifacts.

## Outputs

Each portfolio writes to `run_outputs/portfolio_XX/`:

```text
SUCCESS.json
daily_risk_forecasts.csv
risk_scores.csv
rolling_250_day_exceedances.csv
simulated_state_counts.csv
portfolio_real_returns.csv
representative_simulated_returns_<model>.csv
wgan_loss_report.csv
wgan_critic_loss.png
wgan_ess_diagnostics.png
wgan_checkpoints/
original_paper_risk_diagnostics/
original_paper_stylized_facts/
```

The aggregate job writes:

```text
run_outputs/run_summary.json
run_outputs/aggregate_comparison/all_portfolio_risk_scores.csv
run_outputs/aggregate_comparison/paired_portfolio_scores.csv
run_outputs/aggregate_comparison/primary_scorecard_summary.csv
run_outputs/aggregate_comparison/backtest_p_value_summary.csv
```

## Legacy Research Scripts

The original script-by-script implementation is preserved for reference:

| Legacy phase | Scripts |
|---|---|
| Raw data preparation | `p0_1_raw_to_rets.py` |
| HMM seed search and labeling | `p1_0_hmm_params.py`, `p1_1_best_params.py`, `p1_2_hmm.py` |
| Original WGAN training | `p2_1_wgan_plt.py`, `p2_2_wgan.py` |
| Original simulation and risk | `p3_0_sims.py`, `p3_3_risk_man.py` |
| Legacy orchestration | `p0_0_orchestration.py` |

The integrated `run_pipeline.py` workflow is the supported path for the current
fixed-versus-dynamic experiment and Athena execution.

## Repository Structure

| Path | Description |
|---|---|
| `pipeline_*.py` | Integrated production phases and shared runtime logic. |
| `run_pipeline.py` | Main command-line entry point. |
| `p1_*.py` | HMM and dynamic transition model implementations. |
| `p2_0_utils.py` | WGAN networks, Maxout, gradient penalty, and stopping rules. |
| `p3_*.py` | Original simulation and diagnostic implementations. |
| `slurm/` | Local Athena batch scripts; intentionally excluded from Git. |
| `docs/` | GitHub-visible reference result snapshots. |
| `phase_1_model_specifications.csv` | Candidate enablement and selection policy. |
| `pipeline_config.json` | Production experiment configuration. |
| `pipeline_config_smoke.json` | Small functional-test configuration. |
| `pipeline_config_dynamic_scale.json` | Dynamic-only Group-Weighted MLG plus causal volatility scaling experiment. |
| `pipeline_volatility.py` | GJR-GARCH filtering, fallbacks, cache validation, and standardized-return artifacts. |
| `requirements-athena.txt` | Integrated Linux/Athena dependencies. |
| `ATHENA_SUBMISSION_RUNBOOK.md` | Step-by-step production runbook. |

## Reproducibility and Storage Notes

- Portfolio manifests are deterministic and should not be regenerated midway
  through an experiment.
- Completed portfolios are resumed only when `SUCCESS.json` exists.
- An incomplete non-empty output folder is rejected unless it is moved aside or
  overwrite behavior is explicitly enabled.
- All simulation paths are not stored; daily distribution summaries and one
  representative stock-return vector per transition model are retained.
- `run_outputs/`, environments, logs, checkpoints, archives, inputs, generated
  Phase 1 artifacts, deployment-specific SLURM scripts, and other large
  research outputs are excluded from Git.
- Use the same Python, TensorFlow, CUDA, dependency versions, hardware class,
  configuration, manifests, and random seeds for reproducible reruns.

## Git Branch

The integrated implementation is maintained on:

```text
WQfinalproject
```

Push updates with:

```bash
git push origin WQfinalproject
```
