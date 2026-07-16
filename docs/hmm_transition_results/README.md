# HMM Transition Result Snapshots

This folder contains selected generated outputs from:

```powershell
python benchmark_dynamic_transition.py
python p1_5_transition_comparison.py
python p1_6_duration_hazard.py
```

The original generated files live under `phase_1/` and `phase_1/transition_comparison/`, but `phase_1/` is ignored by Git because it contains generated research artifacts. These snapshots are copied here so reviewers can see the HMM transition comparison directly on GitHub.

All results use the same explicit windows: train to 2009-06-01 (matching the fixed HMM training cutoff), validation 2009-06-02 to 2016-12-30 (lag selection only), prediction 2017-01-03 to 2023-12-05 (final out-of-sample comparison). Every report row carries `SPLIT`, `SPLIT_START`, and `SPLIT_END` columns.

Included files:

| File | Purpose |
|---|---|
| `dynamic_transition_report_q2.csv`, `dynamic_transition_report_q4.csv` | Fixed HMM versus dynamic-logit metrics per lag configuration, per split. |
| `duration_hazard_report_q2.csv`, `duration_hazard_report_q4.csv` | Fixed HMM versus duration-hazard scheme metrics per split. |
| `duration_hazard_betas.csv` | Fitted hazard betas per state: constrained (>= 0) and unconstrained diagnostic. |
| `transition_matrix_summary.csv` | Observed prediction-window transitions versus fixed HMM and dynamic-logit average transitions. |
| `duration_hazard_matrix_summary.csv` | Observed prediction-window transitions versus fixed HMM and hazard-scheme average transitions. |
| `transition_matrix_fixed_only_q1.png` | One-state baseline transition matrix. |
| `transition_matrix_comparison_q2.png`, `transition_matrix_comparison_q4.png` | Observed, fixed, dynamic-logit average, and difference matrices on the prediction window. |
| `transition_matrix_hazard_q2.png`, `transition_matrix_hazard_q4.png` | Observed, fixed, hazard-scheme average, and difference matrices on the prediction window. |
| `duration_stay_probability_q2.png`, `duration_stay_probability_q4.png` | Dynamic-logit stay probability by regime duration on the prediction window. |
| `duration_hazard_stay_q2.png`, `duration_hazard_stay_q4.png` | Hazard-scheme stay probability versus the observed stay curve. The fitted betas are zero, so the scheme collapses onto the fixed HMM line while the observed curve rises with duration. |
| `transition_log_loss_delta_grid_validation.png`, `transition_log_loss_delta_grid_prediction.png` | Dynamic-minus-fixed log-loss across lag settings per split. Negative is better. |
| `transition_brier_score_delta_grid_validation.png`, `transition_brier_score_delta_grid_prediction.png` | Dynamic-minus-fixed Brier score across lag settings per split. Negative is better. |

Headline results on the prediction window: the dynamic logit beats the fixed HMM on `q4` log-loss (0.497 versus 0.752) with tied accuracy and Brier score, and adds nothing on `q2`. The duration-hazard scheme (exit probability forced to rise with duration) is rejected by the data: all fitted betas hit the zero boundary, the unconstrained fit prefers negative betas in every state, and forcing positive betas strictly worsens out-of-sample log-loss. Empirically, the probability of staying in a regime rises with time already spent in it.

The dynamic models do not have one fixed transition matrix. The matrices shown for them are average predicted transition probabilities on the prediction window, grouped by current HMM state.
