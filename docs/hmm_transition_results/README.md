# HMM Transition Result Snapshots

This folder contains selected generated outputs from:

```powershell
python p1_5_transition_comparison.py
```

The original generated files live under `phase_1/transition_comparison/`, but `phase_1/` is ignored by Git because it contains generated research artifacts. These snapshots are copied here so reviewers can see the HMM transition comparison directly on GitHub.

Included files:

| File | Purpose |
|---|---|
| `transition_matrix_summary.csv` | Numeric comparison of observed validation transitions, old fixed HMM transitions, and new dynamic average transitions. |
| `transition_matrix_fixed_only_q1.png` | One-state baseline transition matrix. |
| `transition_matrix_comparison_q2.png` | Observed, old fixed, new dynamic, and difference matrices for the 2-state HMM. |
| `transition_matrix_comparison_q4.png` | Observed, old fixed, new dynamic, and difference matrices for the 4-state HMM. |
| `duration_stay_probability_q2.png` | How dynamic stay probability changes with regime duration for the 2-state HMM. |
| `duration_stay_probability_q4.png` | How dynamic stay probability changes with regime duration for the 4-state HMM. |
| `transition_log_loss_delta_grid.png` | Dynamic-minus-fixed log-loss across lag settings. Negative is better. |
| `transition_brier_score_delta_grid.png` | Dynamic-minus-fixed Brier score across lag settings. Negative is better. |

The dynamic model does not have one fixed transition matrix. The matrix shown for the dynamic model is the average predicted transition probability on the validation period, grouped by current HMM state.
