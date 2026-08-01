from numpy import zeros
from pandas import DataFrame, concat, read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p1_3_transition_features import build_transition_dataset
from project_config import BASE_DIR

# The fixed HMM transition matrix is estimated on data before TRAIN_END
# (TRAIN_DATE in p1_2_hmm.py). The dynamic model is fitted on the same window
# so both models carry the same information set.
TRAIN_END = '2009-06-02'
# Lag configurations are selected on [TRAIN_END, VALIDATION_END). The final
# fixed-versus-dynamic comparison uses the untouched prediction window from
# VALIDATION_END onward.
VALIDATION_END = '2017-01-01'


def _align_probabilities(probabilities, model_classes, state_classes):
    aligned = zeros((probabilities.shape[0], len(state_classes)))
    class_to_column = {state: index for index, state in enumerate(state_classes)}

    for model_column, state in enumerate(model_classes):
        aligned[:, class_to_column[int(state)]] = probabilities[:, model_column]

    return aligned


def multiclass_brier_score(y_true, probabilities, state_classes):
    class_to_column = {state: index for index, state in enumerate(state_classes)}
    encoded = zeros(probabilities.shape)

    for row, state in enumerate(y_true):
        encoded[row, class_to_column[int(state)]] = 1.0

    return ((encoded - probabilities) ** 2).sum(axis=1).mean()


def fixed_transition_probabilities(current_states, transition_matrix, state_classes):
    matrix = transition_matrix.to_numpy(dtype=float)
    probabilities = zeros((len(current_states), len(state_classes)))

    for row, state in enumerate(current_states.astype(int)):
        probabilities[row] = matrix[state]

    return probabilities


def split_transition_dataset(X, y, meta, train_end=TRAIN_END, validation_end=VALIDATION_END):
    masks = {
        'train': X.index < train_end,
        'validation': (X.index >= train_end) & (X.index < validation_end),
        'prediction': X.index >= validation_end,
    }

    splits = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f'The {name} split has no observations.')
        splits[name] = (X[mask], y[mask], meta[mask])

    return splits


def fit_dynamic_transition_model(X_train, y_train):
    model = Pipeline([
        ('scale', StandardScaler()),
        ('logit', LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            random_state=42,
        ))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_probabilities(label, y_true, probabilities, state_classes, extra=None):
    predicted = [state_classes[index] for index in probabilities.argmax(axis=1)]
    row = {
        'MODEL': label,
        'LOG_LOSS': log_loss(y_true, probabilities, labels=state_classes),
        'ACCURACY': accuracy_score(y_true, predicted),
        'BRIER_SCORE': multiclass_brier_score(y_true, probabilities, state_classes),
    }

    if extra:
        row.update(extra)

    return row


def evaluate_dynamic_transition(n_states, max_state_lag, max_exog_lag):
    stock_rets = read_csv(BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv', index_col=[0], parse_dates=[0])
    exog_rets = read_csv(BASE_DIR / 'processed' / 'exog_rets.csv', index_col=[0], parse_dates=[0])
    transition_matrix = read_csv(BASE_DIR / 'phase_1' / f'hmm_trans_mat_q{n_states}.csv', index_col=[0])

    X, y, meta, state_classes = build_transition_dataset(
        stock_rets['STATE'],
        exog_rets,
        max_state_lag=max_state_lag,
        max_exog_lag=max_exog_lag,
    )

    splits = split_transition_dataset(X, y, meta)
    X_train, y_train, _ = splits['train']

    model = fit_dynamic_transition_model(X_train, y_train)
    logit = model.named_steps['logit']

    rows = []
    for split_name in ['validation', 'prediction']:
        X_eval, y_eval, meta_eval = splits[split_name]
        extra = {
            'SPLIT': split_name,
            'SPLIT_START': X_eval.index.min().date(),
            'SPLIT_END': X_eval.index.max().date(),
            'N_STATES': n_states,
            'STATE_LAG': max_state_lag,
            'EXOG_LAG': max_exog_lag,
            'TRAIN_ROWS': len(X_train),
            'EVAL_ROWS': len(X_eval),
        }

        fixed_probs = fixed_transition_probabilities(
            meta_eval['CURRENT_STATE'], transition_matrix, state_classes
        )
        rows.append(evaluate_probabilities('fixed_hmm_transition', y_eval, fixed_probs, state_classes, extra))

        dynamic_probs = _align_probabilities(model.predict_proba(X_eval), logit.classes_, state_classes)
        rows.append(evaluate_probabilities('dynamic_duration_logit', y_eval, dynamic_probs, state_classes, extra))

    return DataFrame(rows)


def run_lag_grid(n_states, state_lags=(0, 1, 2), exog_lags=(0, 1, 2)):
    rows = []
    for state_lag in state_lags:
        for exog_lag in exog_lags:
            rows.append(evaluate_dynamic_transition(
                n_states=n_states,
                max_state_lag=state_lag,
                max_exog_lag=exog_lag,
            ))

    return concat(rows, ignore_index=True)


def save_dynamic_transition_report(n_states, state_lags=(0, 1, 2), exog_lags=(0, 1, 2)):
    report = run_lag_grid(
        n_states=n_states,
        state_lags=state_lags,
        exog_lags=exog_lags,
    )
    output_path = BASE_DIR / 'phase_1' / f'dynamic_transition_report_q{n_states}.csv'
    report.to_csv(output_path, index=False)
    return report, output_path
