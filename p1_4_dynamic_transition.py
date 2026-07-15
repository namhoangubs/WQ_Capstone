from numpy import zeros
from pandas import DataFrame, concat, read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p1_3_transition_features import build_transition_dataset
from project_config import BASE_DIR


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


def split_transition_dataset(X, y, meta, validation_fraction=0.25):
    split_index = int(len(X) * (1 - validation_fraction))

    if split_index <= 0 or split_index >= len(X):
        raise ValueError('Validation split leaves no training or validation observations.')

    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
        meta.iloc[:split_index],
        meta.iloc[split_index:],
    )


def fit_dynamic_transition_model(X_train, y_train):
    model = Pipeline([
        ('scale', StandardScaler()),
        ('logit', LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
        ))
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_probabilities(label, y_valid, probabilities, state_classes, extra=None):
    predicted = [state_classes[index] for index in probabilities.argmax(axis=1)]
    row = {
        'MODEL': label,
        'LOG_LOSS': log_loss(y_valid, probabilities, labels=state_classes),
        'ACCURACY': accuracy_score(y_valid, predicted),
        'BRIER_SCORE': multiclass_brier_score(y_valid, probabilities, state_classes),
    }

    if extra:
        row.update(extra)

    return row


def evaluate_dynamic_transition(n_states, max_state_lag, max_exog_lag, validation_fraction=0.25):
    stock_rets = read_csv(BASE_DIR / 'phase_1' / f'stock_rets_q{n_states}.csv', index_col=[0], parse_dates=[0])
    exog_rets = read_csv(BASE_DIR / 'processed' / 'exog_rets.csv', index_col=[0], parse_dates=[0])
    transition_matrix = read_csv(BASE_DIR / 'phase_1' / f'hmm_trans_mat_q{n_states}.csv', index_col=[0])

    X, y, meta, state_classes = build_transition_dataset(
        stock_rets['STATE'],
        exog_rets,
        max_state_lag=max_state_lag,
        max_exog_lag=max_exog_lag,
    )

    X_train, X_valid, y_train, y_valid, _, meta_valid = split_transition_dataset(
        X, y, meta, validation_fraction=validation_fraction
    )

    extra = {
        'N_STATES': n_states,
        'STATE_LAG': max_state_lag,
        'EXOG_LAG': max_exog_lag,
        'TRAIN_ROWS': len(X_train),
        'VALID_ROWS': len(X_valid),
    }

    fixed_probs = fixed_transition_probabilities(
        meta_valid['CURRENT_STATE'], transition_matrix, state_classes
    )
    fixed_row = evaluate_probabilities('fixed_hmm_transition', y_valid, fixed_probs, state_classes, extra)

    model = fit_dynamic_transition_model(X_train, y_train)
    logit = model.named_steps['logit']
    dynamic_probs = _align_probabilities(model.predict_proba(X_valid), logit.classes_, state_classes)
    dynamic_row = evaluate_probabilities('dynamic_duration_logit', y_valid, dynamic_probs, state_classes, extra)

    return DataFrame([fixed_row, dynamic_row])


def run_lag_grid(n_states, state_lags=(0, 1, 2), exog_lags=(0, 1, 2), validation_fraction=0.25):
    rows = []
    for state_lag in state_lags:
        for exog_lag in exog_lags:
            rows.append(evaluate_dynamic_transition(
                n_states=n_states,
                max_state_lag=state_lag,
                max_exog_lag=exog_lag,
                validation_fraction=validation_fraction,
            ))

    return concat(rows, ignore_index=True)


def save_dynamic_transition_report(n_states, state_lags=(0, 1, 2), exog_lags=(0, 1, 2), validation_fraction=0.25):
    report = run_lag_grid(
        n_states=n_states,
        state_lags=state_lags,
        exog_lags=exog_lags,
        validation_fraction=validation_fraction,
    )
    output_path = BASE_DIR / 'phase_1' / f'dynamic_transition_report_q{n_states}.csv'
    report.to_csv(output_path, index=False)
    return report, output_path
