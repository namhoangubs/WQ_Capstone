from numpy import array
from pandas import DataFrame, concat, get_dummies


def compute_regime_duration(states):
    """Count consecutive days spent in the current inferred HMM state."""
    values = array(states)
    durations = []
    current_duration = 0
    previous_state = None

    for state in values:
        if previous_state is None or state != previous_state:
            current_duration = 1
        else:
            current_duration += 1

        durations.append(current_duration)
        previous_state = state

    return DataFrame({'DURATION': durations}, index=states.index)


def _state_lag_features(states, max_state_lag, state_classes):
    features = []
    for lag in range(max_state_lag + 1):
        lagged = states.shift(lag).rename(f'STATE_LAG_{lag}')
        dummies = get_dummies(lagged, prefix=f'STATE_LAG_{lag}')

        for state in state_classes:
            column = f'STATE_LAG_{lag}_{state}'
            if column not in dummies:
                dummies[column] = 0

        features.append(dummies[[f'STATE_LAG_{lag}_{state}' for state in state_classes]])

    return concat(features, axis=1)


def _exog_lag_features(exog_rets, max_exog_lag):
    features = []
    for lag in range(max_exog_lag + 1):
        lagged = exog_rets.shift(lag).copy()
        lagged.columns = [f'{column}_LAG_{lag}' for column in lagged.columns]
        features.append(lagged)

    return concat(features, axis=1)


def build_transition_dataset(states, exog_rets, max_state_lag=1, max_exog_lag=1):
    """
    Build a supervised transition dataset.

    Features use information available at date t. The target is the next
    inferred HMM state q[t+1].
    """
    states = states.astype(int).rename('STATE')
    state_classes = sorted(states.dropna().unique())

    aligned = concat([states, exog_rets], axis=1, join='inner').dropna()
    states = aligned['STATE'].astype(int)
    exog_rets = aligned.drop(columns=['STATE'])

    duration = compute_regime_duration(states)
    duration_features = [duration]

    for state in state_classes:
        duration_features.append(
            (duration['DURATION'] * (states == state).astype(int)).rename(f'DURATION_STATE_{state}')
        )

    X = concat([
        _state_lag_features(states, max_state_lag, state_classes),
        _exog_lag_features(exog_rets, max_exog_lag),
        concat(duration_features, axis=1)
    ], axis=1)

    y = states.shift(-1).rename('NEXT_STATE')
    meta = DataFrame({
        'CURRENT_STATE': states
    }, index=states.index)

    dataset = concat([X, y, meta], axis=1).dropna()
    feature_columns = [column for column in X.columns if column in dataset.columns]

    return (
        dataset[feature_columns].astype(float),
        dataset['NEXT_STATE'].astype(int),
        dataset[['CURRENT_STATE']].astype(int),
        state_classes,
    )
