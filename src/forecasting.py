from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAGS = [1, 2, 3, 7, 14]


def _feature_frame(series: pd.Series) -> pd.DataFrame:
    s = series.astype(float)
    out = pd.DataFrame({'y': s})
    for lag in LAGS:
        out[f'lag_{lag}'] = s.shift(lag)
    out['roll_mean_7'] = s.shift(1).rolling(7).mean()
    out['roll_std_7'] = s.shift(1).rolling(7).std()
    out['roll_mean_14'] = s.shift(1).rolling(14).mean()
    doy = out.index.dayofyear
    out['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    out['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    return out.dropna()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return {'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape)}


def model_factory(random_state: int = 42):
    return {
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Random Forest': RandomForestRegressor(
            n_estimators=180, max_depth=10, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=160, learning_rate=0.04, max_depth=3,
            random_state=random_state, loss='huber'
        ),
    }


def evaluate_models(daily: pd.DataFrame, target: str = 'temperature_celsius', test_fraction: float = 0.2):
    ts = daily[['date', target]].dropna().set_index('date')[target]
    feat = _feature_frame(ts)
    if len(feat) < 80:
        raise ValueError('Not enough daily observations for reliable time-series evaluation.')

    split = int(len(feat) * (1 - test_fraction))
    train, test = feat.iloc[:split], feat.iloc[split:]
    X_train, y_train = train.drop(columns='y'), train['y']
    X_test, y_test = test.drop(columns='y'), test['y']

    preds = {}
    fitted = {}
    rows = []

    # Seasonal-naive baseline uses the known value seven days earlier.
    baseline = test['lag_7'].to_numpy()
    preds['Seasonal Naive (7d)'] = baseline
    rows.append({'Model': 'Seasonal Naive (7d)', **_metrics(y_test.to_numpy(), baseline)})

    for name, model in model_factory().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        preds[name] = pred
        rows.append({'Model': name, **_metrics(y_test.to_numpy(), pred)})

    # Ensemble only learned models so its members share the same feature set.
    ensemble = np.mean(np.column_stack([preds[k] for k in ['Ridge','Random Forest','Gradient Boosting']]), axis=1)
    preds['Ensemble'] = ensemble
    rows.append({'Model': 'Ensemble', **_metrics(y_test.to_numpy(), ensemble)})

    metric_df = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
    pred_df = pd.DataFrame({'date': test.index, 'Actual': y_test.to_numpy()})
    for name, p in preds.items():
        pred_df[name] = p
    return metric_df, pred_df, fitted, feat.columns.drop('y').tolist()


def _make_one_row(history: pd.Series, future_date: pd.Timestamp) -> pd.DataFrame:
    vals = {}
    for lag in LAGS:
        vals[f'lag_{lag}'] = history.iloc[-lag]
    vals['roll_mean_7'] = history.iloc[-7:].mean()
    vals['roll_std_7'] = history.iloc[-7:].std()
    vals['roll_mean_14'] = history.iloc[-14:].mean()
    doy = future_date.dayofyear
    vals['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    vals['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame([vals])


def future_forecast(daily: pd.DataFrame, horizon: int = 7, target: str = 'temperature_celsius') -> pd.DataFrame:
    ts = daily[['date', target]].dropna().set_index('date')[target].astype(float)
    feat = _feature_frame(ts)
    X, y = feat.drop(columns='y'), feat['y']
    models = model_factory()
    for m in models.values():
        m.fit(X, y)

    histories = {name: ts.copy() for name in models}
    model_forecasts = {name: [] for name in models}
    dates = []

    last_date = ts.index.max()
    for step in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=step)
        dates.append(d)
        for name, model in models.items():
            hist = histories[name]
            row = _make_one_row(hist, d)
            p = float(model.predict(row)[0])
            model_forecasts[name].append(p)
            histories[name] = pd.concat([hist, pd.Series([p], index=[d])])

    out = pd.DataFrame({'date': dates, **model_forecasts})
    out['Ensemble'] = out[['Ridge','Random Forest','Gradient Boosting']].mean(axis=1)
    return out
