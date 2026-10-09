from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

WEATHER_FEATURES = [
    'humidity','wind_kph','pressure_mb','precip_mm','cloud','visibility_km','uv_index',
    'gust_kph','air_quality_PM2.5','air_quality_PM10','air_quality_Ozone'
]


def importance_tables(daily: pd.DataFrame):
    features = [c for c in WEATHER_FEATURES if c in daily.columns]
    work = daily[features + ['temperature_celsius']].dropna()
    if len(work) < 80:
        return pd.DataFrame(), pd.DataFrame()

    X, y = work[features], work['temperature_celsius']
    split = int(len(work) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    intrinsic = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='neg_mean_absolute_error')
    permutation = pd.DataFrame({'Feature': features, 'Importance': perm.importances_mean}).sort_values('Importance', ascending=False)
    return intrinsic, permutation
