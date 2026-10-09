from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ['temperature_celsius','precip_mm','humidity','wind_kph','pressure_mb','air_quality_PM2.5']


def detect_anomalies(daily: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    cols = [c for c in FEATURES if c in daily.columns]
    work = daily[['date'] + cols].dropna().copy()
    if len(work) < 30:
        work['anomaly'] = False
        work['anomaly_score'] = np.nan
        return work

    X = StandardScaler().fit_transform(work[cols])
    model = IsolationForest(n_estimators=250, contamination=contamination, random_state=42)
    label = model.fit_predict(X)
    score = -model.score_samples(X)  # higher = more unusual
    work['anomaly'] = label == -1
    work['anomaly_score'] = score
    return work.sort_values('date')
