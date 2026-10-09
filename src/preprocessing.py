from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = [
    'temperature_celsius','precip_mm','humidity','wind_kph','pressure_mb','cloud',
    'feels_like_celsius','visibility_km','uv_index','gust_kph',
    'air_quality_Carbon_Monoxide','air_quality_Ozone','air_quality_Nitrogen_dioxide',
    'air_quality_Sulphur_dioxide','air_quality_PM2.5','air_quality_PM10',
    'latitude','longitude'
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
    df = df.dropna(subset=['last_updated','country','location_name','temperature_celsius']).copy()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove exact duplicate observations and impossible physical values.
    df = df.drop_duplicates(subset=['country','location_name','last_updated'])
    df = df[df['humidity'].between(0, 100, inclusive='both')]
    df = df[df['temperature_celsius'].between(-90, 65, inclusive='both')]
    df = df[df['pressure_mb'].between(850, 1100, inclusive='both')]
    df = df[df['precip_mm'].ge(0)]
    df = df[df['wind_kph'].ge(0)]

    # Calendar features used for EDA and pooled comparisons.
    df['date'] = df['last_updated'].dt.floor('D')
    df['year'] = df['last_updated'].dt.year
    df['month'] = df['last_updated'].dt.month
    df['dayofyear'] = df['last_updated'].dt.dayofyear
    return df.sort_values('last_updated').reset_index(drop=True)


def location_daily(df: pd.DataFrame, country: str, location: str) -> pd.DataFrame:
    sub = df[(df['country'] == country) & (df['location_name'] == location)].copy()
    if sub.empty:
        return sub

    numeric = [c for c in NUMERIC_COLS if c in sub.columns]
    # One record per day is typical, but aggregate protects against duplicate-day snapshots.
    daily = sub.groupby('date', as_index=False)[numeric].mean(numeric_only=True)
    daily = daily.sort_values('date').set_index('date')

    # Reindex daily and use time-aware interpolation only for small gaps.
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(full_idx)
    daily.index.name = 'date'
    daily[numeric] = daily[numeric].interpolate(method='time', limit=3, limit_direction='both')
    daily = daily.dropna(subset=['temperature_celsius'])
    return daily.reset_index()


def latest_location_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(['country','location_name'])['last_updated'].idxmax()
    cols = ['country','location_name','latitude','longitude','temperature_celsius','precip_mm',
            'humidity','wind_kph','air_quality_PM2.5','condition_text','last_updated']
    return df.loc[idx, [c for c in cols if c in df.columns]].reset_index(drop=True)
