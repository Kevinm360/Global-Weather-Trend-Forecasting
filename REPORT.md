# Advanced Weather Trend Forecasting Report

## 1. Objective
The goal of this project is to analyze daily weather observations from cities around the world, compare multiple forecasting approaches, detect unusual observations, and study geographic and air-quality patterns.

## 2. Dataset
The Global Weather Repository contains weather, location, astronomy, and air-quality variables. The supplied snapshot contains observations from May 2024 through October 2026 across more than 200 countries.

## 3. Data preparation
Timestamps are parsed and observations are ordered chronologically. Exact duplicate location/timestamp records are removed. The pipeline checks physical ranges for humidity, temperature, pressure, precipitation, and wind. For city-level forecasting, observations are aggregated to daily frequency and short gaps are interpolated for at most three consecutive days.

Forecasting features are derived only from information available before the target date: temperature lags, rolling statistics shifted by one day, and cyclical day-of-year variables.

## 4. Exploratory analysis
The dashboard provides temperature history, 30-day moving averages, monthly precipitation, and correlation views for weather and air-quality variables. These views make seasonal changes and local relationships easy to inspect without treating correlation as causation.

## 5. Forecasting models
The project compares four individual approaches plus an ensemble:

1. Seasonal Naive — uses the observed temperature from seven days earlier.
2. Ridge Regression — a regularized linear model using lag, rolling, and calendar features.
3. Random Forest — a nonlinear tree ensemble.
4. Gradient Boosting — a sequential tree ensemble using Huber loss.
5. Ensemble — equal-weight average of Ridge, Random Forest, and Gradient Boosting predictions.

The final 20% of each location's feature-complete observations form the test period. This chronological split is deliberately used instead of random splitting to avoid future-data leakage.

## 6. Advanced EDA: anomaly detection
Isolation Forest is applied to standardized temperature, precipitation, humidity, wind, pressure, and PM2.5 observations. The model flags approximately 3% of observations as unusual. A flagged point can represent genuine extreme weather, unusual pollution, or a data-quality issue, so the dashboard treats anomalies as candidates for investigation rather than automatic errors.

## 7. Environmental analysis
PM2.5 is compared with temperature, humidity, wind, pressure, and precipitation using correlation analysis. The results help identify local associations but are not interpreted as causal environmental effects.

## 8. Feature importance
The dashboard uses two methods to assess which weather variables are most useful for explaining temperature:

- Random Forest impurity importance
- Permutation importance measured on a chronological holdout set

Agreement between the two methods increases confidence that an important feature is not merely an artifact of one importance calculation.

## 9. Spatial and geographic analysis
The latest available observation for each location is displayed on an interactive world map. Country-level and continent-level summaries compare observed temperature, precipitation, and PM2.5.

## 10. Limitation on climate interpretation
The dataset covers roughly 2.4 years. That is suitable for short-horizon forecasting and multi-year weather comparisons, but it is not long enough to support strong claims about long-term climate change. The project therefore uses the phrase **weather trends** instead of treating short-term trends as climate trends.

## 11. PM Accelerator mission
PM Accelerator's mission is to make industry-leading tools and education available to individuals from all backgrounds, helping level the playing field for future PM leaders. It focuses on giving aspiring and experienced PMs access to industry leaders, the right PM ecosystem, and AI product management skills.

## 12. Conclusion
The project demonstrates an end-to-end data-science workflow: cleaning, exploratory analysis, leakage-aware time-series validation, multiple forecasting models, ensemble modeling, anomaly detection, explainability, environmental analysis, and geographic visualization. The Streamlit dashboard makes the workflow reproducible and interactive for different locations.

## 13. Representative benchmark results
To verify the pipeline before submission, the models were evaluated on five locations with near-complete histories. The best holdout model differed by location, which is one reason the dashboard compares multiple models instead of assuming one method always wins.

| Location | Best model | MAE (°C) | RMSE (°C) | MAPE |
|---|---|---:|---:|---:|
| London, United Kingdom | Ensemble | 1.95 | 2.79 | 20.8% |
| Tokyo, Japan | Ridge | 2.64 | 3.44 | 10.9% |
| Cairo, Egypt | Ridge | 1.70 | 2.45 | 6.9% |
| Paris, France | Ensemble | 2.11 | 2.89 | 25.4% |
| Singapore, Singapore | Ridge | 1.26 | 1.80 | 4.2% |

MAPE is relatively high for some cooler locations because percentage error becomes unstable when actual temperatures approach zero. MAE and RMSE are therefore the primary comparison metrics.
