# 1–2 Minute Demo Script

Hi, this is my advanced weather trend forecasting assessment using the Global Weather Repository dataset.

I built the project as a Streamlit dashboard so the analysis can be explored by country and city. The data pipeline cleans duplicate or physically invalid observations, converts each location to a daily time series, and only interpolates short gaps.

In the forecasting section I compare a seven-day seasonal baseline with Ridge Regression, Random Forest, Gradient Boosting, and an ensemble. I use a chronological holdout instead of a random train-test split so future observations never leak into training. The dashboard reports MAE, RMSE, and MAPE and also creates a multi-day future forecast.

For the advanced analysis, I use Isolation Forest to flag unusual weather observations. I also compare two feature-importance methods, analyze how PM2.5 relates to weather variables, and map geographic patterns across countries and continents.

One limitation I kept in mind is that this dataset only covers about two and a half years. Because of that, I describe the results as multi-year weather patterns rather than making long-term climate-change claims.

The full methodology, setup steps, and model details are included in the README.
