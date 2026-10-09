from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from src.preprocessing import load_and_clean, location_daily, latest_location_snapshot
from src.forecasting import evaluate_models, future_forecast
from src.anomaly_detection import detect_anomalies
from src.feature_importance import importance_tables
from src.geo import add_continent

st.set_page_config(page_title='Global Weather Forecasting', page_icon='🌦️', layout='wide')

DATA_PATH = Path(__file__).parent / 'data' / 'GlobalWeatherRepository.csv'

@st.cache_data(show_spinner=False)
def get_data():
    return load_and_clean(str(DATA_PATH))

@st.cache_data(show_spinner=False)
def get_daily(df, country, location):
    return location_daily(df, country, location)

@st.cache_data(show_spinner=False)
def get_eval(daily):
    return evaluate_models(daily)

@st.cache_data(show_spinner=False)
def get_forecast(daily, horizon):
    return future_forecast(daily, horizon=horizon)

@st.cache_data(show_spinner=False)
def get_anomalies(daily):
    return detect_anomalies(daily)

st.title('Global Weather Trend Forecasting')
st.caption('Advanced PM Accelerator Data Science Assessment')

with st.expander('PM Accelerator mission', expanded=False):
    st.write(
        "PM Accelerator's mission is to make industry-leading tools and education available to individuals from all backgrounds, helping level the playing field for future PM leaders. It focuses on giving aspiring and experienced PMs access to industry leaders, the right PM ecosystem, and AI product management skills."
    )

try:
    df = get_data()
except FileNotFoundError:
    st.error('Dataset not found. Place GlobalWeatherRepository.csv inside the data/ folder.')
    st.stop()

pairs = (
    df.groupby(['country','location_name']).size().reset_index(name='n')
    .query('n >= 120').sort_values(['country','location_name'])
)

with st.sidebar:
    st.header('Filters')
    countries = pairs['country'].drop_duplicates().tolist()
    default_country = 'United Kingdom' if 'United Kingdom' in countries else countries[0]
    country = st.selectbox('Country', countries, index=countries.index(default_country))
    locations = pairs.loc[pairs['country'].eq(country), 'location_name'].tolist()
    location = st.selectbox('Location', locations)
    horizon = st.slider('Forecast horizon (days)', 3, 14, 7)
    st.caption('Only locations with at least 120 observations are shown for model stability.')

# Selected city's daily history
daily = get_daily(df, country, location)

latest = daily.iloc[-1]
first_date = daily['date'].min().date()
last_date = daily['date'].max().date()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Latest temperature', f"{latest['temperature_celsius']:.1f} °C")
c2.metric('Humidity', f"{latest['humidity']:.0f}%")
c3.metric('Precipitation', f"{latest['precip_mm']:.1f} mm")
c4.metric('Daily observations', f"{len(daily):,}")
st.caption(f'{location}, {country} • {first_date} to {last_date}')

overview, trends, forecast_tab, advanced, geography = st.tabs(
    ['Overview', 'Weather Trends', 'Forecasting', 'Advanced Analysis', 'Geography']
)

with overview:
    left, right = st.columns([1.55, 1])
    with left:
        fig = px.line(daily, x='date', y='temperature_celsius', title='Daily temperature history')
        fig.update_layout(yaxis_title='Temperature (°C)', xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        monthly = daily.assign(month=daily['date'].dt.month).groupby('month', as_index=False).agg(
            temperature_celsius=('temperature_celsius','mean'),
            precip_mm=('precip_mm','mean')
        )
        fig = px.bar(monthly, x='month', y='precip_mm', title='Average precipitation by month')
        fig.update_layout(yaxis_title='Precipitation (mm)', xaxis_title='Month')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader('Data quality')
    raw_sub = df[(df['country'] == country) & (df['location_name'] == location)]
    qa1, qa2, qa3 = st.columns(3)
    qa1.metric('Raw rows', f'{len(raw_sub):,}')
    qa2.metric('Missing values', f'{int(raw_sub.isna().sum().sum()):,}')
    qa3.metric('Date coverage', f'{(raw_sub.last_updated.max()-raw_sub.last_updated.min()).days:,} days')
    st.caption('Cleaning removes exact duplicate timestamps and physically implausible values. Short daily gaps are time-interpolated only within a three-day limit.')

with trends:
    st.subheader('Trend and seasonality')
    smooth = daily[['date','temperature_celsius']].copy()
    smooth['30-day mean'] = smooth['temperature_celsius'].rolling(30, min_periods=10).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smooth['date'], y=smooth['temperature_celsius'], mode='lines', name='Daily', opacity=.45))
    fig.add_trace(go.Scatter(x=smooth['date'], y=smooth['30-day mean'], mode='lines', name='30-day mean'))
    fig.update_layout(title='Temperature with 30-day moving average', yaxis_title='Temperature (°C)', xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    numeric_for_corr = [c for c in ['temperature_celsius','humidity','wind_kph','pressure_mb','precip_mm','cloud','visibility_km','uv_index','air_quality_PM2.5','air_quality_PM10','air_quality_Ozone'] if c in daily.columns]
    corr = daily[numeric_for_corr].corr(numeric_only=True)
    fig2, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap='vlag', center=0, ax=ax)
    ax.set_title('Weather and air-quality correlation matrix')
    st.pyplot(fig2, clear_figure=True)

with forecast_tab:
    st.subheader('Time-series model comparison')
    st.caption('The holdout set is the most recent 20% of observations. No random split is used, which prevents future-to-past leakage.')
    try:
        with st.spinner('Training forecasting models...'):
            metrics, pred_df, fitted, feature_names = get_eval(daily)
            future = get_forecast(daily, horizon)

        st.dataframe(metrics.style.format({'MAE':'{:.2f}','RMSE':'{:.2f}','MAPE':'{:.1f}%'}), use_container_width=True, hide_index=True)

        best = metrics.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric('Best holdout model', best['Model'])
        m2.metric('Best RMSE', f"{best['RMSE']:.2f} °C")
        m3.metric('Best MAE', f"{best['MAE']:.2f} °C")

        display_models = ['Actual','Seasonal Naive (7d)','Ridge','Random Forest','Gradient Boosting','Ensemble']
        long_pred = pred_df.melt(id_vars='date', value_vars=display_models, var_name='Series', value_name='Temperature')
        fig = px.line(long_pred, x='date', y='Temperature', color='Series', title='Holdout-period forecasts vs. actual temperature')
        fig.update_layout(yaxis_title='Temperature (°C)', xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        future_long = future.melt(id_vars='date', var_name='Model', value_name='Temperature')
        fig = px.line(future_long, x='date', y='Temperature', color='Model', markers=True, title=f'Next {horizon} days: model forecasts')
        fig.update_layout(yaxis_title='Temperature (°C)', xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button('Download forecast CSV', future.to_csv(index=False).encode('utf-8'), file_name=f'{location}_forecast.csv', mime='text/csv')
    except ValueError as e:
        st.warning(str(e))

with advanced:
    st.subheader('Anomaly detection')
    anomalies = get_anomalies(daily)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['temperature_celsius'], mode='lines', name='Temperature'))
    flagged = anomalies[anomalies['anomaly']]
    fig.add_trace(go.Scatter(x=flagged['date'], y=flagged['temperature_celsius'], mode='markers', name='Isolation Forest anomaly', marker=dict(size=9, symbol='x')))
    fig.update_layout(title='Detected unusual weather observations', yaxis_title='Temperature (°C)', xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'{len(flagged)} observations were flagged using multivariate Isolation Forest detection at ~3% contamination.')

    st.subheader('Feature importance — two methods')
    intrinsic, permutation = importance_tables(daily)
    if not intrinsic.empty:
        a, b = st.columns(2)
        with a:
            fig = px.bar(intrinsic.head(10).sort_values('Importance'), x='Importance', y='Feature', orientation='h', title='Random Forest impurity importance')
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.bar(permutation.head(10).sort_values('Importance'), x='Importance', y='Feature', orientation='h', title='Permutation importance')
            st.plotly_chart(fig, use_container_width=True)

    st.subheader('Environmental / air-quality analysis')
    aq_cols = [c for c in ['air_quality_PM2.5','air_quality_PM10','air_quality_Ozone','air_quality_Nitrogen_dioxide','temperature_celsius','humidity','wind_kph','pressure_mb','precip_mm'] if c in daily.columns]
    aq_corr = daily[aq_cols].corr(numeric_only=True)
    target = 'air_quality_PM2.5'
    if target in aq_corr:
        rel = aq_corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False).reset_index()
        rel.columns = ['Feature','Correlation']
        fig = px.bar(rel, x='Correlation', y='Feature', orientation='h', title='Correlation with PM2.5')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('These correlations describe association, not causation or environmental impact by themselves.')

with geography:
    st.subheader('Spatial weather patterns')
    snapshot = latest_location_snapshot(df)
    fig = px.scatter_geo(
        snapshot,
        lat='latitude', lon='longitude', color='temperature_celsius', size='humidity',
        hover_name='location_name', hover_data=['country','temperature_celsius','humidity','precip_mm'],
        projection='natural earth', title='Latest available weather by location'
    )
    st.plotly_chart(fig, use_container_width=True)

    geo_df = add_continent(df)
    valid_geo = geo_df[geo_df['continent'] != 'Other'].copy()
    continent = valid_geo.groupby('continent', as_index=False).agg(
        avg_temp=('temperature_celsius','mean'),
        avg_precip=('precip_mm','mean'),
        avg_pm25=('air_quality_PM2.5','mean')
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(continent.sort_values('avg_temp'), x='continent', y='avg_temp', title='Average observed temperature by continent')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(continent.sort_values('avg_pm25'), x='continent', y='avg_pm25', title='Average observed PM2.5 by continent')
        st.plotly_chart(fig, use_container_width=True)

    country_stats = df.groupby('country', as_index=False).agg(
        avg_temp=('temperature_celsius','mean'),
        avg_precip=('precip_mm','mean'),
        avg_pm25=('air_quality_PM2.5','mean'),
        observations=('temperature_celsius','size')
    ).query('observations >= 100')
    st.dataframe(country_stats.sort_values('avg_temp', ascending=False).head(20), use_container_width=True, hide_index=True)
    st.caption('The dataset spans roughly 2.4 years, so these are multi-year weather patterns—not evidence of long-term climate change.')

st.divider()
st.caption('Built with Streamlit, pandas, scikit-learn, Plotly, Matplotlib and Seaborn. Forecasts are analytical estimates, not operational weather-service predictions.')
