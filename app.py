import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
import io
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Title and description
st.set_page_config(page_title="Weather Forecast", layout="centered")
st.markdown("<h1 style='text-align: center;'>🌤️ Weather Forecast with SARIMAX</h1>", unsafe_allow_html=True)

# User inputs
city = st.text_input("Enter the name of the city (optional)")
temp_unit = st.selectbox("Select temperature unit", ["Celsius", "Fahrenheit"])
graph_type = st.radio("Select graph type", ["Line Graph", "Bar Graph"])
uploaded_file = st.file_uploader("Upload your weather_data_test.csv", type=["csv"])

if uploaded_file:
    # Load and preprocess
    weather_data = pd.read_csv(uploaded_file, parse_dates=['datetime'], sep=';', decimal=',', infer_datetime_format=True)
    df = weather_data[["datetime", "T_mu"]].dropna()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # Date range selector for training
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    st.sidebar.markdown("### 🛠️ Training Range Filter")
    train_range = st.sidebar.date_input("Select training date range", [min_date, max_date])

    # Forecast horizon selector
    forecast_days = st.sidebar.slider("🔮 Forecast days into the future", 1, 30, 7)

    # Filter training range
    if len(train_range) == 2:
        df = df.loc[train_range[0]:train_range[1]]

    st.subheader("📊 Raw Temperature Data")
    st.line_chart(df["T_mu"])

    # Decomposition
    st.subheader("📉 Seasonal Decomposition")
    result = seasonal_decompose(df["T_mu"], model='additive', period=365)
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(result.observed); axs[0].set_ylabel("Observed")
    axs[1].plot(result.trend); axs[1].set_ylabel("Trend")
    axs[2].plot(result.seasonal); axs[2].set_ylabel("Seasonal")
    axs[3].plot(result.resid); axs[3].set_ylabel("Residual")
    st.pyplot(fig)

    # Model training
    st.subheader("🔁 SARIMAX Forecasting")
    model = SARIMAX(df["T_mu"], order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)

    # Forecast future values
    future_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast = results.get_forecast(steps=forecast_days)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_mean.index = future_index
    forecast_ci.index = future_index

    # Plot forecast
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["T_mu"], label='Historical')
    if graph_type == "Line Graph":
        ax.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
    else:
        ax.bar(forecast_mean.index, forecast_mean, label='Forecast', color='red', width=0.8)
    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    ax.legend()
    st.pyplot(fig2)

    # Show forecast table
    st.subheader("📋 Forecasted Values")
    forecast_df = pd.DataFrame({
        "Date": forecast_mean.index,
        "Forecast_T_mu": forecast_mean.values,
        "Lower Bound": forecast_ci.iloc[:, 0].values,
        "Upper Bound": forecast_ci.iloc[:, 1].values
    })
    st.dataframe(forecast_df.set_index("Date"))

    # Download button
    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"forecast_{city or 'weather'}.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a valid `weather_data_test.csv` file to proceed.")
