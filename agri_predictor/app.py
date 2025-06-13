import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from reportlab.pdfgen import canvas # This import is present but not used in the provided app logic
from reportlab.lib.pagesizes import A4 # This import is present but not used in the provided app logic
import os
import requests # This import is present but not used in the provided app logic
from datetime import timedelta

import google.generativeai as genai
from dotenv import load_dotenv # ✅ NEW: Import load_dotenv

# ----------------------------- CONFIG -----------------------------
st.set_page_config(page_title="Agri Commodity Forecasting", layout="wide")

# ✅ NEW: Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------- CUSTOM STYLING -----------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e0f7fa, #ffffff);
        }
        .main {
            background-color: transparent;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        h1, h2, h3, .stMetric {
            color: #004d61;
        }
        .glass-box {
            background: rgba(255, 255, 255, 0.6);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            margin-bottom: 20px;
        }
        .metric-box {
            padding: 15px;
            border-left: 6px solid #004d61;
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 10px;
        }
        .stButton button {
            background-color: #004d61;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stButton button:hover {
            background-color: #00707f;
        }
        .nav-radio .stRadio > div {
            flex-direction: row;
            gap: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- LOAD DATA -----------------------------
@st.cache_data
def load_data():
    # Make sure 'maharashtra_market_daily_complete.csv' is in your GitHub repo
    # in the correct path relative to this app.py file.
    df = pd.read_csv("/mount/src/bazaarbataye/agri_predictor/maharashtra_market_daily_complete.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    return df

data = load_data()
commodities = data['Commodity'].unique()
locations = data['District'].unique()

# ----------------------------- AI PROMPT FUNCTION -----------------------------
def generate_ai_reasoning(latest_row, forecast_end_price, lower_threshold, upper_threshold):
    trend = "high" if forecast_end_price > upper_threshold else "low" if forecast_end_price < lower_threshold else "neutral"
    prompt = f"""
You are an agriculture analyst. Based on the following data, explain why the forecasted commodity price is {trend}:

- Commodity: {latest_row['Commodity']}
- Location: {latest_row['District']}
- Rainfall (mm): {latest_row.get('Rainfall_mm', 'N/A')}
- Soil Quality: {latest_row['Soil_Quality']}
- Latest Price: ₹{latest_row['Price_per_kg']}
- Forecasted Price: ₹{forecast_end_price}
- Thresholds: Lower ₹{lower_threshold}, Upper ₹{upper_threshold}

Your explanation should consider the possible impact of  soil quality, location trends, and economic demand-supply factors.
"""
    return prompt

# ✅ NEW: Use the loaded API key
genai.configure(api_key=GEMINI_API_KEY)
def call_llama2(prompt):
    if not GEMINI_API_KEY:
        return "⚠ Gemini API key not found. Please set it in your environment variables."
    try:
        model = genai.GenerativeModel("gemini-2.0-flash") # Changed to a more recent model like 1.5-flash
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠ Error communicating with Gemini: {e}"

# ----------------------------- NAVIGATION -----------------------------
st.markdown("<h1 style='text-align:center;'>🌾 Agri Commodity Intelligence Platform</h1>", unsafe_allow_html=True)
page = st.radio("Navigate", ["📈 Overview", "📊 Dashboard"], horizontal=True, key="navigation", label_visibility="collapsed")

# ----------------------------- 📈 OVERVIEW -----------------------------
if page == "📈 Overview":
    st.sidebar.header("⚙ Forecast Settings")
    commodity = st.sidebar.selectbox("Select Commodity", commodities)
    location = st.sidebar.selectbox("Select Location", locations)

    filtered_varieties = data[data["Commodity"] == commodity]["Variety"].dropna().unique()
    variety = st.sidebar.selectbox("Select Variety", filtered_varieties)

    investment = st.sidebar.number_input("Investment (₹)", min_value=0.0, step=1000.0, value=5000.0)
    quintal_yield = st.sidebar.number_input("Quintal Yield", min_value=0.0, step=1.0, value=10.0)
    # ✅ Quality Input
    quality = st.sidebar.selectbox("Crop Quality", ["Average", "Good", "Bad"], index=0)
    forecast_days = st.sidebar.slider("Forecast Days", 15, 120, 30)

    today = pd.Timestamp.today()
    one_year_ago = today - pd.DateOffset(years=1)

    df = data[(data['Commodity'] == commodity) &
              (data['District'] == location) &
              (data['Variety'] == variety) &
              (data['Date'] >= one_year_ago)].sort_values('Date')

    if df.empty:
        st.warning("No data available for the selected commodity, location, and variety.")
        st.stop()

    df_prophet = df[['Date', 'Price_per_kg']].rename(columns={'Date': 'ds', 'Price_per_kg': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    st.title(f"📈 {commodity} Price Forecast Overview ({location})")
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Price", f"₹{df_prophet['y'].iloc[-1]:.2f}")

    # Calculate forecasted price
    forecast_end_price = forecast['yhat'].iloc[-1]

    # ✅ Apply Quality Logic
    import random
    if quality == "Good":
        forecast_end_price += random.choice([2.444889, 3.222])
    elif quality == "Bad":
        forecast_end_price -= random.choice([1.78, 2.34])

    col2.metric("Forecast End Price", f"₹{forecast_end_price:.2f}")

    # Profit Calculation
    profit = (quintal_yield * 100 * forecast_end_price) - investment
    col3.metric("💰 Projected Profit", f"₹{profit:,.2f}")

    st.subheader("📅 Monthly Average Price")
    monthly_df = df_prophet.copy()
    monthly_df['month'] = monthly_df['ds'].dt.to_period("M").dt.to_timestamp()
    monthly_avg = monthly_df.groupby('month')['y'].mean().reset_index()
    fig_month = go.Figure()
    fig_month.add_trace(go.Scatter(x=monthly_avg['month'], y=monthly_avg['y'], mode='lines+markers', name='Monthly Avg'))
    fig_month.update_layout(title=f"Monthly Average Price of {commodity} ({location})", template="plotly_white")
    st.plotly_chart(fig_month, use_container_width=True)

    latest_price = df_prophet['y'].iloc[-1]
    lower_threshold = df_prophet['y'].quantile(0.25)
    upper_threshold = df_prophet['y'].quantile(0.75)

    st.subheader("📊 Forecast with Thresholds")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines+markers', name='Actual Price', line=dict(color='blue', width=2)))
    forecast_future = forecast[forecast['ds'] > df_prophet['ds'].max()]
    fig1.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat'], mode='lines', name='Forecast Price', line=dict(color='orange', width=2, dash='dot')))
    fig1.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='red', dash='dash'), opacity=0.4))
    fig1.add_trace(go.Scatter(x=forecast_future['ds'], y=forecast_future['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='green', dash='dash'), opacity=0.4))
    fig1.add_hline(y=lower_threshold, line_dash="dot", line_color="green", annotation_text=f"Lower Threshold (₹{lower_threshold:.2f})", annotation_position="top left")
    fig1.add_hline(y=upper_threshold, line_dash="dot", line_color="red", annotation_text=f"Upper Threshold (₹{upper_threshold:.2f})", annotation_position="top left")
    fig1.update_layout(title=f"{commodity} Price Forecast ({location})", xaxis_title="Date", yaxis_title="Price (₹/kg)", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    latest_row = df.iloc[-1]
    prompt = generate_ai_reasoning(latest_row, forecast_end_price, lower_threshold, upper_threshold)
    ai_reasoning = call_llama2(prompt)

    if forecast_end_price > latest_price:
        recommendation = "🟢 Hold the crop — price is expected to rise"
    elif forecast_end_price < latest_price:
        recommendation = "🔴 Sell now — price may decrease"
    else:
        recommendation = "🟠 Price is stable — decide based on your preference"
    
    # Define color for the recommendation box based on recommendation
    if "🟢" in recommendation:
        color = "green"
    elif "🔴" in recommendation:
        color = "red"
    else:
        color = "orange"

    st.markdown(f"""
    <div class="glass-box" style="border-left: 6px solid {color};">
        <h3>💡 AI-Driven Actionable Advice</h3>
        <strong>Latest Price:</strong> ₹{latest_price:.2f} <br>
        <strong>Forecasted End Price:</strong> ₹{forecast_end_price:.2f} <br>
        <strong>Thresholds:</strong> ₹{lower_threshold:.2f} (Lower), ₹{upper_threshold:.2f} (Upper)<br>
        <strong>Correct Action:</strong> {recommendation} <br>
        <strong>Reason:</strong> {ai_reasoning}<br>
        <strong>Estimated Profit:</strong> ₹{profit:,.2f}
    </div>
    """, unsafe_allow_html=True)

# ----------------------------- 📊 POWER BI STYLE DASHBOARD -----------------------------
elif page == "📊 Dashboard":
    st.sidebar.header("📂 Dashboard Filters")
    selected_commodity = st.sidebar.selectbox("Select Commodity", sorted(commodities))
    selected_location = st.sidebar.selectbox("Select Location", sorted(locations))
    date_range = st.sidebar.date_input("Select Date Range", [data['Date'].min(), data['Date'].max()])

    df_filtered = data[(data['Commodity'] == selected_commodity) &
                       (data['District'] == selected_location) &
                       (data['Date'] >= pd.to_datetime(date_range[0])) &
                       (data['Date'] <= pd.to_datetime(date_range[1]))]

    st.subheader(f"📊 Insights for {selected_commodity} ({selected_location})")

    min_price = df_filtered['Price_per_kg'].min()
    max_price = df_filtered['Price_per_kg'].max()
    mean_price = df_filtered['Price_per_kg'].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("📉 Min Price", f"₹{min_price:.2f}")
    k2.metric("📈 Max Price", f"₹{max_price:.2f}")
    k3.metric("📊 Avg Price", f"₹{mean_price:.2f}")

    monthly_df = df_filtered.copy()
    monthly_df['month'] = monthly_df['Date'].dt.to_period("M").dt.to_timestamp()
    monthly_avg = monthly_df.groupby('month')['Price_per_kg'].mean().reset_index()

    fig_trend = px.line(monthly_avg, x='month', y='Price_per_kg', markers=True, title="📅 Monthly Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    fig_box = px.box(df_filtered, y='Price_per_kg', title="📦 Price Distribution")
    st.plotly_chart(fig_box, use_container_width=True)

    latest_df = data[data['Date'] == data['Date'].max()]
    fig_bar = px.bar(latest_df.groupby(['Commodity', 'District']))
