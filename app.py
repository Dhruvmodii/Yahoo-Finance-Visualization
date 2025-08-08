import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------- App Config ----------
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📊 Stock Price Prediction App")
st.markdown("Enter a stock ticker below to predict the next 30 days of closing prices using a pre-trained model.")

# ---------- User Input ----------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
start_date = st.date_input("Start Date", datetime(2010, 1, 1))
end_date = st.date_input("End Date", datetime.today())

if st.button("🔍 visualize"):
    # ---------- Download Data ----------
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("❌ No data found for this ticker.")
    else:
        st.success(f"✅ Fetched {len(df)} records for {ticker}")

        df.reset_index(inplace=True)
        df = df[["Date", "Close", "Volume"]]  # include volume now

        # ---------- Moving Averages ----------
        df["MA365"] = df["Close"].rolling(window=365).mean()
        df["MA730"] = df["Close"].rolling(window=730).mean()
        df["MA1095"] = df["Close"].rolling(window=1095).mean()

        # ---------- Chart 1: Close + MA ----------
        st.subheader("📊 Close + Moving Averages")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["Date"], df["Close"], label="Close", color='blue')
        ax1.plot(df["Date"], df["MA365"], label="365 MA", color='green')
        ax1.plot(df["Date"], df["MA730"], label="730 MA", color='orange')
        ax1.plot(df["Date"], df["MA1095"], label="1095 MA", color='red')
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)

        # ---------- Chart 2: Daily % Return ----------
        st.subheader("📈 Daily Price Change (% Returns) Chart")
        df["Daily Return %"] = df["Close"].pct_change() * 100
        fig_ret, ax_ret = plt.subplots(figsize=(12, 4))
        ax_ret.plot(df["Date"], df["Daily Return %"], color='purple', label="Daily Return %")
        ax_ret.axhline(0, linestyle='--', color='gray', alpha=0.6)
        ax_ret.set_ylabel("Daily Change (%)")
        ax_ret.set_xlabel("Date")
        ax_ret.set_title(f"{ticker} Daily Price % Change")
        ax_ret.legend()
        st.pyplot(fig_ret)


        # 📉 Volatility Chart (Rolling Std Dev)
        df["Volatility_30"] = df["Close"].rolling(window=30).std()

        st.subheader("📉 30-Day Rolling Volatility")
        fig_volatility, ax_vol = plt.subplots(figsize=(12, 6))
        ax_vol.plot(df.index, df["Volatility_30"], color='purple', label='30-Day Std Dev')
        ax_vol.set_ylabel("Volatility")
        ax_vol.set_title("Volatility Over Time")
        ax_vol.legend()
        st.pyplot(fig_volatility)

        # ---------- Scaling ----------
        close_data = df[["Close"]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        
        
        














