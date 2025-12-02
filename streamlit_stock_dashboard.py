"""
Real-Time Stock Market Dashboard with Watchlist & Price Alerts

Features:
- Enter one or more tickers (comma-separated)
- Watchlist persistence
- Set price alerts for each ticker
- Choose period and interval
- Live refresh (auto) or manual refresh
- Candlestick chart with SMA/EMA overlays
- RSI and MACD indicators
- Volume bar chart
- Auto-refresh and sound alert on price target
- Historical comparison of multiple tickers
- CSV export of latest data
- Compact summary cards for better UI
- Automatic interval/period adjustment for intraday data

Requirements (pip):
streamlit
pandas
numpy
yfinance
plotly

Run:
streamlit run streamlit_stock_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import time
import sys
if sys.platform == "win32":
    import winsound

st.set_page_config(layout="wide", page_title="Real-Time Stock Dashboard")

WATCHLIST_FILE = 'watchlist.json'

# -----------------
# Helper functions
# -----------------
@st.cache_data(ttl=60)
def load_data(ticker, period, interval):
    try:
        # Auto-adjust period for minute intervals (1m/2m)
        if interval in ["1m", "2m"] and period not in ["1d","5d","7d"]:
            period = "7d"
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
        if df.empty:
            return None
        df = df.rename_axis('Datetime').reset_index()
        return df
    except Exception as e:
        st.error(f"Data load error for {ticker}: {e}")
        return None


def sma(series, window):
    return series.rolling(window=window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = ema(series, span_short)
    ema_long = ema(series, span_long)
    macd = ema_short - ema_long
    signal = ema(macd, span_signal)
    hist = macd - signal
    return macd, signal, hist

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    return []

def save_watchlist(tickers):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(tickers, f)

# -----------------
# UI - Sidebar
# -----------------
st.sidebar.title("Settings")

saved_watchlist = load_watchlist()
watchlist_input = st.sidebar.text_input("Tickers (comma-separated)", value=','.join(saved_watchlist) if saved_watchlist else 'AAPL,MSFT,GOOG')
period = st.sidebar.selectbox("Period", options=["1d","5d","1mo","3mo","6mo","1y","2y","5y"], index=0)
interval = st.sidebar.selectbox("Interval", options=["1m","2m","5m","15m","30m","60m","90m","1h","1d","1wk"], index=0)

refresh_mode = st.sidebar.radio("Refresh mode", ("Manual","Auto"))
auto_refresh_seconds = st.sidebar.number_input("Auto refresh every (seconds)", min_value=5, max_value=3600, value=30)

show_sma = st.sidebar.checkbox("Show SMA (20)", value=True)
show_ema = st.sidebar.checkbox("Show EMA (50)", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Save Watchlist"):
    tickers_to_save = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    save_watchlist(tickers_to_save)
    st.success("Watchlist saved!")
    st.experimental_rerun()

if st.sidebar.button("Refresh now"):
    st.experimental_rerun()

# -----------------
# Main layout
# -----------------
st.title("📈 Real-Time Stock Market Dashboard")

col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Charts")
with col2:
    st.subheader("Ticker Summary & Alerts")

tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

ticker_data_dict = {}

# -----------------
# Fetch & display each ticker
# -----------------
for ticker in tickers:
    st.markdown(f"---\n### {ticker}")
    df = load_data(ticker, period, interval)
    if df is None or df.empty:
        st.error(f"No data for {ticker}. Try a different interval/period or check ticker symbol.")
        continue

    ticker_data_dict[ticker] = df.copy()

    # Indicators
    df['SMA20'] = sma(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['Close'])

    # Price alert
    st.sidebar.markdown(f"---\nPrice Alert for {ticker}")
    alert_price = st.sidebar.number_input(f"Target Price for {ticker}", value=float(df['Close'].iloc[-1]))

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev_close = float(prev['Close']) if isinstance(prev['Close'], (pd.Series, np.ndarray)) else prev['Close']
    pct_change = (float(latest['Close']) - prev_close) / prev_close * 100 if prev_close != 0 else 0.0
    alert_triggered = float(latest['Close']) >= alert_price

    left, right = st.columns([3,1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        if show_sma:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['SMA20'], mode='lines', name='SMA20'))
        if show_ema:
            fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA50'], mode='lines', name='EMA50'))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig, use_container_width=True)

        ind_cols = st.columns(2)
        with ind_cols[0]:
            if show_rsi:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df['Datetime'], y=df['RSI14'], name='RSI14'))
                fig_rsi.update_layout(height=250, yaxis=dict(range=[0,100]))
                st.plotly_chart(fig_rsi, use_container_width=True)
        with ind_cols[1]:
            if show_macd:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Bar(x=df['Datetime'], y=df['MACD_hist'], name='MACD Hist'))
                fig_macd.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD'], name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df['Datetime'], y=df['MACD_signal'], name='Signal'))
                fig_macd.update_layout(height=250)
                st.plotly_chart(fig_macd, use_container_width=True)

    with right:
        latest_open = float(latest['Open'])
        latest_high = float(latest['High'])
        latest_low = float(latest['Low'])
        latest_volume = int(latest['Volume'])
        latest_close_val = float(latest['Close'])

        st.metric(label="Last Close", value=f"{latest_close_val:.2f}", delta=f"{pct_change:.2f}%")
        st.write(f"Open: {latest_open:.2f}")
        st.write(f"High: {latest_high:.2f}")
        st.write(f"Low: {latest_low:.2f}")
        st.write(f"Volume: {latest_volume}")

        if alert_triggered:
            st.success(f"🚨 {ticker} has reached your target price of {alert_price}")
            if sys.platform == "win32":
                winsound.Beep(1000, 500)
        else:
            st.info(f"Target price: {alert_price}")

        show_table = st.checkbox(f"Show last rows for {ticker}", key=f"table_{ticker}")
        if show_table:
            st.dataframe(df.tail(10).set_index('Datetime'))

# -----------------
# Historical comparison & CSV export
# -----------------
if len(ticker_data_dict) > 1:
    st.markdown("---")
    st.subheader("📊 Historical Comparison")
    fig_comp = go.Figure()
    for t, data in ticker_data_dict.items():
        fig_comp.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name=t))
    fig_comp.update_layout(height=400, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_comp, use_container_width=True)

    if st.button("Download CSV of latest data"):
        combined_df = pd.concat([df.assign(Ticker=t) for t, df in ticker_data_dict.items()])
        combined_df.to_csv("latest_stock_data.csv", index=False)
        st.success("CSV saved as latest_stock_data.csv")

# -----------------
# Auto-refresh
# -----------------
if refresh_mode == "Auto":
    time.sleep(auto_refresh_seconds)
    st.experimental_rerun()

st.markdown("---")
st.write("Tips: For intraday data use short intervals like 1m/2m and period '1d' or '7d'. yfinance may have limits or slight delays. Consider using a dedicated streaming API for true low-latency feeds.")
