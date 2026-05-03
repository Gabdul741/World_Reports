import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Trading Beobachter", layout="wide")
st.title("📊 Trading Beobachter Dashboard")
st.markdown("**Система анализа и прогнозирования**")

st.sidebar.title("Настройки")
asset = st.sidebar.selectbox("Актив:", [
    "WTI Нефть (CL=F)",
    "Серебро (SI=F)",
    "Золото (GC=F)"
])
period = st.sidebar.selectbox("Период данных:", ["1y", "2y", "3y"])

tickers = {
    "WTI Нефть (CL=F)": "CL=F",
    "Серебро (SI=F)": "SI=F",
    "Золото (GC=F)": "GC=F"
}
ticker = tickers[asset]

if st.sidebar.button("Загрузить и рассчитать"):
    with st.spinner("Загружаем данные..."):
        main = yf.download(ticker, period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        dxy = yf.download("DX-Y.NYB", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        sp500 = yf.download("^GSPC", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        gold = yf.download("GC=F", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        vix = yf.download("^VIX", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        uso = yf.download("USO", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        tlt = yf.download("TLT", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()

        df = pd.DataFrame({
            "Price": main,
            "DXY": dxy,
            "SP500": sp500,
            "Gold": gold,
            "VIX": vix,
            "USO": uso,
            "TLT": tlt
        }).dropna()

        today_price = float(df["Price"].iloc[-1])
        today_vix = float(df["VIX"].iloc[-1])
        daily_move = today_price * (today_vix/100) / (252**0.5)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Цена сегодня", f"${today_price:.2f}")
        col2.metric("VIX", f"{today_vix:.2f}")
        col3.metric("Ожид. движение", f"${daily_move:.2f}")
        col4.metric("Данных дней", len(df))

        st.subheader("История цены")
        fig, ax = plt.subplots(figsize=(12, 4))
