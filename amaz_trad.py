import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from chronos import ChronosPipeline
from xgboost import XGBRegressor
from arch import arch_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import ta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Amazon Chronos Trader", layout="wide")
st.title("🚀 Amazon Chronos Trader")
st.markdown("**Прогноз на основе Amazon Chronos — Foundation Model**")

st.sidebar.title("Настройки")
asset = st.sidebar.selectbox("Актив:", [
    "WTI Нефть (CL=F)",
    "Серебро (SI=F)",
    "Золото (GC=F)",
    "S&P 500 (^GSPC)",
    "Apple (AAPL)",
    "Nvidia (NVDA)",
])
period = st.sidebar.selectbox("Период данных:", ["6mo", "1y", "2y", "3y"])
context_days = st.sidebar.slider("Контекст (дней):", 30, 120, 60)

tickers = {
    "WTI Нефть (CL=F)": "CL=F",
    "Серебро (SI=F)": "SI=F",
    "Золото (GC=F)": "GC=F",
    "S&P 500 (^GSPC)": "^GSPC",
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
}
ticker = tickers[asset]
if st.sidebar.button("Загрузить и предсказать"):
    with st.spinner("Загружаем данные..."):
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
        df = df[["Close", "Volume"]].dropna()
        df.columns = ["Price", "Volume"]
        vix = yf.download("^VIX", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        df["VIX"] = vix
        df = df.dropna()

        today_price = float(df["Price"].iloc[-1])
        today_vix = float(df["VIX"].iloc[-1])
        daily_move = today_price * (today_vix/100) / (252**0.5)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Цена сегодня", f"${today_price:.2f}")
        col2.metric("VIX", f"{today_vix:.2f}")
        col3.metric("Ожид. движение ±", f"${daily_move:.2f}")
        col4.metric("Данных дней", len(df))

        st.subheader(f"История цены — {asset}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Price"], color="blue", linewidth=1)
        ax.set_title(f"{asset}")
        ax.set_ylabel("Цена ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("GARCH Волатильность")
        returns = df["Price"].pct_change().dropna() * 100
        garch_model = arch_model(returns, vol="Garch", p=1, q=1)
        garch_result = garch_model.fit(disp="off")
        garch_vol = garch_result.conditional_volatility
        garch_today = float(garch_vol.iloc[-1])
        garch_annual = garch_today * (252**0.5)

        col1, col2, col3 = st.columns(3)
        col1.metric("GARCH дневная", f"{garch_today:.2f}%")
        col2.metric("GARCH годовая", f"{garch_annual:.2f}%")
        signal = "Спокойно" if today_vix < 20 else "Осторожно" if today_vix < 30 else "Опасно!"
        col3.metric("Сигнал VIX", signal)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        vix_daily = df["VIX"] / (252**0.5)
        ax2.plot(garch_vol.index, garch_vol, color="red", label="GARCH", linewidth=1)
        ax2.plot(vix_daily.index, vix_daily, color="blue", label="VIX/√252", linewidth=1)
        ax2.set_title("GARCH vs VIX")
        ax2.set_ylabel("Волатильность (%)")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
st.subheader("🚀 Amazon Chronos Прогноз")
        with st.spinner("Загружаем Chronos модель..."):
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cpu",
                dtype=torch.float32,
            )
            prices = df["Price"].values.flatten()
            context = torch.tensor(prices[-context_days:], dtype=torch.float32)
            forecast = pipeline.predict([context], prediction_length=1, num_samples=20)
            tomorrow_chronos = float(forecast[0].median())
            low_chronos = float(forecast[0].quantile(0.1))
            high_chronos = float(forecast[0].quantile(0.9))

        change_chronos = ((tomorrow_chronos - today_price) / today_price * 100)
        direction_chronos = "Рост" if change_chronos > 0 else "Падение"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chronos прогноз", f"${tomorrow_chronos:.2f}")
        col2.metric("Изменение", f"{change_chronos:.2f}%")
        col3.metric("Направление", direction_chronos)
        col4.metric("Точность модели", "Foundation")

        col1, col2, col3 = st.columns(3)
        col1.metric("Chronos минимум (10%)", f"${low_chronos:.2f}")
        col2.metric("Chronos медиана", f"${tomorrow_chronos:.2f}")
        col3.metric("Chronos максимум (90%)", f"${high_chronos:.2f}")

        st.subheader("📊 XGBoost прогноз для сравнения")
        df["GARCH_VOL"] = garch_vol.reindex(df.index).ffill().bfill()
        df["RSI"] = ta.momentum.RSIIndicator(df["Price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["Price"]).macd()
        df["EMA20"] = ta.trend.EMAIndicator(df["Price"], window=20).ema_indicator()
        for lag in [1, 2, 3, 5]:
            df[f"lag_{lag}"] = df["Price"].shift(lag)
            df[f"ret_{lag}"] = df["Price"].pct_change(lag)
        df["Target"] = df["Price"].shift(-1)
        df_xgb = df.dropna()

        features = [c for c in df_xgb.columns if c != "Target"]
        X = df_xgb[features]
        y = df_xgb["Target"]
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        with st.spinner("Обучаем XGBoost..."):
            xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=6, random_state=42)
            xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = 100 - (mae / y_test.mean() * 100)
        tomorrow_xgb = float(xgb_model.predict(X.iloc[-1:])[0])
        change_xgb = ((tomorrow_xgb - today_price) / today_price * 100)

        col1, col2, col3 = st.columns(3)
        col1.metric("XGBoost прогноз", f"${tomorrow_xgb:.2f}")
        col2.metric("Изменение XGBoost", f"{change_xgb:.2f}%")
        col3.metric("Точность XGBoost", f"{accuracy:.1f}%")
