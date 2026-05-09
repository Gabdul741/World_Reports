import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(page_title="Trading Beobachter", layout="wide")
st.title("📊 Trading Beobachter — GARCH + LSTM")
st.markdown("**Система прогнозирования на основе GARCH + LSTM**")

st.sidebar.title("Настройки")
asset = st.sidebar.selectbox("Актив:", [
    "WTI Нефть (CL=F)",
    "Серебро (SI=F)",
    "Золото (GC=F)",
    "S&P 500 (^GSPC)"
])
period = st.sidebar.selectbox("Период данных:", ["1y", "2y", "3y"])
epochs = st.sidebar.slider("Эпохи обучения LSTM:", 10, 100, 30)

tickers = {
    "WTI Нефть (CL=F)": "CL=F",
    "Серебро (SI=F)": "SI=F",
    "Золото (GC=F)": "GC=F",
    "S&P 500 (^GSPC)": "^GSPC"
}
ticker = tickers[asset]
if st.sidebar.button("Загрузить и рассчитать"):
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
        col3.metric("Ожид. движение", f"${daily_move:.2f}")
        col4.metric("Данных дней", len(df))

        st.subheader("История цены")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Price"], color="blue", linewidth=1)
        ax.set_title(f"{asset}")
        ax.set_xlabel("Дата")
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

        df["GARCH_VOL"] = garch_vol.reindex(df.index).fillna(method="ffill")
        st.subheader("LSTM Прогноз на завтра")
        with st.spinner("Обучаем LSTM..."):
            features = ["Price", "VIX", "GARCH_VOL"]
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[features])

            price_scaler = MinMaxScaler()
            price_scaler.fit(df[["Price"]])

            window = 30
            X, y = [], []
            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i])
                y.append(scaled[i, 0])
            X, y = np.array(X), np.array(y)

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(window, len(features))),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)

            y_pred = model.predict(X_test, verbose=0)
            y_pred_inv = price_scaler.inverse_transform(y_pred)
            y_test_inv = price_scaler.inverse_transform(y_test.reshape(-1,1))
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            accuracy = 100 - (mae / y_test_inv.mean() * 100)

            last_seq = scaled[-window:].reshape(1, window, len(features))
            tomorrow_scaled = model.predict(last_seq, verbose=0)
            tomorrow = float(price_scaler.inverse_transform(tomorrow_scaled)[0][0])
            change = ((tomorrow - today_price) / today_price * 100)
        direction = "Рост" if change > 0 else "Падение"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Прогноз LSTM", f"${tomorrow:.2f}")
        col2.metric("Изменение", f"{change:.2f}%")
        col3.metric("Точность", f"{accuracy:.1f}%")
        col4.metric("Направление", direction)

        st.subheader("Ожидаемый диапазон завтра")
        low = tomorrow - daily_move
        high = tomorrow + daily_move

        col1, col2, col3 = st.columns(3)
        col1.metric("Минимум", f"${low:.2f}")
        col2.metric("Прогноз", f"${tomorrow:.2f}")
        col3.metric("Максимум", f"${high:.2f}")

        st.subheader("Рекомендация Trading Beobachter")
        if today_vix < 15:
            risk = "Низкий риск - можно торговать"
        elif today_vix < 25:
            risk = "Умеренный риск - осторожно"
        elif today_vix < 35:
            risk = "Высокий риск - уменьшить позиции"
        else:
            risk = "Экстремальный риск - лучше не торговать!"

        st.info(f"Уровень риска: {risk}")
        st.info(f"Прогноз GARCH+LSTM: {direction} до ${tomorrow:.2f} (плюс-минус ${daily_move:.2f})")
        st.success("Модель: GARCH волатильность + LSTM нейросеть + VIX")

st.markdown("---")
st.caption("Trading Beobachter - Gabdul741 и Claude Sonnet 4.6 - Anthropic")

            
