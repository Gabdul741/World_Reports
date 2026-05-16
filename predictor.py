import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import ta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Predictor")
st.markdown("**Прогноз цены акций на следующий день — XGBoost + GARCH + VIX**")

st.sidebar.title("Настройки")
ticker_input = st.sidebar.text_input("Тикер акции:", value="AAPL")
period = st.sidebar.selectbox("Период данных:", ["1y", "2y", "3y"])

popular = st.sidebar.expander("Популярные тикеры")
with popular:
    st.markdown("""
    **Акции США:**
    - AAPL — Apple
    - MSFT — Microsoft
    - GOOGL — Google
    - AMZN — Amazon
    - NVDA — Nvidia
    - TSLA — Tesla
    
    **Индексы:**
    - ^GSPC — S&P 500
    - ^DJI — Dow Jones
    - ^IXIC — Nasdaq
    
    **Сырьё:**
    - CL=F — Нефть WTI
    - GC=F — Золото
    - SI=F — Серебро
    """)

if st.sidebar.button("Загрузить и предсказать"):
    with st.spinner("Загружаем данные..."):
        df = yf.download(ticker_input, period=period, interval="1d", auto_adjust=True)
        if df.empty:
            st.error("Тикер не найден! Проверьте название.")
        else:
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            vix = yf.download("^VIX", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
            df["VIX"] = vix
            df = df.dropna()

            today_price = float(df["Close"].iloc[-1])
            today_vix = float(df["VIX"].iloc[-1])
            daily_move = today_price * (today_vix/100) / (252**0.5)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Цена сегодня", f"${today_price:.2f}")
            col2.metric("VIX", f"{today_vix:.2f}")
            col3.metric("Ожид. движение ±", f"${daily_move:.2f}")
            col4.metric("Данных дней", len(df))

            st.subheader("История цены")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df["Close"], color="blue", linewidth=1)
            ax.set_title(f"{ticker_input} — История цены")
            ax.set_ylabel("Цена ($)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.subheader("GARCH Волатильность")
            returns = df["Close"].pct_change().dropna() * 100
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

            st.subheader("XGBoost Прогноз на завтра")
            df["GARCH_VOL"] = garch_vol.reindex(df.index).ffill().bfill()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["MACD"] = ta.trend.MACD(df["Close"]).macd()
            df["EMA20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
            for lag in [1, 2, 3, 5]:
                df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
                df[f"Return_lag_{lag}"] = df["Close"].pct_change(lag)
            df["Target"] = df["Close"].shift(-1)
            df = df.dropna()

            features = [c for c in df.columns if c != "Target"]
            X = df[features]
            y = df["Target"]
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            with st.spinner("Обучаем XGBoost..."):
                model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=6, random_state=42)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            accuracy = 100 - (mae / y_test.mean() * 100)
            tomorrow = float(model.predict(X.iloc[-1:])[0])
            change = ((tomorrow - today_price) / today_price * 100)
            direction = "Рост" if change > 0 else "Падение"

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Прогноз XGBoost", f"${tomorrow:.2f}")
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

            st.subheader("Рекомендация")
            if today_vix < 15:
                risk = "Низкий риск - можно торговать"
            elif today_vix < 25:
                risk = "Умеренный риск - осторожно"
            elif today_vix < 35:
                risk = "Высокий риск - уменьшить позиции"
            else:
                risk = "Экстремальный риск - лучше не торговать!"

            st.info(f"Уровень риска: {risk}")
            st.info(f"Прогноз: {direction} до ${tomorrow:.2f} (плюс-минус ${daily_move:.2f})")
            st.success("Модель: XGBoost + GARCH + VIX + Технические индикаторы")

st.markdown("---")
st.caption("Stock Predictor - Gabdul741 и Claude Sonnet 4.6 - Anthropic")
