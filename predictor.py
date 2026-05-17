import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import ta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Predictor", layout="wide")
st.title("📊 Smart Predictor — Умный выбор модели")
st.markdown("**Каждый актив получает свою оптимальную модель прогнозирования**")

st.sidebar.title("Настройки")
asset = st.sidebar.selectbox("Актив:", [
    "WTI Нефть (CL=F) — SARIMAX",
    "Серебро (SI=F) — GARCH+XGBoost",
    "S&P 500 (^GSPC) — XGBoost",
    "Золото (GC=F) — SARIMAX",
])
period = st.sidebar.selectbox("Период данных:", ["1y", "2y", "3y"])

asset_map = {
    "WTI Нефть (CL=F) — SARIMAX": ("CL=F", "SARIMAX", "WTI Нефть"),
    "Серебро (SI=F) — GARCH+XGBoost": ("SI=F", "GARCH+XGBoost", "Серебро"),
    "S&P 500 (^GSPC) — XGBoost": ("^GSPC", "XGBoost", "S&P 500"),
    "Золото (GC=F) — SARIMAX": ("GC=F", "SARIMAX", "Золото"),
}
ticker, model_type, asset_name = asset_map[asset]
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
        col3.metric("Модель", model_type)
        col4.metric("Ожид. движение ±", f"${daily_move:.2f}")

        st.subheader(f"История цены — {asset_name}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Price"], color="blue", linewidth=1)
        ax.set_title(f"{asset_name}")
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
        df["GARCH_VOL"] = garch_vol.reindex(df.index).ffill().bfill()

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
        st.subheader(f"Прогноз на завтра — {model_type}")

        if model_type == "SARIMAX":
            dxy = yf.download("DX-Y.NYB", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
            gold = yf.download("GC=F", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
            sp500 = yf.download("^GSPC", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
            uso = yf.download("USO", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
            tlt = yf.download("TLT", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()

            df["DXY"] = dxy
            df["Gold"] = gold
            df["SP500"] = sp500
            df["USO"] = uso
            df["TLT"] = tlt
            df = df.dropna()

            exog_cols = ["DXY", "Gold", "SP500", "VIX", "USO", "TLT", "GARCH_VOL"]
            train = df.iloc[:-30]
            test = df.iloc[-30:]

            with st.spinner("Обучаем SARIMAX..."):
                sarimax_model = SARIMAX(
                    train["Price"],
                    exog=train[exog_cols],
                    order=(2,1,2),
                    seasonal_order=(1,1,1,5),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                sarimax_result = sarimax_model.fit(disp=False)
                pred = sarimax_result.forecast(steps=30, exog=test[exog_cols])
                mae = mean_absolute_error(test["Price"], pred)
                accuracy = 100 - (mae / test["Price"].mean() * 100)
                last_exog = df[exog_cols].iloc[-1:]
                tomorrow = float(sarimax_result.forecast(steps=1, exog=last_exog).iloc[0])

        elif model_type in ["XGBoost", "GARCH+XGBoost"]:
            df["RSI"] = ta.momentum.RSIIndicator(df["Price"]).rsi()
            df["MACD"] = ta.trend.MACD(df["Price"]).macd()
            df["EMA20"] = ta.trend.EMAIndicator(df["Price"], window=20).ema_indicator()
            df["BB_high"] = ta.volatility.BollingerBands(df["Price"]).bollinger_hband()
            df["BB_low"] = ta.volatility.BollingerBands(df["Price"]).bollinger_lband()
            for lag in [1, 2, 3, 5]:
                df[f"lag_{lag}"] = df["Price"].shift(lag)
                df[f"ret_{lag}"] = df["Price"].pct_change(lag)
            df["Target"] = df["Price"].shift(-1)
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
        col1.metric("Прогноз", f"${tomorrow:.2f}")
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

        if model_type == "SARIMAX":
            model_info = "SARIMAX + внешние переменные (DXY, Gold, SP500, VIX, USO, TLT, GARCH)"
        elif model_type == "GARCH+XGBoost":
            model_info = "GARCH волатильность + XGBoost + технические индикаторы"
        else:
            model_info = "XGBoost + технические индикаторы + VIX + GARCH"

        st.info(f"Уровень риска: {risk}")
        st.info(f"Прогноз: {direction} до ${tomorrow:.2f} (плюс-минус ${daily_move:.2f})")
        st.success(f"Модель: {model_info}")

st.markdown("---")
st.caption("Smart Predictor - Gabdul741 и Claude Sonnet 4.6 - Anthropic")
