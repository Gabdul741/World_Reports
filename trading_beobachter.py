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
        garch_vol = float(garch_result.conditional_volatility.iloc[-1])
        garch_annual = garch_vol * (252**0.5)

        col1, col2, col3 = st.columns(3)
        col1.metric("GARCH дневная", f"{garch_vol:.2f}%")
        col2.metric("GARCH годовая", f"{garch_annual:.2f}%")
        signal = "Спокойно" if today_vix < 20 else "Осторожно" if today_vix < 30 else "Опасно!"
        col3.metric("Сигнал VIX", signal)

        st.subheader("Прогноз SARIMAX на завтра")
        exog_cols = ["DXY", "SP500", "Gold", "VIX", "USO", "TLT"]
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

        st.info(f"Уровень риска: {risk}")
        st.info(f"Прогноз: {direction} до ${tomorrow:.2f} (плюс-минус ${daily_move:.2f})")
        st.success("Источники: SARIMAX + VIX + GARCH | Данные: Yahoo Finance")

st.markdown("---")
st.caption("Trading Beobachter - Gabdul741 и Claude Sonnet 4.6 - Anthropic")
