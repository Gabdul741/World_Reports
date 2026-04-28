import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(layout="wide")
st.title("📊 Портфельный контролёр с ИИ (прямой WTI)")
st.markdown("Прогноз на 7 дней, сигналы: 🟢 купить / 🟡 держать / 🔴 продавать")

# -------------------------------------------------------------------
# 1. АКТИВЫ — ТОЛЬКО WTI (CL=F) + эталоны
# -------------------------------------------------------------------
TICKERS = {
    "CL=F": "Нефть WTI (прямой фьючерс)",
    "GLD": "Золото ETF",
    "SLV": "Серебро ETF",
    "QQQ": "Nasdaq 100 ETF",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp."
}

selected = st.multiselect(
    "Выберите активы (2–5 шт)",
    options=list(TICKERS.keys()),
    format_func=lambda x: TICKERS[x],
    default=["CL=F", "GLD", "SLV"]
)

HISTORY_YEARS = st.slider("Глубина истории (лет)", 2, 5, 3)
FORECAST_DAYS = 7

# -------------------------------------------------------------------
# 2. ЗАГРУЗКА ДАННЫХ
# -------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, years):
    end = datetime.now()
    start = end - timedelta(days=365 * years)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    return df

def make_forecast(df, days):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days, include_history=False)
    forecast = model.predict(future)
    return forecast

def get_signal(current_price, forecast_row):
    low = forecast_row["yhat_lower"]
    high = forecast_row["yhat_upper"]
    if current_price < low:
        return "🟢 Купить"
    elif current_price > high:
        return "🔴 Продавать"
    else:
        return "🟡 Держать"

# -------------------------------------------------------------------
# 3. ОСНОВНОЙ ЦИКЛ
# -------------------------------------------------------------------
results = []
for ticker in selected:
    with st.spinner(f"Загружаю {TICKERS[ticker]}..."):
        df = load_data(ticker, HISTORY_YEARS)
        if df is None or len(df) < 50:
            st.warning(f"⚠️ Недостаточно данных для {TICKERS[ticker]}")
            continue

        current_price = df["y"].iloc[-1]

        # ----- ЗАЩИТА ОТ АНОМАЛИЙ WTI -----
        if ticker == "CL=F" and (current_price > 200 or current_price < 10):
            st.warning(f"⚠️ Аномальные данные по WTI (${current_price:.2f}) — пропускаем")
            continue
        # ---------------------------------

        forecast = make_forecast(df, FORECAST_DAYS)
        signal = get_signal(current_price, forecast.iloc[0])

        results.append({
            "Актив": TICKERS[ticker],
            "Цена сейчас": f"${current_price:.2f}",
            "Прогноз завтра": f"${forecast.iloc[0]['yhat']:.2f}",
            "Нижняя граница": f"${forecast.iloc[0]['yhat_lower']:.2f}",
            "Верхняя граница": f"${forecast.iloc[0]['yhat_upper']:.2f}",
            "Сигнал": signal,
        })

        # График
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="История"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Прогноз"))
        fig.update_layout(title=f"{TICKERS[ticker]} — прогноз Prophet на {FORECAST_DAYS} дней")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# 4. ТАБЛИЦА СИГНАЛОВ
# -------------------------------------------------------------------
st.subheader("📋 Сигналы Prophet по активам")
if results:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
else:
    st.warning("Нет данных для отображения")

buys = [r["Актив"] for r in results if "🟢" in r["Сигнал"]]
sells = [r["Актив"] for r in results if "🔴" in r["Сигнал"]]
if buys:
    st.success(f"🟢 Рассмотрите покупку: {', '.join(buys)}")
if sells:
    st.error(f"🔴 Рассмотрите продажу: {', '.join(sells)}")
if not buys and not sells:
    st.info("🟡 Ничего не делайте, наблюдайте")

# -------------------------------------------------------------------
# 5. АНСАМБЛЬ (XGBoost + SARIMAX) – отдельная опция
# -------------------------------------------------------------------
st.sidebar.markdown("---")
ensemble_on = st.sidebar.checkbox("🔬 Сравнение XGBoost / SARIMAX", value=False)

if ensemble_on and results:
    st.subheader("🧪 Сравнение прогнозов моделей")
    for ticker in selected:
        with st.expander(f"{TICKERS[ticker]} ({ticker})", expanded=False):
            end = datetime.now()
            start = end - timedelta(days=365*4)
            df_long = yf.download(ticker, start=start, end=end, progress=False)
            if df_long.empty or len(df_long) < 100:
                st.warning(f"Недостаточно данных для {TICKERS[ticker]}")
                continue

            prices = df_long["Close"].values
            last_price = prices[-1]

            # XGBoost
            try:
                window = 10
                X, y = [], []
                for i in range(window, len(prices)):
                    X.append(prices[i-window:i])
                    y.append(prices[i])
                X = np.array(X)
                split = int(len(X)*0.8)
                model_xgb = XGBRegressor(n_estimators=50, random_state=42)
                model_xgb.fit(X[:split], y[:split])
                pred_xgb = model_xgb.predict(X[-1:])[0]
                change_xgb = (pred_xgb - last_price)/last_price*100
                signal_xgb = "🔴 Продавать" if change_xgb < -1 else ("🟢 Покупать" if change_xgb > 1 else "🟡 Держать")
                st.metric("XGBoost", f"${pred_xgb:.2f}", f"{change_xgb:+.2f}%")
                st.caption(signal_xgb)
            except Exception as e:
                st.error(f"XGBoost ошибка: {e}")

            # SARIMAX
            try:
                y_series = pd.Series(prices[-200:], index=df_long.index[-200:])
                model_sar = SARIMAX(y_series, order=(1,0,1), seasonal_order=(0,0,0,0))
                res_sar = model_sar.fit(disp=False, maxiter=200)
                pred_sar = res_sar.forecast(steps=1).iloc[0]
                change_sar = (pred_sar - last_price)/last_price*100
                signal_sar = "🔴 Продавать" if change_sar < -1 else ("🟢 Покупать" if change_sar > 1 else "🟡 Держать")
                st.metric("SARIMAX", f"${pred_sar:.2f}", f"{change_sar:+.2f}%")
                st.caption(signal_sar)
                st.caption(f"AIC: {res_sar.aic:.1f} | Ширина интервала: {(res_sar.get_forecast(steps=1).conf_int().iloc[0,1] - res_sar.get_forecast(steps=1).conf_int().iloc[0,0])/2:.2f}")
            except Exception as e:
                st.error(f"SARIMAX ошибка: {e}")
