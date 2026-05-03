import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
from arch import arch_model

st.set_page_config(layout="wide")
st.title("📊 Портфельный контролёр с ИИ")
st.markdown("Прогноз Prophet + GARCH волатильность + VIX (рыночный)")

# ======================
# 1. АКТИВЫ
# ======================
TICKERS = {
    "CL=F": "Нефть WTI",
    "SLV": "Серебро ETF",
    "^GSPC": "S&P 500",
}

selected = st.multiselect(
    "Выберите активы (2–5 шт)",
    options=list(TICKERS.keys()),
    format_func=lambda x: TICKERS[x],
    default=["CL=F", "SLV", "^GSPC"]
)

# ======================
# 2. ПАРАМЕТРЫ
# ======================
HISTORY_YEARS = st.slider("Глубина истории (лет)", 2, 5, 3)
FORECAST_DAYS = 7

vol_threshold = st.slider(
    "⚠️ Порог волатильности для отмены сделок (%)",
    min_value=1.0, max_value=10.0, value=3.0, step=0.1,
    help="Если прогнозная волатильность (GARCH) превышает этот порог, сигналы покупки блокируются"
)

# ======================
# 3. ЗАГРУЗКА ДАННЫХ
# ======================
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

# ======================
# 4. ГЛОБАЛЬНЫЙ VIX (ОДИН ДЛЯ ВСЕХ АКТИВОВ)
# ======================
vix_global = None
try:
    vix_data = yf.download("^VIX", period="2d", progress=False)['Close']
    vix_global = float(vix_data.iloc[-1].values[0])
except:
    vix_global = None

# ======================
# 5. ОСНОВНОЙ ЦИКЛ
# ======================
results = []
for ticker in selected:
    with st.spinner(f"Загружаю {TICKERS[ticker]}..."):
        df = load_data(ticker, HISTORY_YEARS)
        if df is None or len(df) < 50:
            st.warning(f"⚠️ Недостаточно данных для {TICKERS[ticker]}")
            continue

        current_price = df["y"].iloc[-1]
        forecast = make_forecast(df, FORECAST_DAYS)
        signal = get_signal(current_price, forecast.iloc[0])

        # GARCH для актива
        vol_forecast = None
        try:
            prices_series = df['y'].iloc[-252:]
            returns = prices_series.pct_change().dropna() * 100
            if len(returns) > 10:
                model_garch = arch_model(returns, vol='Garch', p=1, q=1)
                garch_result = model_garch.fit(update_freq=5, disp='off')
                vol_forecast = garch_result.conditional_volatility.iloc[-1]
                if hasattr(vol_forecast, 'values'):
                    vol_forecast = float(vol_forecast.values[0])
                else:
                    vol_forecast = float(vol_forecast)
                # Блокировка сигнала покупки при высокой волатильности
                if vol_forecast > vol_threshold and "🟢" in signal:
                    signal = "🟡 Держать (высокая волатильность)"
        except Exception as e:
            # Если GARCH не рассчитался, оставляем None
            pass

        results.append({
            "Актив": TICKERS[ticker],
            "Цена сейчас": f"${current_price:.2f}",
            "Прогноз завтра": f"${forecast.iloc[0]['yhat']:.2f}",
            "Нижняя граница": f"${forecast.iloc[0]['yhat_lower']:.2f}",
            "Верхняя граница": f"${forecast.iloc[0]['yhat_upper']:.2f}",
            "Сигнал": signal,
            "VIX (рыночный)": f"{vix_global:.2f}" if vix_global is not None else "-",
            "GARCH (%)": f"{vol_forecast:.2f}%" if vol_forecast is not None else "-",
        })

        # График Prophet
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="История"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Прогноз"))
        fig.update_layout(title=f"{TICKERS[ticker]} — прогноз Prophet на {FORECAST_DAYS} дней")
        st.plotly_chart(fig, use_container_width=True)

# ======================
# 6. СВОДНАЯ ТАБЛИЦА
# ======================
st.subheader("📋 Сигналы Prophet по активам")
if results:
    df_results = pd.DataFrame(results)
    # Переупорядочим колонки для удобства
    column_order = ["Актив", "Цена сейчас", "Прогноз завтра", "Сигнал", "VIX (рыночный)", "GARCH (%)"]
    df_results = df_results[column_order]
    st.dataframe(df_results, use_container_width=True)

    buys = [r["Актив"] for r in results if "🟢" in r["Сигнал"]]
    sells = [r["Актив"] for r in results if "🔴" in r["Сигнал"]]
    if buys:
        st.success(f"🟢 Рассмотрите покупку: {', '.join(buys)}")
    if sells:
        st.error(f"🔴 Рассмотрите продажу: {', '.join(sells)}")
    if not buys and not sells:
        st.info("🟡 Ничего не делайте, наблюдайте")
else:
    st.warning("Нет данных для отображения")
