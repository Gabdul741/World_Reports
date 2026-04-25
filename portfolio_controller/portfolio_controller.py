import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Портфельный контролёр с ИИ (Prophet)")
st.markdown("Прогноз на 7 дней, сигналы: 🟢 купить / 🟡 держать / 🔴 продавать")

# Список активов (тикер : название)
TICKERS = {
    "BZ=F": "Нефть Brent",
    "GC=F": "Золото",
    "^GSPC": "S&P 500",
    "AAPL": "Apple",
    "XOM": "Exxon Mobil"
}

# Выбор активов
selected = st.multiselect(
    "Выберите активы для анализа (2–5 штук)",
    options=list(TICKERS.keys()),
    format_func=lambda x: TICKERS[x],
    default=["BZ=F", "GC=F", "^GSPC"]
)

HISTORY_YEARS = st.slider("Глубина истории (лет)", 2, 5, 3)
FORECAST_DAYS = 7

@st.cache_data(ttl=3600)
def load_data(ticker, years):
    end = datetime.now()
    start = end - timedelta(days=365*years)
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
        return "🟢", "Купить (дешевле прогноза)"
    elif current_price > high:
        return "🔴", "Продавать (дороже прогноза)"
    else:
        return "🟡", "Держать (в диапазоне)"

# Основной блок
results = []

for ticker in selected:
    with st.spinner(f"Загружаю {TICKERS[ticker]}..."):
        df = load_data(ticker, HISTORY_YEARS)
        if df is None or len(df) < 50:
            st.warning(f"❌ Недостаточно данных для {TICKERS[ticker]}")
            continue
        
        forecast = make_forecast(df, FORECAST_DAYS)
        current_price = df["y"].iloc[-1]
        signal, comment = get_signal(current_price, forecast.iloc[0])
        
        results.append({
            "Актив": TICKERS[ticker],
            "Цена сейчас": f"${current_price:.2f}",
            "Прогноз завтра": f"${forecast.iloc[0]['yhat']:.2f}",
            "Нижняя граница": f"${forecast.iloc[0]['yhat_lower']:.2f}",
            "Верхняя граница": f"${forecast.iloc[0]['yhat_upper']:.2f}",
            "Сигнал": signal,
            "Комментарий": comment
        })
        
        # График
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="История"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Прогноз"))
        fig.update_layout(title=f"{TICKERS[ticker]} — прогноз на {FORECAST_DAYS} дней")
        st.plotly_chart(fig, use_container_width=True)

# Сводная таблица
st.subheader("📋 Сводная таблица сигналов")
df_results = pd.DataFrame(results)
st.dataframe(df_results, use_container_width=True)

# Итоговый совет
st.subheader("🧠 Итоговая рекомендация")
buys = [r["Актив"] for r in results if r["Сигнал"] == "🟢"]
sells = [r["Актив"] for r in results if r["Сигнал"] == "🔴"]

if buys:
    st.success(f"🟢 Рассмотрите покупку: {', '.join(buys)}")
if sells:
    st.warning(f"🔴 Рассмотрите продажу: {', '.join(sells)}")
if not buys and not sells:
    st.info("🟡 Ничего не делайте, наблюдайте")
