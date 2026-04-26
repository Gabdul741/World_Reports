import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Портфельный контролёр с ИИ (Prophet)")
st.markdown("Прогноз на 7 дней, цветные сигналы: 🟢 купить / 🟡 держать / 🔴 продавать")

# --- Выбор активов ---
TICKERS = {
    "BZ=F": "Нефть Brent",
    "GC=F": "Золото",
    "^GSPC": "S&P 500",
    "AAPL": "Apple",
    "XOM": "Exxon Mobil"
}

selected = st.multiselect(
    "Выберите активы (2–5 шт)",
    options=list(TICKERS.keys()),
    format_func=lambda x: TICKERS[x],
    default=["BZ=F", "GC=F", "^GSPC"]
)

HISTORY_YEARS = st.slider("Глубина истории (лет)", 2, 5, 3)
FORECAST_DAYS = 7

# --- Загрузка данных ---
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
        return "Купить"
    elif current_price > high:
        return "Продавать"
    else:
        return "Держать"

def signal_to_color(signal):
    if signal == "Купить":
        return "background-color: #90EE90"
    elif signal == "Продавать":
        return "background-color: #FFCCCC"
    else:
        return "background-color: #FFFACD"

# --- Основной цикл ---
results = []

for ticker in selected:
    with st.spinner(f"Загружаю {TICKERS[ticker]}..."):
        df = load_data(ticker, HISTORY_YEARS)
        if df is None or len(df) < 50:
            st.warning(f"❌ Недостаточно данных для {TICKERS[ticker]}")
            continue
        
        # Модель и прогноз
        forecast = make_forecast(df, FORECAST_DAYS)
        current_price = df["y"].iloc[-1]
        signal = get_signal(current_price, forecast.iloc[0])
        
        # МЕТРИКИ: точность прогноза на исторических данных (блок expander)
        # (не влияет на основной расчёт, добавляет контекст)
        with st.expander(f"📊 Точность прогноза для {TICKERS[ticker]}"):
            # Ограничим расчёт последними 100 днями для скорости
            test_len = min(100, len(df))
            test_df = df.iloc[-test_len:].copy()
            # Переобучаем модель на данных до последних test_len дней
            train_df = df.iloc[:-test_len].copy()
            if len(train_df) > 30:
                test_model = Prophet(daily_seasonality=True)
                test_model.fit(train_df)
                test_forecast = test_model.predict(test_df[["ds"]])
                actual = test_df["y"].values
                predicted = test_forecast["yhat"].values
                mae = abs(actual - predicted).mean()
                coverage = ((actual >= test_forecast["yhat_lower"].values) & 
                            (actual <= test_forecast["yhat_upper"].values)).mean() * 100
                st.metric("Средняя ошибка (MAE)", f"${mae:.2f}")
                st.metric("Попадание в интервал", f"{coverage:.1f}%")
                if coverage > 70:
                    st.success("✅ Доверительные интервалы надёжны")
                elif coverage > 50:
                    st.warning("⚠️ Интервалы работают средне")
                else:
                    st.error("❌ Интервалы ненадёжны, прогноз с осторожностью")
            else:
                st.info("Недостаточно данных для оценки точности")
        
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
        fig.update_layout(title=f"{TICKERS[ticker]} — прогноз на {FORECAST_DAYS} дней")
        st.plotly_chart(fig, use_container_width=True)

# --- Сводная таблица с цветом ---
st.subheader("📋 Сигналы по активам")
if results:
    df_results = pd.DataFrame(results)
    styled_df = df_results.style.applymap(signal_to_color, subset=["Сигнал"])
    st.dataframe(styled_df, use_container_width=True)
else:
    st.warning("Нет данных для отображения. Проверьте выбранные активы.")
# --- Итоговая рекомендация ---
st.subheader("🧠 Итоговая рекомендация")
buys = [r["Актив"] for r in results if r["Сигнал"] == "Купить"]
sells = [r["Актив"] for r in results if r["Сигнал"] == "Продавать"]

if buys:
    st.success(f"🟢 Рассмотрите покупку: {', '.join(buys)}")
if sells:
    st.error(f"🔴 Рассмотрите продажу: {', '.join(sells)}")
if not buys and not sells:
    st.info("🟡 Ничего не делайте, наблюдайте")
