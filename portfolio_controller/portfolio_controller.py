import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("📊 Портфельный контролёр с ИИ (Prophet)")
st.markdown("Прогноз на 7 дней, цветные сигналы: 🟢 купить / 🟡 держать / 🔴 продавать")

# --- АКТИВЫ (новые: WTI и серебро) ---
TICKERS = {
    "CL=F": "Нефть WTI",
    "SI=F": "Серебро",
    "^GSPC": "S&P 500",
    "AAPL": "Apple",
    "XOM": "Exxon Mobil"
}

selected = st.multiselect(
    "Выберите активы (2–5 шт)",
    options=list(TICKERS.keys()),
    format_func=lambda x: TICKERS[x],
    default=["CL=F", "SI=F", "^GSPC"]
)

HISTORY_YEARS = st.slider("Глубина истории (лет)", 2, 5, 3)
FORECAST_DAYS = 7
show_interval = st.checkbox("📈 Показать доверительный интервал на графике", value=False)

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
        return "🟢 Купить"
    elif current_price > high:
        return "🔴 Продавать"
    else:
        return "🟡 Держать"

results = []

for ticker in selected:
    with st.spinner(f"Загружаю {TICKERS[ticker]}..."):
        df = load_data(ticker, HISTORY_YEARS)
        if df is None or len(df) < 50:
            st.warning(f"❌ Недостаточно данных для {TICKERS[ticker]}")
            continue
        
        forecast = make_forecast(df, FORECAST_DAYS)
        current_price = df["y"].iloc[-1]
        signal = get_signal(current_price, forecast.iloc[0])
        
        with st.expander(f"📊 Точность прогноза для {TICKERS[ticker]}"):
            test_len = min(100, len(df))
            train_df = df.iloc[:-test_len].copy()
            if len(train_df) > 30:
                test_model = Prophet(daily_seasonality=True)
                test_model.fit(train_df)
                test_forecast = test_model.predict(df.iloc[-test_len:][["ds"]])
                actual = df.iloc[-test_len:]["y"].values
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
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="История"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Прогноз"))
        if show_interval:
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(dash="dash"), name="Верхняя граница"))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line=dict(dash="dash"), name="Нижняя граница"))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill='toself', fillcolor='rgba(255,255,0,0.2)', line=dict(color='rgba(255,255,0,0)'),
                name="Доверительный интервал", showlegend=True
            ))
        fig.update_layout(title=f"{TICKERS[ticker]} — прогноз на {FORECAST_DAYS} дней")
        st.plotly_chart(fig, use_container_width=True)

# --- Сводная таблица ---
st.subheader("📋 Сигналы по активам")
if results:
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
else:
    st.warning("Нет данных для отображения. Проверьте выбранные активы.")

# --- Итоговая рекомендация ---
st.subheader("🧠 Итоговая рекомендация")
buys = [r["Актив"] for r in results if "Купить" in r["Сигнал"]]
sells = [r["Актив"] for r in results if "Продавать" in r["Сигнал"]]

if buys:
    st.success(f"🟢 Рассмотрите покупку: {', '.join(buys)}")
if sells:
    st.error(f"🔴 Рассмотрите продажу: {', '.join(sells)}")
if not buys and not sells:
    st.info("🟡 Ничего не делайте, наблюдайте")
