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
st.markdown("Прогноз Prophet + GARCH волатильность + VIX рыночный страх")

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
# 4. ОСНОВНОЙ ЦИКЛ
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

        # ===== GARCH ДЛЯ НЕФТИ =====
        if ticker == "CL=F":
            try:
                prices_series = df['y'].iloc[-252:]
                returns = prices_series.pct_change().dropna() * 100
                model_garch = arch_model(returns, vol='Garch', p=1, q=1)
                garch_result = model_garch.fit(update_freq=5, disp='off')
                garch_vol_series = garch_result.conditional_volatility
                vol_forecast = garch_vol_series.iloc[-1]
                
                if vol_forecast > vol_threshold:
                    st.warning(f"⚠️ Волатильность нефти {vol_forecast:.2f}% (выше порога {vol_threshold}%)")
                    if "🟢" in signal:
                        signal = "🟡 Держать (высокая волатильность)"

                # ===== VIX И СРАВНИТЕЛЬНЫЙ ГРАФИК =====
                start_date = df['ds'].min()
                end_date = df['ds'].max()
                vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False)['Close']
                vix_data = vix_data.reindex(df['ds'], method='ffill')
                vix_daily = vix_data / (252 ** 0.5)

                fig_vix = go.Figure()
                fig_vix.add_trace(go.Scatter(x=df['ds'], y=garch_vol_series, mode='lines', name='GARCH (дневная)', line=dict(color='red')))
                fig_vix.add_trace(go.Scatter(x=vix_data.index, y=vix_daily, mode='lines', name='VIX / √252', line=dict(color='blue', dash='dot')))
                fig_vix.update_layout(title="Сравнение волатильности: GARCH vs VIX", xaxis_title="Дата", yaxis_title="Волатильность (%)")
                st.plotly_chart(fig_vix, use_container_width=True)
            
                current_vix = float(vix_data.iloc[-1].values[0])
            
                if current_vix < 15:
                    vix_signal = "🟢 Спокойно"
                elif current_vix < 25:
                    vix_signal = "🟡 Осторожно"
                elif current_vix < 35:
                    vix_signal = "🟠 Высокий риск"
                else:
                    vix_signal = "🔴 Экстремальный риск"
                st.info(f"📊 Текущий VIX: **{current_vix:.2f}** → {vix_signal}")

            except Exception as e:
                st.caption(f"GARCH/VIX не рассчитан: {e}")

        results.append({
            "Актив": TICKERS[ticker],
            "Цена сейчас": f"${current_price:.2f}",
            "Прогноз завтра": f"${forecast.iloc[0]['yhat']:.2f}",
            "Нижняя граница": f"${forecast.iloc[0]['yhat_lower']:.2f}",
            "Верхняя граница": f"${forecast.iloc[0]['yhat_upper']:.2f}",
            "Сигнал": signal,
        })

        # График Prophet
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="История"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines+markers", name="Прогноз"))
        fig.update_layout(title=f"{TICKERS[ticker]} — прогноз Prophet на {FORECAST_DAYS} дней")
        st.plotly_chart(fig, use_container_width=True)

# ======================
# 5. СВОДНАЯ ТАБЛИЦА
# ======================
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
