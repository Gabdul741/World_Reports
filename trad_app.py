import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import datetime
import plotly.graph_objects as go

# --- 1. КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Market Observer 2026", layout="wide")

st.title("📊 Market Observer & Backtest AI")
st.caption(f"Система наблюдения за точностью прогнозов | Сегодня: {datetime.date.today()} | Aydın, Türkiye")

# Список тикеров
assets = {
    "Нефть WTI": "CL=F",
    "Серебро": "SI=F",
    "S&P 500": "^GSPC",
    "Индекс Доллара": "DX-Y.NYB",
    "VIX Index": "^VIX"
}

# --- 2. ФУНКЦИЯ АНАЛИЗА (GARCH + ТРЕНД) ---
def get_analysis(ticker):
    # Загружаем данные (с запасом для расчета тренда и GARCH)
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if data.empty or len(data) < 10: return None
    
    prices = data['Close'].squeeze()
    returns = 100 * prices.pct_change().dropna()
    
    # GARCH волатильность (Дыхание рынка)
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        res_fit = model.fit(disp="off")
        forecast_vol = np.sqrt(res_fit.forecast(horizon=1).variance.values[-1, 0])
    except:
        forecast_vol = returns.std() # Фолбэк на стандартное отклонение
    
    # Расчет цен
    current_price = float(prices.iloc[-1])
    prev_price = float(prices.iloc[-2])
    short_trend = returns.tail(5).mean()
    predicted_price = current_price * (1 + (short_trend / 100))
    
    return {
        "current": current_price,
        "prev": prev_price,
        "prediction": predicted_price,
        "garch": forecast_vol,
        "trend": short_trend
    }

# --- 3. ИНДИКАТОРЫ УВЕРЕННОСТИ (VIX и DXY) ---
try:
    vix_val = float(yf.download(assets["VIX Index"], period="2d", progress=False)['Close'].squeeze().iloc[-1])
    dxy_val = float(yf.download(assets["Индекс Доллара"], period="2d", progress=False)['Close'].squeeze().iloc[-1])
except:
    vix_val, dxy_val = 0.0, 0.0

st.subheader("🌐 Внешний контекст")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Страх (VIX)", f"{vix_val:.2f}")
    st.caption("GARCH точнее, когда VIX < 25")
with c2:
    st.metric("Доллар (DXY)", f"{dxy_val:.2f}")
    st.caption("Рост DXY давит на сырье")
with c3:
    st.info("Режим: Наблюдатель. Сравнение прогноза (красный) и реальности (синий).")

st.divider()

# --- 4. ОСНОВНЫЕ АКТИВЫ И ГРАФИКИ ---
main_assets = ["Нефть WTI", "Серебро", "S&P 500"]
cols = st.columns(3)

for i, name in enumerate(main_assets):
    with cols[i]:
        raw_data = yf.download(assets[name], period="4mo", interval="1d", progress=False)
        
        if not raw_data.empty:
            prices = raw_data['Close'].squeeze()
            prices = prices[prices.index.dayofweek < 5] 
            
            res_now = get_analysis(assets[name])
            if res_now:
                current_p = res_now['current']
                prev_p = res_now['prev']
                pred_p = res_now['prediction']
                
                # 1. Расчет текущей динамики дня
                day_diff = current_p - prev_p
                day_label = "▲ РОСТ" if day_diff > 0 else "▼ ПАДЕНИЕ"
                
                # 2. Вектор прогноза на завтра
                forecast_diff = pred_p - current_p
                forecast_label = "▲ РОСТ" if forecast_diff > 0 else "▼ ПАДЕНИЕ"

                st.header(name)
                
                # Визуальный блок метрик (Исправлено: теперь значения не выпадают)
                m1, m2 = st.columns(2)
                m1.metric(
                    label="Сегодня (Факт)", 
                    value=f"{current_p:.2f}", 
                    delta=f"{day_diff:+.2f} {day_label}"
                )
                m2.metric(
                    label="Цель на завтра", 
                    value=f"{pred_p:.2f}", 
                    delta=f"{forecast_diff:+.2f} {forecast_label}"
                )

                # Статус синхронности
                is_synced = (day_diff > 0 and forecast_diff > 0) or (day_diff < 0 and forecast_diff < 0)
                if is_synced:
                    st.success("✅ СИНХРОННО")
                else:
                    st.warning("⚠️ ДИВЕРГЕНЦИЯ")

                # --- БЭКТЕСТ (Расчет истории) ---
                dates_list, actual_prices, predicted_prices = [], [], []
                for day_idx in range(len(prices) - 30, len(prices)):
                    actual_val = float(prices.iloc[day_idx])
                    history = prices.iloc[:day_idx]
                    returns_hist = 100 * history.pct_change().dropna()
                    if len(returns_hist) > 20:
                        trend = returns_hist.tail(5).mean()
                        last_p = float(history.iloc[-1])
                        dates_list.append(prices.index[day_idx])
                        actual_prices.append(actual_val)
                        predicted_prices.append(last_p * (1 + (trend / 100)))

                # Отрисовка графика
                fig = go.Figure()
                vol_factor = res_now['garch'] / 100
                upper_bound = [p * (1 + vol_factor) for p in predicted_prices]
                lower_bound = [p * (1 - vol_factor) for p in predicted_prices]

                fig.add_trace(go.Scatter(
                    x=dates_list + dates_list[::-1], y=upper_bound + lower_bound[::-1],
                    fill='toself', fillcolor='rgba(255, 0, 0, 0.08)',
                    line=dict(color='rgba(255,255,255,0)'), name='Дыхание (GARCH)', hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(x=dates_list, y=actual_prices, mode='lines', name='Рынок (Факт)', line=dict(color='#00d4ff', width=3)))
                fig.add_trace(go.Scatter(x=dates_list, y=predicted_prices, mode='lines', name='Прогноз', line=dict(color='#FF4B4B', width=2, dash='dash')))

                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white", legend=dict(orientation="h", y=-0.2), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

# --- 5. HELP & ABOUT ---
st.divider()
with st.expander("❓ HELP & ABOUT (Методология и Авторы)"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 📈 Легенда и Визуализация
        *   **Синяя линия:** Реальное движение цены (Market Fact). 
        *   **Красная линия:** Прогнозная траектория модели.
        *   **Розовое облако (GARCH):** «Дыхание рынка». Доверительный интервал волатильности. Широкое облако — глубокое дыхание (риск), узкое — штиль.
        """)
    with col_b:
        st.markdown("""
        ### 🧠 Модель предсказания
        1. **GARCH:** Оценка волатильности и амплитуды.
        2. **Impulse:** Вектор текущего дня (Рост/Падение).
        3. **Context:** Влияние DXY и VIX на устойчивость прогноза.
        """)
    st.divider()
    st.info("**Создатели:** Gabdul741 & Gemini AI | **Версия:** 2.1 (GARCH Breath Edition)")
