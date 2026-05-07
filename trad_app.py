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
st.caption(f"Система наблюдения за точностью прогнозов | Сегодня: {datetime.date.today()}")

# Список тикеров
assets = {
    "Нефть WTI": "CL=F",
    "Серебро": "SI=F",
    "S&P 500": "^GSPC",
    "Индекс Доллара": "DX-Y.NYB",
    "VIX Index": "^VIX"
}

# --- 2. ФУНКЦИЯ АНАЛИЗА (ДЛЯ ПРОГНОЗА НА ЗАВТРА) ---
def get_analysis(ticker):
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if data.empty: return None
    
    prices = data['Close'].squeeze()
    returns = 100 * prices.pct_change().dropna()
    
    # GARCH волатильность
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    res_fit = model.fit(disp="off")
    forecast_vol = np.sqrt(res_fit.forecast(horizon=1).variance.values[-1, 0])
    
    # Прогноз цены (инерционный дрифт)
    current_price = float(prices.iloc[-1])
    short_trend = returns.tail(5).mean()
    predicted_price = current_price * (1 + (short_trend / 100))
    
    return {
        "current": current_price,
        "prediction": predicted_price,
        "garch": forecast_vol,
        "trend": short_trend
    }

# --- 3. ИНДИКАТОРЫ УВЕРЕННОСТИ (VIX и DXY) ---
vix_data = yf.download(assets["VIX Index"], period="2d", progress=False)['Close'].squeeze()
dxy_data = yf.download(assets["Индекс Доллара"], period="2d", progress=False)['Close'].squeeze()
vix_val = float(vix_data.iloc[-1])
dxy_val = float(dxy_data.iloc[-1])

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
main_assets = ["Нефть WTI", "Серебро", "S&P 500"] # Та самая переменная
cols = st.columns(3)

for i, name in enumerate(main_assets):
    with cols[i]:
        # Загружаем данные для графика и бэктеста
        raw_data = yf.download(assets[name], period="4mo", interval="1d", progress=False)
        
        if not raw_data.empty:
            prices = raw_data['Close'].squeeze()
            prices = prices[prices.index.dayofweek < 5] # Убираем выходные
            
            # Получаем расчеты на текущий момент
            res_now = get_analysis(assets[name])
            current_p = res_now['current']
            pred_p = res_now['prediction']
            delta_val = pred_p - current_p
            
            # --- ОТОБРАЖЕНИЕ ТЕКУЩЕЙ ЦЕНЫ ---
            st.subheader(name)
            st.metric(label="Текущая цена", value=f"{current_p:.2f}")
            
            # --- БЛОК БЭКТЕСТА (Расчет истории прогнозов) ---
            dates_list = []
            actual_prices = []
            predicted_prices = []
            
            # Считаем историю за последние 30 торговых дней
            for day_idx in range(len(prices) - 30, len(prices)):
                actual_val = float(prices.iloc[day_idx])
                history = prices.iloc[:day_idx]
                returns_hist = 100 * history.pct_change().dropna()
                
                if len(returns_hist) > 20:
                    trend = returns_hist.tail(5).mean()
                    last_p = float(history.iloc[-1])
                    p_val = last_p * (1 + (trend / 100))
                    
                    dates_list.append(prices.index[day_idx])
                    actual_prices.append(actual_val)
                    predicted_prices.append(p_val)

            # Отрисовка графика
# --- ОТРИСОВКА ГРАФИКА С ОБЛАКОМ GARCH ---
            fig = go.Figure()

            # Рассчитываем границы облака (Прогноз +/- GARCH волатильность)
            # Для простоты визуализации используем текущее значение GARCH из анализа
            vol_factor = res_now['garch'] / 100
            upper_bound = [p * (1 + vol_factor) for p in predicted_prices]
            lower_bound = [p * (1 - vol_factor) for p in predicted_prices]

            # Облако волатильности (GARCH)
            fig.add_trace(go.Scatter(
                x=dates_list + dates_list[::-1], # Идем вперед по датам и возвращаемся назад
                y=upper_bound + lower_bound[::-1], # Верхняя граница, затем нижняя в обратном порядке
                fill='toself',
                fillcolor='rgba(255, 75, 75, 0.1)', # Бледный красный цвет
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Зона риска (GARCH)'
            ))

            # Линия реальных цен (Факт)
            fig.add_trace(go.Scatter(
                x=dates_list, y=actual_prices, 
                mode='lines', name='Рынок (Факт)', 
                line=dict(color='#00d4ff', width=3)
            ))

            # Линия прогноза (Центр)
            fig.add_trace(go.Scatter(
                x=dates_list, y=predicted_prices, 
                mode='lines', name='Прогноз', 
                line=dict(color='#FF4B4B', width=2, dash='dash')
            ))
            fig.update_layout(
                height=300, 
                margin=dict(l=0, r=0, t=10, b=0), 
                legend=dict(orientation="h", y=-0.1, font=dict(color="white")), 
                template="plotly_dark",
                font=dict(color="white"), 
                # Заменяем прозрачность на конкретный темно-серый цвет
                paper_bgcolor='#1E1E1E', 
                plot_bgcolor='#1E1E1E',
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            # --- ИТОГИ НА ЗАВТРА ---
            st.metric("Цель на завтра", f"{pred_p:.2f}", f"{delta_val:+.2f}")
            
            if delta_val > 0 and vix_val < 25:
                st.success("🟢 СИГНАЛ: BUY")
            elif delta_val < 0 or vix_val > 30:
                st.error("🔴 СИГНАЛ: SELL")
            else:
                st.warning("⚪️ СИГНАЛ: HOLD")
