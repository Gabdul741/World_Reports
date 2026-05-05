import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import datetime

st.set_page_config(page_title="Market Predictor 2026", layout="wide")

st.title("📊 Market Observer & AI Forecast")
st.caption(f"Статистический анализ рынка | Сегодня: {datetime.date.today()}")

assets = {
    "Нефть WTI": "CL=F",
    "Серебро": "SI=F",
    "S&P 500": "^GSPC",
    "Индекс Доллара": "DX-Y.NYB",
    "VIX Index": "^VIX"
}

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

# --- ИНДИКАТОРЫ УВЕРЕННОСТИ (VIX и DXY) ---
vix_raw = yf.download(assets["VIX Index"], period="2d", progress=False)['Close'].squeeze()
dxy_raw = yf.download(assets["Индекс Доллара"], period="2d", progress=False)['Close'].squeeze()
vix_val = float(vix_raw.iloc[-1])
dxy_val = float(dxy_raw.iloc[-1])

st.subheader("🌐 Внешние факторы влияния")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Страх (VIX)", f"{vix_val:.2f}")
    st.caption("GARCH работает точнее при VIX < 25")
with c2:
    st.metric("Индекс Доллара (DXY)", f"{dxy_val:.2f}")
    st.caption("Рост DXY давит на нефть и серебро")
with c3:
    st.info("Политический контекст: Высокая активность (Трамп-фактор)")

st.divider()

# --- ОСНОВНЫЕ АКТИВЫ ---
cols = st.columns(3)
for i, name in enumerate(["Нефть WTI", "Серебро", "S&P 500"]):
    with cols[i]:
        res = get_analysis(assets[name])
        if res:
            st.subheader(name)
            st.metric("Текущая цена", f"{res['current']:.2f}")
            
            # ВИЗУАЛЬНЫЙ АКЦЕНТ НА ПРОГНОЗ
            st.markdown("---")
            st.markdown(f"### 🔮 Прогноз на завтра:\n # {res['prediction']:.2f}")
            
            # Блок уверенности
            with st.expander("📊 Показатели уверенности", expanded=True):
                st.write(f"Риск (GARCH): **{res['garch']:.2f}%**")
                st.write(f"Тренд (5 дней): **{res['trend']:+.2f}%**")
            
            # Рекомендация
            delta = res['prediction'] - res['current']
            if delta > 0 and vix_val < 25:
                st.success(f"🟢 КУПИТЬ (Ожидается {delta:+.2f})")
            elif delta < 0 or vix_val > 30:
                st.error(f"🔴 ПРОДАТЬ (Ожидается {delta:+.2f})")
            else:
                st.warning("⚪️ ДЕРЖАТЬ (Нейтрально)")
