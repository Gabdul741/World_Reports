import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from chronos import ChronosPipeline
from xgboost import XGBRegressor
from arch import arch_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import ta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Amazon Chronos Trader", layout="wide")
st.title("🚀 Amazon Chronos Trader")
st.markdown("**Прогноз на основе Amazon Chronos — Foundation Model**")

st.sidebar.title("Настройки")
asset = st.sidebar.selectbox("Актив:", [
    "WTI Нефть (CL=F)",
    "Серебро (SI=F)",
    "Золото (GC=F)",
    "S&P 500 (^GSPC)",
    "Apple (AAPL)",
    "Nvidia (NVDA)",
])
period = st.sidebar.selectbox("Период данных:", ["6mo", "1y", "2y", "3y"])
context_days = st.sidebar.slider("Контекст (дней):", 30, 120, 60)

tickers = {
    "WTI Нефть (CL=F)": "CL=F",
    "Серебро (SI=F)": "SI=F",
    "Золото (GC=F)": "GC=F",
    "S&P 500 (^GSPC)": "^GSPC",
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
}
ticker = tickers[asset]

if st.sidebar.button("Загрузить и предсказать"):
    with st.spinner("Загружаем данные..."):
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
        df = df[["Close", "Volume"]].dropna()
        df.columns = ["Price", "Volume"]
        vix = yf.download("^VIX", period=period, interval="1d", auto_adjust=True)["Close"].squeeze()
        df["VIX"] = vix
        df = df.dropna()

        today_price = float(df["Price"].iloc[-1])
        today_vix = float(df["VIX"].iloc[-1])
        today_date = df.index[-1]
        daily_move = today_price * (today_vix/100) / (252**0.5)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Цена сегодня", f"${today_price:.2f}")
        col2.metric("VIX", f"{today_vix:.2f}")
        col3.metric("Ожид. движение ±", f"${daily_move:.2f}")
        col4.metric("Данных дней", len(df))

        st.subheader(f"История цены — {asset}")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["Price"], color="blue", linewidth=1)
        ax.set_title(f"{asset}")
        ax.set_ylabel("Цена ($)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("GARCH Волатильность")
        returns = df["Price"].pct_change().dropna() * 100
        garch_model = arch_model(returns, vol="Garch", p=1, q=1)
        garch_result = garch_model.fit(disp="off")
        garch_vol = garch_result.conditional_volatility
        garch_today = float(garch_vol.iloc[-1])
        garch_annual = garch_today * (252**0.5)

        col1, col2, col3 = st.columns(3)
        col1.metric("GARCH дневная", f"{garch_today:.2f}%")
        col2.metric("GARCH годовая", f"{garch_annual:.2f}%")
        signal = "Спокойно" if today_vix < 20 else "Осторожно" if today_vix < 30 else "Опасно!"
        col3.metric("Сигнал VIX", signal)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        vix_daily = df["VIX"] / (252**0.5)
        ax2.plot(garch_vol.index, garch_vol, color="red", label="GARCH", linewidth=1)
        ax2.plot(vix_daily.index, vix_daily, color="blue", label="VIX/√252", linewidth=1)
        ax2.set_title("GARCH vs VIX")
        ax2.set_ylabel("Волатильность (%)")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.subheader("🚀 Amazon Chronos Прогноз")
        with st.spinner("Загружаем Chronos модель..."):
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cpu",
                dtype=torch.float32,
            )
            prices = df["Price"].values.flatten()
            context = torch.tensor(prices[-context_days:], dtype=torch.float32)
            forecast = pipeline.predict([context], prediction_length=1, num_samples=20)
            tomorrow_chronos = float(forecast[0].median())
            low_chronos = float(forecast[0].quantile(0.1))
            high_chronos = float(forecast[0].quantile(0.9))

        change_chronos = ((tomorrow_chronos - today_price) / today_price * 100)
        direction_chronos = "Рост" if change_chronos > 0 else "Падение"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chronos прогноз", f"${tomorrow_chronos:.2f}")
        col2.metric("Изменение", f"{change_chronos:.2f}%")
        col3.metric("Направление", direction_chronos)
        col4.metric("Точность модели", "Foundation")

        col1, col2, col3 = st.columns(3)
        col1.metric("Chronos минимум (10%)", f"${low_chronos:.2f}")
        col2.metric("Chronos медиана", f"${tomorrow_chronos:.2f}")
        col3.metric("Chronos максимум (90%)", f"${high_chronos:.2f}")

        st.subheader("📊 XGBoost прогноз для сравнения")
        df["GARCH_VOL"] = garch_vol.reindex(df.index).ffill().bfill()
        df["RSI"] = ta.momentum.RSIIndicator(df["Price"]).rsi()
        df["MACD"] = ta.trend.MACD(df["Price"]).macd()
        df["EMA20"] = ta.trend.EMAIndicator(df["Price"], window=20).ema_indicator()
        for lag in [1, 2, 3, 5]:
            df[f"lag_{lag}"] = df["Price"].shift(lag)
            df[f"ret_{lag}"] = df["Price"].pct_change(lag)
        df["Target"] = df["Price"].shift(-1)
        df_xgb = df.dropna()

        features = [c for c in df_xgb.columns if c != "Target"]
        X = df_xgb[features]
        y = df_xgb["Target"]
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        with st.spinner("Обучаем XGBoost..."):
            xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=6, random_state=42)
            xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = 100 - (mae / y_test.mean() * 100)

        # Прогноз на завтра — используем df (не df_xgb) чтобы включить сегодня
        df_today = df.copy()
        df_today["GARCH_VOL"] = garch_vol.reindex(df_today.index).ffill().bfill()
        df_today["RSI"] = ta.momentum.RSIIndicator(df_today["Price"]).rsi()
        df_today["MACD"] = ta.trend.MACD(df_today["Price"]).macd()
        df_today["EMA20"] = ta.trend.EMAIndicator(df_today["Price"], window=20).ema_indicator()
        for lag in [1, 2, 3, 5]:
            df_today[f"lag_{lag}"] = df_today["Price"].shift(lag)
            df_today[f"ret_{lag}"] = df_today["Price"].pct_change(lag)
        df_today = df_today.dropna()
        X_today = df_today[[c for c in df_today.columns if c != "Target"]] if "Target" in df_today.columns else df_today
        X_today = df_today[features] if all(f in df_today.columns for f in features) else df_today.iloc[:, :len(features)]

        tomorrow_xgb = float(xgb_model.predict(X.iloc[-1:])[0])
        change_xgb = ((tomorrow_xgb - today_price) / today_price * 100)

        col1, col2, col3 = st.columns(3)
        col1.metric("XGBoost прогноз", f"${tomorrow_xgb:.2f}")
        col2.metric("Изменение XGBoost", f"{change_xgb:.2f}%")
        col3.metric("Точность XGBoost", f"{accuracy:.1f}%")

        st.subheader("🎯 Консенсус моделей")
        tomorrow_ensemble = np.mean([tomorrow_chronos, tomorrow_xgb])
        change_ensemble = ((tomorrow_ensemble - today_price) / today_price * 100)
        spread = abs(tomorrow_chronos - tomorrow_xgb)
        direction_ensemble = "Рост" if change_ensemble > 0 else "Падение"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chronos", f"${tomorrow_chronos:.2f}", f"{change_chronos:.2f}%")
        col2.metric("XGBoost", f"${tomorrow_xgb:.2f}", f"{change_xgb:.2f}%")
        col3.metric("Консенсус", f"${tomorrow_ensemble:.2f}", f"{change_ensemble:.2f}%")
        col4.metric("Разброс", f"${spread:.2f}")

        if spread > today_price * 0.05:
            st.warning(f"⚠️ Высокая неопределённость! Разброс: ${spread:.2f} ({(spread/today_price*100):.1f}%)")
        elif spread > today_price * 0.02:
            st.info(f"ℹ️ Умеренная неопределённость. Разброс: ${spread:.2f}")
        else:
            st.success(f"✅ Консенсус моделей! Разброс: ${spread:.2f} ({(spread/today_price*100):.1f}%)")

        # ── История прогнозов ──────────────────────────────────────────────
        st.subheader("📅 История прогнозов — последние 7 дней")
        st.caption("Прогноз каждой модели сравнивается с реальным движением на следующий день. Последняя строка — сегодняшний прогноз на завтра.")

        history_rows = []
        history_days = min(7, len(df_xgb))

        with st.spinner("Считаем историю прогнозов..."):
            for i in range(-history_days, 0):
                date = df_xgb.index[i]
                real_today = float(df_xgb["Price"].iloc[i])
                # Следующий день: из df_xgb если есть, иначе из df (сегодняшняя цена)
                if i < -1:
                    real_next = float(df_xgb["Price"].iloc[i + 1])
                else:
                    real_next = float(df["Price"].iloc[-1])
                real_dir   = "🟢 Рост" if real_next > real_today else "🔴 Падение"
                real_chg   = (real_next - real_today) / real_today * 100

                # XGBoost прогноз
                xgb_pred = float(xgb_model.predict(X.iloc[[i]])[0])
                xgb_dir  = "🟢 Рост" if xgb_pred > real_today else "🔴 Падение"
                xgb_ok   = "✅" if xgb_dir == real_dir else "❌"

                # Chronos прогноз
                ctx_vals = df_xgb["Price"].values[:i][-context_days:]
                ctx = torch.tensor(ctx_vals, dtype=torch.float32)
                fc = pipeline.predict([ctx], prediction_length=1, num_samples=20)
                chr_pred = float(fc[0].median())
                chr_dir  = "🟢 Рост" if chr_pred > real_today else "🔴 Падение"
                chr_ok   = "✅" if chr_dir == real_dir else "❌"

                history_rows.append({
                    "Дата": date.strftime("%d.%m.%Y"),
                    "Цена": f"${real_today:.2f}",
                    "Факт след. день": f"${real_next:.2f}",
                    "Факт движение": f"{real_chg:+.2f}% {real_dir}",
                    "XGBoost прогноз": f"${xgb_pred:.2f} {xgb_dir}",
                    "XGBoost ✓": xgb_ok,
                    "Chronos прогноз": f"${chr_pred:.2f} {chr_dir}",
                    "Chronos ✓": chr_ok,
                })

            # Последняя строка — сегодня, прогноз на завтра
            history_rows.append({
                "Дата": f"🔮 {today_date.strftime('%d.%m.%Y')} (сегодня)",
                "Цена": f"${today_price:.2f}",
                "Факт след. день": "—",
                "Факт движение": "—",
                "XGBoost прогноз": f"${tomorrow_xgb:.2f} {'🟢 Рост' if change_xgb > 0 else '🔴 Падение'}",
                "XGBoost ✓": "⏳",
                "Chronos прогноз": f"${tomorrow_chronos:.2f} {'🟢 Рост' if change_chronos > 0 else '🔴 Падение'}",
                "Chronos ✓": "⏳",
            })

        hist_df = pd.DataFrame(history_rows)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Итоговая точность за неделю (без строки "сегодня")
        past_rows = history_rows[:-1]
        xgb_acc_week = sum(1 for r in past_rows if r["XGBoost ✓"] == "✅") / len(past_rows) * 100
        chr_acc_week = sum(1 for r in past_rows if r["Chronos ✓"] == "✅") / len(past_rows) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("XGBoost точность (7д)", f"{xgb_acc_week:.0f}%")
        col2.metric("Chronos точность (7д)", f"{chr_acc_week:.0f}%")
        col3.metric("Дней в выборке", len(past_rows))

        st.subheader("💡 Рекомендация")
        if today_vix < 15:
            risk = "Низкий риск - можно торговать"
        elif today_vix < 25:
            risk = "Умеренный риск - осторожно"
        elif today_vix < 35:
            risk = "Высокий риск - уменьшить позиции"
        else:
            risk = "Экстремальный риск - лучше не торговать!"

        if abs(change_ensemble) > 10:
            st.warning(f"⚠️ Экстремальный прогноз {change_ensemble:.1f}% — форс-мажор на рынке!")

        st.info(f"Уровень риска: {risk}")
        st.info(f"Прогноз Chronos: {direction_chronos} до ${tomorrow_chronos:.2f}")
        st.info(f"Консенсус: {direction_ensemble} до ${tomorrow_ensemble:.2f} (плюс-минус ${daily_move:.2f})")
        st.success("Модель: Amazon Chronos Foundation + XGBoost + GARCH + VIX")

st.markdown("---")
st.caption("Amazon Chronos Trader - Gabdul741 и Claude Sonnet 4.6 - Anthropic")
