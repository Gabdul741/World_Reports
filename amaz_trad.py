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
import os, json
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

FORECAST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"forecasts_{ticker.replace('=','_').replace('^','')}.json")

def next_trading_day(d, trading_days_set):
    """Следующий торговый день из известных дат."""
    nxt = d + pd.Timedelta(days=1)
    for _ in range(10):
        if nxt in trading_days_set or nxt.weekday() < 5:
            if nxt.weekday() < 5:
                return nxt
        nxt += pd.Timedelta(days=1)
    return nxt

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
        today_key = today_date.strftime("%Y-%m-%d")
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

        # ── Трекер прогнозов (скользящее окно 7 дней) ─────────────────────
        st.subheader("📅 Трекер прогнозов — 7 дней")
        st.caption("Прогнозы сохраняются ежедневно. Окно скользит автоматически.")

        # Загружаем сохранённые прогнозы
        if os.path.exists(FORECAST_FILE):
            with open(FORECAST_FILE, "r") as f:
                saved = json.load(f)
        else:
            saved = {}

        # Сохраняем сегодняшний прогноз на завтра
        saved[today_key] = {
            "xgb": round(tomorrow_xgb, 2),
            "chronos": round(tomorrow_chronos, 2),
            "price": round(today_price, 2),
        }
        with open(FORECAST_FILE, "w") as f:
            json.dump(saved, f, indent=2)

        # Строим окно: все календарные дни от -4 торговых до +2 торговых,
        # включая выходные для непрерывности картины
        trading_days = list(df.index)
        today_idx = len(trading_days) - 1

        # Крайние точки окна
        start_trading = trading_days[max(0, today_idx - 4)]
        last_trading  = trading_days[today_idx]

        # Два следующих торговых дня
        future = []
        last = last_trading
        for _ in range(2):
            nxt = last + pd.Timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += pd.Timedelta(days=1)
            future.append(nxt)
            last = nxt

        end_date = future[-1]

        # Все календарные дни в окне (включая выходные)
        window = []
        cur = start_trading
        while cur <= end_date:
            window.append(cur)
            cur += pd.Timedelta(days=1)

        # Реальная дата сегодня (может быть выходной)
        real_today = pd.Timestamp.now().normalize()

        # Индекс цен по дате для быстрого доступа
        price_by_date = {d.strftime("%Y-%m-%d"): float(df["Price"].loc[d]) 
                         for d in df.index if d in df.index}

        # Строим 3 строки
        dates_row     = {}
        facts_row     = {}
        forecasts_row = {}

        for d in window:
            key   = d.strftime("%Y-%m-%d")
            label = d.strftime("%a %d.%m")

            # Строка 1: Дата
            if key == today_key:
                dates_row[label] = f"📍 {label}"
            else:
                dates_row[label] = label

            # Строка 2: Факт (реальная цена если известна)
            if key in price_by_date:
                facts_row[label] = f"${price_by_date[key]:.2f}"
            else:
                facts_row[label] = "—"

            # Строка 3: Прогноз (сделан накануне)
            prev = d - pd.Timedelta(days=1)
            while prev.weekday() >= 5:
                prev -= pd.Timedelta(days=1)
            prev_key = prev.strftime("%Y-%m-%d")

            if prev_key in saved:
                p = saved[prev_key]
                xgb_p   = p["xgb"]
                chr_p   = p["chronos"]
                base    = p["price"]
                dir_x   = "🟢" if xgb_p >= base else "🔴"
                dir_c   = "🟢" if chr_p >= base else "🔴"

                # Если факт известен — сравниваем направление
                if key in price_by_date and key != today_key:
                    fact_p    = price_by_date[key]
                    fact_dir  = "🟢" if fact_p >= base else "🔴"
                    ok_x = "✅" if dir_x == fact_dir else "❌"
                    ok_c = "✅" if dir_c == fact_dir else "❌"
                    forecasts_row[label] = f"XGB ${xgb_p:.2f}{dir_x}{ok_x} | CHR ${chr_p:.2f}{dir_c}{ok_c}"
                else:
                    mark = "⏳" if key == today_key else "🔮"
                    forecasts_row[label] = f"XGB ${xgb_p:.2f}{dir_x}{mark} | CHR ${chr_p:.2f}{dir_c}{mark}"
            else:
                # Будущие дни: прогноз на завтра уже посчитан
                if prev_key == today_key:
                    dir_x = "🟢" if tomorrow_xgb >= today_price else "🔴"
                    dir_c = "🟢" if tomorrow_chronos >= today_price else "🔴"
                    forecasts_row[label] = f"XGB ${tomorrow_xgb:.2f}{dir_x}🔮 | CHR ${tomorrow_chronos:.2f}{dir_c}🔮"
                else:
                    forecasts_row[label] = "—"

        # Собираем DataFrame 3 строки
        cols = [d.strftime("%a %d.%m") for d in window]
        tracker_df = pd.DataFrame([
            {c: dates_row.get(c, c) for c in cols},
            {c: facts_row.get(c, "—") for c in cols},
            {c: forecasts_row.get(c, "—") for c in cols},
        ], index=["📅 Дата", "💰 Факт", "🔮 Прогноз"])

        st.dataframe(tracker_df, use_container_width=True)

        # Точность за окно (прошлые дни где есть и факт и прогноз)
        correct_xgb = correct_chr = total = 0
        for d in window[:5]:
            label = d.strftime("%a %d.%m")
            fc = forecasts_row.get(label, "")
            if "✅" in fc or "❌" in fc:
                total += 1
                parts = fc.split("|")
                if len(parts) == 2:
                    correct_xgb += "✅" in parts[0]
                    correct_chr  += "✅" in parts[1]

        if total > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("XGBoost точность (окно)", f"{correct_xgb/total*100:.0f}%")
            col2.metric("Chronos точность (окно)", f"{correct_chr/total*100:.0f}%")
            col3.metric("Дней с прогнозом", total)

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
