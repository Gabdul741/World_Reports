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
# ========== БЭКТЕСТ (обкатка) ==========
# ========== УПРОЩЁННЫЙ БЭКТЕСТ (без ежедневного переобучения) ==========
st.sidebar.markdown("---")
if st.sidebar.button("🔁 Быстрый бэктест (обкатка)"):
    st.subheader("📊 Бэктест стратегии на истории")
    st.info("Модель обучается один раз на последних 3 годах, затем проверяется на следующих 6 месяцах")
    
    results_backtest = []
    TEST_MONTHS = 6
    
    for ticker in selected:
        with st.spinner(f"Обрабатываю {TICKERS[ticker]}..."):
            # Скачиваем длинную историю (4 года)
            end = datetime.now()
            start = end - timedelta(days=365*4)
            df_full = yf.download(ticker, start=start, end=end, progress=False)
            if df_full.empty or len(df_full) < 500:
                st.warning(f"Недостаточно данных для {TICKERS[ticker]}")
                continue
            
            # Разделяем на обучение и тест (первые 3 года обучение, последние 6 месяцев тест)
            split_date = datetime.now() - timedelta(days=30*TEST_MONTHS)
            train = df_full[df_full.index < split_date]
            test = df_full[df_full.index >= split_date]
            
            if len(train) < 100 or len(test) < 50:
                st.warning(f"Недостаточно тестовых данных для {TICKERS[ticker]}")
                continue
            
            # Обучаем модель на тренировочных данных
            train_df = train.reset_index()[["Date", "Close"]]
            train_df.columns = ["ds", "y"]
            model = Prophet(daily_seasonality=True)
            model.fit(train_df)
            
            # Делаем прогноз на тестовый период
            test_df = test.reset_index()[["Date"]]
            test_df.columns = ["ds"]
            forecast = model.predict(test_df)
            
            # Считаем, сколько раз сигнал совпал с движением вверх/вниз
            # (упрощённо: сравнение прогноза с реальностью)
            actual = test["Close"].values
            predicted = forecast["yhat"].values
            correct_up = 0
            correct_down = 0
            total = len(actual) - 1
            
            for i in range(total):
                if predicted[i] > actual[i] and actual[i+1] > actual[i]:
                    correct_up += 1
                elif predicted[i] < actual[i] and actual[i+1] < actual[i]:
                    correct_down += 1
            
            accuracy = (correct_up + correct_down) / total * 100 if total > 0 else 0
            
            st.metric(f"{TICKERS[ticker]} — точность направления прогноза", f"{accuracy:.1f}%")
            if accuracy > 55:
                st.success("✅ Прогноз лучше случайного")
            else:
                st.warning("⚠️ Прогноз слабее случайного")
            
            results_backtest.append({
                "Актив": TICKERS[ticker],
                "Точность направления": f"{accuracy:.1f}%",
                "Тестовых дней": total,
                "Оценка": "✅ лучше случайного" if accuracy > 55 else "⚠️ хуже случайного"
            })
    
    if results_backtest:
        st.subheader("📋 Сводка по активам")
        st.dataframe(pd.DataFrame(results_backtest), use_container_width=True)
# ========== МОДУЛЬ АНСАМБЛЯ (XGBoost/LSTM/SARIMAX) ==========
st.sidebar.markdown("---")
ensemble_on = st.sidebar.checkbox("🔮 Ансамбль (XGBoost + LSTM + SARIMAX)", value=False)

if ensemble_on and selected:
    st.subheader("🤖 Мнения моделей (XGBoost / LSTM / SARIMAX)")
    
    # --- Загрузка внешних данных для SARIMAX (экзогенные переменные) ---
    @st.cache_data(ttl=3600)
    def get_exogenous_data():
        # Индекс доллара DXY
        dxy = yf.download("DX-Y.NYB", period="2y", progress=False)['Close']
        # Запасы нефти в США (Cushing, OK Crude Oil Futures Contract)
        # yfinance тикер: "CL=F" — контракт, запасы отдельно сложно, используем тикер WCBRENT для proxy
        # Для простоты пока берём второй тикер — ETF нефти USO
        uso = yf.download("USO", period="2y", progress=False)['Close']
        # Объединяем в один DataFrame
        exog = pd.DataFrame({'DXY': dxy, 'USO': uso}).dropna()
        return exog
    
    try:
        exog_data = get_exogenous_data()
    except:
        exog_data = None
        st.info("⚠️ Внешние данные для SARIMAX не загружены, используется авторегрессионный SARIMAX")
    
    for ticker in selected:
        st.markdown(f"### {TICKERS[ticker]} ({ticker})")
        
        # Скачиваем историю
        end = datetime.now()
        start = end - timedelta(days=365*3)  # глубокая история для LSTM / SARIMAX
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty or len(df) < 100:
            st.warning(f"❌ Недостаточно данных для {TICKERS[ticker]}")
            continue
        
        prices = df['Close'].values
        dates = df.index
        
        # 1. XGBoost (быстрое дерево, подходит для волатильных рядов)
        try:
            from xgboost import XGBRegressor
            # Используем предсказание по последним 20 дням
            window = 20
            if len(prices) > window:
                X = np.arange(len(prices)).reshape(-1,1)
                y = prices
                split = int(len(X) * 0.9)
                model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
                model_xgb.fit(X[:split], y[:split])
                next_day = np.array([[len(prices)]])
                pred_xgb = model_xgb.predict(next_day)[0]
                change_xgb = (pred_xgb - prices[-1]) / prices[-1] * 100
                signal_xgb = "🔴 Продавать" if change_xgb < -1 else ("🟢 Покупать" if change_xgb > 1 else "🟡 Держать")
                st.write(f"**XGBoost:** прогноз ${pred_xgb:.2f}  ({change_xgb:+.2f}%) → {signal_xgb}")
            else:
                st.write("**XGBoost:** недостаточно данных")
        except Exception as e:
            st.write(f"**XGBoost:** ошибка ({e})")
        
        # 2. LSTM (нейросеть — долго, но для краткосрочного прогноза часто избыточна)
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense
            # Упрощённая LSTM (чтобы не грузить CPU/GPU)
            # Можно пропустить, если времени нет, но для демонстрации добавим заглушку
            # Для реального использования требуется масштабирование и много эпох
            # Пока заглушка: используем XGBoost как прокси, но в боевой версии нужно полноценно реализовать
            # (поскольку это сложная модель, мы её здесь не разворачиваем полностью, а отмечаем как требующую доработки)
            st.write(f"**LSTM:** ⏳ требует отдельного раздела (сложная нейросеть)")
        except:
            st.write(f"**LSTM:** библиотека не установлена (`pip install tensorflow`)")
        
        # 3. SARIMAX с экзогенными переменными
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            # Берём последние 200 точек для скорости
            y_series = pd.Series(prices[-200:], index=dates[-200:])
            if exog_data is not None:
                # Выравниваем экзогенные данные по датам
                exog_aligned = exog_data.reindex(y_series.index).dropna()
                if len(exog_aligned) > 10:
                    model_sarimax = SARIMAX(y_series, exog=exog_aligned, order=(1,1,1), seasonal_order=(0,0,0,0))
                    res_sarimax = model_sarimax.fit(disp=False)
                    forecast = res_sarimax.forecast(steps=1, exog=exog_aligned.iloc[[-1]])
                    pred_sarimax = forecast.iloc[0]
                    change_sarimax = (pred_sarimax - prices[-1]) / prices[-1] * 100
                    signal_sarimax = "🔴 Продавать" if change_sarimax < -1 else ("🟢 Покупать" if change_sarimax > 1 else "🟡 Держать")
                    st.write(f"**SARIMAX:** прогноз ${pred_sarimax:.2f}  ({change_sarimax:+.2f}%) → {signal_sarimax}")
                else:
                    st.write("**SARIMAX:** недостаточно выровненных внешних данных")
            else:
                st.write("**SARIMAX:** внешние данные отсутствуют")
        except Exception as e:
            st.write(f"**SARIMAX:** ошибка ({e})")
        
        st.markdown("---")
