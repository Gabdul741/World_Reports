# wti_sarimax_module.py
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta

def get_wti_forecast_with_sarimax(ticker="CL=F", days_back=750, forecast_horizon=1):
    """
    Возвращает прогноз SARIMAX для WTI с метриками качества
    """
    end = datetime.now()
    start = end - timedelta(days=days_back)
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty or len(df) < 100:
        return None, None, None, None, None, None
    
    prices = df['Close'].values
    dates = df.index
    last_price = prices[-1]
    
    # Обучаем SARIMAX (экзогенные — можно добавить позже)
    y_series = pd.Series(prices, index=dates)
    model = SARIMAX(y_series, order=(1,0,1), seasonal_order=(0,0,0,0))
    res = model.fit(disp=False, maxiter=200)
    
    # Прогноз
    forecast = res.forecast(steps=forecast_horizon)
    pred_price = forecast.iloc[0]
    change_pct = (pred_price - last_price) / last_price * 100
    
    # Метрики
    aic = res.aic
    bic = res.bic
    
    # Доверительный интервал
    forecast_result = res.get_forecast(steps=forecast_horizon)
    conf_int = forecast_result.conf_int()
    width = (conf_int.iloc[0, 1] - conf_int.iloc[0, 0]) / 2
    width_pct = (width / pred_price) * 100
    
    return pred_price, change_pct, aic, bic, width_pct, last_price
