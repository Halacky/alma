import numpy as np
import pandas as pd
import yfinance as yf
import pymc as pm
import matplotlib.pyplot as plt

# Загрузка данных о ценах акций компании Apple
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2020-01-01')

# Используем только данные о ценах закрытия
close_prices = data['Close'].values

# Нормализация данных
normalized_prices = (close_prices - np.mean(close_prices)) / np.std(close_prices)

# Определение временных шагов
n = len(normalized_prices)
time = np.arange(n)

# Построение байесовской модели с использованием PyMC
with pm.Model() as model:
    # Параметры модели (коэффициенты и шум)
    a = pm.Normal('a', mu=0, sigma=10)  # Заменили 'sd' на 'sigma'
    b = pm.HalfNormal('b', sigma=10)  # Заменили 'sd' на 'sigma'
    sigma = pm.HalfNormal('sigma', sigma=10)  # Заменили 'sd' на 'sigma'

    # Шум скрытых состояний (динамика состояния)
    x = pm.Normal('x', mu=0, sigma=1, shape=n)  # Заменили 'sd' на 'sigma'

    # Модель наблюдений (отношение между скрытыми состояниями и наблюдаемыми данными)
    y = pm.Normal('y', mu=a * x, sigma=sigma, observed=normalized_prices)  # Заменили 'sd' на 'sigma'

    # Сэмплирование
    trace = pm.sample(2000, tune=1000, return_inferencedata=False)

# Визуализация результатов
pm.plot_trace(trace)
plt.show()

# Оценка предсказаний
a_posterior = trace['a'].mean()
x_posterior = trace['x'].mean(axis=0)
predicted_prices = a_posterior * x_posterior * np.std(close_prices) + np.mean(close_prices)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(time, close_prices, label='Real Data (Close Prices)', color='blue', alpha=0.5)
plt.plot(time, predicted_prices, label='Predicted Prices (Bayesian Model)', color='green')
plt.title(f'Bayesian Prediction of {ticker} Stock Prices')
plt.legend()
plt.show()
