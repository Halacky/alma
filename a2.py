import lightgbm as lgb
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка датасета California Housing
california = fetch_california_housing()
X, y = california.data, california.target

# Масштабирование признаков для улучшения производительности модели
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразование в формат данных LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Установка параметров модели для quantile regression
params = {
    'objective': 'quantile',  # Используем quantile regression
    'alpha': 0.9,             # Квантиль 90% (для верхней границы доверительного интервала)
    'metric': 'l2',           # Используем MSE как метрику
    'boosting_type': 'gbdt',  # Градиентный бустинг
    'num_leaves': 31,         # Количество листьев в дереве
    'learning_rate': 0.05,    # Шаг обучения
    'verbose': -1
}

# Используем callbacks для ранней остановки
callbacks = [lgb.early_stopping(stopping_rounds=100)]

# Обучение модели
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, callbacks=callbacks)

# Прогнозирование на тестовых данных
y_pred_quantile = model.predict(X_test, num_iteration=model.best_iteration)

# Пример вывода прогнозов
print(f"Прогноз для тестового набора данных: {y_pred_quantile[:5]}")

# Для построения доверительных интервалов обучим модель для 10% (нижний) и 90% (верхний) квантилей:
params['alpha'] = 0.1  # Нижний квантиль (10%)
model_lower = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, callbacks=callbacks)

params['alpha'] = 0.9  # Верхний квантиль (90%)
model_upper = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, callbacks=callbacks)

# Прогнозирование для нижней и верхней границы
y_pred_lower = model_lower.predict(X_test, num_iteration=model_lower.best_iteration)
y_pred_upper = model_upper.predict(X_test, num_iteration=model_upper.best_iteration)

# Доверительный интервал
lower_bound = y_pred_lower
upper_bound = y_pred_upper

# Вывод доверительных интервалов для первых 5 точек
print("Доверительный интервал для первых 5 тестовых точек:")
for i in range(5):
    print(f"Точка {i}: {lower_bound[i]} <= Предсказание <= {upper_bound[i]}")
