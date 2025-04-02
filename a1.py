import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета
url = "/content/AirQuality.csv"
data = pd.read_csv(url, sep=";", decimal=",")

# Убираем ненужные столбцы с датой и временем
data = data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'])

# Проверка на пропуски в данных
print(data.isnull().sum())

# Убираем строки с пропущенными значениями
data = data.dropna()

# Преобразуем Concentration NO2 в 3 категории
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
data['NO2_category'] = kbins.fit_transform(data[['NO2(GT)']])

# Используем все доступные признаки (кроме 'NO2_category' как целевой переменной)
X = data.drop(columns=['NO2_category'])
y = data['NO2_category']

# Преобразуем все данные в числовой формат
X = X.apply(pd.to_numeric, errors='coerce')

# Разделим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели (GradientBoostingClassifier)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Калибровка с использованием Platt Scaling
# Используем обученную модель с помощью CalibratedClassifierCV
platt = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
platt.fit(X_train, y_train)
probs_platt = platt.predict_proba(X_test)

# Калибровка с использованием Isotonic Regression
isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
isotonic.fit(X_train, y_train)
probs_isotonic = isotonic.predict_proba(X_test)

# Оценка предсказаний с использованием confusion matrix
y_pred = model.predict(X_test)
y_pred_platt = platt.predict(X_test)
y_pred_isotonic = isotonic.predict(X_test)

# Построение confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Построим confusion matrix для всех моделей
plot_confusion_matrix(y_test, y_pred, "Confusion Matrix (Raw Gradient Boosting)")
plot_confusion_matrix(y_test, y_pred_platt, "Confusion Matrix (Platt Scaling)")
plot_confusion_matrix(y_test, y_pred_isotonic, "Confusion Matrix (Isotonic Regression)")

# Вывод метрик
print(f"Accuracy (Raw Gradient Boosting): {accuracy_score(y_test, y_pred):.4f}")
print(f"Accuracy (Platt Scaling): {accuracy_score(y_test, y_pred_platt):.4f}")
print(f"Accuracy (Isotonic Regression): {accuracy_score(y_test, y_pred_isotonic):.4f}")
