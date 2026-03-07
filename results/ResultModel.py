import time
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool


# Если X и y уже определены в сессии, используем их; иначе — создаём заглушку 
try:
    X  # проверка существования
    y
except NameError:
    # Заменить на загрузку реальных данных
    print("X/y не найдены в сессии — создаю примерные данные (замени на свои).")
    n_samples = 2000
    X = pd.DataFrame(np.random.randn(n_samples, 1015), columns=[f'v{i}' for i in range(1015)])
    y = np.random.randint(0, 3, size=n_samples)

# Если валидационные наборы не определены — делаем stratified split
try:
    X_train; X_val; y_train; y_val
    print("X_train/X_val уже есть в сессии — использую их.")
except NameError:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    print(f"Сделан split: train={len(X_train)} val={len(X_val)}")

# Параметры CatBoost для быстрого прогона
params = {
    "iterations": 400,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "MultiClass",
    "random_seed": 42,
    "verbose": 100,
    "train_dir": "catboost_logs",
    "use_best_model": True
}

model = CatBoostClassifier(**params)

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

start = time.time()
model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=50,
    verbose=100
)
elapsed = time.time() - start
print(f"Обучение завершено. Время (s): {elapsed:.1f}")

# Сохранение модели
model.save_model("catboost_model.cbm")
print("Модель сохранена: catboost_model.cbm")
