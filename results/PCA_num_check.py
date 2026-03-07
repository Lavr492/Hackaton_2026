import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_csv("results/all_processed_spectra.csv")

# Удаляем не-спектральные столбцы
X = df.drop(columns=["class", "brain_area", "window", "x", "y"])
y = df["class"]

# Кодируем классы
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.25, stratify=y_encoded, random_state=42
)

# Диапазон PCA компонент
components_range = range(2, 51)

accuracies = []

for n in components_range:
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train_full)
    X_test = pca.transform(X_test_full)

    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        verbose=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).flatten()

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"PCA={n}, accuracy={acc:.4f}")

# График
plt.figure(figsize=(10,5))
plt.plot(components_range, accuracies, marker='o')
plt.xlabel("Число PCA-компонент")
plt.ylabel("Accuracy")
plt.title("Зависимость accuracy от числа PCA-компонент (CatBoost)")
plt.grid(True)
plt.show()
