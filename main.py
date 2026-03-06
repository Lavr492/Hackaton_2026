import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Загрузка данных
file_path = 'cerebellum_right_control_3group_633nm_center2900_obj100_power100_1s_5acc_map35x15_step2_place3_4.txt'
df = pd.read_csv(file_path, sep='\t', skipinitialspace=True)
# Убираем символ # из названий колонок
df.columns = df.columns.str.replace('#', '').str.strip()

print("Первые строки данных:")
print(df.head())
print("\nРазмер датасета:", df.shape)

# 2. Преобразование в матрицу "спектры × признаки"
# Уникальные значения X (координаты точек)
unique_x = df['X'].unique()
print(f"Найдено {len(unique_x)} уникальных точек (спектров)")

# Предполагаем, что волновые числа одинаковы для всех точек
waves = df['Wave'].unique()
print(f"Количество волновых чисел (признаков): {len(waves)}")

# Создаём массив признаков
X_spectra = []
x_labels = []

for x in unique_x:
    subset = df[df['X'] == x].sort_values('Wave')  # сортируем по волновому числу (на всякий случай)
    intensities = subset['Intensity'].values
    X_spectra.append(intensities)
    x_labels.append(x)

X = np.array(X_spectra)  # shape (n_spectra, n_features)
print(f"Матрица признаков: {X.shape}")

# 3. Создание целевых меток (делим на 3 равные группы по порядку X)
n_spectra = len(unique_x)
spectra_per_class = n_spectra // 3
y = np.array([0]*spectra_per_class + [1]*spectra_per_class + [2]*spectra_per_class)
# Если остаются лишние (n_spectra не кратно 3), можно отбросить или распределить иначе
if len(y) < n_spectra:
    y = np.append(y, [2]*(n_spectra - len(y)))  # добавим остаток в последний класс
print("Распределение по классам:", np.bincount(y))

# 4. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Обучение Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 6. Оценка
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nТочность на тестовой выборке: {acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Кросс-валидация
cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Кросс-валидация (5-fold): средняя точность = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 7. Визуализация средних спектров по классам
plt.figure(figsize=(12, 6))
for class_label in np.unique(y):
    mean_spectrum = X[y == class_label].mean(axis=0)
    plt.plot(waves, mean_spectrum, label=f'Class {class_label}')
plt.xlabel('Wave number')
plt.ylabel('Mean Intensity')
plt.title('Average spectra for each class')
plt.legend()
plt.grid(True)
plt.show()

# Важность признаков (топ-10 волновых чисел)
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), waves[indices])
plt.xlabel('Feature importance')
plt.title('Top 10 important wave numbers')
plt.show()