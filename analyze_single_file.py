import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter

URL = "https://raw.githubusercontent.com/Lavr492/Hackaton_2026/refs/heads/my_start/exo/mexo1/cortex_exo_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place4_1.txt"

# ---------------------------------------------------------
# 2. Загрузка данных
# ---------------------------------------------------------
print("Загружаю файл из GitHub...")
df = pd.read_csv(URL, sep=r"\s+", engine="python")
df.columns = [c.strip().lstrip('#') for c in df.columns]

print("Первые строки файла:")
print(df.head())


# ---------------------------------------------------------
# 3. Сборка спектров по координатам
# ---------------------------------------------------------
groups = df.groupby(["X", "Y"])

coords = []
spectra = []

for (x, y), g in groups:
    g_sorted = g.sort_values("Wave")
    coords.append([x, y])
    spectra.append(g_sorted["Intensity"].values)

coords = np.array(coords)
spectra = np.vstack(spectra)
waves = g_sorted["Wave"].values

print(f"Количество спектров: {spectra.shape[0]}")
print(f"Длина спектра: {spectra.shape[1]}")


# ---------------------------------------------------------
# 4. Графики сырых спектров
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
for i in range(min(5, spectra.shape[0])):
    plt.plot(waves, spectra[i], alpha=0.7)
plt.title("Первые 5 сырых спектров")
plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity")
plt.grid(alpha=0.2)
plt.show()


# ---------------------------------------------------------
# 5. Предобработка
# ---------------------------------------------------------

# 5.1 Baseline correction
baseline_corrected = []
for spec in spectra:
    coefs = Polynomial.fit(waves, spec, deg=3).convert().coef
    poly = Polynomial(coefs)
    baseline = poly(waves)
    corrected = spec - baseline
    baseline_corrected.append(corrected)

baseline_corrected = np.vstack(baseline_corrected)

# 5.2 Smoothing
smoothed = savgol_filter(
    baseline_corrected,
    window_length=11,
    polyorder=3,
    axis=1
)

# 5.3 Normalization
norm_spectra = smoothed / np.max(smoothed, axis=1, keepdims=True)


# ---------------------------------------------------------
# 6. Графики после обработки
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
for i in range(min(5, norm_spectra.shape[0])):
    plt.plot(waves, norm_spectra[i], alpha=0.7)
plt.title("Первые 5 спектров после обработки")
plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Normalized intensity")
plt.grid(alpha=0.2)
plt.show()


# ---------------------------------------------------------
# 7. График сравнения одного спектра ДО и ПОСЛЕ
# ---------------------------------------------------------
idx = 0  # индекс спектра для сравнения

plt.figure(figsize=(10, 5))
plt.plot(waves, spectra[idx], label="Raw", alpha=0.7)
plt.plot(waves, norm_spectra[idx], label="Processed", alpha=0.7)
plt.title("Сравнение спектра: до и после обработки")
plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Intensity")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

print("Готово.")