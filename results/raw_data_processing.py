import io
import os
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Если используем XGBoost:
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost не установлен, блок XGBoost будет пропущен")


# ---------------------------------------------------------
# 1. Настройки
# ---------------------------------------------------------

GITHUB_USER = "Lavr492"
REPO = "Hackaton_2026"
DATA_BRANCH = "my_start"   # данные лежат здесь
FOLDERS = ["control", "endo", "exo"]

FILE_PATTERN = re.compile(
    r"(cortex|striatum).*?(control|endo|exo).*?(1500|2900)",
    re.IGNORECASE
)


# ---------------------------------------------------------
# 2. Загрузка файлов из GitHub
# ---------------------------------------------------------

def list_files_from_github(folder):
    url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO}/contents/data/{folder}?ref={DATA_BRANCH}"
    r = requests.get(url)
    r.raise_for_status()
    files = r.json()
    return [f["name"] for f in files if f["name"].endswith(".txt")]

def load_file_from_github(folder, filename):
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{DATA_BRANCH}/data/{folder}/{filename}"
    r = requests.get(raw_url)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------
# 3. Функции предобработки
# ---------------------------------------------------------

def baseline_correction(waves, spec, deg=3):
    coefs = Polynomial.fit(waves, spec, deg=deg).convert().coef
    poly = Polynomial(coefs)
    return spec - poly(waves)

def smooth(spec):
    return savgol_filter(spec, window_length=11, polyorder=3)

def normalize(spec):
    m = np.max(spec)
    return spec / m if m != 0 else spec

def spectrum_energy(spec):
    return np.sum(spec**2)


# ---------------------------------------------------------
# 4. Чтение одного файла (из текста)
# ---------------------------------------------------------

def load_single_file_from_text(text):
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
    df.columns = [c.strip().lstrip('#') for c in df.columns]

    groups = df.groupby(["X", "Y"])
    spectra = []
    coords = []

    for (x, y), g in groups:
        g_sorted = g.sort_values("Wave")
        coords.append([x, y])
        spectra.append(g_sorted["Intensity"].values)

    waves = g_sorted["Wave"].values
    spectra = np.vstack(spectra)

    return waves, coords, spectra


# ---------------------------------------------------------
# 5. Обработка всех файлов
# ---------------------------------------------------------

all_records = []
waves_global = None

for folder in FOLDERS:
    print(f"\n=== Папка: {folder} ===")
    file_list = list_files_from_github(folder)

    for fname in file_list:
        match = FILE_PATTERN.search(fname)
        if not match:
            print(f"⚠ Не удалось разобрать имя файла: {fname}")
            continue

        brain_area, cls, window = match.groups()
        brain_area = brain_area.lower()
        cls = cls.lower()
        window = int(window)

        print(f"Читаю файл из GitHub: data/{folder}/{fname}")
        text = load_file_from_github(folder, fname)
        waves, coords, spectra = load_single_file_from_text(text)

        if waves_global is None:
            waves_global = waves
        else:
            if not np.allclose(waves_global, waves):
                print("⚠ Разные сетки по Wave, файл будет пропущен:", fname)
                continue

        processed = []
        energies = []
        for spec in spectra:
          spec = baseline_correction(waves, spec)
          spec = smooth(spec)
          spec = normalize(spec)
          processed.append(spec)
          energies.append(spectrum_energy(spec))

        processed = np.vstack(processed)
        energies = np.array(energies)

        # фильтрация: оставим только спектры с энергией выше медианы
        thr = np.median(energies)
        mask_good = energies >= thr

        for (x, y), spec, en in zip(coords, processed, energies):
            if en < thr:
                continue
            all_records.append({
                "brain_area": brain_area,
                "class": cls,
                "window": window,
                "x": x,
                "y": y,
                **{f"v{i}": v for i, v in enumerate(spec)}
            })

# ---------------------------------------------------------
# 6. Итоговая таблица
# ---------------------------------------------------------

df_all = pd.DataFrame(all_records)
print("\nГотово! Размер итоговой таблицы:", df_all.shape)

os.makedirs("results", exist_ok=True)
df_all.to_csv("results/all_processed_spectra.csv", index=False)
print("Файл сохранён: results/all_processed_spectra.csv")

feature_cols = [c for c in df_all.columns if c.startswith("v")]
X = df_all[feature_cols].values
y = df_all["class"].values


# ---------------------------------------------------------
# 7. Средние спектры по классам
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))
for cls in FOLDERS:
    mean_spec = df_all[df_all["class"] == cls][feature_cols].mean().values
    plt.plot(waves_global, mean_spec, label=cls)

plt.title("Средние спектры по классам")
plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Normalized intensity")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/mean_spectra_by_class.png", dpi=200)
plt.close()

pd.DataFrame(
    {cls: df_all[df_all["class"] == cls][feature_cols].mean().values
     for cls in FOLDERS},
    index=waves_global
).to_csv("results/mean_spectra_by_class.csv")


# ---------------------------------------------------------
# 8. Сравнение окон 1500 vs 2900
# ---------------------------------------------------------

plt.figure(figsize=(10, 5))
for w in [1500, 2900]:
    if (df_all["window"] == w).any():
        mean_spec = df_all[df_all["window"] == w][feature_cols].mean().values
        plt.plot(waves_global, mean_spec, label=f"window {w}")

plt.title("Сравнение окон 1500 и 2900")
plt.xlabel("Raman shift (cm⁻¹)")
plt.ylabel("Normalized intensity")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/mean_spectra_by_window.png", dpi=200)
plt.close()


# ---------------------------------------------------------
# 9. PCA
# ---------------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
for cls in FOLDERS:
    mask = (y == cls)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=10, label=cls)

plt.title("PCA: разделение классов")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/pca_classes.png", dpi=200)
plt.close()

pd.DataFrame(
    X_pca, columns=["PC1", "PC2"]
).assign(label=y).to_csv("results/pca_embedding.csv", index=False)

pd.DataFrame({
    "component": ["PC1", "PC2"],
    "explained_variance_ratio": pca.explained_variance_ratio_
}).to_csv("results/pca_explained_variance.csv", index=False)


# ---------------------------------------------------------
# 10. t-SNE
# ---------------------------------------------------------

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="random", random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
for cls in FOLDERS:
    mask = (y == cls)
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=10, label=cls)

plt.title("t-SNE: локальные кластеры")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("results/tsne_classes.png", dpi=200)
plt.close()

pd.DataFrame(
    X_tsne, columns=["TSNE1", "TSNE2"]
).assign(label=y).to_csv("results/tsne_embedding.csv", index=False)