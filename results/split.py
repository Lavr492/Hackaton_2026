import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загружаем PCA
pca = pd.read_csv("results/pca.csv")

X = pca[["PC1", "PC2"]]
y = pca["label"]

# Масштабирование (важно для SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42
)
