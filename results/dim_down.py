import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pca = pd.read_csv("results/pca.csv")
tsne = pd.read_csv("results/tsne.csv")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
sns.scatterplot(
    data=pca,
    x="PC1",
    y="PC2",
    hue="label",
    palette="tab10",
    s=8,
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title("PCA: глобальная структура данных")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

# t-SNE
sns.scatterplot(
    data=tsne,
    x="TSNE1",
    y="TSNE2",
    hue="label",
    palette="tab10",
    s=8,
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("t-SNE: локальные кластеры")
axes[1].set_xlabel("TSNE1")
axes[1].set_ylabel("TSNE2")

plt.tight_layout()
plt.show()
