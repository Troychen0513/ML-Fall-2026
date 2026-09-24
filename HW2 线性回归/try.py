import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# 442个样本 × 10个输入特征
data = load_diabetes(scaled=False)
X = StandardScaler().fit_transform(data.data)
colour = data.target  # 只用于着色

pca = PCA(n_components=2, svd_solver="full")

embeddings = {
    "PCA": pca.fit_transform(X),

    "t-SNE": TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=42,
    ).fit_transform(X),

    "UMAP": umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,
    ).fit_transform(X),
}

fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.5), layout="constrained"
)

for ax, (name, Z) in zip(axes, embeddings.items()):
    sc = ax.scatter(
        Z[:, 0], Z[:, 1],
        c=colour, cmap="viridis", s=18, alpha=0.75
    )
    ax.set_title(name)
    ax.set_xlabel(f"{name} 1")
    ax.set_ylabel(f"{name} 2")
    ax.set_aspect("equal", adjustable="box")

fig.colorbar(sc, ax=axes.tolist(), label="Outcome (colour only)")
print("PCA前两维解释方差：", pca.explained_variance_ratio_.sum())
plt.show()