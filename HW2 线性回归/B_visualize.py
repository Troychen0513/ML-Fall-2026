from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1. 读取训练集
data_path = Path(
    r"C:\Users\26491\Desktop\ML-Fall-2026\HW2 线性回归\data\B_train.csv"
)
figure_dir = Path(__file__).resolve().parent / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

features = [
    "area_sqm", "building_age_years", "metro_m", "center_km", "bedrooms",
    "floor", "amenities_1km", "renovation_score", "has_elevator", "sunlight_hours",
]
feature_names = [
    "面积", "楼龄", "距地铁距离", "距中心距离", "卧室数",
    "楼层", "周边配套数", "装修评分", "有无电梯", "日照时长",
]

df = pd.read_csv(data_path)
X = df[features]
rent = df["monthly_rent"].to_numpy()

if not np.isfinite(df[features + ["monthly_rent"]].to_numpy(dtype=float)).all():
    raise ValueError("数据包含缺失值或无穷值，请先检查。")

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# 2. 特征相关性热力图
correlation = X.corr()
n_features = len(features)

fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
heatmap = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)

ax.set_xticks(range(n_features), feature_names, rotation=45, ha="right")
ax.set_yticks(range(n_features), feature_names)
ax.set_title("训练集输入特征的 Pearson 相关系数")

for i in range(n_features):
    for j in range(n_features):
        value = correlation.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center", color=color, fontsize=9,
        )

fig.colorbar(heatmap, ax=ax, label="Pearson r", shrink=0.8)
fig.savefig(
    figure_dir / "B_特征相关性热力图.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)


# 标准化后进行 PCA，租金不参与拟合
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=3, svd_solver="full")
coordinates = pca.fit_transform(X_scaled)

variance = pca.explained_variance_ratio_
weights = pca.components_.T

print(f"前两个主成分累计解释方差：{variance[:2].sum():.2%}")
print(f"前三个主成分累计解释方差：{variance.sum():.2%}")


# PCA 方向系数热力图
fig, ax = plt.subplots(figsize=(6.5, 7), layout="constrained")
limit = np.abs(weights).max()
heatmap = ax.imshow(
    weights, cmap="RdBu_r",
    vmin=-limit, vmax=limit, aspect="auto",
)

ax.set_xticks(range(3), ["PC1", "PC2", "PC3"])
ax.set_yticks(range(n_features), feature_names)
ax.set_title("前三个主成分的方向系数")

for i in range(n_features):
    for j in range(3):
        value = weights[i, j]
        color = "white" if abs(value) > 0.6 * limit else "black"
        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center", color=color,
        )

fig.colorbar(heatmap, ax=ax, label="标准化特征的方向系数", shrink=0.8)
fig.savefig(
    figure_dir / "B_PCA方向系数热力图.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)


# PCA投影图
order = np.random.default_rng(42).permutation(len(df))
rent_norm = plt.Normalize(rent.min(), rent.max())

fig, axes = plt.subplots(
    1, 2, figsize=(13, 5.5), layout="constrained"
)

for ax, (a, b) in zip(axes, [(0, 1), (1, 2)]):
    scatter = ax.scatter(
        coordinates[order, a],
        coordinates[order, b],
        c=rent[order],
        cmap="viridis",
        norm=rent_norm,
        s=9,
        alpha=0.6,
        linewidths=0,
    )
    ax.set_xlabel(f"PC{a + 1}（{variance[a]:.2%}）")
    ax.set_ylabel(f"PC{b + 1}（{variance[b]:.2%}）")
    ax.set_title(f"PC{a + 1}–PC{b + 1}：按月租金着色")
    ax.set_aspect("equal", adjustable="box")

fig.colorbar(
    scatter, ax=axes.tolist(),
    label="月租金（元/月）", shrink=0.9,
)
fig.savefig(
    figure_dir / "B_PCA投影图.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)

plt.show()