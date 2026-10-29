from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


BASE_DIR = Path("/Users/tuyuchen/Downloads/HW3 感知机算法")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "preprocessing_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

colors = {-1: "#377eb8", 1: "#f28e2b"}

datasets = {}

# 读取数据并检查
for name in ["A", "B"]:
    train = pd.read_csv(DATA_DIR / f"{name}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{name}_test.csv")
    datasets[name] = {"train": train, "test": test}

    print(f"\n数据集 {name}")
    print(f"训练集形状：{train.shape}")
    print(f"测试集形状：{test.shape}")
    print("训练集数据类型：")
    print(train.dtypes)
    print("训练集缺失值数量：")
    print(train.isna().sum())
    print("测试集缺失值数量：")
    print(test.isna().sum())
    print(f"训练集重复行数量：{train.duplicated().sum()}")
    print(f"测试集重复行数量：{test.duplicated().sum()}")
    print("训练集标签数量：")
    print(train["y"].value_counts().sort_index())
    print("测试集标签数量：")
    print(test["y"].value_counts().sort_index())

A_train = datasets["A"]["train"]
A_test = datasets["A"]["test"]
B_train = datasets["B"]["train"]
B_test = datasets["B"]["test"]

# A 散点图和 B PCA 二维图
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

ax = axes[0]
for label in [-1, 1]:
    train_part = A_train[A_train["y"] == label]
    test_part = A_test[A_test["y"] == label]
    ax.scatter(
        train_part["x1"], train_part["x2"], s=22, alpha=0.60,
        color=colors[label], label=f"train, y={label}"
    )
    ax.scatter(
        test_part["x1"], test_part["x2"], s=38, marker="x",
        color=colors[label], label=f"test, y={label}"
    )
ax.set_title("Dataset A: x1-x2 scatter")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

feature_columns = [column for column in B_train.columns if column.startswith("x")]
B_train_features = B_train[feature_columns].to_numpy(dtype=float)
B_test_features = B_test[feature_columns].to_numpy(dtype=float)

# 只用训练集的均值和标准差进行标准化
mean = B_train_features.mean(axis=0)
std = B_train_features.std(axis=0)
std[std == 0] = 1.0
B_train_scaled = (B_train_features - mean) / std
B_test_scaled = (B_test_features - mean) / std

pca = PCA(n_components=2)
B_train_pca = pca.fit_transform(B_train_scaled)
B_test_pca = pca.transform(B_test_scaled)

print("\nB 数据集 PCA 方差解释率：")
print(pca.explained_variance_ratio_)
print("前两个主成分累计解释率：", pca.explained_variance_ratio_.sum())

ax = axes[1]
for label in [-1, 1]:
    train_part = B_train_pca[B_train["y"].to_numpy() == label]
    test_part = B_test_pca[B_test["y"].to_numpy() == label]
    ax.scatter(
        train_part[:, 0], train_part[:, 1], s=22, alpha=0.60,
        color=colors[label], label=f"train, y={label}"
    )
    ax.scatter(
        test_part[:, 0], test_part[:, 1], s=38, marker="x",
        color=colors[label], label=f"test, y={label}"
    )
ax.set_title("Dataset B: PCA projection")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.savefig(OUTPUT_DIR / "A_scatter_B_PCA.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# B 按类别分组的箱线图（备用）
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
for ax, column in zip(axes.ravel(), feature_columns):
    values = [
        B_train.loc[B_train["y"] == -1, column],
        B_train.loc[B_train["y"] == 1, column],
    ]
    box = ax.boxplot(values, tick_labels=["y=-1", "y=1"], patch_artist=True)
    box["boxes"][0].set_facecolor(colors[-1])
    box["boxes"][1].set_facecolor(colors[1])
    ax.set_title(column)
    ax.grid(axis="y", alpha=0.25)

fig.suptitle("Dataset B: feature distributions by class", fontsize=16)
fig.savefig(OUTPUT_DIR / "B_boxplot_by_class.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"\n图像已保存到：{OUTPUT_DIR}")
