"""只分析 B_train.csv：原始关系、PCA、UMAP。运行方法见同目录说明。"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 保存图片，不弹出阻塞执行的窗口。
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


# 1. 读取训练集
data_path = Path(r"C:\Users\26491\Desktop\ML-Fall-2026\HW2 线性回归\data\B_train.csv")
if len(sys.argv) > 1:
    data_path = Path(sys.argv[1])
output_dir = Path(__file__).resolve().parent
figure_dir = output_dir / "figures"
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
rent = df["monthly_rent"]
if len(df) < 51 or not np.isfinite(df[features + ["monthly_rent"]]).all().all():
    raise ValueError("本脚本要求至少 51 个样本，且特征、租金均为有限数值。")
if (X.nunique() < 2).any() or not df["has_elevator"].isin([0, 1]).all():
    raise ValueError("请检查常量特征或有无电梯的编码。")

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 180,
})
# 随机绘制顺序，减轻文件顺序造成的颜色遮挡；所有 7500 个点均参与分析。
order = np.random.default_rng(42).permutation(len(df))
points = df.iloc[order]
rent_norm = plt.Normalize(rent.min(), rent.max())


def save_figure(fig, filename):
    """唯一的辅助函数：统一保存、关闭图片。"""
    fig.savefig(figure_dir / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存：{filename}", flush=True)


# 2. 原始关系：横纵轴保留实际含义，颜色补充另一项房源属性。
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
settings = [
    ("area_sqm", "center_km", "面积（平方米）", "距中心距离（千米）"),
    ("center_km", "area_sqm", "距中心距离（千米）", "面积（平方米）"),
]
for ax, (x_column, color_column, x_label, color_label) in zip(axes, settings):
    scatter = ax.scatter(
        points[x_column], points["monthly_rent"], c=points[color_column],
        cmap="viridis", s=9, alpha=0.5, linewidths=0,
    )
    r = df[x_column].corr(rent)
    ax.set(xlabel=x_label, ylabel="月租金（原始数据单位）", title=f"与月租金的关系（Pearson r={r:.3f}）")
    fig.colorbar(scatter, ax=ax, label=color_label, shrink=0.9)
save_figure(fig, "01_raw_relationships.png")

# 3. 特征相关性：只放输入特征，查看规模、区位等指标的相关结构。
correlation = X.corr()
fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
heatmap = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(10), feature_names, rotation=45, ha="right")
ax.set_yticks(range(10), feature_names)
ax.set_title("训练集输入特征的 Pearson 相关系数")
for i in range(10):
    for j in range(10):
        value = correlation.iloc[i, j]
        color = "white" if abs(value) > 0.6 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
fig.colorbar(heatmap, ax=ax, label="Pearson r", shrink=0.8)
save_figure(fig, "02_feature_correlations.png")

# 4. 标准化和 PCA 只使用十个房源特征，ID 和月租金不参与拟合。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(svd_solver="full")
pca_coordinates = pca.fit_transform(X_scaled)
variance = pca.explained_variance_ratio_
cumulative = variance.cumsum()
weights = pd.DataFrame(pca.components_[:3].T, index=features, columns=["PC1", "PC2", "PC3"])

# 左图查看信息损失，右图查看每个主成分对应哪些原始特征。
fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
axes[0].bar(range(1, 11), variance * 100, color="#557da7", label="单个主成分")
axes[0].plot(range(1, 11), cumulative * 100, "o-", color="#b34f39", label="累计")
axes[0].axhline(90, linestyle="--", color="gray", linewidth=1, label="90%参考线")
axes[0].set(xticks=range(1, 11), ylim=(0, 105), xlabel="主成分序号", ylabel="解释方差比例（%）", title="PCA 方差谱")
axes[0].legend(frameon=False)
limit = np.abs(weights.to_numpy()).max()
heatmap = axes[1].imshow(weights, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
axes[1].set_xticks(range(3), weights.columns)
axes[1].set_yticks(range(10), feature_names)
axes[1].set_title("前三个主成分的方向系数")
for i in range(10):
    for j in range(3):
        value = weights.iloc[i, j]
        color = "white" if abs(value) > 0.6 * limit else "black"
        axes[1].text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
fig.colorbar(heatmap, ax=axes[1], label="标准化特征的方向系数", shrink=0.8)
save_figure(fig, "03_pca_diagnostics.png")

# 两个二维视角：坐标轴标明解释方差，颜色均为月租金且色标一致。
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
for ax, (a, b) in zip(axes, [(0, 1), (1, 2)]):
    scatter = ax.scatter(
        pca_coordinates[order, a], pca_coordinates[order, b], c=points["monthly_rent"],
        cmap="viridis", norm=rent_norm, s=9, alpha=0.6, linewidths=0,
    )
    ax.set_xlabel(f"PC{a + 1}（{variance[a]:.2%}）")
    ax.set_ylabel(f"PC{b + 1}（{variance[b]:.2%}）")
    ax.set_title(f"PC{a + 1}–PC{b + 1}：按月租金着色")
    ax.set_aspect("equal", adjustable="box")
fig.colorbar(scatter, ax=axes.tolist(), label="月租金（原始数据单位）", shrink=0.9)
save_figure(fig, "04_pca_projections.png")

# 5. UMAP 使用完整的标准化十维输入，不能只把前两个主成分交给它。
print("正在计算 UMAP：n_neighbors=30，min_dist=0.3，random_state=42", flush=True)
umap_model = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.3,
    metric="euclidean", random_state=42, n_jobs=1,
)
umap_coordinates = umap_model.fit_transform(X_scaled)

# 同一套坐标仅更换颜色，检查租金梯度与电梯分组的关系。
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), layout="constrained", sharex=True, sharey=True)
scatter = axes[0].scatter(
    umap_coordinates[order, 0], umap_coordinates[order, 1], c=points["monthly_rent"],
    cmap="viridis", norm=rent_norm, s=9, alpha=0.6, linewidths=0,
)
axes[0].set_title("UMAP：按月租金着色")
fig.colorbar(scatter, ax=axes[0], label="月租金（原始数据单位）", shrink=0.85)
elevator = points["has_elevator"].to_numpy()
for value, label, color in [(0, "无电梯", "#b66323"), (1, "有电梯", "#286aa6")]:
    rows = order[elevator == value]
    axes[1].scatter(
        umap_coordinates[rows, 0], umap_coordinates[rows, 1],
        s=9, alpha=0.6, linewidths=0, color=color, label=f"{label}（{len(rows)}套）",
    )
axes[1].set_title("相同 UMAP 坐标：按电梯状态着色")
axes[1].legend(frameon=False, markerscale=2)
for ax in axes:
    ax.set(xlabel="UMAP 1（无实际单位）", ylabel="UMAP 2（无实际单位）")
    ax.set_aspect("equal", adjustable="box")
save_figure(fig, "05_umap_views.png")

# 6. 保存坐标和 PCA 结果，以后改变着色时无需重新拟合。
result = df.copy()
result[["PC1", "PC2", "PC3"]] = pca_coordinates[:, :3]
result[["UMAP1", "UMAP2"]] = umap_coordinates
result.to_csv(output_dir / "training_coordinates.csv", index=False, encoding="utf-8-sig")
weights.to_csv(output_dir / "pca_weights.csv", encoding="utf-8-sig")
pd.DataFrame({"component": range(1, 11), "variance_ratio": variance, "cumulative_ratio": cumulative}).to_csv(
    output_dir / "pca_variance.csv", index=False, encoding="utf-8-sig",
)
print(f"训练样本：{len(df)}；输入特征：{len(features)}")
print(f"PCA 前两维：{cumulative[1]:.2%}；前三维：{cumulative[2]:.2%}")
print(f"解释至少 90% 方差所需主成分数：{np.searchsorted(cumulative, 0.9) + 1}")
print("图形和坐标已保存到：", output_dir)
