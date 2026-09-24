import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FEATURES = ["area_dm2", "building_age_years"]
TARGET = "monthly_rent"
folder = Path(__file__).resolve().parent / "data" 
df = pd.read_csv(folder / "A_train.csv")
df_test = pd.read_csv(folder / "A_test.csv")
X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

if len(y) == 0 or not (np.isfinite(X).all() and np.isfinite(y).all()):
    raise ValueError("数据有误")

mu = X.mean(axis=0)
std = X.std(axis=0, ddof=0)
Z = (X - mu) / std
A = np.column_stack([np.ones(len(y)), Z])
n = len(y)

# 梯度下降
learning_rate = 0.1
tol = 1e-10
theta_gd = np.zeros(A.shape[1])
iteration = 0
theta_history = [theta_gd.copy()]
loss_history = [np.mean((A @ theta_gd - y) ** 2) / 2]

while True:
    iteration += 1
    error = A @ theta_gd - y
    gradient = A.T @ error / n
    theta_new = theta_gd - learning_rate * gradient
    converged = np.linalg.norm(theta_new - theta_gd) <= tol

    theta_gd = theta_new
    theta_history.append(theta_gd.copy())
    loss_history.append(np.mean((A @ theta_gd - y) ** 2) / 2)
    if converged:
        break

print(f"梯度下降迭代次数：{iteration}")

# 正规方程
theta_normal = np.linalg.pinv(A.T @ A) @ (A.T @ y)


def show_result(name, theta):
    weights = theta[1:] / std
    intercept = theta[0] - mu @ weights
    y_pred = A @ theta
    residual = y - y_pred
    mse_train = np.mean(residual ** 2)
    rmse_train = np.sqrt(mse_train)
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_train = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    
    mse_test = np.mean((df_test[TARGET].to_numpy(dtype=float) - (np.column_stack([np.ones(len(df_test)), (df_test[FEATURES].to_numpy(dtype=float) - mu) / std]) @ theta)) ** 2)
    rmse_test = np.sqrt(mse_test)
    r2_test = 1 - np.sum((df_test[TARGET].to_numpy(dtype=float) - (np.column_stack([np.ones(len(df_test)), (df_test[FEATURES].to_numpy(dtype=float) - mu) / std]) @ theta)) ** 2) / np.sum((df_test[TARGET].to_numpy(dtype=float) - df_test[TARGET].mean()) ** 2) if np.sum((df_test[TARGET].to_numpy(dtype=float) - df_test[TARGET].mean()) ** 2) > 0 else np.nan

    print(f"\n{name}")
    print("标准化模型参数 [b, w_a, w_h]：", theta)
    print(
        f"原始尺度模型：y = {intercept:.6f}"
        f" {weights[0]:+.6f} * area_dm2"
        f" {weights[1]:+.6f} * building_age_years"
    )
    print(f"训练集 MSE：{mse_train:.6f}")
    print(f"训练集 RMSE：{rmse_train:.6f}")
    print(f"训练集 R²：{r2_train:.6f}")
    
    print(f"测试集 MSE：{mse_test:.6f}")
    print(f"测试集 RMSE：{rmse_test:.6f}")
    print(f"测试集 R²：{r2_test:.6f}")

show_result("批量梯度下降法", theta_gd)
show_result("正规方程法", theta_normal)

history = np.array(theta_history)
normal_loss = np.mean((A @ theta_normal - y) ** 2) / 2


# 绘图
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titlepad": 14,
    "axes.labelsize": 11,
    "axes.labelcolor": "#253746",
    "text.color": "#253746",
    "xtick.color": "#526574",
    "ytick.color": "#526574",
    "axes.edgecolor": "#C8D3DC",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 130,
    "legend.frameon": False,
    "legend.fontsize": 10,
})

BLUE = "#236978"
ORANGE = "#D87849"
INK = "#253746"


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(color="#E6EDF2", linewidth=0.8)
    ax.tick_params(length=0, pad=7)


# 回归平面
area = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
age = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
aa, hh = np.meshgrid(area, age)
yy = (theta_gd[0] + theta_gd[1] * (aa - mu[0]) / std[0]
      + theta_gd[2] * (hh - mu[1]) / std[1])

fig = plt.figure(figsize=(9.5, 7.2))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], y, s=15, color=BLUE,
           alpha=0.65, edgecolors="white", linewidths=0.25, depthshade=False)
ax.plot_surface(aa, hh, yy, color="#E9B989", alpha=0.38,
                edgecolor="#C9AE94", linewidth=0.25, rstride=3, cstride=3)
ax.set(xlabel="房屋面积（平方分米）", ylabel="房龄（年）", zlabel="月租金")
ax.xaxis.labelpad = 11
ax.yaxis.labelpad = 9
ax.zaxis.labelpad = 10
ax.view_init(elev=23, azim=-58)
ax.set_box_aspect((1.25, 1, 0.85))
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((0.96, 0.98, 0.99, 0.4))
    axis.pane.set_edgecolor("#E6EDF2")
    axis._axinfo["grid"].update(color="#DEE7EC", linewidth=0.7)
fig.suptitle("房源样本与线性回归平面", fontsize=16, fontweight="bold", y=0.96)
fig.subplots_adjust(left=0.03, right=0.94, bottom=0.13, top=0.91)
fig.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", color=BLUE,
           markersize=6, label="房源样本"),
    Patch(facecolor="#E9B989", alpha=0.6, label="拟合回归平面"),
], loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, columnspacing=3)
fig.savefig(r"C:\Users\26491\Desktop\ML-Fall-2026\HW2 线性回归\figures\回归平面.png", dpi=300, bbox_inches="tight")

# 损失函数等高线
slopes = np.vstack([history[:, 1:], theta_normal[1:]])
lo, hi = slopes.min(axis=0), slopes.max(axis=0)
center = (lo + hi) / 2
radius = max((hi - lo).max() * 0.72, 1)
wa, wh = np.meshgrid(
    np.linspace(center[0] - radius, center[0] + radius, 220),
    np.linspace(center[1] - radius, center[1] + radius, 220),
)
G = Z.T @ Z / n
da, dh = wa - theta_normal[1], wh - theta_normal[2]
L = (G[0, 0] * da**2 + 2 * G[0, 1] * da * dh + G[1, 1] * dh**2) / 2
levels = np.geomspace(L.max() * 1e-3, L.max() * 0.95, 10)
start = history[0, 1:]
end = theta_normal[1:]
marks = np.unique(np.geomspace(1, len(history), 12).astype(int) - 1).tolist()

fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), sharex=True, sharey=True)
for ax in axes:
    ax.contourf(wa, wh, L, levels=np.r_[0, levels], cmap="Blues",
                alpha=0.13, extend="max")
    cs = ax.contour(wa, wh, L, levels=levels, colors="#A6BBC9", linewidths=0.8)
    ax.clabel(cs, levels[2::3], fontsize=8, fmt="%.2g", colors="#738B9B")
    ax.scatter(*start, s=55, color=INK, edgecolors="white", zorder=5)
    ax.scatter(*end, s=180, marker="*", color=ORANGE,
               edgecolors="white", linewidths=0.7, zorder=6)
    ax.set(xlabel=r"面积系数 $w_a$", ylabel=r"房龄系数 $w_h$")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(length=0, pad=7)

axes[0].plot(history[:, 1], history[:, 2], color=BLUE, linewidth=2.3,
             marker="o", markersize=3.5, markevery=marks, zorder=4)
axes[1].annotate("", xy=end, xytext=start, arrowprops={
    "arrowstyle": "-|>", "color": ORANGE, "lw": 2.5,
    "mutation_scale": 18, "shrinkA": 6, "shrinkB": 9,
})
axes[0].set_title("批量梯度下降")
axes[1].set_title("正规方程")
fig.suptitle("损失等高线与参数求解过程", fontsize=16, fontweight="bold", y=0.97)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.84, wspace=0.13)
fig.legend(handles=[
    Line2D([], [], marker="o", linestyle="none", color=INK, label="初始点"),
    Line2D([], [], color=BLUE, marker="o", markersize=3, lw=2,
           label="梯度下降迭代"),
    Line2D([], [], color=ORANGE, lw=2, label="正规方程直接求解"),
    Line2D([], [], marker="*", linestyle="none", color=ORANGE,
           markersize=12, label="最优解"),
], loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, columnspacing=2.5)
fig.savefig(r"C:\Users\26491\Desktop\ML-Fall-2026\HW2 线性回归\figures\参数比较.png", dpi=300, bbox_inches="tight")


# 模型随迭代的变化
prediction_gap = np.sqrt(np.mean((history @ A.T - A @ theta_normal) ** 2, axis=1))
steps = np.unique([
    *[int(np.flatnonzero(prediction_gap <= ratio * prediction_gap[0])[0])
      for ratio in (0.7, 0.25)],
    iteration,
])
area_z = (area - mu[0]) / std[0]
age_z = (age - mu[1]) / std[1]
coordinates = [area, age]
standardized = [area_z, age_z]
point_indices = np.linspace(0, len(area) - 1, 7, dtype=int)
stage_colors = ["#8198AE", "#409197", "#173D57"]
stage_styles = [":", "--", "-"]

fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
all_predictions = []
for j, ax in enumerate(axes):
    for i, step in enumerate(steps):
        final = step == iteration
        phase = "收敛" if final else ["初期", "中期"][i]
        color = stage_colors[-1] if final else stage_colors[i]
        style = "-" if final else stage_styles[i]
        prediction = history[step, 0] + history[step, j + 1] * standardized[j]
        all_predictions.append(prediction)
        ax.plot(coordinates[j], prediction, color=color, linestyle=style,
                 lw=2.8 if final else 2.3, label=f"{phase}（第 {step} 次）")

    reference = theta_normal[0] + theta_normal[j + 1] * standardized[j]
    ax.plot(coordinates[j][point_indices], reference[point_indices],
             linestyle="none", marker="o", markersize=6.5,
             markerfacecolor="white", markeredgecolor=ORANGE,
             markeredgewidth=1.8, label="正规方程解", zorder=5)
    style_axis(ax)

low = min(values.min() for values in all_predictions)
high = max(values.max() for values in all_predictions)
margin = max((high - low) * 0.08, 1)
axes[0].set_ylim(low - margin, high + margin)
axes[0].set(xlabel="房屋面积（平方分米）", ylabel="预测月租金",
            title=f"固定房龄 {mu[1]:.2f} 年")
axes[1].set(xlabel="房龄（年）",
            title=f"固定面积 {mu[0]:.2f} 平方分米")

for j, ax in enumerate(axes):
    zoom = ax.inset_axes([0.49, 0.22, 0.46, 0.23])
    final_prediction = theta_gd[0] + theta_gd[j + 1] * standardized[j]
    reference = theta_normal[0] + theta_normal[j + 1] * standardized[j]
    zoom.plot(coordinates[j], final_prediction, color=stage_colors[-1], lw=2)
    zoom.plot(coordinates[j][point_indices], reference[point_indices],
               linestyle="none", marker="o", markersize=4,
               markerfacecolor="white", markeredgecolor=ORANGE,
               markeredgewidth=1.2)
    zoom.set_title("收敛细节（纵轴放大）", fontsize=9.5, pad=6)
    zoom.set_xticks([coordinates[j][0], coordinates[j][-1]])
    zoom.yaxis.set_major_locator(plt.MaxNLocator(3))
    zoom.tick_params(labelsize=8, length=2, pad=2)
    zoom.grid(axis="y", color="#E6EDF2", linewidth=0.6)
    zoom.margins(x=0.04, y=0.2)
    for spine in zoom.spines.values():
        spine.set_visible(True)
        spine.set_color("#CAD7E0")

fig.suptitle("梯度下降逐步逼近最终回归关系", fontsize=17,
              fontweight="bold", y=0.97)

fig.subplots_adjust(left=0.075, right=0.98, bottom=0.28, top=0.81, wspace=0.16)

effects = [100 * theta_gd[1] / std[0], theta_gd[2] / std[1]]
for j, ax in enumerate(axes):
    center_x = (ax.get_position().x0 + ax.get_position().x1) / 2
    direction = "增加" if effects[j] >= 0 else "减少"

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045),
            ncol=4, columnspacing=2.5, handlelength=3.2)
fig.savefig(r"C:\Users\26491\Desktop\ML-Fall-2026\HW2 线性回归\figures\模型演变.png", dpi=300, bbox_inches="tight")


# 损失函数变化
fig, ax = plt.subplots(figsize=(9, 5.8))
ax.plot(loss_history, color=BLUE, lw=2.3, label="批量梯度下降")
ax.axhline(normal_loss, color=ORANGE, linestyle="--", lw=2,
            label="正规方程最小损失")
if normal_loss > 0 and np.all(np.asarray(loss_history) > 0):
    ax.set_yscale("log")
    ax.set_ylabel(r"损失 $J(\theta)$（对数刻度）")
else:
    ax.set_ylabel(r"损失 $J(\theta)$")
ax.set(xlabel="迭代次数", title="训练损失随迭代次数的变化")
style_axis(ax)
fig.subplots_adjust(left=0.12, right=0.96, bottom=0.22, top=0.87)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
            ncol=2, columnspacing=3, handlelength=3)
fig.savefig(r"C:\Users\26491\Desktop\ML-Fall-2026\HW2 线性回归\figures\损失曲线.png", dpi=300, bbox_inches="tight")

plt.show()
