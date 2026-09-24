from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent / "data"
FEATURES = [
    "area_sqm", "building_age_years", "metro_m", "center_km", "bedrooms",
    "floor", "amenities_1km", "renovation_score", "has_elevator", "sunlight_hours",
]
TARGET = "monthly_rent"

train = pd.read_csv(DATA_DIR / "B_train.csv")
test = pd.read_csv(DATA_DIR / "B_test.csv")
X_train = train[FEATURES].to_numpy(dtype=float)
y_train = train[TARGET].to_numpy(dtype=float)
X_test = test[FEATURES].to_numpy(dtype=float)
y_test = test[TARGET].to_numpy(dtype=float)

# 添加截距，使用最小二乘法拟合
A_train = np.column_stack([np.ones(len(y_train)), X_train])
A_test = np.column_stack([np.ones(len(y_test)), X_test])
theta = np.linalg.lstsq(A_train, y_train, rcond=None)[0]

coefficient_table = pd.DataFrame({
    "参数": ["截距"] + FEATURES,
    "系数": theta,
})
print("拟合系数：")
print(coefficient_table.to_string(index=False, float_format="{:.6f}".format))
print("\n模型评估：")

results = []
for name, A, y in [("训练集", A_train, y_train), ("测试集", A_test, y_test)]:
    prediction = A @ theta
    mse = np.mean((prediction - y) ** 2)
    rmse = np.sqrt(mse)
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((prediction - y) ** 2) / total if total > 0 else np.nan
    results.append({
        "数据集": name, "样本数": len(y), "MSE": mse, "R²": r2, "RMSE": rmse,
    })

result_table = pd.DataFrame(results)
print(result_table.to_string(index=False, float_format="{:.6f}".format))

# 测试集拟合效果
test_prediction = A_test @ theta
residual = y_test - test_prediction
test_metrics = result_table.loc[result_table["数据集"] == "测试集"].iloc[0]

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titlepad": 14,
    "text.color": "#2D4055", "axes.labelcolor": "#2D4055",
    "xtick.color": "#64788C", "ytick.color": "#64788C",
    "axes.edgecolor": "#CDD8E3", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "figure.dpi": 130,
})

lower = min(y_test.min(), test_prediction.min())
upper = max(y_test.max(), test_prediction.max())
padding = (upper - lower) * 0.05
limits = [lower - padding, upper + padding]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.5))
axes[0].scatter(y_test, test_prediction, s=16, alpha=0.38,
                color="#2778B5", edgecolors="none", label="测试集房源")
axes[0].plot(limits, limits, "--", color="#ED962B", lw=1.8, label="理想预测线")
axes[0].set(title="(a) 真实租金与预测租金", xlabel="真实月租金", ylabel="预测月租金",
            xlim=limits, ylim=limits)
axes[0].set_aspect("equal", adjustable="box")

axes[1].scatter(test_prediction, residual, s=16, alpha=0.38,
                color="#18A395", edgecolors="none", label="测试集残差")
axes[1].axhline(0, color="#ED962B", linestyle="--", lw=1.8, label="零残差线")
residual_limit = np.max(np.abs(residual)) * 1.1
axes[1].set(title="(b) 预测租金与残差", xlabel="预测月租金", ylabel="残差（真实值 − 预测值）",
            xlim=limits, ylim=(-residual_limit, residual_limit))
axes[1].set_box_aspect(1)

for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(color="#EAF0F5", linewidth=0.8)
    ax.tick_params(length=0, pad=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
              handlelength=2.5, columnspacing=1.5, markerscale=1.5)

fig.subplots_adjust(left=0.075, right=0.98, bottom=0.21, top=0.81, wspace=0.28)
figure_path = Path(__file__).resolve().parent / "figures" / "B_线性回归拟合效果.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
