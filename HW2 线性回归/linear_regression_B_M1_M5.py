"""M1—M5：内部训练/验证选模型，最优模型全量拟合后评估测试集。"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "B_model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES = [
    "area_sqm", "building_age_years", "metro_m", "center_km", "bedrooms",
    "floor", "amenities_1km", "renovation_score", "has_elevator", "sunlight_hours",
]


# 训练、测试使用相同的逐行公式，不估计总体统计量。
def add_features(frame):
    frame = frame.copy()
    if not np.isfinite(frame[FEATURES + ["monthly_rent"]]).all().all():
        raise ValueError("数据中存在缺失值或无穷值。")
    if (frame["metro_m"] < 0).any():
        raise ValueError("距地铁距离不能为负数。")
    frame["area_x_center"] = frame["area_sqm"] * frame["center_km"]
    frame["floor_x_elevator"] = frame["floor"] * frame["has_elevator"]
    frame["area_sq"] = (frame["area_sqm"] / 100) ** 2
    frame["metro_log"] = np.log1p(frame["metro_m"] / 1000)
    frame["age_sq"] = (frame["building_age_years"] / 10) ** 2
    return frame


# 1. 6000 条内部训练、1500 条内部验证；不使用测试集选择模型。
train = add_features(pd.read_csv(DATA_DIR / "B_train.csv"))
settings = {"M1": FEATURES}
settings["M2"] = settings["M1"] + ["area_x_center", "floor_x_elevator"]
settings["M3"] = settings["M2"] + ["area_sq"]
settings["M4"] = settings["M3"] + ["metro_log"]
settings["M5"] = settings["M4"] + ["age_sq"]
fit_rows, valid_rows = train_test_split(np.arange(len(train)), test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models, rows, validation_residuals = {}, [], {}
for name, columns in settings.items():
    X, y = train[columns], train["monthly_rent"]
    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-4, 4, 17)},
        cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise",
    )
    search.fit(X.iloc[fit_rows], y.iloc[fit_rows])
    model = search.best_estimator_
    models[name] = model
    for label, indices in [("训练", fit_rows), ("验证", valid_rows)]:
        actual = y.iloc[indices].to_numpy()
        prediction = model.predict(X.iloc[indices])
        mse = mean_squared_error(actual, prediction)
        rows.append({"方案": name, "数据集": label, "样本数": len(indices),
                     "特征数": len(columns), "alpha": search.best_params_["ridge__alpha"],
                     "CV_RMSE": -search.best_score_, "MSE": mse, "RMSE": np.sqrt(mse),
                     "MAE": mean_absolute_error(actual, prediction), "R2": r2_score(actual, prediction)})
        if label == "验证":
            validation_residuals[name] = actual - prediction
metrics = pd.DataFrame(rows)
metrics.to_csv(OUTPUT_DIR / "train_validation_metrics.csv", index=False, encoding="utf-8-sig")

# 2. 按验证 RMSE 选择最优模型，固定 alpha 后用全部训练集重新拟合。
validation = metrics[metrics["数据集"] == "验证"].set_index("方案")
selected = validation["RMSE"].idxmin()
columns = settings[selected]
final_model = clone(models[selected])
final_model.fit(train[columns], train["monthly_rent"])

# 3. 只评估最优模型，导出测试指标和每套房源的预测结果。
test = add_features(pd.read_csv(DATA_DIR / "B_test.csv"))
actual = test["monthly_rent"].to_numpy()
predicted = final_model.predict(test[columns])
mse = mean_squared_error(actual, predicted)
test_metrics = pd.DataFrame([{
    "方案": selected, "特征数": len(columns), "alpha": validation.loc[selected, "alpha"],
    "重新拟合样本数": len(train), "测试样本数": len(test),
    "MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(actual, predicted),
    "R2": r2_score(actual, predicted),
}])
test_metrics.to_csv(OUTPUT_DIR / "best_model_test_metrics.csv", index=False, encoding="utf-8-sig")
predictions = test[["listing_id", "monthly_rent"]].copy()
predictions["predicted_rent"] = predicted
predictions["residual"] = actual - predicted
predictions.to_csv(OUTPUT_DIR / "B_test_predictions.csv", index=False, encoding="utf-8-sig")
scale = final_model.named_steps["standardscaler"]
regression = final_model.named_steps["ridge"]
coef = regression.coef_ / scale.scale_
intercept = regression.intercept_ - scale.mean_ @ coef
pd.DataFrame({"参数": ["截距"] + columns, "原始基函数系数": np.r_[intercept, coef]}).to_csv(
    OUTPUT_DIR / "best_model_coefficients.csv", index=False, encoding="utf-8-sig")
print("训练与验证指标：\n", metrics.to_string(index=False, float_format="{:.6f}".format))
print("\n最优模型测试指标：\n", test_metrics.to_string(index=False, float_format="{:.6f}".format))

# 4. 拟合对比图：左图与训练/验证表对应，右图展示最优模型的测试拟合。
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#C9D4DF", "text.color": "#263E52", "axes.labelcolor": "#263E52",
})
comparison = metrics.pivot(index="方案", columns="数据集", values="RMSE").reindex(settings)
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), layout="constrained")
x = np.arange(len(settings))
axes[0].plot(x, comparison["训练"], "o--", color="#6A7F99", lw=1.8,
             markerfacecolor="white", label=f"内部训练集 · {len(fit_rows)} 条")
axes[0].plot(x, comparison["验证"], "s-", color="#078A84", lw=2,
             label=f"内部验证集 · {len(valid_rows)} 条")
for i, value in enumerate(comparison["验证"]):
    axes[0].annotate(f"{value:.2f}", (i, value), xytext=(0, -19), textcoords="offset points",
                     ha="center", color="#087D78", fontsize=10)
axes[0].set_xticks(x, ["M1\n原始特征", "M2\n＋交互项", "M3\n＋面积平方",
                        "M4\n＋地铁对数", "M5\n＋楼龄平方"])
axes[0].set(title="(a) 训练与验证误差", ylabel="RMSE（元/月，越低越好）",
            ylim=(105, 210), xlim=(-0.35, 4.35))
axes[0].legend(loc="upper right", frameon=False, fontsize=10)
axes[0].set_box_aspect(1)
low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
padding = (high - low) * 0.06
limits = [low - padding, high + padding]
axes[1].scatter(actual, predicted, s=14, alpha=0.32, color="#078A84", edgecolors="none")
axes[1].plot(limits, limits, "--", color="#D88949", lw=1.6, label="理想预测线")
axes[1].set(title=f"(b) {selected} 测试集拟合", xlabel="真实租金（元/月）",
            ylabel="预测租金（元/月）", xlim=limits, ylim=limits)
axes[1].set_aspect("equal", adjustable="box")
score = test_metrics.iloc[0]
axes[1].text(0.05, 0.95, f"RMSE = {score['RMSE']:.2f} 元/月\nR² = {score['R2']:.5f}",
             transform=axes[1].transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "none"})
axes[1].legend(loc="lower right", frameon=False, fontsize=10)
for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(color="#E8EEF3", linewidth=0.8)
    ax.tick_params(length=0, pad=6)
fig.savefig(OUTPUT_DIR / "M1_M5_fit.png", dpi=240, facecolor="white")
plt.close(fig)

# 5. 用同一内部验证集比较 M2 与最优模型，观察非线性扩展前后的残差趋势。
# 正残差表示低估，负残差表示高估；分箱均值只用于诊断，不参与拟合。
bin_tables = []
for feature, label in [("area_sqm", "面积（平方米）"), ("metro_m", "距地铁距离（米）")]:
    values = train.iloc[valid_rows][feature].to_numpy()
    bins = pd.qcut(values, 8, duplicates="drop")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), layout="constrained", sharex=True, sharey=True)
    for ax, name in zip(axes, ["M2", selected]):
        residual = validation_residuals[name]
        grouped = pd.DataFrame({"x": values, "残差": residual, "分箱": bins})
        summary = grouped.groupby("分箱", observed=True).agg(
            横坐标=("x", "mean"), 平均残差=("残差", "mean"), 样本数=("残差", "size"))
        ax.scatter(values, residual, s=9, alpha=0.18, color="#557DA7", linewidths=0)
        ax.plot(summary["横坐标"], summary["平均残差"], "o-", color="#BE6239", lw=2,
                label="8 个分位区间的平均残差")
        ax.axhline(0, color="#6A7F99", linestyle="--", lw=1)
        ax.set(title=f"{name} · 内部验证残差", xlabel=label,
                ylabel="残差（元/月，真实值−预测值）")
        ax.grid(color="#E8EEF3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
        summary["方案"], summary["特征"] = name, feature
        bin_tables.append(summary.reset_index())
    fig.savefig(OUTPUT_DIR / f"residual_{feature}.png", dpi=240, facecolor="white")
    plt.close(fig)
pd.concat(bin_tables, ignore_index=True).to_csv(
    OUTPUT_DIR / "residual_bins.csv", index=False, encoding="utf-8-sig")
print("\n全部结果已保存至：", OUTPUT_DIR)
