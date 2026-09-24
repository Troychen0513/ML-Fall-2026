from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "gd_experiment_results"
FEATURES = ["area_dm2", "building_age_years"]
TARGET = "monthly_rent"
SEED = 2026
FOLDS = 5
TOL = 1e-8
MAX_ITER = 10000
TIMING_REPEATS = 5
L1_GRID = [0.01, 0.1, 1, 5, 10, 25, 50]
L2_GRID = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]
METHODS = ["无正则化", "L1", "L2", "L1+L2"]
COLORS = ["#76869B", "#2778B5", "#18A395", "#9666C4"]
STYLES = ["--", "-", "-.", ":"]


def run_gd(A, y, alpha, lambda1=0.0, lambda2=0.0):
    theta = np.zeros(A.shape[1])
    theta[0] = y.mean()
    mse_history, objective_history = [], []
    status = "达到迭代上限"
    begin = perf_counter()

    for iteration in range(MAX_ITER + 1):
        error = A @ theta - y
        mse = float(np.mean(error ** 2))
        objective = (mse / 2 + lambda1 * np.abs(theta[1:]).sum()
                     + lambda2 * np.sum(theta[1:] ** 2) / 2)
        mse_history.append(mse)
        objective_history.append(objective)
        if not np.isfinite(objective) or objective > max(objective_history[0], 1) * 1e10:
            status = "发散"
            break

        gradient = A.T @ error / len(y)
        gradient[1:] += lambda2 * theta[1:]

        residual = gradient.copy()
        residual[1:] = np.where(
            theta[1:] != 0,
            gradient[1:] + lambda1 * np.sign(theta[1:]),
            np.sign(gradient[1:]) * np.maximum(np.abs(gradient[1:]) - lambda1, 0),
        )
        if np.max(np.abs(residual)) <= TOL:
            status = "收敛"
            break
        if iteration == MAX_ITER:
            break

        theta = theta - alpha * gradient
        theta[1:] = np.sign(theta[1:]) * np.maximum(np.abs(theta[1:]) - alpha * lambda1, 0)

    return {
        "theta": theta, "iterations": iteration, "status": status,
        "time_ms": (perf_counter() - begin) * 1000,
        "mse": np.array(mse_history), "objective": np.array(objective_history),
    }

# 读取+标准化
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_data = pd.read_csv(DATA_DIR / "A_train.csv")
test_data = pd.read_csv(DATA_DIR / "A_test.csv")
X = train_data[FEATURES].to_numpy(dtype=float)
y = train_data[TARGET].to_numpy(dtype=float)
X_test = test_data[FEATURES].to_numpy(dtype=float)
y_test = test_data[TARGET].to_numpy(dtype=float)

for name, features, target in [("训练集", X, y), ("测试集", X_test, y_test)]:
    if len(target) < FOLDS or not (np.isfinite(features).all() and np.isfinite(target).all()):
        raise ValueError(f"{name}的数据不足或包含无效值。")

mu, std = X.mean(axis=0), X.std(axis=0, ddof=0)
if np.any(std == 0):
    raise ValueError("存在标准差为零的特征。")

A = np.column_stack([np.ones(len(y)), (X - mu) / std])
A_test = np.column_stack([np.ones(len(y_test)), (X_test - mu) / std])
theta_reference = np.linalg.pinv(A.T @ A) @ (A.T @ y)
reference_mse = float(np.mean((A @ theta_reference - y) ** 2))

# 学习率敏感性分析
learning_rates = [0.05, 0.1, 0.2, 1, 1.8, 2.2]
lr_runs, lr_rows = [], []
for alpha in learning_rates:
    runs = [run_gd(A, y, alpha) for _ in range(TIMING_REPEATS)]
    result = runs[-1]
    result["time_ms"] = float(np.median([run["time_ms"] for run in runs]))
    result["alpha"] = alpha
    lr_runs.append(result)

    train_mse = float(np.mean((A @ result["theta"] - y) ** 2))
    test_mse = float(np.mean((A_test @ result["theta"] - y_test) ** 2))
    test_total = float(np.sum((y_test - y_test.mean()) ** 2))
    test_r2 = 1 - test_mse * len(y_test) / test_total if test_total > 0 else np.nan
    lr_rows.append({
        "学习率": alpha, "状态": result["status"], "迭代次数": result["iterations"],
        "耗时中位数_ms": result["time_ms"], "训练_RMSE": np.sqrt(train_mse),
        "测试_RMSE": np.sqrt(test_mse), "测试_R2": test_r2,
        "参数相对误差": np.linalg.norm(result["theta"] - theta_reference)
        / max(np.linalg.norm(theta_reference), 1),
    })
fastest = min((run for run in lr_runs if run["status"] == "收敛"),
              key=lambda run: run["iterations"])
lr_table = pd.DataFrame(lr_rows)


# 五折交叉验证选择正则强度
indices = np.random.default_rng(SEED).permutation(len(y))
folds = np.array_split(indices, FOLDS)
fold_data = []
for k in range(FOLDS):
    train = np.concatenate([folds[j] for j in range(FOLDS) if j != k])
    valid = folds[k]
    fold_mu = X[train].mean(axis=0)
    fold_std = X[train].std(axis=0, ddof=0)
    if np.any(fold_std == 0):
        raise ValueError("存在标准差为零的特征。")
    fold_A = np.column_stack([np.ones(len(train)), (X[train] - fold_mu) / fold_std])
    fold_V = np.column_stack([np.ones(len(valid)), (X[valid] - fold_mu) / fold_std])
    fold_L = np.linalg.eigvalsh(fold_A.T @ fold_A / len(train)).max()
    fold_data.append((fold_A, y[train], fold_V, y[valid], fold_L))

candidates = {
    "无正则化": [(0.0, 0.0)],
    "L1": [(value, 0.0) for value in L1_GRID],
    "L2": [(0.0, value) for value in L2_GRID],
    "L1+L2": [(a, b) for a in L1_GRID for b in L2_GRID],
}
tuning_rows, selected = [], {}
for method, pairs in candidates.items():
    for lambda1, lambda2 in pairs:
        squared_errors, count = 0.0, 0
        for fold_A, fold_y, fold_V, valid_y, fold_L in fold_data:
            result = run_gd(fold_A, fold_y, 1 / (fold_L + lambda2), lambda1, lambda2)
            if result["status"] != "收敛":
                raise RuntimeError(f"交叉验证未收敛：{method}, {lambda1}, {lambda2}")
            squared_errors += np.sum((fold_V @ result["theta"] - valid_y) ** 2)
            count += len(valid_y)
        tuning_rows.append({
            "方法": method, "lambda1": lambda1, "lambda2": lambda2,
            "CV_RMSE": np.sqrt(squared_errors / count),
        })
    selected[method] = min((row for row in tuning_rows if row["方法"] == method),
                           key=lambda row: row["CV_RMSE"])
tuning_table = pd.DataFrame(tuning_rows)

# 比较交叉验证选出的正则化模型
common_alpha = fastest["alpha"]
reg_runs, reg_rows = [], []
for method in METHODS:
    parameters = selected[method]
    runs = [run_gd(A, y, common_alpha, parameters["lambda1"], parameters["lambda2"])
            for _ in range(TIMING_REPEATS)]
    result = runs[-1]
    result["time_ms"] = float(np.median([run["time_ms"] for run in runs]))
    result["method"] = method
    reg_runs.append(result)

    scores = {}
    for label, matrix, target in [("训练", A, y), ("测试", A_test, y_test)]:
        mse = float(np.mean((matrix @ result["theta"] - target) ** 2))
        total = float(np.sum((target - target.mean()) ** 2))
        scores[f"{label}_MSE"] = mse
        scores[f"{label}_RMSE"] = np.sqrt(mse)
        scores[f"{label}_R2"] = 1 - mse * len(target) / total if total > 0 else np.nan
    reg_rows.append({
        **parameters, "共同学习率": common_alpha, "状态": result["status"],
        "迭代次数": result["iterations"], "耗时中位数_ms": result["time_ms"],
        **scores,
        "标准化面积系数": result["theta"][1], "标准化房龄系数": result["theta"][2],
    })
reg_table = pd.DataFrame(reg_rows)


# 保存并输出指标
lr_table.to_csv(OUTPUT_DIR / "learning_rate_results.csv", index=False, encoding="utf-8-sig")
reg_table.to_csv(OUTPUT_DIR / "regularization_results.csv", index=False, encoding="utf-8-sig")
tuning_table.to_csv(OUTPUT_DIR / "regularization_cv.csv", index=False, encoding="utf-8-sig")
print(f"按相同停止条件选出的最快学习率：{fastest['alpha']:.6f}")
print(lr_table[["学习率", "状态", "迭代次数", "耗时中位数_ms", "训练_RMSE", "测试_RMSE", "测试_R2"]]
      .to_string(index=False, float_format="{:.6g}".format))
print(f"\n正则化比较的共同学习率：{common_alpha:.6f}")
print(reg_table[["方法", "lambda1", "lambda2", "CV_RMSE", "迭代次数", "耗时中位数_ms", "测试_RMSE", "测试_R2"]]
      .to_string(index=False, float_format="{:.6g}".format))

# 绘图
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


# 学习率敏感性分析
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
stable = [run for run in lr_runs if run["status"] == "收敛"]
unstable = [run for run in lr_runs if run["status"] != "收敛"]
palette = ["#ED962B", "#71869C", "#2878B5", "#11A38E", "#9666C4"]
for i, run in enumerate(stable):
    axes[0].plot(run["mse"], color=palette[i % len(palette)],
                 linestyle=["--", (0, (5, 2)), "-", "-.", ":"][i % 5],
                 lw=2.1, label=fr"$\alpha={run['alpha']:.4f}$（{run['iterations']} 次）")
axes[0].axhline(reference_mse, color="#B6C4CF", linestyle="--", lw=1, zorder=0)
axes[0].set_xscale("symlog", linthresh=1)
axes[0].set_xlim(0, max(run["iterations"] for run in stable) * 1.05)
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
axes[0].set(title="(a) 稳定学习率：效率与收敛精度", xlabel="更新次数", ylabel="训练 MSE")
for run in unstable:
    axes[1].plot(run["mse"], color="#E26044", lw=2.1,
                 label=fr"$\alpha={run['alpha']:.4f}$（{run['status']}）")
axes[1].set_yscale("log")
axes[1].set(title="(b) 过大的学习率", xlabel="更新次数", ylabel="训练 MSE（对数刻度）")
handles, labels = [], []
for ax in axes:
    ax.set_axisbelow(True)
    ax.grid(color="#EAF0F5", linewidth=0.8)
    ax.tick_params(length=0, pad=6)
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
fig.subplots_adjust(left=0.075, right=0.98, bottom=0.25, top=0.85, wspace=0.28)
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
           ncol=3, handlelength=2.6, columnspacing=2, labelspacing=0.9, fontsize=10)
fig.savefig(OUTPUT_DIR / "学习率敏感性分析.png", dpi=300, bbox_inches="tight", facecolor="white")

plt.show()
