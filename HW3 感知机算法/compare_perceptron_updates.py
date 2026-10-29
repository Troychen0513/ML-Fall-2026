from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


for font_name in ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = Path("/Users/tuyuchen/Downloads/HW3 感知机算法")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "comparison_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = ["A", "B"]
LEARNING_RATES = [0.01, 0.1, 1.0, 10.0]
INITIALIZATIONS = ["zero", "small_random", "random"]
RANDOM_SEED = 42
MAX_EPOCHS = 10000


def load_dataset(name, standardize=False):
    train = pd.read_csv(DATA_DIR / f"{name}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{name}_test.csv")
    feature_columns = [column for column in train.columns if column.startswith("x")]

    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    X_test = test[feature_columns].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)

    if standardize:
        mean = X_train.mean(axis=0)
        scale = X_train.std(axis=0)
        scale[scale == 0] = 1.0
        X_train = (X_train - mean) / scale
        X_test = (X_test - mean) / scale

    X_train = np.c_[X_train, np.ones(X_train.shape[0])]
    X_test = np.c_[X_test, np.ones(X_test.shape[0])]
    return X_train, y_train, X_test, y_test, train.shape, test.shape


def make_initial_parameter(kind, dimension, seed):
    if kind == "zero":
        return np.zeros(dimension)

    rng = np.random.default_rng(seed)
    if kind == "small_random":
        return rng.normal(0.0, 0.01, dimension)
    if kind == "random":
        return rng.normal(0.0, 1.0, dimension)
    raise ValueError(f"Unknown initialization: {kind}")


def perceptron_criterion(X, y, theta):
    """
    感知机准则函数：只对当前误分类样本累加 -y_i(theta^T x_tilde_i)。
    对正确分类样本不计入准则，因此不使用 max 表达式。
    """
    margins = y * (X @ theta)
    misclassified = margins <= 0
    return float(-margins[misclassified].sum())


def accuracy(X, y, theta):
    prediction = np.where(X @ theta >= 0, 1, -1)
    return float(np.mean(prediction == y))


def sgd_perceptron(
    X_train,
    y_train,
    X_test,
    y_test,
    eta,
    theta0,
    seed=RANDOM_SEED,
    max_epochs=MAX_EPOCHS,
):
    theta = theta0.copy()
    rng = np.random.default_rng(seed)
    history = []
    sample_updates = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(X_train))
        epoch_updates = 0

        for index in order:
            margin = y_train[index] * (X_train[index] @ theta)
            if margin <= 0:
                theta = theta + eta * y_train[index] * X_train[index]
                epoch_updates += 1
                sample_updates += 1

        train_mistakes = int(np.sum(y_train * (X_train @ theta) <= 0))
        history.append(
            {
                "epoch": epoch,
                "mistakes": train_mistakes,
                "criterion": perceptron_criterion(X_train, y_train, theta),
                "train_accuracy": accuracy(X_train, y_train, theta),
                "test_accuracy": accuracy(X_test, y_test, theta),
            }
        )

        if epoch_updates == 0:
            return theta, history, sample_updates, sample_updates, True

    return theta, history, sample_updates, sample_updates, False


def batch_perceptron(
    X_train,
    y_train,
    X_test,
    y_test,
    eta,
    theta0,
    max_epochs=MAX_EPOCHS,
):
    theta = theta0.copy()
    history = []
    batch_updates = 0
    sample_updates = 0

    for epoch in range(1, max_epochs + 1):
        margins = y_train * (X_train @ theta)
        misclassified = margins <= 0
        mistakes = int(np.sum(misclassified))

        if mistakes == 0:
            history.append(
                {
                    "epoch": epoch,
                    "mistakes": 0,
                    "criterion": 0.0,
                    "train_accuracy": accuracy(X_train, y_train, theta),
                    "test_accuracy": accuracy(X_test, y_test, theta),
                }
            )
            return theta, history, batch_updates, sample_updates, True

        update = (y_train[misclassified, None] * X_train[misclassified]).sum(axis=0)
        theta = theta + eta * update
        batch_updates += 1
        sample_updates += mistakes

        new_mistakes = int(np.sum(y_train * (X_train @ theta) <= 0))
        history.append(
            {
                "epoch": epoch,
                "mistakes": new_mistakes,
                "criterion": perceptron_criterion(X_train, y_train, theta),
                "train_accuracy": accuracy(X_train, y_train, theta),
                "test_accuracy": accuracy(X_test, y_test, theta),
            }
        )

    return theta, history, batch_updates, sample_updates, False


def run_one(name, algorithm, eta=1.0, initialization="zero", standardize=False):
    X_train, y_train, X_test, y_test, train_shape, test_shape = load_dataset(
        name, standardize=standardize
    )
    theta0 = make_initial_parameter(
        initialization, X_train.shape[1], seed=RANDOM_SEED
    )

    if algorithm == "SGD":
        theta, history, update_steps, sample_updates, converged = sgd_perceptron(
            X_train,
            y_train,
            X_test,
            y_test,
            eta,
            theta0,
        )
    elif algorithm == "BGD":
        theta, history, update_steps, sample_updates, converged = batch_perceptron(
            X_train,
            y_train,
            X_test,
            y_test,
            eta,
            theta0,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return {
        "dataset": name,
        "algorithm": algorithm,
        "eta": eta,
        "initialization": initialization,
        "standardized": standardize,
        "train_shape": str(train_shape),
        "test_shape": str(test_shape),
        "epochs": len(history),
        "update_steps": update_steps,
        "sample_updates": sample_updates,
        "sample_evaluations": len(history) * len(X_train),
        "converged": converged,
        "train_accuracy": accuracy(X_train, y_train, theta),
        "test_accuracy": accuracy(X_test, y_test, theta),
        "theta": theta,
        "history": history,
    }


# Baseline comparison
baseline_rows = []
baseline_runs = {}
for name in DATASETS:
    for algorithm in ["SGD", "BGD"]:
        result = run_one(name, algorithm)
        baseline_runs[(name, algorithm)] = result
        baseline_rows.append({key: value for key, value in result.items() if key not in ["theta", "history"]})

baseline = pd.DataFrame(baseline_rows)
baseline.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)

# Learning-rate sensitivity
learning_rate_rows = []
for name in DATASETS:
    for algorithm in ["SGD", "BGD"]:
        for eta in LEARNING_RATES:
            result = run_one(name, algorithm, eta=eta)
            learning_rate_rows.append({key: value for key, value in result.items() if key not in ["theta", "history"]})

learning_rate_results = pd.DataFrame(learning_rate_rows)
learning_rate_results.to_csv(OUTPUT_DIR / "learning_rate_sensitivity.csv", index=False)

# Initialization sensitivity
initialization_rows = []
for name in DATASETS:
    for algorithm in ["SGD", "BGD"]:
        for initialization in INITIALIZATIONS:
            result = run_one(name, algorithm, initialization=initialization)
            initialization_rows.append({key: value for key, value in result.items() if key not in ["theta", "history"]})

initialization_results = pd.DataFrame(initialization_rows)
initialization_results.to_csv(OUTPUT_DIR / "initialization_sensitivity.csv", index=False)

# Feature-scale sensitivity
scale_rows = []
for name in DATASETS:
    for algorithm in ["SGD", "BGD"]:
        for standardized in [False, True]:
            result = run_one(name, algorithm, standardize=standardized)
            scale_rows.append({key: value for key, value in result.items() if key not in ["theta", "history"]})

scale_results = pd.DataFrame(scale_rows)
scale_results.to_csv(OUTPUT_DIR / "feature_scale_sensitivity.csv", index=False)

# Convergence curves for the baseline runs
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for ax, name in zip(axes, DATASETS):
    for algorithm, color in [("SGD", "#377eb8"), ("BGD", "#e45756")]:
        history = baseline_runs[(name, algorithm)]["history"]
        epochs = [item["epoch"] for item in history]
        mistakes = [item["mistakes"] for item in history]
        ax.plot(epochs, mistakes, marker="o", linewidth=2, color=color, label=algorithm)
    ax.set_title(f"Dataset {name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training mistakes")
    ax.grid(alpha=0.25)
    ax.legend()

fig.savefig(OUTPUT_DIR / "baseline_convergence.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Sensitivity plots

# Publication-style palette and layout.
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Heiti SC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
COLORS = {"SGD": "#1F77B4", "BGD": "#D55E00"}
MARKERS = {"SGD": "o", "BGD": "s"}


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA5B1")
    axis.spines["bottom"].set_color("#9AA5B1")
    axis.grid(axis="y", color="#D9E1E8", linewidth=0.8)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#485460", labelsize=10)


def add_line_panel(axis, data, x_column, metric, x_order=None, percent=False, log_x=False):
    for algorithm in ["SGD", "BGD"]:
        current = data[data["algorithm"] == algorithm].copy()
        if x_order is not None:
            current[x_column] = pd.Categorical(current[x_column], categories=x_order, ordered=True)
            current = current.sort_values(x_column)
        else:
            current = current.sort_values(x_column)
        x_values = current[x_column].to_numpy()
        y_values = current[metric].to_numpy(dtype=float)
        if percent:
            y_values = y_values * 100
        axis.plot(
            x_values,
            y_values,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            markersize=6,
            linewidth=2.2,
            label=algorithm,
            zorder=3,
        )
        for x_value, y_value in zip(x_values, y_values):
            label = f"{y_value:.1f}" if percent else f"{y_value:.0f}"
            axis.annotate(
                label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8.5,
                color=COLORS[algorithm],
            )
    if log_x:
        axis.set_xscale("log")
    style_axis(axis)


def add_bar_panel(axis, data, x_column, order, metric, percent=False):
    positions = np.arange(len(order))
    width = 0.34
    for offset, algorithm in [(-width / 2, "SGD"), (width / 2, "BGD")]:
        current = data[data["algorithm"] == algorithm].set_index(x_column).reindex(order)
        values = current[metric].to_numpy(dtype=float)
        if percent:
            values = values * 100
        bars = axis.bar(
            positions + offset,
            values,
            width=width,
            color=COLORS[algorithm],
            alpha=0.92,
            label=algorithm,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            label = f"{value:.1f}" if percent else f"{value:.0f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.7 if not percent else 0.08),
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#485460",
            )
    axis.set_xticks(positions)
    axis.set_xticklabels(order)
    style_axis(axis)


# Learning-rate sensitivity: rows are datasets and columns are the three reported measures.
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), dpi=220, constrained_layout=True)
for row, name in enumerate(DATASETS):
    subset = learning_rate_results[learning_rate_results["dataset"] == name]
    add_line_panel(axes[row, 0], subset, "eta", "epochs", log_x=True)
    add_line_panel(axes[row, 1], subset, "eta", "update_steps", log_x=True)
    add_line_panel(axes[row, 2], subset, "eta", "test_accuracy", percent=True, log_x=True)
    for column in range(3):
        axes[row, column].set_title(f"数据集 {name} · {['训练轮数', '参数更新次数', '测试准确率'][column]}", fontsize=12, pad=9)
        axes[row, column].set_xticks(LEARNING_RATES)
        axes[row, column].set_xticklabels(["0.01", "0.1", "1", "10"])
        axes[row, column].set_xlabel("学习率 η（对数坐标）", fontsize=10)
axes[0, 0].set_ylabel("训练轮数", fontsize=10)
axes[1, 0].set_ylabel("训练轮数", fontsize=10)
axes[0, 1].set_ylabel("参数更新次数", fontsize=10)
axes[1, 1].set_ylabel("参数更新次数", fontsize=10)
axes[0, 2].set_ylabel("测试准确率（%）", fontsize=10)
axes[1, 2].set_ylabel("测试准确率（%）", fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.035))
fig.suptitle("学习率敏感性分析", fontsize=16, fontweight="bold", y=1.08)
fig.savefig(OUTPUT_DIR / "sensitivity_learning_rate.png", dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)


# Initialization sensitivity: rows are datasets and columns are the three reported measures.
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), dpi=220, constrained_layout=True)
initialization_order = ["zero", "small_random", "random"]
initialization_labels = ["零初始化", "小随机", "随机"]
for row, name in enumerate(DATASETS):
    subset = initialization_results[initialization_results["dataset"] == name].copy()
    subset["initialization"] = subset["initialization"].map(dict(zip(initialization_order, initialization_labels)))
    add_bar_panel(axes[row, 0], subset, "initialization", initialization_labels, "epochs")
    add_bar_panel(axes[row, 1], subset, "initialization", initialization_labels, "update_steps")
    add_bar_panel(axes[row, 2], subset, "initialization", initialization_labels, "test_accuracy", percent=True)
    axes[row, 0].set_title(f"数据集 {name} · 训练轮数", fontsize=12, pad=9)
    axes[row, 1].set_title(f"数据集 {name} · 参数更新次数", fontsize=12, pad=9)
    axes[row, 2].set_title(f"数据集 {name} · 测试准确率", fontsize=12, pad=9)
    for column in range(3):
        axes[row, column].set_xlabel("参数初始化方式", fontsize=10)
axes[0, 0].set_ylabel("训练轮数", fontsize=10)
axes[1, 0].set_ylabel("训练轮数", fontsize=10)
axes[0, 1].set_ylabel("参数更新次数", fontsize=10)
axes[1, 1].set_ylabel("参数更新次数", fontsize=10)
axes[0, 2].set_ylabel("测试准确率（%）", fontsize=10)
axes[1, 2].set_ylabel("测试准确率（%）", fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.035))
fig.suptitle("参数初始化敏感性分析", fontsize=16, fontweight="bold", y=1.08)
fig.savefig(OUTPUT_DIR / "sensitivity_initialization.png", dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)


# Feature-scale sensitivity: rows are datasets and columns are the three reported measures.
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), dpi=220, constrained_layout=True)
scale_order = ["原始尺度", "标准化"]
for row, name in enumerate(DATASETS):
    subset = scale_results[scale_results["dataset"] == name].copy()
    subset["standardized"] = subset["standardized"].map({False: "原始尺度", True: "标准化"})
    add_bar_panel(axes[row, 0], subset, "standardized", scale_order, "epochs")
    add_bar_panel(axes[row, 1], subset, "standardized", scale_order, "update_steps")
    add_bar_panel(axes[row, 2], subset, "standardized", scale_order, "test_accuracy", percent=True)
    axes[row, 0].set_title(f"数据集 {name} · 训练轮数", fontsize=12, pad=9)
    axes[row, 1].set_title(f"数据集 {name} · 参数更新次数", fontsize=12, pad=9)
    axes[row, 2].set_title(f"数据集 {name} · 测试准确率", fontsize=12, pad=9)
    for column in range(3):
        axes[row, column].set_xlabel("特征尺度处理", fontsize=10)
axes[0, 0].set_ylabel("训练轮数", fontsize=10)
axes[1, 0].set_ylabel("训练轮数", fontsize=10)
axes[0, 1].set_ylabel("参数更新次数", fontsize=10)
axes[1, 1].set_ylabel("参数更新次数", fontsize=10)
axes[0, 2].set_ylabel("测试准确率（%）", fontsize=10)
axes[1, 2].set_ylabel("测试准确率（%）", fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.035))
fig.suptitle("特征尺度敏感性分析", fontsize=16, fontweight="bold", y=1.08)
fig.savefig(OUTPUT_DIR / "sensitivity_feature_scale.png", dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)


# Console output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")

print("数据集形状")
for name in DATASETS:
    _, _, _, _, train_shape, test_shape = load_dataset(name)
    print(f"{name}: train={train_shape}, test={test_shape}")

print("\n基准比较")
print(
    baseline[
        [
            "dataset",
            "algorithm",
            "epochs",
            "update_steps",
            "sample_updates",
            "sample_evaluations",
            "converged",
            "train_accuracy",
            "test_accuracy",
        ]
    ].to_string(index=False)
)

print("\n学习率敏感性")
print(
    learning_rate_results[
        ["dataset", "algorithm", "eta", "epochs", "update_steps", "converged", "train_accuracy", "test_accuracy"]
    ].to_string(index=False)
)

print("\n初始化敏感性")
print(
    initialization_results[
        ["dataset", "algorithm", "initialization", "epochs", "update_steps", "converged", "train_accuracy", "test_accuracy"]
    ].to_string(index=False)
)

print("\n特征尺度敏感性")
print(
    scale_results[
        ["dataset", "algorithm", "standardized", "epochs", "update_steps", "converged", "train_accuracy", "test_accuracy"]
    ].to_string(index=False)
)

print(f"\n结果文件目录：{OUTPUT_DIR}")
