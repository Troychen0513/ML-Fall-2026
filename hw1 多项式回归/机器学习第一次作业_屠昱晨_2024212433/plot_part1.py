"""绘制 part1.py 生成的结果。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DEGREES = np.arange(1, 9)


def read_results():
    train = np.loadtxt(DATA / "dataset_A_train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(DATA / "dataset_A_test.csv", delimiter=",", skiprows=1)
    metrics = np.loadtxt(BASE / "part1_metrics.csv", delimiter=",", skiprows=1)
    curves = np.loadtxt(BASE / "part1_predictions.csv", delimiter=",", skiprows=1)
    test_predictions = np.loadtxt(BASE / "part1_test_predictions.csv", delimiter=",", skiprows=1)
    cv_scores = np.loadtxt(BASE / "part1_cv_scores.csv", delimiter=",", skiprows=1)
    if not all(np.isfinite(data).all() for data in (metrics, curves, test_predictions, cv_scores)):
        raise ValueError("结果文件包含 NaN/inf，请先重新运行 part1.py。")
    return train, test, metrics, curves, test_predictions, cv_scores


def draw_fits(ax, train, test, curves, selected):
    """绘制数据点和不同阶数的拟合曲线。"""
    ax.scatter(train[:, 0], train[:, 1], s=20, alpha=0.65, label="Train")
    ax.scatter(test[:, 0], test[:, 1], s=20, alpha=0.50, label="Test")

    for i, degree in enumerate(DEGREES):
        chosen = degree == selected
        ax.plot(
            curves[:, 0], curves[:, i + 1],
            color="#D62728" if chosen else "#777777",
            linewidth=2.5 if chosen else 0.9,
            alpha=1.0 if chosen else 0.22,
            label=f"Selected degree {degree}" if chosen else None,
        )

    ax.set_title("Polynomial fits")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()


def draw_errors(ax, metrics, selected):
    """绘制训练误差、CV误差和测试误差。"""
    train_mse = metrics[:, 1]
    cv_mse = metrics[:, 2]
    cv_std = metrics[:, 3]
    test_mse = metrics[:, 4]

    ax.plot(DEGREES, train_mse, "o-", label="Train MSE")
    ax.plot(DEGREES, cv_mse, "o-", label="CV MSE")
    ax.fill_between(DEGREES, np.maximum(0, cv_mse - cv_std), cv_mse + cv_std,
                    alpha=0.18, label="CV ± 1 std")
    ax.plot(DEGREES, test_mse, "o-", label="Test MSE")
    ax.axvline(selected, color="#D62728", linestyle="--", label="Selected")

    ax.set_title("Error versus degree")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("MSE")
    ax.set_xticks(DEGREES)
    ax.grid(alpha=0.3)
    ax.legend()


def draw_residuals(ax, test_predictions, selected):
    """绘制最佳模型的测试残差。"""
    residual = test_predictions[:, 1] - test_predictions[:, selected + 1]
    ax.scatter(test_predictions[:, 0], residual, s=20, alpha=0.65)
    ax.axhline(0, color="#D62728", linestyle="--")
    ax.set_title("Test residuals")
    ax.set_xlabel("x")
    ax.set_ylabel("y - prediction")
    ax.grid(alpha=0.3)


def draw_prediction_comparison(ax, test_predictions, selected):
    """绘制测试集真实值和预测值的对比。"""
    y_true = test_predictions[:, 1]
    y_pred = test_predictions[:, selected + 1]
    lower = min(y_true.min(), y_pred.min())
    upper = max(y_true.max(), y_pred.max())

    ax.scatter(y_true, y_pred, s=20, alpha=0.65)
    ax.plot([lower, upper], [lower, upper], "r--", label="y = prediction")
    ax.set_title("Predicted versus true values")
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.grid(alpha=0.3)
    ax.legend()


def draw_cv_distribution(ax, cv_scores):
    """绘制每个阶数在各折中的CV误差分布。"""
    ax.boxplot(cv_scores[:, 1:].T, tick_labels=DEGREES)
    ax.set_title("CV error distribution")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Validation MSE")
    ax.grid(axis="y", alpha=0.3)


def draw_residual_histogram(ax, test_predictions, selected):
    """绘制最佳模型残差的直方图。"""
    residual = test_predictions[:, 1] - test_predictions[:, selected + 1]
    ax.hist(residual, bins=15, color="#4C78A8", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="#D62728", linestyle="--")
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)


def main():
    train, test, metrics, curves, test_predictions, cv_scores = read_results()
    selected = int(DEGREES[np.argmin(metrics[:, 2])])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 每个子图只负责一种结果，便于阅读。
    draw_fits(axes[0, 0], train, test, curves, selected)
    draw_errors(axes[0, 1], metrics, selected)
    draw_cv_distribution(axes[0, 2], cv_scores)
    draw_residuals(axes[1, 0], test_predictions, selected)
    draw_prediction_comparison(axes[1, 1], test_predictions, selected)
    draw_residual_histogram(axes[1, 2], test_predictions, selected)

    fig.suptitle(f"HW1 Polynomial Regression | Selected degree = {selected}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(BASE / "part1_plot.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"selected degree: {selected}")
    print(f"saved: {BASE / 'part1_plot.png'}")


if __name__ == "__main__":
    main()
