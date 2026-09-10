"""绘制第二部分的偏差-方差实验结果。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parent
DEGREES = [1, 6, 8]


def load_results():
    result = np.load(BASE / "part2_results.npz")
    metrics = np.loadtxt(BASE / "part2_metrics.csv", delimiter=",", skiprows=1)
    return result, metrics


def plot_prediction_curves(result):
    """分别绘制三种模型的重复预测、平均预测和真实函数。"""
    x = result["x_eval"]
    y_true = result["y_true"]
    predictions = result["predictions"]
    mean_prediction = result["mean_prediction"]
    std = np.sqrt(result["pointwise_variance"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for i, degree in enumerate(DEGREES):
        ax = axes[i]
        for curve in predictions[::10, i]:
            ax.plot(x, curve, color="#9ECAE1", alpha=0.28, linewidth=0.8)
        ax.plot(x, y_true, color="#222222", linewidth=2, label="True response")
        ax.plot(x, mean_prediction[i], color="#D62728", linewidth=2, label="Mean prediction")
        ax.fill_between(x, mean_prediction[i] - std[i], mean_prediction[i] + std[i],
                        color="#D62728", alpha=0.18, label="±1 std")
        ax.set_title(f"Degree {degree}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("response")
    axes[0].legend(fontsize=8)
    fig.suptitle("Repeated predictions and model stability")
    fig.tight_layout()
    fig.savefig(BASE / "part2_prediction_curves.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_bias_variance(metrics):
    """比较偏差平方、方差和两种预测误差。"""
    degrees = metrics[:, 0].astype(int)
    bias = metrics[:, 1]
    variance = metrics[:, 2]
    mse_true = metrics[:, 3]
    mse_observed = metrics[:, 4]
    bias_plus_variance = metrics[:, 5]

    x = np.arange(len(degrees))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * width, bias, width, label="Bias²")
    ax.bar(x - 0.5 * width, variance, width, label="Variance")
    ax.bar(x + 0.5 * width, mse_true, width, label="MSE(true)")
    ax.bar(x + 1.5 * width, mse_observed, width, label="MSE(observed)")
    ax.plot(x, bias_plus_variance, "ko--", label="Bias² + Variance")
    ax.set_xticks(x)
    ax.set_xticklabels(degrees)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Error")
    ax.set_title("Bias-variance comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / "part2_bias_variance.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_error_distributions(result):
    """比较每次重复实验的测试误差分布。"""
    true_errors = result["run_mse_true"]
    observed_errors = result["run_mse_observed"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].boxplot(true_errors, tick_labels=DEGREES)
    axes[0].set_title("MSE against true response")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(observed_errors, tick_labels=DEGREES)
    axes[1].set_title("MSE against observed response")
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Prediction error across repeated samples")
    fig.tight_layout()
    fig.savefig(BASE / "part2_error_distributions.png", dpi=200, bbox_inches="tight")
    plt.show()


def main():
    result, metrics = load_results()
    plot_prediction_curves(result)
    plot_bias_variance(metrics)
    plot_error_distributions(result)
    print("saved: part2_prediction_curves.png")
    print("saved: part2_bias_variance.png")
    print("saved: part2_error_distributions.png")


if __name__ == "__main__":
    main()
