"""Q1-3：KNN 概率输出与 one-hot 输出的偏差—方差分解。

两种输出共用真实条件概率 eta 和类别维度求和的平方损失：
总误差 = 偏差平方 + 方差 + 不可约噪声。
与 Q1-1.py 放在同一目录；运行：python Q1-3.py --data-dir data1
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


K_VALUES = np.array([1, 3, 5, 9, 15, 21, 31, 51, 81, 121, 181, 251])
PILOT_K_VALUES = np.array([5, 15, 31, 61])
N_TRAIN = 900
N_REPEATS = 500
FIRST_SEED = 60000
N_CLASSES = 3


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q1 = load_module(Path(__file__).resolve().with_name("Q1-1.py"), "q1_1")
standardize_features = q1.standardize_features
DISTANCE_FUNCTIONS = q1.DISTANCE_FUNCTIONS
set_plot_style = q1.set_plot_style


def knn_probabilities(x_train, y_train, x_query, k_values, metric):
    """稳定排序找近邻，累积类别计数，一次计算所有 K 的概率。"""
    k_values = np.asarray(k_values)
    if k_values.ndim != 1 or k_values.size == 0:
        raise ValueError("K 必须是非空的一维数组。")
    if np.any(k_values != k_values.astype(int)) or np.any(k_values < 1) or np.any(k_values > len(x_train)):
        raise ValueError("K 必须是 1 到训练样本数之间的整数。")
    distances = DISTANCE_FUNCTIONS[metric](x_query, x_train)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, : int(k_values.max())]
    neighbor_labels = y_train[neighbors]
    counts = np.eye(N_CLASSES)[neighbor_labels].cumsum(axis=1)
    return np.stack([counts[:, int(k) - 1] / k for k in k_values])


def select_distance(data_dir):
    """只用 train/val，按预定 K 集合上的平均 Brier 损失选择距离。"""
    train = pd.read_csv(data_dir / "B_train.csv")
    val = pd.read_csv(data_dir / "B_val.csv")
    x_train = train[["x1", "x2"]].to_numpy(dtype=float)
    x_val = val[["x1", "x2"]].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=int)
    y_val = val["y"].to_numpy(dtype=int)
    train_scaled, val_scaled = standardize_features(x_train, x_val, enabled=True)
    targets = np.eye(N_CLASSES)[y_val]
    records = []
    for metric in DISTANCE_FUNCTIONS:
        probabilities = knn_probabilities(train_scaled, y_train, val_scaled, PILOT_K_VALUES, metric)
        for k, predicted in zip(PILOT_K_VALUES, probabilities):
            records.append({
                "metric": metric,
                "k": int(k),
                "val_brier": np.mean(np.sum((predicted - targets) ** 2, axis=1)),
                "val_accuracy": np.mean(predicted.argmax(axis=1) == y_val),
            })
    results = pd.DataFrame(records)
    means = results.groupby("metric", as_index=False)[["val_brier", "val_accuracy"]].mean()
    ranking = means.sort_values(["val_brier", "metric"], kind="stable")
    return ranking.iloc[0]["metric"], results, ranking


def load_evaluation(data_dir, generator):
    frame = pd.read_csv(data_dir / "B_evaluation_points.csv")
    x_eval = frame[["x1", "x2"]].to_numpy(dtype=float)
    eta = frame[["eta_0", "eta_1", "eta_2"]].to_numpy(dtype=float)
    if not np.isfinite(x_eval).all() or not np.isfinite(eta).all() or np.any((eta < 0) | (eta > 1)):
        raise ValueError("固定评估点或真实概率包含无效值。")
    np.testing.assert_allclose(eta.sum(axis=1), 1, atol=1e-8, rtol=0)
    eta = eta / eta.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(eta, generator.probabilities(x_eval), atol=1e-8, rtol=1e-6)
    return x_eval, eta


def repeat_training(generator, x_eval, metric, n_repeats):
    predictions = np.empty((n_repeats, len(K_VALUES), len(x_eval), N_CLASSES))
    for repeat in range(n_repeats):
        x_train, y_train = generator.generate_b(n=N_TRAIN, seed=FIRST_SEED + repeat, label_mode="noisy")
        train_scaled, eval_scaled = standardize_features(x_train, x_eval, enabled=True)
        predictions[repeat] = knn_probabilities(train_scaled, y_train, eval_scaled, K_VALUES, metric)
        if (repeat + 1) % 50 == 0 or repeat + 1 == n_repeats:
            print(f"重复抽样：{repeat + 1}/{n_repeats}", flush=True)
    np.testing.assert_allclose(predictions.sum(axis=-1), 1, atol=1e-12, rtol=0)
    return predictions


def decompose(predictions, eta):
    """输入形状为重复次数 × 评估点 × 类别；方差使用 ddof=0。"""
    mean_prediction = predictions.mean(axis=0)
    bias_squared = np.mean(np.sum((mean_prediction - eta) ** 2, axis=-1))
    variance = np.mean(np.sum((predictions - mean_prediction) ** 2, axis=-1))
    noise = np.mean(1 - np.sum(eta ** 2, axis=-1))
    reducible_error = np.mean(np.sum((predictions - eta) ** 2, axis=-1))

    # 对随机测试标签的平方误差求条件期望，独立于分量相加来计算总误差。
    conditional_loss = (
        np.sum(predictions ** 2, axis=-1)
        - 2 * np.sum(predictions * eta, axis=-1) + 1
    )
    risk_per_repeat = conditional_loss.mean(axis=1)
    total_error = risk_per_repeat.mean()
    np.testing.assert_allclose(total_error, bias_squared + variance + noise, atol=1e-12, rtol=0)
    np.testing.assert_allclose(reducible_error, bias_squared + variance, atol=1e-12, rtol=0)
    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "noise": noise,
        "reducible_error": reducible_error,
        "total_error": total_error,
        "total_error_mc_se": risk_per_repeat.std(ddof=1) / np.sqrt(len(predictions)),
        "decomposition_residual": total_error - bias_squared - variance - noise,
    }


def compare_outputs(probabilities, eta):
    records = []
    for column, k in enumerate(K_VALUES):
        soft = probabilities[:, column]
        predicted_class = soft.argmax(axis=-1)
        hard = np.eye(N_CLASSES)[predicted_class]
        if k == 1:
            np.testing.assert_array_equal(soft, hard)
        for output, predictions in (("probability", soft), ("one_hot", hard)):
            metrics = decompose(predictions, eta)
            if output == "one_hot":
                error_01 = np.mean(1 - eta[np.arange(len(eta))[None, :], predicted_class])
                np.testing.assert_allclose(metrics["total_error"], 2 * error_01, atol=1e-12, rtol=0)
            records.append({"output": output, "k": int(k), **metrics})
    return pd.DataFrame(records)


def plot_decomposition(results, output_path):
    curves = [
        ("bias_squared", "偏差平方", "#D88743"),
        ("variance", "方差", "#287F8E"),
        ("total_error", "总误差", "#665794"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.83, wspace=0.13)
    for ax, output, title in zip(axes, ["probability", "one_hot"], ["类别概率预测", "argmax 后的 one-hot 标签"]):
        data = results[results["output"] == output].sort_values("k")
        for key, label, color in curves:
            ax.plot(data.k, data[key], color=color, lw=2.5, marker="o", ms=4.5, label=label)
        ax.axhline(data.noise.iloc[0], color="#87929F", lw=1.5, ls="--", label="不可约噪声")
        ax.set(xscale="log", xlabel="近邻数量 K", title=title)
        ticks = [int(K_VALUES[0]), 5, 15, 51, int(K_VALUES[-1])]
        ax.set_xticks(sorted(set(ticks)), sorted(set(ticks)))
        ax.set_ylim(0, results.total_error.max() * 1.10)
        ax.spines[["left", "bottom"]].set_color("#A3ADB7")
        ax.grid(alpha=0.13)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("平方损失")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.99), ncol=4, frameon=False)
    fig.savefig(output_path, dpi=220, facecolor="white", transparent=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data1")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "Q1-3_results")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("重复次数至少为 2。")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric, metric_results, metric_ranking = select_distance(args.data_dir)
    print("距离验证结果：\n" + metric_ranking.to_string(index=False), flush=True)
    print(f"固定距离：{metric}", flush=True)
    generator = load_module(args.data_dir / "B_generator.py", "b_generator")
    x_eval, eta = load_evaluation(args.data_dir, generator)
    probabilities = repeat_training(generator, x_eval, metric, args.repeats)
    results = compare_outputs(probabilities, eta)

    metric_results.to_csv(args.output_dir / "metric_validation.csv", index=False)
    results.to_csv(args.output_dir / "decomposition.csv", index=False)
    settings = {
        "n_train_per_repeat": N_TRAIN,
        "n_repeats": args.repeats,
        "first_seed": FIRST_SEED,
        "last_seed": FIRST_SEED + args.repeats - 1,
        "k_values": K_VALUES.tolist(),
        "metric": metric,
        "metric_selection": "Minimum mean validation Brier over pilot K values; tie by metric name.",
        "pilot_k_values": PILOT_K_VALUES.tolist(),
        "metric_ranking": metric_ranking.to_dict(orient="records"),
        "standardize": "Fit separately on each training sample; transform evaluation coordinates.",
        "label_mode": "noisy",
        "evaluation_points": len(x_eval),
        "evaluation_source": "B_evaluation_points.csv; eta checked against generator in original coordinates.",
        "data_roles": {
            "B_train/B_val": "Distance selection only, before repeated sampling.",
            "B_generator": "Independent training samples, shared across all K and both outputs.",
            "B_evaluation_points": "Fixed evaluation points and true conditional probabilities; never train.",
            "B_test": "Not used for this decomposition experiment.",
        },
        "loss": "Squared Euclidean distance to random one-hot Y; sum across classes.",
        "total_error": "bias_squared + variance + noise",
        "reducible_error": "Squared error to eta = bias_squared + variance",
        "variance_ddof": 0,
        "estimation": "Finite-repeat Monte Carlo estimates, averaged over the fixed evaluation points.",
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    set_plot_style()
    plot_decomposition(results, args.output_dir / "bias_variance.png")
    print(results[["output", "k", "bias_squared", "variance", "noise", "total_error"]].to_string(index=False))
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
