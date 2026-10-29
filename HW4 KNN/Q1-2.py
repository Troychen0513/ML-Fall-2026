"""Q1-2：比较 K 和训练样本变化对二维 KNN 分类边界的影响。

与 Q1-1.py 放在同一目录；运行：python Q1-2.py --data-dir data1
"""

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.lines import Line2D


N_TRAIN = 900
SEEDS = np.array([50000, 50001, 50002])
K_VALUES = np.array([1, 15, 101])
METRIC = "euclidean"
LABEL_MODE = "noisy"
GRID_SIZE = 240
BATCH_SIZE = 1024
LIMITS = (-2, 2)
CLASS_COLORS = ["#CB784F", "#3B9294", "#8580BA"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q1 = load_module(Path(__file__).resolve().with_name("Q1-1.py"), "q1_1")
knn_predict = q1.knn_predict
standardize_features = q1.standardize_features
set_plot_style = q1.set_plot_style


def make_grid(grid_size):
    edges = np.linspace(*LIMITS, grid_size + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    grid_x, grid_y = np.meshgrid(centers, centers)
    return edges, np.column_stack([grid_x.ravel(), grid_y.ravel()])


def predict_grid(x_train, y_train, grid, grid_size):
    """每组样本单独拟合标准化；分批预测以控制距离矩阵的内存。"""
    train_scaled, grid_scaled = standardize_features(x_train, grid, enabled=True)
    predictions = np.empty((len(K_VALUES), len(grid)), dtype=np.uint8)
    for start in range(0, len(grid), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(grid))
        batch = knn_predict(train_scaled, y_train, grid_scaled[start:stop], K_VALUES, METRIC)
        for column, k in enumerate(K_VALUES):
            predictions[column, start:stop] = batch[int(k)]
    return predictions.reshape(len(K_VALUES), grid_size, grid_size)


def boundary_segments(labels, edges):
    """沿相邻网格的类别变化画线，避免把类别编号当成连续数值插值。"""
    rows, cols = np.nonzero(labels[:, :-1] != labels[:, 1:])
    vertical = np.stack([
        np.column_stack([edges[cols + 1], edges[rows]]),
        np.column_stack([edges[cols + 1], edges[rows + 1]]),
    ], axis=1)
    rows, cols = np.nonzero(labels[:-1, :] != labels[1:, :])
    horizontal = np.stack([
        np.column_stack([edges[cols], edges[rows + 1]]),
        np.column_stack([edges[cols + 1], edges[rows + 1]]),
    ], axis=1)
    return np.concatenate([vertical, horizontal])


def sampling_sensitivity(predictions):
    """同一网格上不同训练样本的预测不一致率；不代表分类错误率。"""
    pairs = list(combinations(range(len(predictions)), 2))
    records = []
    for column, k in enumerate(K_VALUES):
        disagreement = [
            np.mean(predictions[i, column] != predictions[j, column])
            for i, j in pairs
        ]
        records.append({
            "k": int(k),
            "mean_pairwise_disagreement": float(np.mean(disagreement)),
        })
    return pd.DataFrame(records)


def plot_boundaries(training_sets, predictions, edges, output_path):
    fill_colors = ListedColormap([to_rgba(color, 0.17) for color in CLASS_COLORS])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], len(CLASS_COLORS))
    fig, axes = plt.subplots(
        len(training_sets), len(K_VALUES), figsize=(4.2 * len(K_VALUES), 4 * len(training_sets)),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.92, wspace=0.08, hspace=0.10)
    for row, (x_train, y_train) in enumerate(training_sets):
        for column, k in enumerate(K_VALUES):
            ax = axes[row, column]
            labels = predictions[row, column]
            ax.pcolormesh(edges, edges, labels, cmap=fill_colors, norm=norm, shading="flat")
            ax.add_collection(LineCollection(
                boundary_segments(labels, edges), colors="#45576A", linewidths=0.65, alpha=0.72,
            ))
            ax.scatter(
                x_train[:, 0], x_train[:, 1], c=np.array(CLASS_COLORS)[y_train],
                s=7, alpha=0.72, edgecolors="none",
            )
            ax.set(xlim=LIMITS, ylim=LIMITS, aspect="equal", xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            ax.tick_params(length=0, labelsize=10)
            ax.spines[["left", "bottom"]].set_color("#A3ADB7")
            if row == 0:
                ax.set_title(f"K = {k}", fontsize=17, pad=14)
            if column == 0:
                ax.set_ylabel(f"样本 {row + 1}", fontsize=14, labelpad=13)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, label=f"类别 {label}", markersize=7)
        for label, color in enumerate(CLASS_COLORS)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.997), ncol=3, frameon=False)
    fig.supxlabel("$x_1$", fontsize=15, y=0.015)
    fig.supylabel("$x_2$", fontsize=15, x=0.015)
    fig.savefig(output_path, dpi=220, facecolor="white", transparent=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data1")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "Q1-2_results")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    args = parser.parse_args()
    if args.grid_size < 2 or len(SEEDS) < 2 or len(np.unique(SEEDS)) != len(SEEDS):
        raise ValueError("网格边长至少为 2，且需要至少两个不同的随机种子。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = load_module(args.data_dir / "B_generator.py", "b_generator")
    edges, grid = make_grid(args.grid_size)
    training_sets, predictions, frames = [], [], []

    for seed in SEEDS:
        x_train, y_train = generator.generate_b(n=N_TRAIN, seed=int(seed), label_mode=LABEL_MODE)
        training_sets.append((x_train, y_train))
        predictions.append(predict_grid(x_train, y_train, grid, args.grid_size))
        frames.append(pd.DataFrame({
            "seed": int(seed), "x_1": x_train[:, 0], "x_2": x_train[:, 1], "y": y_train,
        }))
        print(f"完成样本：seed={seed}, n={len(y_train)}", flush=True)

    predictions = np.stack(predictions)
    sensitivity = sampling_sensitivity(predictions)
    pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "training_samples.csv", index=False)
    sensitivity.to_csv(args.output_dir / "sampling_sensitivity.csv", index=False)
    np.savez_compressed(
        args.output_dir / "grid_predictions.npz", edges=edges, predictions=predictions,
        seeds=SEEDS, k_values=K_VALUES,
    )
    settings = {
        "n_train": N_TRAIN,
        "seeds": SEEDS.tolist(),
        "k_values": K_VALUES.tolist(),
        "metric": METRIC,
        "standardize": True,
        "label_mode": LABEL_MODE,
        "grid_size": args.grid_size,
        "limits": list(LIMITS),
        "row_order": "seeds",
        "column_order": "k_values",
        "sensitivity_measure": "Mean pairwise prediction disagreement on the plotting grid; not accuracy.",
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    set_plot_style()
    plot_boundaries(training_sets, predictions, edges, args.output_dir / "decision_boundaries.png")
    print(sensitivity.to_string(index=False))
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
