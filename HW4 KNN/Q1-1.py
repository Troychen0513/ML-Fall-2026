"""Q1-1：KNN 的近邻数量、距离、标准化与 PCA 维数比较。

依赖：numpy pandas scipy scikit-learn matplotlib
运行：python Q1-1.py --data-dir data1
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


K_VALUES = np.array([1, 3, 5, 7, 11, 15, 21, 31, 51])
PCA_DIMS = np.array([2, 5, 10, 20, 40, 80, 160])
STANDARDIZE_OPTIONS = (False, True)
COLORS = {"raw": "#D88743", "standardized": "#287F8E", "ink": "#203448"}


def load_data(data_dir):
    data = {}
    feature_names = None
    for split in ("train", "val", "test"):
        path = data_dir / f"A_{split}.csv"
        frame = pd.read_csv(path)
        if "y" not in frame or frame.empty or frame.isna().any().any():
            raise ValueError(f"{path.name} 必须包含 y 列、非空样本且没有缺失值。")
        features = frame.drop(columns="y")
        if feature_names is None:
            feature_names = list(features.columns)
        if list(features.columns) != feature_names:
            raise ValueError("train、val、test 的特征列及顺序必须相同。")
        x = features.to_numpy(dtype=float)
        y = frame["y"].to_numpy()
        if not np.isfinite(x).all():
            raise ValueError(f"{path.name} 包含非有限特征值。")
        data[split] = (x, y)
    if len(np.unique(data["train"][1])) != 2:
        raise ValueError("本题的预处理诊断图要求 A 为二分类数据。")
    return data, feature_names


def standardize_features(x_train, x_query, enabled):
    if not enabled:
        return x_train, x_query
    scaler = StandardScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_query)


def reduce_dimension(x_train, x_query, n_components):
    """PCA 只在训练集拟合；None 表示保留原始特征、不做 PCA。"""
    if n_components is None:
        return x_train, x_query, None
    pca = PCA(n_components=n_components, svd_solver="full")
    return pca.fit_transform(x_train), pca.transform(x_query), pca


def euclidean_distance(x_query, x_train):
    return cdist(x_query, x_train, metric="euclidean")


def manhattan_distance(x_query, x_train):
    return cdist(x_query, x_train, metric="cityblock")


def cosine_distance(x_query, x_train):
    query_norm = np.linalg.norm(x_query, axis=1, keepdims=True)
    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    query_unit = np.divide(x_query, query_norm, out=np.zeros_like(x_query, dtype=float), where=query_norm > 0)
    train_unit = np.divide(x_train, train_norm, out=np.zeros_like(x_train, dtype=float), where=train_norm > 0)
    distances = np.clip(1 - query_unit @ train_unit.T, 0, 2)
    # 约定：两个零向量距离为 0；零向量与非零向量距离为 1。
    distances[np.ix_(query_norm.ravel() == 0, train_norm.ravel() == 0)] = 0
    return distances


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "cosine": cosine_distance,
}
DISTANCE_LABELS = {"euclidean": "欧氏距离", "manhattan": "曼哈顿距离", "cosine": "余弦距离"}


def knn_predict(x_train, y_train, x_query, k_values, metric):
    """距离 → 近邻 → 投票；距离并列按训练行序，票数并列取较小标签。"""
    k_values = np.asarray(k_values)
    if k_values.size == 0 or k_values.ndim != 1:
        raise ValueError("K 必须是非空的一维数组。")
    if np.any(k_values != k_values.astype(int)) or np.any(k_values < 1) or np.any(k_values > len(x_train)):
        raise ValueError("K 必须是 1 到训练样本数之间的整数。")
    classes, encoded_labels = np.unique(y_train, return_inverse=True)
    distances = DISTANCE_FUNCTIONS[metric](x_query, x_train)
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, : int(k_values.max())]
    neighbor_labels = encoded_labels[neighbor_indices]

    predictions = {}
    for k in k_values.astype(int):
        votes = np.column_stack([
            np.sum(neighbor_labels[:, :k] == label, axis=1)
            for label in range(len(classes))
        ])
        predictions[int(k)] = classes[np.argmax(votes, axis=1)]
    return predictions


def run_experiments(x_train, y_train, x_val, y_val):
    max_components = min(x_train.shape[0] - 1, x_train.shape[1])
    if np.any(PCA_DIMS < 1) or np.any(PCA_DIMS > max_components):
        raise ValueError(f"PCA_DIMS 必须介于 1 和 {max_components} 之间。")
    records, diagnostics = [], {}
    for standardize in STANDARDIZE_OPTIONS:
        train_scaled, val_scaled = standardize_features(x_train, x_val, standardize)
        train_pca, val_pca, pca = reduce_dimension(train_scaled, val_scaled, max_components)
        diagnostics[standardize] = {
            "variance_ratio": pca.explained_variance_ratio_,
        }
        representations = [("Original", train_scaled, val_scaled)]
        for dim in PCA_DIMS:
            representations.append((f"PCA {dim}", train_pca[:, :dim], val_pca[:, :dim]))

        for representation, train_features, val_features in representations:
            for metric in DISTANCE_FUNCTIONS:
                predictions = knn_predict(train_features, y_train, val_features, K_VALUES, metric)
                for k, predicted in predictions.items():
                    records.append({
                        "standardize": standardize,
                        "representation": representation,
                        "n_features": train_features.shape[1],
                        "metric": metric,
                        "k": k,
                        "val_accuracy": accuracy_score(y_val, predicted),
                        "val_macro_f1": f1_score(y_val, predicted, average="macro", zero_division=0),
                    })
        print(f"完成验证集实验：standardize={standardize}", flush=True)
    return pd.DataFrame(records), diagnostics


def select_best(results):
    # 验证准确率并列时依次选择：低维、大 K、距离名称、未标准化。
    ranked = results.sort_values(
        ["val_accuracy", "n_features", "k", "metric", "standardize"],
        ascending=[False, True, False, True, True],
        kind="stable",
    )
    return ranked.iloc[0]


def evaluate_test(data, best):
    x_train, y_train = data["train"]
    x_test, y_test = data["test"]
    standardize = bool(best["standardize"])
    n_components = None if best["representation"] == "Original" else int(best["n_features"])
    train_features, test_features = standardize_features(x_train, x_test, standardize)
    train_features, test_features, _ = reduce_dimension(train_features, test_features, n_components)
    k = int(best["k"])
    predicted = knn_predict(train_features, y_train, test_features, [k], best["metric"])[k]
    classes = np.unique(y_train)
    summary = {
        "selected_by": "validation accuracy; ties: fewer features, larger K, metric name, no standardization",
        "training_samples": len(y_train),
        "standardize": standardize,
        "representation": best["representation"],
        "n_features": int(best["n_features"]),
        "metric": best["metric"],
        "k": k,
        "val_accuracy": float(best["val_accuracy"]),
        "val_macro_f1": float(best["val_macro_f1"]),
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        "class_order": classes.tolist(),
        "test_confusion_matrix": confusion_matrix(y_test, predicted, labels=classes).tolist(),
    }
    return summary, pd.DataFrame({"y_true": y_test, "y_pred": predicted})


def set_plot_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
    })


def plot_preprocessing(x_train, y_train, diagnostics, output_path):
    """训练集二维 PCA 点云与类条件密度；等高线为峰值的 25%、50%、75%。"""
    options = list(diagnostics)
    classes, encoded = np.unique(y_train, return_inverse=True)
    palette = [COLORS["raw"], COLORS["standardized"]]
    point_colors = np.array(palette)[encoded]
    order = np.random.default_rng(0).permutation(len(y_train))
    fig, axes = plt.subplots(1, len(options), figsize=(6.4 * len(options), 5.1), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.86, wspace=0.25)

    for ax, standardize in zip(axes.ravel(), options):
        scaled, _ = standardize_features(x_train, x_train[:1], standardize)
        projection, _, pca = reduce_dimension(scaled, scaled[:1], 2)
        lower, upper = projection.min(axis=0), projection.max(axis=0)
        margin = (upper - lower) * 0.04
        lower, upper = lower - margin, upper + margin
        grid_x, grid_y = np.meshgrid(
            np.linspace(lower[0], upper[0], 100),
            np.linspace(lower[1], upper[1], 100),
        )
        grid = np.vstack([grid_x.ravel(), grid_y.ravel()])
        ax.scatter(
            projection[order, 0], projection[order, 1], s=14,
            c=point_colors[order], alpha=0.22, edgecolors="none",
        )
        for label, color in zip(classes, palette):
            points = projection[y_train == label]
            density = gaussian_kde(points.T)(grid).reshape(grid_x.shape)
            ax.contour(
                grid_x, grid_y, density, levels=np.array([0.25, 0.5, 0.75]) * density.max(),
                colors=[color], linewidths=[0.9, 1.3, 1.7], alpha=0.9,
            )
        ax.set(
            xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
            title="标准化后" if standardize else "未标准化",
            xlabel=f"PC1（{pca.explained_variance_ratio_[0]:.1%}）",
            ylabel=f"PC2（{pca.explained_variance_ratio_[1]:.1%}）",
        )
        ax.spines[["left", "bottom"]].set_color("#A3ADB7")
        ax.spines[["left", "bottom"]].set_linewidth(0.8)
        ax.grid(alpha=0.12)
        ax.set_axisbelow(True)
    handles = [
        Line2D([], [], color=color, lw=2, label=f"类别 {str(label).removesuffix('.0')}")
        for label, color in zip(classes, palette)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
    fig.savefig(output_path, dpi=200, transparent=True)
    plt.close(fig)


def plot_results(results, best, output_path):
    """两幅热力图呈现整个验证网格，测试集只报告最终选定方案。"""
    representations = ["Original"] + [f"PCA {dim}" for dim in PCA_DIMS]
    metrics = list(DISTANCE_FUNCTIONS)
    index = pd.MultiIndex.from_product([metrics, K_VALUES], names=["metric", "k"])
    original_dim = int(results.loc[results.representation == "Original", "n_features"].iloc[0])
    labels = [f"原始\n{original_dim} 维"] + [f"PCA\n{dim} 维" for dim in PCA_DIMS]
    scores = results["val_accuracy"] * 100
    norm = Normalize(vmin=np.floor(scores.min() / 5) * 5, vmax=np.ceil(scores.max() / 5) * 5)
    fig, axes = plt.subplots(1, len(STANDARDIZE_OPTIONS), figsize=(15, 11), squeeze=False)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.09, top=0.94, wspace=0.31)

    for ax, standardize in zip(axes, STANDARDIZE_OPTIONS):
        subset = results[results["standardize"] == standardize]
        table = subset.pivot(index=["metric", "k"], columns="representation", values="val_accuracy")
        table = table.reindex(index=index, columns=representations)
        matrix = table.to_numpy() * 100
        im = ax.imshow(matrix, cmap="YlGnBu", norm=norm, aspect="auto")
        ax.set_title("标准化后" if standardize else "未标准化", pad=15, weight="bold")
        ax.set_xticks(range(len(labels)), labels, fontsize=10)
        ax.set_yticks(range(len(index)), [f"K={k}" for _, k in index], fontsize=10)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7, alpha=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        for group, metric in enumerate(metrics):
            center = group * len(K_VALUES) + (len(K_VALUES) - 1) / 2
            ax.text(
                -0.18, 1 - (center + 0.5) / len(index), DISTANCE_LABELS[metric],
                transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=12,
            )
            if group:
                ax.axhline(group * len(K_VALUES) - 0.5, color="white", lw=4)
        if standardize == bool(best["standardize"]):
            row = index.get_loc((best["metric"], best["k"]))
            col = representations.index(best["representation"])
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, ec="#F1AF38", lw=2.5))
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(im, ax=axes, location="right", fraction=0.025, pad=0.025, shrink=0.65)
    colorbar.set_label("验证准确率（%）", labelpad=12)
    colorbar.outline.set_visible(False)
    fig.savefig(output_path, dpi=200, transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data1")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "Q1-1_results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, _ = load_data(args.data_dir)
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]

    results, diagnostics = run_experiments(x_train, y_train, x_val, y_val)
    best = select_best(results)
    summary, predictions = evaluate_test(data, best)
    summary["grid"] = {
        "k_values": K_VALUES.tolist(),
        "metrics": list(DISTANCE_FUNCTIONS),
        "standardize": list(STANDARDIZE_OPTIONS),
        "pca_dimensions": PCA_DIMS.tolist(),
    }
    results.to_csv(args.output_dir / "validation_results.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "best_model.json").write_text(summary_json, encoding="utf-8")
    set_plot_style()
    plot_preprocessing(x_train, y_train, diagnostics, args.output_dir / "preprocessing.png")
    plot_results(results, best, args.output_dir / "comparison.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存至：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
