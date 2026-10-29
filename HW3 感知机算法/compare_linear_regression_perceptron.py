from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/Users/tuyuchen/Downloads/HW3 感知机算法")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "linear_regression_comparison_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = ["A", "B"]
ORDER_RUNS = 12
ORDER_SEED = 2026
LEARNING_RATE = 1.0
MAX_EPOCHS = 10000

plt.rcParams["axes.unicode_minus"] = False


def load_dataset(name):
    train = pd.read_csv(DATA_DIR / f"{name}_train.csv")
    test = pd.read_csv(DATA_DIR / f"{name}_test.csv")
    feature_names = [column for column in train.columns if column.startswith("x")]

    X_train = train[feature_names].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    X_test = test[feature_names].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)

    X_train_aug = np.c_[X_train, np.ones(len(X_train))]
    X_test_aug = np.c_[X_test, np.ones(len(X_test))]
    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_aug,
        X_test_aug,
        feature_names,
        train,
        test,
    )


def predict(X_aug, theta):
    return np.where(X_aug @ theta >= 0, 1, -1)


def evaluate(theta, X_train_aug, y_train, X_test_aug, y_test):
    train_score = X_train_aug @ theta
    test_score = X_test_aug @ theta
    train_prediction = np.where(train_score >= 0, 1, -1)
    test_prediction = np.where(test_score >= 0, 1, -1)
    weight_norm = np.linalg.norm(theta[:-1])

    return {
        "train_accuracy": float(np.mean(train_prediction == y_train)),
        "test_accuracy": float(np.mean(test_prediction == y_test)),
        "train_mse": float(np.mean((train_score - y_train) ** 2)),
        "test_mse": float(np.mean((test_score - y_test) ** 2)),
        "train_mistakes": int(np.sum(train_prediction != y_train)),
        "test_mistakes": int(np.sum(test_prediction != y_test)),
        "minimum_train_margin": float(np.min(y_train * train_score)),
        "geometric_train_margin": float(
            np.min(y_train * train_score) / weight_norm
        )
        if weight_norm > 0
        else 0.0,
        "weight_norm": float(weight_norm),
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
    }


def train_perceptron(
    X_train_aug,
    y_train,
    X_test_aug,
    y_test,
    order,
    eta=LEARNING_RATE,
    max_epochs=MAX_EPOCHS,
):
    theta = np.zeros(X_train_aug.shape[1], dtype=float)
    initial_result = evaluate(theta, X_train_aug, y_train, X_test_aug, y_test)
    history = [{
        "update": 0,
        "epoch": 0,
        "mistakes": initial_result["train_mistakes"],
        "train_accuracy": initial_result["train_accuracy"],
        "test_accuracy": initial_result["test_accuracy"],
    }]
    update_count = 0

    for epoch in range(1, max_epochs + 1):
        epoch_updates = 0

        for index in order:
            margin = y_train[index] * (X_train_aug[index] @ theta)
            if margin <= 0:
                theta = theta + eta * y_train[index] * X_train_aug[index]
                update_count += 1
                epoch_updates += 1

                result = evaluate(theta, X_train_aug, y_train, X_test_aug, y_test)
                history.append(
                    {
                        "update": update_count,
                        "epoch": epoch,
                        "mistakes": result["train_mistakes"],
                        "train_accuracy": result["train_accuracy"],
                        "test_accuracy": result["test_accuracy"],
                    }
                )

        if epoch_updates == 0:
            break

    result = evaluate(theta, X_train_aug, y_train, X_test_aug, y_test)
    result.update(
        {
            "theta": theta,
            "history": history,
            "epochs": epoch,
            "updates": update_count,
            "converged": epoch_updates == 0,
        }
    )
    return result


def train_least_squares(X_train_aug, y_train, X_test_aug, y_test):
    # A、B 的扩展设计矩阵均满列秩，因此直接求解正规方程即可。
    gram = X_train_aug.T @ X_train_aug
    theta = np.linalg.solve(gram, X_train_aug.T @ y_train)
    result = evaluate(theta, X_train_aug, y_train, X_test_aug, y_test)
    result.update({"theta": theta})
    return result


def format_expression(theta, feature_names):
    terms = [f"{theta[index]:+.6f}*{name}" for index, name in enumerate(feature_names)]
    terms.append(f"{theta[-1]:+.6f}")
    return "sign(" + " ".join(terms) + ")"


def boundary_y(theta, x_values, fixed_values=None):
    weights = theta[:-1]
    bias = theta[-1]
    if fixed_values is None:
        return -(weights[0] * x_values + bias) / weights[1]
    adjusted_bias = bias + np.dot(weights[2:], fixed_values)
    return -(weights[0] * x_values + adjusted_bias) / weights[1]


def plot_model_comparison(model_results):
    A = load_dataset("A")
    B = load_dataset("B")
    _, y_A_train, X_A_test, y_A_test, X_A_train_aug, X_A_test_aug, names_A, train_A, test_A = A
    X_A_train = train_A[names_A].to_numpy(dtype=float)
    _, y_B_train, X_B_test, y_B_test, X_B_train_aug, X_B_test_aug, names_B, train_B, test_B = B
    X_B_train = train_B[names_B].to_numpy(dtype=float)

    # Figure 4-1 retains only the two decision-boundary panels.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Dataset A: actual two-dimensional decision boundaries.
    ax = axes[0]
    for label, color in [(-1, "#377eb8"), (1, "#f28e2b")]:
        ax.scatter(
            X_A_train[y_A_train == label, 0],
            X_A_train[y_A_train == label, 1],
            s=20,
            alpha=0.55,
            color=color,
            label=f"train y={label}",
        )
        ax.scatter(
            X_A_test[y_A_test == label, 0],
            X_A_test[y_A_test == label, 1],
            s=35,
            marker="x",
            color=color,
            alpha=0.8,
            label=f"test y={label}",
        )

    x_line = np.linspace(X_A_train[:, 0].min() - 0.4, X_A_train[:, 0].max() + 0.4, 300)
    for algorithm, color, linestyle in [
        ("Perceptron", "#222222", "-"),
        ("Least squares", "#d62728", "--"),
    ]:
        theta = model_results[("A", algorithm)]["theta"]
        if abs(theta[1]) > 1e-12:
            ax.plot(
                x_line,
                boundary_y(theta, x_line),
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=algorithm,
            )
    ax.set_title("Dataset A: decision boundaries")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)

    # Dataset B: x1-x2 slice with x3...x6 fixed at training means.
    ax = axes[1]
    fixed_B = X_B_train[:, 2:].mean(axis=0)
    for label, color in [(-1, "#377eb8"), (1, "#f28e2b")]:
        ax.scatter(
            X_B_train[y_B_train == label, 0],
            X_B_train[y_B_train == label, 1],
            s=20,
            alpha=0.55,
            color=color,
            label=f"y={label}",
        )
    x_line = np.linspace(X_B_train[:, 0].min() - 0.5, X_B_train[:, 0].max() + 0.5, 300)
    for algorithm, color, linestyle in [
        ("Perceptron", "#222222", "-"),
        ("Least squares", "#d62728", "--"),
    ]:
        theta = model_results[("B", algorithm)]["theta"]
        if abs(theta[1]) > 1e-12:
            ax.plot(
                x_line,
                boundary_y(theta, x_line, fixed_B),
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=algorithm,
            )
    ax.set_title("Dataset B: x1-x2 slice, x3...x6 fixed at train mean")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

def run_order_experiment(name):
    (
        _,
        y_train,
        _,
        _,
        X_train_aug,
        X_test_aug,
        feature_names,
        train,
        test,
    ) = load_dataset(name)
    y_test = test["y"].to_numpy(dtype=float)
    rng = np.random.default_rng(ORDER_SEED)
    orders = [np.arange(len(y_train))]
    orders.extend(rng.permutation(len(y_train)) for _ in range(ORDER_RUNS - 1))

    results = []
    for order_id, order in enumerate(orders):
        result = train_perceptron(
            X_train_aug,
            y_train,
            X_test_aug,
            y_test,
            order,
        )
        result["order_id"] = order_id
        result["order"] = order
        result["feature_names"] = feature_names
        results.append(result)
    return results


def pairwise_order_metrics(results, X_test_aug, name):
    rows = []
    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            theta_left = results[left]["theta"]
            theta_right = results[right]["theta"]
            w_left = theta_left[:-1]
            w_right = theta_right[:-1]
            norm_product = np.linalg.norm(w_left) * np.linalg.norm(w_right)
            cosine = float(np.dot(w_left, w_right) / norm_product) if norm_product else 0.0
            prediction_left = predict(X_test_aug, theta_left)
            prediction_right = predict(X_test_aug, theta_right)
            rows.append(
                {
                    "dataset": name,
                    "left_order": results[left]["order_id"],
                    "right_order": results[right]["order_id"],
                    "parameter_cosine": cosine,
                    "test_disagreement_rate": float(
                        np.mean(prediction_left != prediction_right)
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_order_experiment(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, name in zip(axes, DATASETS):
        for result in all_results[name]:
            history = result["history"]
            x = [item["update"] for item in history]
            y = [item["mistakes"] for item in history]
            ax.plot(x, y, linewidth=1.2, alpha=0.55)
        ax.set_title(f"Dataset {name}: learning curves by sample order")
        ax.set_xlabel("Perceptron update")
        ax.set_ylabel("Training mistakes")
        ax.grid(alpha=0.2)
    fig.savefig(OUTPUT_DIR / "order_learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 4-4 combines the final boundaries for datasets A and B.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    results_A = all_results["A"]
    X_train_A, y_train_A, _, _, _, _, _, _, _ = load_dataset("A")
    ax = axes[0]
    for label, color in [(-1, "#377eb8"), (1, "#f28e2b")]:
        ax.scatter(
            X_train_A[y_train_A == label, 0],
            X_train_A[y_train_A == label, 1],
            s=20,
            alpha=0.55,
            color=color,
            label=f"y={label}",
        )
    x_line = np.linspace(X_train_A[:, 0].min() - 0.4, X_train_A[:, 0].max() + 0.4, 300)
    for result in results_A:
        theta = result["theta"]
        ax.plot(x_line, boundary_y(theta, x_line), linewidth=1.2, alpha=0.5)
    ax.set_title("Dataset A: final boundaries")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.legend()

    results_B = all_results["B"]
    X_train_B, y_train_B, _, _, _, _, _, _, _ = load_dataset("B")
    fixed_B = X_train_B[:, 2:].mean(axis=0)
    ax = axes[1]
    for label, color in [(-1, "#377eb8"), (1, "#f28e2b")]:
        ax.scatter(
            X_train_B[y_train_B == label, 0],
            X_train_B[y_train_B == label, 1],
            s=20,
            alpha=0.55,
            color=color,
            label=f"y={label}",
        )
    x_line = np.linspace(X_train_B[:, 0].min() - 0.5, X_train_B[:, 0].max() + 0.5, 300)
    for result in results_B:
        theta = result["theta"]
        ax.plot(x_line, boundary_y(theta, x_line, fixed_B), linewidth=1.2, alpha=0.5)
    ax.set_title("Dataset B: final boundaries, x1-x2 slice")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.savefig(OUTPUT_DIR / "order_final_boundaries_combined.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

def main():
    model_results = {}
    model_rows = []
    expression_lines = []

    for name in DATASETS:
        (
            _,
            y_train,
            _,
            y_test,
            X_train_aug,
            X_test_aug,
            feature_names,
            train,
            test,
        ) = load_dataset(name)

        perceptron = train_perceptron(
            X_train_aug,
            y_train,
            X_test_aug,
            y_test,
            np.arange(len(y_train)),
        )
        least_squares = train_least_squares(
            X_train_aug,
            y_train,
            X_test_aug,
            y_test,
        )

        model_results[(name, "Perceptron")] = perceptron
        model_results[(name, "Least squares")] = least_squares

        for algorithm, result in [
            ("Perceptron", perceptron),
            ("Least squares", least_squares),
        ]:
            row = {
                "dataset": name,
                "algorithm": algorithm,
                "train_shape": str(train.shape),
                "test_shape": str(test.shape),
                "updates": result.get("updates", 0),
                "epochs": result.get("epochs", ""),
                "converged": result.get("converged", ""),
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "train_mse": result["train_mse"],
                "test_mse": result["test_mse"],
                "train_mistakes": result["train_mistakes"],
                "test_mistakes": result["test_mistakes"],
                "minimum_train_margin": result["minimum_train_margin"],
                "geometric_train_margin": result["geometric_train_margin"],
                "weight_norm": result["weight_norm"],
            }
            model_rows.append(row)
            expression_lines.append(
                f"{name} - {algorithm}: {format_expression(result['theta'], feature_names)}"
            )

    model_table = pd.DataFrame(model_rows)
    model_table.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    (OUTPUT_DIR / "model_expressions.txt").write_text("\n".join(expression_lines))
    plot_model_comparison(model_results)

    all_order_results = {}
    order_rows = []
    pairwise_tables = []
    for name in DATASETS:
        results = run_order_experiment(name)
        all_order_results[name] = results
        _, _, _, y_test, _, X_test_aug, _, _, _ = load_dataset(name)
        pairwise_tables.append(
            pairwise_order_metrics(results, X_test_aug, name)
        )

        for result in results:
            row = {
                "dataset": name,
                "order_id": result["order_id"],
                "epochs": result["epochs"],
                "updates": result["updates"],
                "converged": result["converged"],
                "train_accuracy": result["train_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "minimum_train_margin": result["minimum_train_margin"],
                "geometric_train_margin": result["geometric_train_margin"],
                "weight_norm": result["weight_norm"],
            }
            order_rows.append(row)

    order_table = pd.DataFrame(order_rows)
    pairwise_table = pd.concat(pairwise_tables, ignore_index=True)
    order_table.to_csv(OUTPUT_DIR / "order_experiment_results.csv", index=False)
    pairwise_table.to_csv(OUTPUT_DIR / "order_pairwise_metrics.csv", index=False)
    plot_order_experiment(all_order_results)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda value: f"{value:.4f}")

    print("模型表达式")
    for line in expression_lines:
        print(line)

    print("\n四个模型分类结果")
    print(
        model_table[
            [
                "dataset",
                "algorithm",
                "train_accuracy",
                "test_accuracy",
                "train_mistakes",
                "test_mistakes",
                "train_mse",
                "test_mse",
                "minimum_train_margin",
                "geometric_train_margin",
            ]
        ].to_string(index=False)
    )

    print("\n样本访问顺序实验")
    print(order_table.to_string(index=False))

    print("\n样本顺序两两量化结果")
    print(
        pairwise_table.groupby("dataset")[["parameter_cosine", "test_disagreement_rate"]]
        .agg(["mean", "std", "min", "max"])
        .to_string()
    )

    print(f"\n结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
