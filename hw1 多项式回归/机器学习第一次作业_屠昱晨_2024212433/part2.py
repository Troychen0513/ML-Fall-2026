"""第二部分：重复抽样，比较偏差、方差和预测误差。"""

import csv
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DEGREES = [1, 6, 8]
SAMPLE_SIZE = 50
REPEATS = 500
SEED = 42


def load_data(filename):
    return np.loadtxt(DATA / filename, delimiter=",", skiprows=1)


def features(x, degree):
    return np.column_stack([x ** i for i in range(degree + 1)])


def fit(x, y, degree):
    mean = np.mean(x)
    scale = np.max(np.abs(x - mean))
    if scale == 0:
        scale = 1
    z = (x - mean) / scale
    X = features(z, degree)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w, mean, scale


def predict(x, model):
    w, mean, scale = model
    z = (x - mean) / scale
    return features(z, len(w) - 1) @ w


def main():
    pool = load_data("dataset_B_pool.csv")
    evaluation = load_data("dataset_B_evaluation.csv")
    x_pool, y_pool = pool.T
    x_eval = evaluation[:, 0]
    y_true = evaluation[:, 1]
    y_observed = evaluation[:, 2]

    rng = np.random.default_rng(SEED)
    predictions = np.zeros((REPEATS, len(DEGREES), len(x_eval)))

    for repeat in range(REPEATS):
        indices = rng.choice(len(x_pool), SAMPLE_SIZE, replace=False)
        x_train = x_pool[indices]
        y_train = y_pool[indices]

        for model_index, degree in enumerate(DEGREES):
            model = fit(x_train, y_train, degree)
            predictions[repeat, model_index] = predict(x_eval, model)

    mean_prediction = predictions.mean(axis=0)
    pointwise_variance = predictions.var(axis=0)
    bias_squared = np.mean((mean_prediction - y_true) ** 2, axis=1)
    variance = np.mean(pointwise_variance, axis=1)
    mse_true = np.mean((predictions - y_true) ** 2, axis=(0, 2))
    mse_observed = np.mean((predictions - y_observed) ** 2, axis=(0, 2))
    run_mse_true = np.mean((predictions - y_true) ** 2, axis=2)
    run_mse_observed = np.mean((predictions - y_observed) ** 2, axis=2)

    np.savez(
        BASE / "part2_results.npz",
        x_eval=x_eval,
        y_true=y_true,
        y_observed=y_observed,
        predictions=predictions,
        mean_prediction=mean_prediction,
        pointwise_variance=pointwise_variance,
        run_mse_true=run_mse_true,
        run_mse_observed=run_mse_observed,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
    )

    with open(BASE / "part2_metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["degree", "bias_squared", "variance", "mse_true", "mse_observed", "bias_plus_variance"])
        for i, degree in enumerate(DEGREES):
            writer.writerow([degree, bias_squared[i], variance[i], mse_true[i], mse_observed[i], bias_squared[i] + variance[i]])

    print(f"sample size: {SAMPLE_SIZE}, repeats: {REPEATS}")
    print("degree | bias^2 | variance | MSE(true) | MSE(observed)")
    for i, degree in enumerate(DEGREES):
        print(f"{degree:6d} | {bias_squared[i]:7.4f} | {variance[i]:8.4f} | {mse_true[i]:9.4f} | {mse_observed[i]:13.4f}")
    print(f"saved: {BASE / 'part2_results.npz'}")
    print(f"saved: {BASE / 'part2_metrics.csv'}")


if __name__ == "__main__":
    main()
