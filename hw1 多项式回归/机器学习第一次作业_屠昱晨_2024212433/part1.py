"""HW1 part 1: polynomial regression and model selection."""

import csv
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
DEGREES = range(1, 9) # 定义阶数



def load_data(filename):
    data = pd.read_csv(DATA / filename)
    return data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy()


def features(x, degree):
    """先写好特征：Return [1, x, x^2, ..., x^degree]."""
    return np.column_stack([x ** i for i in range(degree + 1)])


def fit(x, y, degree, alpha=0.001, epochs=20000):
    """拟合函数"""
    mean = np.mean(x)
    std = np.std(x)
    x_normalized = (x - mean) / std
    
    X = features(x_normalized, degree)
    w = np.zeros(degree+1)
    
    for _ in range(epochs):
        gradient = 2 / len(y) * X.T @ (X @ w - y)
        w -= gradient*alpha
    
    if not np.isfinite(w).all():
        raise ValueError("梯度下降发散，请减小 alpha 或增加数据缩放。")

    return w, mean, std


def predict(x, model):
    coefficients, mean, std = model
    x_normalized = (x - mean) / std
    
    return features(x_normalized, len(coefficients) - 1) @ coefficients


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def cross_validate(x, y, degree, k=10, seed=42):
    """使用K折交叉验证计算MSE和std"""
    indices = np.random.default_rng(seed).permutation(len(x))
    scores = []
    for validation in np.array_split(indices, k):
        training = np.setdiff1d(indices, validation)
        model = fit(x[training], y[training], degree)
        scores.append(mse(y[validation], predict(x[validation], model)))
    return np.mean(scores), np.std(scores), scores




def main():
    x_train, y_train = load_data("dataset_A_train.csv")
    x_test, y_test = load_data("dataset_A_test.csv")
    x_grid = np.linspace(min(x_train.min(), x_test.min()), max(x_train.max(), x_test.max()), 600)

    rows = []
    predictions = [x_grid]
    test_predictions = [x_test, y_test]
    cv_scores = []
    for degree in DEGREES:
        model = fit(x_train, y_train, degree)
        cv_mean, cv_std, scores = cross_validate(x_train, y_train, degree)
        test_prediction = predict(x_test, model)
        cv_scores.append([degree] + scores)
        rows.append([
            degree,
            mse(y_train, predict(x_train, model)),
            cv_mean,
            cv_std,
            mse(y_test, test_prediction),
        ])
        predictions.append(predict(x_grid, model))
        test_predictions.append(test_prediction)

    rows = np.asarray(rows)
    selected = int(rows[np.argmin(rows[:, 2]), 0])

    with open(BASE / "part1_metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["degree", "train_mse", "cv_mse", "cv_std", "test_mse"])
        writer.writerows(rows)

    np.savetxt(
        BASE / "part1_predictions.csv",
        np.column_stack(predictions),
        delimiter=",",
        header="x," + ",".join(f"degree_{d}" for d in DEGREES),
        comments="",
    )
    np.savetxt(
        BASE / "part1_test_predictions.csv",
        np.column_stack(test_predictions),
        delimiter=",",
        header="x,y," + ",".join(f"degree_{d}" for d in DEGREES),
        comments="",
    )
    np.savetxt(
        BASE / "part1_cv_scores.csv",
        np.asarray(cv_scores),
        delimiter=",",
        header="degree," + ",".join(f"fold_{i}" for i in range(1, 11)),
        comments="",
    )

    print("degree | train MSE | CV MSE | test MSE")
    for row in rows:
        print(f"{int(row[0]):6d} | {row[1]:9.6f} | {row[2]:7.6f} | {row[4]:8.6f}")
    print(f"selected degree: {selected}")
    print(f"saved: {BASE / 'part1_metrics.csv'}")
    print(f"saved: {BASE / 'part1_predictions.csv'}")
    print(f"saved: {BASE / 'part1_test_predictions.csv'}")
    print(f"saved: {BASE / 'part1_cv_scores.csv'}")


if __name__ == "__main__":
    main()
