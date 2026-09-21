"""第一题的线性回归模型：所有训练算法均由 NumPy 实现。"""

import csv
import numpy as np


def load_data(path):
    """面积由平方分米换成平方米，房源编号仅作记录。"""
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = [row["listing_id"] for row in rows]
    x = np.array([[float(row["area_dm2"]) / 100,
                   float(row["building_age_years"])] for row in rows])
    y = np.array([float(row["monthly_rent"]) for row in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("数据含缺失值或非有限数值。")
    if len(set(ids)) != len(ids):
        raise ValueError("房源编号重复。")
    return ids, x, y


def prepare_features(x, mode):
    """只在训练集调用；测试集复用返回的均值和标准差。"""
    if mode not in {"none", "center", "standard"}:
        raise ValueError("未知的预处理方式。")
    shift = x.mean(axis=0) if mode != "none" else np.zeros(x.shape[1])
    scale = x.std(axis=0) if mode == "standard" else np.ones(x.shape[1])
    if np.any(scale == 0):
        raise ValueError("特征标准差为零。")
    return design_matrix(x, shift, scale), shift, scale


def design_matrix(x, shift, scale):
    z = (x - shift) / scale
    return np.column_stack([np.ones(len(x)), z])


def original_parameters(theta, shift, scale):
    slopes = theta[1:] / scale
    intercept = theta[0] - shift @ slopes
    return np.r_[intercept, slopes]


def normal_equation(X, y, l2=0.0):
    penalty = np.diag([0.0, l2, l2])
    hessian = X.T @ X / len(y) + penalty
    return np.linalg.solve(hessian, X.T @ y / len(y))


def objective_gradient(X, y, theta, l1=0.0, l2=0.0):
    residual = X @ theta - y
    data_loss = np.mean(residual ** 2) / 2
    penalty = l1 * np.abs(theta[1:]).sum()
    penalty += l2 * np.sum(theta[1:] ** 2) / 2
    gradient = X.T @ residual / len(y)
    gradient[1:] += l2 * theta[1:]
    return data_loss, penalty, gradient


def soft_threshold(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def gradient_descent(X, y, alpha, max_iter=20000, tol=1e-8,
                     l2=0.0, divergence_ratio=1e12):
    """批量梯度下降；l2=0 时就是普通最小二乘。"""
    theta = np.array([y.mean(), 0.0, 0.0])
    history = []
    initial_loss = np.var(y) / 2
    status = "max_iter"
    for step in range(max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            loss, penalty, gradient = objective_gradient(X, y, theta, l2=l2)
        total = loss + penalty
        if not np.isfinite(total) or not np.isfinite(gradient).all():
            status = "diverged"
            break
        norm = np.max(np.abs(gradient))
        history.append([step, loss, penalty, total, norm, *theta])
        if total > max(initial_loss, 1) * divergence_ratio:
            status = "diverged"
            break
        if norm <= tol:
            status = "converged"
            break
        if step < max_iter:
            theta = theta - alpha * gradient
    history = np.asarray(history)
    return history[-1, 5:].copy(), history, status


def proximal_gradient(X, y, alpha, l1, l2=0.0,
                      max_iter=20000, tol=1e-8, divergence_ratio=1e12):
    """L1/Elastic Net：先更新光滑部分，再对斜率做软阈值。"""
    theta = np.array([y.mean(), 0.0, 0.0])
    history = []
    initial_loss = np.var(y) / 2
    status = "max_iter"
    for step in range(max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            loss, penalty, gradient = objective_gradient(X, y, theta, l1, l2)
            next_theta = theta - alpha * gradient
            next_theta[1:] = soft_threshold(next_theta[1:], alpha * l1)
        total = loss + penalty
        if not np.isfinite(total) or not np.isfinite(next_theta).all():
            status = "diverged"
            break
        norm = np.max(np.abs(theta - next_theta)) / alpha
        history.append([step, loss, penalty, total, norm, *theta])
        if total > max(initial_loss, 1) * divergence_ratio:
            status = "diverged"
            break
        if norm <= tol:
            status = "converged"
            break
        if step < max_iter:
            theta = next_theta
    history = np.asarray(history)
    return history[-1, 5:].copy(), history, status


def reference_solution(X, y, l1=0.0, l2=0.0):
    """独立校验用；不参与手写算法的参数更新。"""
    if l1 == 0:
        return normal_equation(X, y, l2)
    from sklearn.linear_model import Lasso, ElasticNet
    settings = dict(fit_intercept=False, tol=1e-13, max_iter=100000)
    if l2 == 0:
        estimator = Lasso(alpha=l1, **settings)
    else:
        estimator = ElasticNet(alpha=l1 + l2,
                               l1_ratio=l1 / (l1 + l2), **settings)
    estimator.fit(X[:, 1:], y - y.mean())
    return np.r_[y.mean(), estimator.coef_]


def prediction_metrics(y, prediction):
    mse = np.mean((y - prediction) ** 2)
    return {"mse": float(mse), "rmse": float(np.sqrt(mse))}
