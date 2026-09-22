"""问题二：固定特征映射、折内标准化及线性/Ridge 回归。"""

import csv
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


FIELDS = ["area_sqm", "building_age_years", "metro_m", "center_km",
          "bedrooms", "floor", "amenities_1km", "renovation_score",
          "has_elevator", "sunlight_hours"]
LABELS = ["建筑面积", "房龄", "地铁距离", "市中心距离", "卧室数量",
          "楼层", "设施数量", "装修评分", "电梯", "采光时长"]
FEATURE_NAMES = FIELDS.copy()
FEATURE_NAMES[2] = "metro_km"
EXTRA_NAMES = ["log1p_metro_km", "log1p_center_km", "floor_x_elevator"]
MODEL_NAMES = {"basic": "双特征 OLS", "full": "完整 OLS",
               "ridge": "完整 Ridge", "expanded": "扩展回归"}


def load_data(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    ids = np.array([row["listing_id"] for row in rows])
    x = np.array([[float(row[name]) for name in FIELDS] for row in rows])
    y = np.array([float(row["monthly_rent"]) for row in rows])
    if len(set(ids)) != len(ids):
        raise ValueError("房源编号重复。")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("数据包含缺失值或非有限值。")
    if np.any(x < 0) or np.any(y <= 0):
        raise ValueError("发现负特征值或非正租金，请检查数据。")
    if not np.isin(x[:, 8], [0, 1]).all():
        raise ValueError("电梯字段必须为 0 或 1。")
    if not np.isin(x[:, 7], [1, 2, 3, 4, 5]).all():
        raise ValueError("装修评分必须为 1 至 5 的整数。")
    return ids, x, y


def feature_matrix(x, kind):
    """变换不估计数据统计量；距离统一以千米代入模型。"""
    if kind == "basic":
        return x[:, :2].copy(), FEATURE_NAMES[:2]
    features = x.copy()
    features[:, 2] /= 1000
    if kind in {"full", "ridge"}:
        return features, FEATURE_NAMES.copy()
    if kind != "expanded":
        raise ValueError(f"未知模型：{kind}")
    extra = np.column_stack([np.log1p(features[:, 2]),
                             np.log1p(features[:, 3]),
                             features[:, 5] * features[:, 8]])
    return np.column_stack([features, extra]), FEATURE_NAMES + EXTRA_NAMES


def fit_model(x, y, kind, penalty=0.0):
    features, names = feature_matrix(x, kind)
    scaler = StandardScaler().fit(features)
    z = scaler.transform(features)
    # sklearn 使用平方误差之和，乘 n 后对应本文的平均损失定义。
    estimator = (Ridge(alpha=len(y) * penalty, solver="svd")
                 if penalty > 0 else LinearRegression())
    estimator.fit(z, y)
    beta = estimator.coef_ / scaler.scale_
    intercept = float(estimator.intercept_ - scaler.mean_ @ beta)
    error = np.max(np.abs(estimator.predict(z) - (intercept + features @ beta)))
    if error > 1e-7:
        raise AssertionError("原单位参数还原未通过核验。")
    return dict(kind=kind, penalty=float(penalty), features=names,
                intercept=intercept, coefficients=beta.tolist(),
                standardized_coefficients=estimator.coef_.tolist(),
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                training_count=len(y), roundtrip_error=float(error))


def predict(model, x):
    features, _ = feature_matrix(x, model["kind"])
    return model["intercept"] + features @ np.array(model["coefficients"])


def metrics(y, prediction):
    error = y - prediction
    mse = np.mean(error ** 2)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(error))),
                r2=float(1 - np.sum(error ** 2) / np.sum((y-y.mean()) ** 2)),
                mse=float(mse), bias=float(error.mean()))
