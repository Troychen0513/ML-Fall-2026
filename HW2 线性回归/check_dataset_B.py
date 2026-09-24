from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
datasets = {}

for filename in ["B_train.csv", "B_test.csv"]:
    df = pd.read_csv(DATA_DIR / filename)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    datasets[filename] = df

    print(f"\n{filename}：{len(df)} 行，{len(df.columns)} 列")
    print("整行重复数（不计首次）：", df.duplicated().sum())
    print("排除编号后的重复记录数：", df.drop(columns="listing_id").duplicated().sum())
    print("重复房源编号数：", df["listing_id"].dropna().duplicated().sum())

    raw = df.drop(columns="listing_id")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    non_numeric = raw.notna() & numeric.isna()
    infinite = np.isinf(numeric)
    finite = numeric.replace([np.inf, -np.inf], np.nan)

    # IQR 
    q1 = finite.quantile(0.25)
    q3 = finite.quantile(0.75)
    iqr = q3 - q1
    outliers = (finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)
    outliers[["has_elevator", "renovation_score"]] = False

    summary = pd.DataFrame({
        "缺失数": df.isna().sum(),
        "非数字数": non_numeric.sum(),
        "无穷值数": infinite.sum(),
        "IQR可疑值数": outliers.sum(),
    }).fillna(0).astype(int)
    print(summary.to_string())
    print("含 IQR 可疑值的行数：", outliers.any(axis=1).sum())

    # 取值范围
    invalid_range = (numeric.drop(columns="floor") < 0).any(axis=1)
    invalid_range |= (numeric[["area_sqm", "monthly_rent"]] <= 0).any(axis=1)
    invalid_range |= numeric["sunlight_hours"] > 24
    invalid_range |= numeric["has_elevator"].notna() & ~numeric["has_elevator"].isin([0, 1])
    print("取值范围异常的行数：", invalid_range.sum())
    print("单一值或全缺失列：", df.columns[df.nunique(dropna=True) <= 1].tolist())

train_ids = set(datasets["B_train.csv"]["listing_id"].dropna())
test_ids = set(datasets["B_test.csv"]["listing_id"].dropna())
print("\n训练集与测试集重叠的房源编号数：", len(train_ids & test_ids))