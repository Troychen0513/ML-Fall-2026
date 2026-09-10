"""查看四个数据集的基本情况。"""

from pathlib import Path

import pandas as pd


DATA = Path(__file__).resolve().parent / "data"
FILES = [
    "dataset_A_train.csv",
    "dataset_A_test.csv",
    "dataset_B_pool.csv",
    "dataset_B_evaluation.csv",
]


for filename in FILES:
    data = pd.read_csv(DATA / filename)

    print(f"\n{filename}")
    print("shape:", data.shape)
    print("columns:", list(data.columns))
    print("first 3 rows:\n", data.head(3))
    print("missing values:", data.isnull().sum().sum())
    print("duplicate rows:", data.duplicated().sum())
    print("statistics:\n", data.describe())
