from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent / "data" 
PRICE = "monthly_rent"

datasets = {
    "训练集": pd.read_csv(folder / "A_train.csv"),
    "测试集": pd.read_csv(folder / "A_test.csv"),
}

for name, df in datasets.items():
    print(f"\n{name}")
    print("缺失值数量：", df.isna().sum().sum())
    print("完全重复行数：", df.duplicated().sum())
    print("编号重复行数：", df["listing_id"].duplicated().sum())
    
    
for name, df in datasets.items():
    x = df.filter(items=["area_dm2", "building_age_years", PRICE])
    x = x.apply(pd.to_numeric, errors="coerce")

    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    abnormal = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)

    print(f"\n{name}各列异常值数量：")
    print(abnormal.sum())
    
data = pd.concat(datasets.values(), ignore_index=True)
counts = data.groupby("listing_id")[PRICE].nunique()
conflict_ids = counts[counts > 1].index

print("\n对应多个价格的编号数量：", len(conflict_ids))
print(data.loc[data["listing_id"].isin(conflict_ids), ["listing_id", PRICE]])