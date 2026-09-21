"""用保存的模型预测新房源；输入只需编号和十个特征。"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from model import FIELDS, predict


BASE = Path(__file__).resolve().parents[1]


def predict_file(model_path, input_path, output_path):
    model = json.loads(model_path.read_text(encoding="utf-8"))
    with open(input_path, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("输入文件没有房源记录。")
    ids = [row["listing_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("房源编号重复。")
    x = np.array([[float(row[name]) for name in FIELDS] for row in rows])
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("输入包含缺失值、非有限值或负特征值。")
    if not np.isin(x[:, 8], [0, 1]).all() or not np.isin(x[:, 7], [1, 2, 3, 4, 5]).all():
        raise ValueError("电梯或装修评分的取值不合法。")
    prediction = predict(model, x)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["listing_id", "predicted_monthly_rent"])
        writer.writerows(zip(ids, prediction))
    print(f"已保存 {len(rows)} 条预测：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/question2_complete/final_model.json")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict_file(BASE / args.model, BASE / args.input, BASE / args.output)
