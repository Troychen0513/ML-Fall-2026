"""独立最小二乘核验、折外预测核验及交付文件检查。"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from model import load_data, feature_matrix, predict, metrics, MODEL_NAMES


BASE = Path(__file__).resolve().parents[1]


def independent_fit(x, y, kind, penalty, evaluation):
    """用增广矩阵的 NumPy 最小二乘独立核验 sklearn 的解。"""
    features, _ = feature_matrix(x, kind)
    values, _ = feature_matrix(evaluation, kind)
    mean, scale = features.mean(axis=0), features.std(axis=0)
    design = np.column_stack([np.ones(len(y)), (features-mean)/scale])
    regularizer = np.diag(np.r_[0.0, np.repeat(np.sqrt(len(y)*penalty), features.shape[1])])
    augmented_x = np.vstack([design, regularizer])
    augmented_y = np.r_[y, np.zeros(design.shape[1])]
    theta = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    return np.column_stack([np.ones(len(values)), (values-mean)/scale]) @ theta


def verify(output, report):
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    data = np.load(output / "analysis.npz")
    ids, x, y = load_data(BASE / "data/B_train.csv")
    test_ids, xt, yt = load_data(BASE / "data/B_test.csv")
    assert len(y) == 7500 and len(yt) == 1500
    assert not set(ids) & set(test_ids)
    np.testing.assert_array_equal(x, data["x"])
    np.testing.assert_array_equal(yt, data["y_test"])
    environment = read("environment.json")
    for name, digest in environment["data_sha256"].items():
        assert hashlib.sha256((BASE / "data" / name).read_bytes()).hexdigest() == digest
    models, scores = read("models.json"), read("comparison.json")
    final_model, selection = read("final_model.json"), read("selection.json")
    reference_errors, oof_errors = [], []
    fold_ids = data["fold_ids"]
    assert np.bincount(fold_ids).tolist() == [0, 1500, 1500, 1500, 1500, 1500]
    for index, model in enumerate(models):
        reference = independent_fit(x, y, model["kind"], model["penalty"], xt)
        predicted = predict(model, xt)
        error = float(np.max(abs(reference-predicted)))
        assert error < 1e-7
        reference_errors.append(error)
        np.testing.assert_allclose(predicted, data["predictions"][:, index], atol=1e-9, rtol=0)
        for name, value in metrics(yt, predicted).items():
            assert abs(value-scores[index]["test"][name]) < 1e-8
        rmses = []
        for fold in range(1, 6):
            validation = fold_ids == fold
            expected = independent_fit(x[~validation], y[~validation], model["kind"],
                                       model["penalty"], x[validation])
            observed = data["oof"][validation, index]
            oof_error = float(np.max(abs(expected-observed)))
            assert oof_error < 1e-7
            oof_errors.append(oof_error)
            rmses.append(metrics(y[validation], observed)["rmse"])
        assert abs(np.mean(rmses)-scores[index]["rmse_mean"]) < 1e-9
        assert abs(np.std(rmses, ddof=1)-scores[index]["rmse_std"]) < 1e-9
    with open(output / "cv_summary.csv", encoding="utf-8-sig") as file:
        candidates = list(csv.DictReader(file))
    best = min(candidates, key=lambda row: float(row["rmse_mean"]))
    assert best["model"] == final_model["kind"] == selection["selected"]["model"]
    assert float(best["penalty"]) == final_model["penalty"]
    with open(output / "test_predictions.csv", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))
    assert [row["listing_id"] for row in records] == test_ids.tolist()
    np.testing.assert_allclose([float(row["predicted"]) for row in records],
                               predict(final_model, xt), rtol=0, atol=1e-9)
    formula_features, _ = feature_matrix(xt, final_model["kind"])
    rounded = round(final_model["intercept"], 6) + formula_features @ np.round(final_model["coefficients"], 6)
    formula_error = float(np.max(abs(rounded-predict(final_model, xt))))
    assert formula_error < .001
    figure_paths = [output / "figures" / f"q2_fig{i:02d}.png" for i in range(1, 9)]
    for path in figure_paths:
        assert path.is_file() and path.stat().st_size > 10000, path
    text = report.read_text(encoding="utf-8")
    # 忽略代码块后检查数学定界符；实际阅读器预览另行说明。
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    prose = re.sub(r"`[^`]*`", "", prose)
    assert "\\[" not in prose and "\\(" not in prose
    assert prose.count("$$") % 2 == 0
    math_blocks = re.findall(r"(?m)^\$\$\s*\n(.*?)\n\$\$\s*$", prose, flags=re.S)
    assert len(math_blocks) >= 10
    for line in prose.splitlines():
        if "$$" in line:
            assert line.strip() == "$$"
    for block in math_blocks:
        assert block.count("{") == block.count("}")
    inline = re.sub(r"(?m)^\$\$\s*\n.*?\n\$\$\s*$", "", prose, flags=re.S)
    assert len(re.findall(r"(?<!\\)\$", inline)) % 2 == 0
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(images) == 8
    for path in images:
        assert (report.parent / path).resolve().is_file(), path
    outcome = dict(status="passed", independent_test_max_error=max(reference_errors),
                   independent_oof_max_error=max(oof_errors),
                   rounded_formula_max_error=formula_error,
                   test_prediction_rows=len(records), math_blocks=len(math_blocks),
                   figures=len(images),
                   checks=["输入数据哈希与样本划分", "四个模型的独立最小二乘解",
                           "各折独立标准化及折外预测", "全部测试指标与预测文件",
                           "训练验证选择与最终模型一致", "六位小数公式的预测误差",
                           "Markdown 数学定界符与图片路径"],
                   reader_preview="未在 Markpad 或 Obsidian 实际界面核验")
    (output / "verification.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/question2_complete")
    parser.add_argument("--report", default="docs/第二题_实验报告.md")
    args = parser.parse_args()
    verify(BASE / args.output, BASE / args.report)
