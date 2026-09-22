"""验证数值正确性、退化情形和复现性。"""

import argparse
import json
from pathlib import Path
import numpy as np
from model import (design_matrix, gradient_descent, proximal_gradient,
                   original_parameters, objective_gradient)


def verify(folder):
    folder = Path(folder)
    rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    results = {r["name"]: r for r in rows}
    data = np.load(folder / "data.npz")
    x, y = data["x"], data["y"]
    checks = []
    for result in rows:
        shift, scale = np.array(result["shift"]), np.array(result["scale"])
        X = design_matrix(x, shift, scale)
        theta = np.array(result["theta"])
        beta = original_parameters(theta, shift, scale)
        np.testing.assert_allclose(X @ theta, beta[0]+x @ beta[1:], rtol=1e-12, atol=1e-8)
        if result["status"] == "converged":
            np.testing.assert_allclose(theta, result["reference"], atol=5e-8, rtol=0)
            _, _, gradient = objective_gradient(X, y, theta, result["l1"], result["l2"])
            for j in [1, 2]:
                if abs(theta[j]) > 1e-8:
                    assert abs(gradient[j]+result["l1"]*np.sign(theta[j])) < 2e-8
                else:
                    assert abs(gradient[j]) <= result["l1"]+2e-8
            assert abs(gradient[0]) < 2e-8
    checks.append("全部模型参数还原后的预测一致；收敛解通过独立参考与 KKT 检验")
    normal = np.array(results["normal"]["beta"])
    fitted = np.array(results["baseline"]["beta"])
    predictions_a = normal[0]+data["x_test"] @ normal[1:]
    predictions_b = fitted[0]+data["x_test"] @ fitted[1:]
    error = float(np.max(np.abs(predictions_a-predictions_b)))
    assert error < 1e-6
    checks.append(f"正规方程与梯度下降测试预测最大差异 {error:.3e} 元")
    baseline = results["baseline"]
    X = design_matrix(x, np.array(baseline["shift"]), np.array(baseline["scale"]))
    ordinary, history, status = gradient_descent(X, y, baseline["alpha"])
    proximal, _, _ = proximal_gradient(X, y, baseline["alpha"], l1=0)
    np.testing.assert_allclose(ordinary, proximal, atol=2e-8, rtol=0)
    saved = np.loadtxt(folder / "baseline/history.csv", delimiter=",", skiprows=1, ndmin=2)
    np.testing.assert_array_equal(history, saved)
    checks.append("零正则近端法退化正确；基准梯度下降逐步历史可精确复现")
    if "l1_1" in results:
        assert results["l1_1"]["steps"] == 0
        np.testing.assert_array_equal(results["l1_1"]["theta"][1:], [0, 0])
        checks.append("强 L1 的初始最优点正确记录为 0 次更新")
    for name in ["raw_same", "lr_2.2"]:
        if name in results:
            assert results[name]["status"] == "diverged"
    if "raw_adapted" in results:
        assert results["raw_adapted"]["status"] == "max_iter"
    checks.append("发散与迭代上限状态正确区分")
    (folder / "verification.json").write_text(json.dumps(dict(passed=True, checks=checks),
                                          ensure_ascii=False, indent=2), encoding="utf-8")
    for check in checks:
        print("PASS:", check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    verify(parser.parse_args().result_directory)
