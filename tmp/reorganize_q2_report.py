from pathlib import Path
from zipfile import ZipFile
import ast
import csv
import hashlib
import io
import json
import math
import re
import shutil


workspace = Path(r"C:\Users\26491\Desktop\ML-Fall-2026")
source = Path(r"C:\Users\26491\Documents\Obsidian Vault\机器学习ML\机器学习作业\第二次作业_第二题实验报告.md")
document = workspace / "HW2 线性回归/docs/第二题重组稿/第二题_房源租金预测实验报告.md"
source_text = source.read_text(encoding="utf-8")
text = document.read_text(encoding="utf-8")
fence = chr(96) * 3


def first_table(section):
    return re.search(r"(?m)^\|[^\n]+\n(?:\|[^\n]+\n?)+", section).group(0).strip()


def source_function(name):
    pattern = r"(?ms)^def " + name + r"\(.*?(?=\n\ndef |\n\nclass |\n" + fence + ")"
    code = re.search(pattern, source_text).group(0).strip()
    ast.parse(code)
    return fence + "python\n" + code + "\n" + fence


model_section = source_text.split("### 2.6.1 完整数学表达式", 1)[1]
equation = re.search(r"(?ms)^\$\$\n.*?^\$\$$", model_section).group(0)
assert r"\hat y={}&771.8941993" in equation

replacements = {
    "FINAL_EQUATION": equation,
    "PREDICTION_TABLE": first_table(source_text[source_text.index("| 房源编号 | 实际租金 |"):]),
    "CANDIDATE_TABLE": first_table(source_text.split("### 全部主要候选结果", 1)[1]),
    "PACMAP_TABLE": "**表B2 PaCMAP 与 UMAP 的结构指标**\n\n" + first_table(
        source_text[source_text.index("| 方法 | 近邻数 | 邻域可信度 |"):]),
    "PIPELINE_CODE": source_function("pipeline_for"),
    "GRID_CODE": source_function("grid_records"),
    "EXPORT_CODE": source_function("export_formula"),
}
for key, value in replacements.items():
    marker = "{{" + key + "}}"
    if marker in text:
        assert text.count(marker) == 1
        text = text.replace(marker, value)
assert "{{" not in text
document.write_text(text, encoding="utf-8")

links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
local_links = sorted({link for link in links if not link.startswith(("https://", "http://"))})
for link in local_links:
    original = source.parent / link
    target = document.parent / link
    assert original.is_file(), str(original)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, target)
    assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
    if original.suffix == ".png" and original.with_suffix(".svg").exists():
        shutil.copy2(original.with_suffix(".svg"), target.with_suffix(".svg"))

archive_path = source.parent / "附件/HW2_第二题/HW2_第二题_完整交付.zip"
with ZipFile(archive_path) as archive:
    prefix = "HW2_第二题/"
    comparison = list(csv.DictReader(io.StringIO(archive.read(prefix + "results/complete/comparison.csv").decode("utf-8-sig"))))
    final_metrics = json.loads(archive.read(prefix + "results/complete/final_metrics.json"))
    test_inputs = list(csv.DictReader(io.StringIO(archive.read(prefix + "data/B_test.csv").decode("utf-8-sig"))))
    figure_records = json.loads(archive.read(prefix + "results/complete/figures.json"))
    for name in ["comparison.csv", "final_metrics.json"]:
        path = document.parent / "附件/HW2_第二题/results/complete" / name
        path.write_bytes(archive.read(prefix + "results/complete/" + name))

results_root = document.parent / "附件/HW2_第二题/results/complete"
predictions = list(csv.DictReader((results_root / "test_predictions.csv").open(encoding="utf-8-sig")))
formula = json.loads((results_root / "final_formula.json").read_text(encoding="utf-8"))
selection = json.loads((results_root / "selection.json").read_text(encoding="utf-8"))
assert len(predictions) == 1500
assert len(test_inputs) == 1500
assert selection["selected"] == "quadratic:lasso"
assert abs(selection["threshold"] - 130.298217) < 1e-6
assert sum(abs(v) > 1e-8 for v in formula["standardized_coefficients"]) == 30

# Independently recompute the stored test metrics.
actual = [float(row["actual_rent"]) for row in predictions]
predicted = [float(row["predicted_rent"]) for row in predictions]
residuals = [y - p for y, p in zip(actual, predicted)]
mean_actual = sum(actual) / len(actual)
rmse = math.sqrt(sum(e * e for e in residuals) / len(residuals))
mae = sum(abs(e) for e in residuals) / len(residuals)
r2 = 1 - sum(e * e for e in residuals) / sum((y - mean_actual) ** 2 for y in actual)
for name, value in [("rmse", rmse), ("mae", mae), ("r2", r2)]:
    assert abs(value - final_metrics["test"][name]) < 1e-8
    assert f"{value:.6f}" in text

# Compare displayed final equation against all stored model coefficients.
variable_names = [
    "area_sqm", "building_age_years", "metro_km", "center_km",
    "bedrooms", "floor", "amenities_1km", "renovation_score",
    "has_elevator", "sunlight_hours",
]
compact = equation.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
compact = compact.replace(r"\hat y={}", "").replace("$", "").replace("&", "")
compact = compact.replace("\\\\", "").replace("\n", "").replace(" ", "")
displayed_intercept = float(re.match(r"[+-]?\d+(?:\.\d+)?", compact).group(0))
term_pattern = r"([+-]\d+(?:\.\d+)?)(x_\{\d+\}(?:\^\{2\})?(?:x_\{\d+\})?)"
equation_terms = re.findall(term_pattern, compact)
displayed_coefficients = {}
for coefficient, term in equation_terms:
    variables = re.findall(r"x_\{(\d+)\}(?:\^\{(\d+)\})?", term)
    name = " ".join(variable_names[int(index) - 1] + ("^" + power if power else "") for index, power in variables)
    assert name not in displayed_coefficients
    displayed_coefficients[name] = float(coefficient)
nonzero = {name: value for name, value in zip(formula["names"], formula["coefficients"])
           if abs(value) > 1e-12}
assert set(displayed_coefficients) == set(nonzero)
assert math.isclose(displayed_intercept, formula["intercept"], rel_tol=1e-9)
for name, value in displayed_coefficients.items():
    assert math.isclose(value, nonzero[name], rel_tol=1e-9, abs_tol=1e-10), name

# Evaluate the displayed rounded equation against all test predictions.
maximum_error = 0.0
for inputs, output in zip(test_inputs, predictions):
    assert inputs["listing_id"] == output["listing_id"]
    values = {key: float(inputs[key]) for key in variable_names if key != "metro_km"}
    values["metro_km"] = float(inputs["metro_m"]) / 1000
    estimate = displayed_intercept
    for term, coefficient in displayed_coefficients.items():
        product = 1.0
        for factor in term.split():
            base, separator, power = factor.partition("^")
            product *= values[base] ** (int(power) if separator else 1)
        estimate += coefficient * product
    maximum_error = max(maximum_error, abs(estimate - float(output["predicted_rent"])))
assert maximum_error < 1e-4

# Check the reorganized fixed-OLS comparison directly against saved results.
indexed = {row["candidate"]: row for row in comparison}
for candidate in ["raw:ols", "encoded:ols", "domain:ols", "quadratic:ols", "spline:ols"]:
    row = indexed[candidate]
    expected = f'{float(row["rmse"]):.3f} ± {float(row["rmse_std"]):.3f}'
    assert expected in text, (candidate, expected)
    assert f'{float(row["mae"]):.3f}' in text
    assert f'{float(row["r2"]):.5f}' in text

# Check Markdown structure without claiming a reader preview.
plain = re.sub(r"(?ms)^" + fence + r"[^\n]*\n.*?^" + fence + r"[ \t]*$", "", text)
plain = re.sub(r"(?ms)^~~~[^\n]*\n.*?^~~~[ \t]*$", "", plain)
assert r"\(" not in plain and r"\[" not in plain
lines = plain.splitlines()
delimiters = [i for i, line in enumerate(lines) if "$$" in line]
assert len(delimiters) % 2 == 0
for i in delimiters:
    assert lines[i] == "$$"
for start, end in zip(delimiters[::2], delimiters[1::2]):
    assert lines[start - 1] == ""
    assert end + 1 == len(lines) or lines[end + 1] == ""
    content = "\n".join(lines[start + 1:end])
    assert content.count("{") == content.count("}")
outside_math = re.sub(r"(?ms)^\$\$\n.*?^\$\$", "", plain)
for line in outside_math.splitlines():
    assert len(re.findall(r"(?<!\\)\$", line)) % 2 == 0, line
    if line.startswith("|"):
        for expression in re.findall(r"\$([^$]+)\$", line):
            assert "|" not in expression
for link in local_links:
    assert (document.parent / link).is_file()
assert equation in text
assert not re.search(r"\{\{[A-Z_]+\}\}", text)
for code in re.findall(r"(?ms)^" + fence + r"python\n(.*?)^" + fence, text):
    ast.parse(code)
for number in range(1, 9):
    assert len(re.findall(r"(?m)^## 2\." + str(number) + r" ", text)) == 1
assert all(item["panels"] <= 2 for item in figure_records)
image_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
assert len(image_links) == 14

report = {
    "source": str(source),
    "document": str(document),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "document_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
    "local_links_checked": len(local_links),
    "images": len(image_links),
    "main_figures": 8,
    "appendix_figures": 6,
    "retained_nonzero_terms": len(equation_terms),
    "test_rows": len(predictions),
    "recomputed_test_metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    "displayed_equation_max_prediction_difference": maximum_error,
    "checks": [
        "source equation copied exactly",
        "displayed coefficients match saved formula",
        "all 1500 displayed-equation predictions agree with stored predictions",
        "fixed OLS comparison matches saved comparison.csv",
        "local links resolve and copied assets are byte-identical",
        "Markdown delimiters, braces and extracted Python snippets checked",
    ],
    "reader_preview": "Markpad / Obsidian UI not verified",
}
check_path = workspace / "tmp/q2_reorganized_verification.json"
check_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
