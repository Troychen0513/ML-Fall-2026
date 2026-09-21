from pathlib import Path
from zipfile import ZipFile
import ast
import hashlib
import json
import re
import textwrap
import warnings


workspace = Path(r"C:\Users\26491\Desktop\ML-Fall-2026")
document = workspace / "HW2 线性回归/docs/第二题重组稿/第二题_房源租金预测实验报告.md"
archive_path = document.parent / "附件/HW2_第二题/HW2_第二题_完整交付.zip"
backup = workspace / "tmp/第二题_插入正文代码前.md"
original = document.read_text(encoding="utf-8")
assert "<!-- q2-section-code:" not in original, "Code has already been inserted."
backup.write_bytes(document.read_bytes())
fence = chr(96) * 3
with ZipFile(archive_path) as archive:
    sources = {
        name.removeprefix("HW2_第二题/"): archive.read(name).decode("utf-8-sig")
        for name in archive.namelist()
        if name.endswith((".py", ".yaml", ".txt"))
    }

F = "question2/features.py"
E = "question2/experiment.py"
P = "question2/plot.py"
R = "question2/report.py"
V = "question2/verify.py"
EM = "question2/embedding.py"
PA = "experiments/pacmap_trial.py"
records = []
insertions = {}


def block(path, start=None, end=None, symbol=None, language="python"):
    source = sources[path]
    if symbol is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
        node = next(n for n in tree.body if getattr(n, "name", None) == symbol)
        start, end = node.lineno, node.end_lineno
    if start is not None:
        code = textwrap.dedent("\n".join(source.splitlines()[start - 1:end])).strip()
        label = f"{path}，第 {start}—{end} 行"
    else:
        code = source.rstrip()
        label = path
    if language == "python":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
    records.append({"source": path, "start": start, "end": end, "symbol": symbol,
                    "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest()})
    return f"**对应代码（{label}）**\n\n{fence}{language}\n{code}\n{fence}\n"


def function(path, name):
    return block(path, symbol=name)


def display_code(code):
    code = textwrap.dedent(code).strip()
    ast.parse(code)
    records.append({"source": "读取已有结果的展示代码",
                    "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest()})
    return f"**对应代码（读取已保存的结果，不重新训练）**\n\n{fence}python\n{code}\n{fence}\n"


def add(section, *parts):
    insertions[section] = "\n\n".join(parts).rstrip() + "\n\n"


add("2.1",
    "以下代码按对应内容插入，完整函数保留原实现，局部语句仅统一展示缩进。代码中的相对导入、config、output、训练折索引和绘图辅助函数沿用原项目上下文；完整程序及依赖见2.8.3节。绘图保存编号沿用原程序，正文图号以本报告题注为准。",
    function(E, "run_training"))

add("2.2.1", block(F, 3, 32), function(E, "audit"))
add("2.2.2", block(P, 3, 40), block(P, 163, 232))
add("2.2.3", block(F, 12, 18), function(E, "domain_variants"), function(E, "search_grids"))

add("2.3.1", block(E, 14, 14), function(E, "metrics"))
add("2.3.2", block(E, 3, 23), function(E, "pipeline_for"), function(E, "compare"))
add("2.3.3",
    block("configs/q2.yaml", language="yaml"),
    function(E, "grid_records"), function(E, "tune"), function(E, "best_record"))

add("2.4", function(E, "pipeline_for"), function(F, "CheckedLinear"),
    display_code("""
        import pandas as pd

        comparison = pd.read_csv(output / "comparison.csv")
        baseline_keys = ["mean:mean", "raw:ols"]
        baseline_results = comparison.set_index("candidate").loc[
            baseline_keys, ["rmse", "rmse_std", "mae", "r2"]
        ]
        print(baseline_results)
    """))

add("2.5.1", block(F, 12, 18), function(F, "FeatureMap"),
    function(E, "search_grids"),
    display_code("""
        import pandas as pd

        comparison = pd.read_csv(output / "comparison.csv")
        families = ["raw", "encoded", "domain", "quadratic", "spline"]
        ols_keys = [family + ":ols" for family in families]
        ols_results = comparison.set_index("candidate").loc[
            ols_keys, ["rmse", "rmse_std", "mae", "r2"]
        ]
        print(ols_results)
    """))

add("2.5.2", function(E, "domain_variants"),
    block(E, 173, 187), function(R, "table"), block(R, 192, 199))

add("2.5.3", function(F, "CheckedLinear"), block(R, 203, 210),
    block(P, 276, 278), block(P, 304, 322))

add("2.5.4", block(E, 195, 211), block(R, 182, 189))

add("2.6.1", function(E, "finalize"), function(P, "term_counts"),
    function(P, "lasso_selection"), block(P, 325, 341))
add("2.6.2", function(F, "export_formula"), function(R, "term_math"),
    function(R, "polynomial_expression"))
add("2.6.3", function(V, "independent_polynomial"), block(R, 229, 231),
    function(P, "ale_curve"), block(P, 342, 353))

add("2.7.1", function(E, "metrics"), function(E, "finalize"), block(P, 355, 366))
add("2.7.2", block(R, 242, 255), block(P, 368, 379))
add("2.7.3", block(E, 232, 242), block(R, 256, 257))

add("2.8.1", display_code("""
        import json
        import pandas as pd

        comparison = pd.read_csv(output / "comparison.csv").set_index("candidate")
        selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
        final = json.loads((output / "final_metrics.json").read_text(encoding="utf-8"))
        formula = json.loads((output / "final_formula.json").read_text(encoding="utf-8"))
        retained = sum(value != 0 for value in formula["coefficients"])
        print("最低验证误差方案：", selection["lowest_rmse"])
        print("最终选择方案：", selection["selected"])
        print("最终候选的验证指标：")
        print(comparison.loc[selection["selected"], ["rmse", "mae", "r2"]])
        print("最终非零项数：", retained)
        print("测试指标：", final["test"])
    """))
add("2.8.2",
    "以下代码提供模型项保留情况与局部影响分析的依据；数据适用范围和因果解释限制属于讨论，不由程序自动判定。",
    function(P, "term_counts"), function(P, "ale_curve"))
add("2.8.3",
    block("question2/__init__.py"),
    block("question2/run.py"),
    function(F, "save_json"),
    block("requirements-q2.txt", language="text"),
    block(V))

add("A.1", function(R, "table"),
    block(R, 47, 64), block(R, 287, 289))
add("A.2", block(P, 275, 302))
add("A.3", block(P, 405, 441))
add("B.1", function(E, "pipeline_for"), function(E, "search_grids"),
    function(EM, "neighborhood"), function(EM, "run_embeddings"),
    function(P, "pca_analysis"))
add("B.2", block(EM), function(P, "embedding_comparison"), block(P, 234, 273))
add("B.3",
    block("configs/pacmap_trial.yaml", language="yaml"),
    block("requirements-pacmap.txt", language="text"),
    block(PA))

# Insert only at section boundaries. No original byte content is removed.
headings = list(re.finditer(r"(?m)^(#{2,3}) ([^\n]+)\n", original))
positions = []
for section, content in insertions.items():
    matches = [i for i, match in enumerate(headings)
               if match.group(2).startswith(section + " ")]
    assert len(matches) == 1, section
    index = matches[0]
    position = headings[index + 1].start() if index + 1 < len(headings) else len(original)
    wrapped = (
        f"<!-- q2-section-code:{section}:begin -->\n\n"
        + content
        + f"<!-- q2-section-code:{section}:end -->\n\n"
    )
    positions.append((position, wrapped))

updated = original
for position, content in sorted(positions, reverse=True):
    updated = updated[:position] + content + updated[position:]

restored = re.sub(
    r"(?ms)^<!-- q2-section-code:[^:]+:begin -->\n.*?^<!-- q2-section-code:[^:]+:end -->\n\n",
    "", updated
)
assert restored == original, "Existing content was changed."
assert re.findall(r"(?ms)^\$\$\n.*?^\$\$$", original) == re.findall(
    r"(?ms)^\$\$\n.*?^\$\$$",
    re.sub(r"(?ms)^" + fence + r"[^\n]*\n.*?^" + fence + r"[ \t]*$", "", updated)
)
python_blocks = re.findall(r"(?ms)^" + fence + r"python\n(.*?)^" + fence + r"[ \t]*$", updated)
for index, code in enumerate(python_blocks):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        ast.parse(code, filename=f"report_python_block_{index + 1}")
    assert not re.search(r"(?m)^\s*(?:\.\.\.|#\s*省略)", code)

plain = re.sub(r"(?ms)^" + fence + r"[^\n]*\n.*?^" + fence + r"[ \t]*$", "", updated)
plain = re.sub(r"(?ms)^~~~[^\n]*\n.*?^~~~[ \t]*$", "", plain)
for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", plain):
    if not link.startswith(("http://", "https://")):
        assert (document.parent / link).exists(), link
assert updated.count(fence + "python\n") == len(python_blocks)
assert len(insertions) == 27
document.write_text(updated, encoding="utf-8")

report = {
    "document": str(document),
    "backup": str(backup),
    "sections_with_inserted_code": list(insertions),
    "sections_count": len(insertions),
    "python_blocks_after": len(python_blocks),
    "source_blocks_inserted": len(records),
    "original_text_preserved_exactly": restored == original,
    "all_python_blocks_parse": True,
    "original_formulas_preserved": True,
    "existing_links_resolve": True,
    "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    "document_sha256": hashlib.sha256(document.read_bytes()).hexdigest(),
    "source_records": records,
    "reader_preview": "No Markpad or Obsidian UI verification",
}
verification = workspace / "tmp/q2_section_code_verification.json"
verification.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "source_records"},
                 ensure_ascii=False, indent=2))
