"""固定五折验证选择模型，选择结束后才读取测试集。"""

import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from model import FIELDS, MODEL_NAMES, load_data, fit_model, predict, metrics


BASE = Path(__file__).resolve().parents[1]
SEED = 42
PENALTIES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_validate(x, y, splits, kind, penalty):
    oof = np.zeros(len(y))
    rows = []
    for fold, (training, validation) in enumerate(splits, 1):
        model = fit_model(x[training], y[training], kind, penalty)
        oof[validation] = predict(model, x[validation])
        rows.append(dict(model=kind, penalty=penalty, fold=fold,
                         **metrics(y[validation], oof[validation])))
    rmse = [row["rmse"] for row in rows]
    summary = dict(model=kind, label=MODEL_NAMES[kind], penalty=penalty,
                   rmse_mean=float(np.mean(rmse)),
                   rmse_std=float(np.std(rmse, ddof=1)),
                   mae_mean=float(np.mean([row["mae"] for row in rows])),
                   parameter_count=(3 if kind == "basic" else
                                    14 if kind == "expanded" else 11))
    return summary, rows, oof


def group_errors(x_train, y_train, x_test, y_test, prediction):
    rows = []
    for group, training, testing in [("rent", y_train, y_test),
                                     ("area", x_train[:, 0], x_test[:, 0])]:
        cuts = np.quantile(training, [0.25, 0.5, 0.75])
        bins = np.digitize(testing, cuts, right=True)
        for index in range(4):
            mask = bins == index
            rows.append(dict(group=group, bin=index+1, count=int(mask.sum()),
                             low=None if index == 0 else float(cuts[index-1]),
                             high=None if index == 3 else float(cuts[index]),
                             **metrics(y_test[mask], prediction[mask])))
    for value in [0, 1]:
        mask = x_test[:, 8] == value
        rows.append(dict(group="elevator", bin=value, count=int(mask.sum()),
                         low=None, high=None,
                         **metrics(y_test[mask], prediction[mask])))
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    train_path = BASE / "data/B_train.csv"
    ids, x, y = load_data(train_path)
    splits = list(KFold(5, shuffle=True, random_state=SEED).split(x))
    fold_ids = np.zeros(len(y), dtype=int)
    for fold, (_, validation) in enumerate(splits, 1):
        fold_ids[validation] = fold
    save_csv(output / "fold_assignments.csv",
             [dict(listing_id=name, fold=int(fold)) for name, fold in zip(ids, fold_ids)])
    summaries, fold_scores, winners, oof_predictions = [], [], [], []
    for kind in MODEL_NAMES:
        penalties = ([0.0] if kind in {"basic", "full"} else
                     PENALTIES[1:] if kind == "ridge" else PENALTIES)
        candidates = []
        for penalty in penalties:
            summary, rows, oof = cross_validate(x, y, splits, kind, penalty)
            summaries.append(summary)
            fold_scores.extend(rows)
            candidates.append((summary, oof))
        # 先按 RMSE；精确并列时使用更小的惩罚。
        best, best_oof = min(candidates, key=lambda item:
                             (item[0]["rmse_mean"], item[0]["penalty"]))
        winners.append(best)
        oof_predictions.append(best_oof)
        print(f'{best["label"]}: CV RMSE={best["rmse_mean"]:.4f}, '
              f'lambda={best["penalty"]:g}', flush=True)
    selected = min(winners, key=lambda row: (row["rmse_mean"], row["parameter_count"]))
    selection = dict(seed=SEED, folds=5, penalties=PENALTIES, selected=selected,
                     winners=winners, rule="minimum mean validation RMSE",
                     feature_extension=["log1p(metro_m/1000)", "log1p(center_km)",
                                        "floor * has_elevator"])
    # 该文件在任何测试数据读取前落盘，记录已确定的选择。
    save_json(output / "selection.json", selection)
    save_csv(output / "cv_summary.csv", summaries)
    save_csv(output / "cv_folds.csv", fold_scores)
    models = [fit_model(x, y, row["model"], row["penalty"]) for row in winners]

    test_path = BASE / "data/B_test.csv"
    test_ids, x_test, y_test = load_data(test_path)
    if set(ids) & set(test_ids):
        raise ValueError("训练集与测试集存在重叠编号。")
    predictions = np.column_stack([predict(model, x_test) for model in models])
    comparison = []
    for index, (row, model) in enumerate(zip(winners, models)):
        comparison.append(dict(**row, train=metrics(y, predict(model, x)),
                               test=metrics(y_test, predictions[:, index])))
    chosen_index = list(MODEL_NAMES).index(selected["model"])
    final_model = models[chosen_index]
    final_prediction = predictions[:, chosen_index]
    error = y_test-final_prediction
    save_json(output / "models.json", models)
    save_json(output / "final_model.json", final_model)
    save_json(output / "comparison.json", comparison)
    save_csv(output / "test_predictions.csv",
             [dict(listing_id=name, actual=float(actual), predicted=float(predicted),
                   residual=float(actual-predicted), absolute_error=float(abs(actual-predicted)))
              for name, actual, predicted in zip(test_ids, y_test, final_prediction)])
    save_csv(output / "coefficients.csv",
             [dict(term="intercept", coefficient=final_model["intercept"])] +
             [dict(term=name, coefficient=value) for name, value in
              zip(final_model["features"], final_model["coefficients"])])
    groups = group_errors(x, y, x_test, y_test, final_prediction)
    save_json(output / "group_metrics.json", groups)
    statistics = []
    for name, values in zip(FIELDS+["monthly_rent"], np.column_stack([x, y]).T):
        statistics.append(dict(field=name, mean=float(values.mean()),
                               std=float(values.std(ddof=1)), minimum=float(values.min()),
                               q25=float(np.quantile(values, .25)), median=float(np.median(values)),
                               q75=float(np.quantile(values, .75)), maximum=float(values.max())))
    save_csv(output / "training_statistics.csv", statistics)
    diagnostics = dict(training_count=len(y), test_count=len(y_test),
                       missing_values=0, duplicate_ids=0, overlapping_ids=0,
                       duplicate_training_features=int(len(x)-len(np.unique(x, axis=0))),
                       correlation=np.corrcoef(np.column_stack([x, y]).T).tolist(),
                       error_quantiles={str(q): float(np.quantile(abs(error), q))
                                        for q in [.5, .9, .95]},
                       within_200=float(np.mean(abs(error) <= 200)),
                       within_300=float(np.mean(abs(error) <= 300)))
    save_json(output / "diagnostics.json", diagnostics)
    np.savez_compressed(output / "analysis.npz", x=x, y=y, x_test=x_test, y_test=y_test,
                        test_ids=test_ids, fold_ids=fold_ids,
                        oof=np.column_stack(oof_predictions), predictions=predictions)
    environment = dict(python=platform.python_version(),
                       packages={name: version(name) for name in
                                 ["numpy", "matplotlib", "scipy", "scikit-learn"]},
                       data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [train_path, test_path]})
    save_json(output / "environment.json", environment)
    print(f'Selected: {selected["label"]}; test metrics: '
          f'{comparison[chosen_index]["test"]}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/question2_complete")
    args = parser.parse_args()
    run(BASE / args.output)
