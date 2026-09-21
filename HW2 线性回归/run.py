"""YAML 配置驱动的轻量实验管理器。"""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import numpy as np
import yaml
from model import (load_data, prepare_features, design_matrix,
                   original_parameters, normal_equation, gradient_descent,
                   proximal_gradient, reference_solution, objective_gradient,
                   prediction_metrics)

BASE = Path(__file__).resolve().parent
HISTORY_COLUMNS = "step,data_loss,penalty,objective,optimality,b,w_area,w_age"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


class ExperimentManager:
    def __init__(self, config_path, output=None):
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.base = self.config["base"]
        self.train_path = BASE / self.config["data"]["train"]
        self.test_path = BASE / self.config["data"]["test"]
        self.ids, self.x, self.y = load_data(self.train_path)
        self.test_ids, self.x_test, self.y_test = load_data(self.test_path)
        if set(self.ids) & set(self.test_ids):
            raise ValueError("训练集与测试集房源编号重叠。")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output = Path(output) if output else BASE / self.config["output"]["root"] / stamp
        self.output.mkdir(parents=True, exist_ok=False)
        self.results = []

    def experiment_list(self, group):
        standard, _, _ = prepare_features(self.x, "standard")
        L = np.linalg.eigvalsh(standard.T @ standard / len(self.y))[-1]
        gmax = np.max(np.abs(standard[:, 1:].T @ (self.y-self.y.mean()) / len(self.y)))
        experiments = [dict(name="normal", method="normal", group="baseline"),
                       dict(name="baseline", group="baseline")]
        if group in {"all", "scaling"}:
            experiments += [dict(name="raw_adapted", mode="none", group="scaling"),
                            dict(name="center_adapted", mode="center", group="scaling"),
                            dict(name="raw_same", mode="none", group="scaling",
                                 alpha=self.base["step_factor"] / L)]
        if group in {"all", "learning_rate"}:
            for factor in self.config["experiments"]["learning_rate"]:
                experiments.append(dict(name=f"lr_{factor:g}", group="learning_rate",
                                        step_factor=factor))
        if group in {"all", "regularization"}:
            for kind in ["l1", "l2", "elastic"]:
                for strength in self.config["experiments"]["regularization"]:
                    l1 = strength * gmax if kind != "l2" else 0.0
                    l2 = strength * L if kind != "l1" else 0.0
                    if kind == "elastic":
                        l1, l2 = l1 / 2, l2 / 2
                    experiments.append(dict(name=f"{kind}_{strength:g}",
                                            group="regularization", kind=kind,
                                            strength=strength, l1=l1, l2=l2))
        return experiments

    def run_one(self, spec):
        mode = spec.get("mode", "standard")
        X, shift, scale = prepare_features(self.x, mode)
        l1, l2 = spec.get("l1", 0.0), spec.get("l2", 0.0)
        hessian = X.T @ X / len(self.y)
        hessian += np.diag([0.0, l2, l2])
        eigenvalues = np.linalg.eigvalsh(hessian)
        factor = spec.get("step_factor", self.base["step_factor"])
        alpha = spec.get("alpha", factor / eigenvalues[-1])
        reference = reference_solution(X, self.y, l1, l2)
        ref_loss, ref_penalty, _ = objective_gradient(X, self.y, reference, l1, l2)
        if spec.get("method") == "normal":
            theta = normal_equation(X, self.y)
            history = np.empty((0, 8))
            status = "analytic"
        else:
            options = dict(max_iter=self.base["max_iter"], tol=self.base["tol"],
                           divergence_ratio=self.base["divergence_ratio"])
            if l1 > 0:
                theta, history, status = proximal_gradient(X, self.y, alpha, l1, l2, **options)
            else:
                theta, history, status = gradient_descent(X, self.y, alpha, l2=l2, **options)
        beta = original_parameters(theta, shift, scale)
        metrics = prediction_metrics(self.y, X @ theta)
        result = dict(spec, mode=mode, alpha=float(alpha), l1=float(l1), l2=float(l2),
                      status=status, steps=int(history[-1, 0]) if len(history) else 0,
                      theta=theta.tolist(), beta=beta.tolist(), shift=shift.tolist(),
                      scale=scale.tolist(), reference=reference.tolist(),
                      reference_objective=float(ref_loss + ref_penalty),
                      condition=float(eigenvalues[-1] / eigenvalues[0]),
                      train_mse=metrics["mse"], train_rmse=metrics["rmse"],
                      nonzero=int(np.count_nonzero(np.abs(theta[1:]) > 1e-8)),
                      reference_error=float(np.max(np.abs(theta-reference))))
        if len(history):
            result["optimality"] = float(history[-1, 4])
        if spec["group"] == "baseline":
            X_test = design_matrix(self.x_test, shift, scale)
            prediction = X_test @ theta
            result["test"] = prediction_metrics(self.y_test, prediction)
            with (self.output / f"{spec['name']}_predictions.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["listing_id", "actual_rent", "predicted_rent"])
                writer.writerows(zip(self.test_ids, self.y_test, prediction))
        folder = self.output / spec["name"]
        folder.mkdir()
        np.savetxt(folder / "history.csv", history, delimiter=",", header=HISTORY_COLUMNS, comments="")
        save_json(folder / "result.json", result)
        self.results.append(result)
        print(f"{spec['name']:18s} {status:10s} steps={result['steps']:5d} MSE={metrics['mse']:.6f}")

    def run(self, group="all"):
        for spec in self.experiment_list(group):
            self.run_one(spec)
        np.savez_compressed(self.output / "data.npz", x=self.x, y=self.y,
                            x_test=self.x_test, y_test=self.y_test)
        resolved = dict(self.config, resolved_experiments=self.results)
        (self.output / "config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True,
                                                              sort_keys=False), encoding="utf-8")
        save_json(self.output / "summary.json", self.results)
        packages = ["numpy", "matplotlib", "PyYAML", "scikit-learn", "scipy"]
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [self.train_path, self.test_path]}
        save_json(self.output / "environment.json", dict(python=platform.python_version(),
                  packages={name: version(name) for name in packages}, data_sha256=hashes))
        with (self.output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
            columns = ["name", "mode", "status", "steps", "alpha", "l1", "l2", "train_mse", "nonzero"]
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.results)
        print("RESULTS:", self.output)
        return self.output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BASE / "configs/q1.yaml"))
    parser.add_argument("--output")
    parser.add_argument("--group", choices=["all", "baseline", "scaling", "learning_rate", "regularization"], default="all")
    parser.add_argument("--plot-only", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        output = args.plot_only
    else:
        output = ExperimentManager(args.config, args.output).run(args.group)
    if not args.no_plots:
        from plot import draw_all
        draw_all(output)


if __name__ == "__main__":
    main()
