"""读取实验结果生成论文插图；不执行训练。"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from model import design_matrix, original_parameters

BLUE, ORANGE, GREEN, PURPLE = "#3B6FA5", "#C98338", "#33877B", "#9270A5"
GRAY, RED = "#747E87", "#B94E48"
COLORS = [BLUE, ORANGE, GREEN, PURPLE]


def set_style():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10, "axes.labelsize": 10.5,
        "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9BA3AA", "axes.linewidth": 0.7,
        "xtick.color": "#59636D", "ytick.color": "#59636D",
        "text.color": "#283644", "axes.labelcolor": "#283644",
        "grid.color": "#E6EBEF", "grid.linewidth": 0.6,
        "svg.fonttype": "none", "savefig.facecolor": "white"})


def two_axes():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), layout="constrained")
    for ax in axes:
        ax.grid(alpha=0.7)
        ax.set_axisbelow(True)
    return fig, axes


def end_labels(ax, x_end, endpoints):
    """仅调整文字位置，不移动曲线；短引线连接真实端点。"""
    low, high = ax.get_ylim()
    spacing = (high-low) * 0.075
    positions = []
    for y, text, color in sorted(endpoints):
        position = max(y, positions[-1][0]+spacing) if positions else y
        positions.append((position, y, text, color))
    overflow = max(0, positions[-1][0] - (high-spacing))
    dx = (ax.get_xlim()[1]-ax.get_xlim()[0]) * 0.025
    for position, y, text, color in positions:
        ax.annotate(text, (x_end, y), (x_end+dx, position-overflow),
                    fontsize=8.5, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6))


class Figures:
    def __init__(self, output):
        self.output = Path(output)
        self.folder = self.output / "figures"
        self.folder.mkdir(exist_ok=True)
        self.config = yaml.safe_load((self.output / "config.yaml").read_text(encoding="utf-8"))
        self.results = {r["name"]: r for r in json.loads((self.output / "summary.json").read_text(encoding="utf-8"))}
        self.data = np.load(self.output / "data.npz")
        self.x, self.y = self.data["x"], self.data["y"]
        self.mean = self.x.mean(axis=0)
        self.reference = np.array(self.results["normal"]["beta"])
        rng = np.random.default_rng(self.config["base"]["seed"])
        count = min(self.config["figures"]["sample_count"], len(self.y))
        self.sample = rng.choice(len(self.y), count, replace=False)
        self.manifest = []

    def history(self, name):
        return np.loadtxt(self.output / name / "history.csv", delimiter=",", skiprows=1, ndmin=2)

    def beta(self, name, step=None):
        result = self.results[name]
        if step is None:
            return np.array(result["beta"])
        history = self.history(name)
        row = history[min(step, len(history)-1)]
        return original_parameters(row[5:], np.array(result["shift"]), np.array(result["scale"]))

    def gap(self, name):
        history = self.history(name)
        best = self.results[name]["reference_objective"]
        denominator = history[0, 3] - best
        if abs(denominator) < 1e-9:
            return history[:, 0], np.zeros(len(history))
        return history[:, 0], np.maximum((history[:, 3]-best)/denominator, 1e-12)

    def save(self, number, title, fig):
        stem = f"fig{number:02d}"
        for suffix in ["png", "svg"]:
            fig.savefig(self.folder / f"{stem}.{suffix}", dpi=self.config["figures"]["dpi"], bbox_inches="tight")
        plt.close(fig)
        self.manifest.append(dict(number=number, title=title, png=f"{stem}.png", svg=f"{stem}.svg"))
        print(stem, title)

    def conditional(self, entries):
        """entries: (原单位参数, 标签, 颜色, 线型)。"""
        fig, axes = two_axes()
        for j, ax in enumerate(axes):
            other = 1-j
            adjusted = self.y - self.reference[other+1] * (self.x[:, other]-self.mean[other])
            ax.scatter(self.x[self.sample, j], adjusted[self.sample], s=12,
                       color="#AEBAC4", alpha=0.35, edgecolors="none", rasterized=True)
            lo, hi = self.x[:, j].min(), self.x[:, j].max()
            grid = np.linspace(lo, hi, 100)
            endpoints = []
            for beta, label, color, style in entries:
                prediction = beta[0] + beta[j+1]*grid + beta[other+1]*self.mean[other]
                ax.plot(grid, prediction, color=color, linestyle=style, lw=1.8)
                if label:
                    endpoints.append((prediction[-1], label, color))
            ax.set_xlim(lo-(hi-lo)*0.035, hi+(hi-lo)*0.36)
            ax.set_xticks(np.linspace(lo, hi, 4))
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.0f}"))
            ax.set_xlabel(["建筑面积（平方米）", "房龄（年）"][j])
            ax.set_ylabel("校正月租金（元/月）")
            ax.set_title(["固定房龄为训练均值", "固定面积为训练均值"][j], loc="left")
            end_labels(ax, hi, endpoints)
        return fig

    def loss_curve(self, ax, name, label, color, style="-"):
        steps, gap = self.gap(name)
        ax.plot(steps, gap, label=label, color=color, ls=style, lw=1.7)
        ax.set_yscale("log")
        ax.set_ylim(bottom=5e-13)
        ax.set_xlabel("更新次数")
        ax.set_ylabel("归一化目标差距")

    def contours(self, ax, name, trajectory=True):
        result = self.results[name]
        X = design_matrix(self.x, np.array(result["shift"]), np.array(result["scale"]))
        optimum = np.array(result["reference"])[1:]
        hessian = X[:, 1:].T @ X[:, 1:] / len(self.y)
        width = max(np.abs(optimum).max(), 1) * 1.35
        midpoint = optimum / 2
        a = np.linspace(midpoint[0]-width/2, midpoint[0]+width/2, 240)
        b = np.linspace(midpoint[1]-width/2, midpoint[1]+width/2, 240)
        aa, bb = np.meshgrid(a, b)
        da, db = aa-optimum[0], bb-optimum[1]
        excess = (hessian[0, 0]*da**2 + 2*hessian[0, 1]*da*db + hessian[1, 1]*db**2)/2
        start_loss = optimum @ hessian @ optimum / 2
        levels = start_loss * np.array([0.001, 0.01, 0.04, 0.12, 0.3, 0.6, 1, 1.7])
        ax.contour(aa, bb, excess, levels=levels, colors="#A8BACB", linewidths=0.7)
        if trajectory:
            history = self.history(name)
            ax.plot(history[:, 6], history[:, 7], color=BLUE, lw=1.7)
            for t in [0, 2, 8]:
                if t < len(history):
                    xy = history[t, 6:8]
                    ax.scatter(*xy, color=BLUE, s=17, zorder=4)
                    ax.annotate(f"t={t}", xy, xytext=(-7, 10 if t != 2 else -16),
                                textcoords="offset points", fontsize=8, color=BLUE)
        ax.scatter(*optimum, marker="*", s=120, color=ORANGE, zorder=5)
        ax.annotate("最优解", optimum, xytext=(8, -14), textcoords="offset points", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("面积系数" + ("（标准化）" if result["mode"] == "standard" else "（原尺度）"))
        ax.set_ylabel("房龄系数" + ("（标准化）" if result["mode"] == "standard" else "（原尺度）"))

    def baseline_figures(self):
        fig, axes = two_axes()
        for j, ax in enumerate(axes):
            ax.scatter(self.x[self.sample, j], self.y[self.sample], s=16,
                       color=BLUE, alpha=0.45, edgecolors="none")
            ax.set_xlabel(["建筑面积（平方米）", "房龄（年）"][j])
            ax.set_ylabel("月租金（元/月）")
        self.save(1, "房屋特征与月租金的样本分布", fig)
        fig, axes = two_axes()
        self.contours(axes[0], "baseline", False)
        self.contours(axes[1], "baseline", True)
        axes[0].set_title("正规方程：直接求解", loc="left")
        axes[1].set_title("梯度下降：逐步更新", loc="left")
        self.save(2, "正规方程解与梯度下降轨迹", fig)
        fig = self.conditional([(self.reference, "", GRAY, "--"),
                                (self.beta("baseline"), "正规≈GD", BLUE, "-")])
        self.save(3, "两种求解方法的条件回归直线", fig)
        fig = plt.figure(figsize=(6.8, 4.5), layout="constrained")
        ax = fig.add_subplot(111, projection="3d")
        points = self.x[self.sample]
        ax.scatter(points[:, 0], points[:, 1], self.y[self.sample], s=12,
                   color=BLUE, alpha=0.48, edgecolors="none")
        a, h = np.meshgrid(np.linspace(self.x[:, 0].min(), self.x[:, 0].max(), 12),
                           np.linspace(self.x[:, 1].min(), self.x[:, 1].max(), 12))
        rent = self.reference[0] + self.reference[1]*a + self.reference[2]*h
        ax.plot_surface(a, h, rent, color="#B2CADD", alpha=0.42, edgecolor="white", linewidth=0.25)
        ax.view_init(elev=23, azim=-58)
        ax.set_xlabel("建筑面积（平方米）", labelpad=8)
        ax.set_ylabel("房龄（年）", labelpad=6)
        ax.text2D(1.16, 0.5, "月租金（元/月）", transform=ax.transAxes,
                  rotation=90, va="center", fontsize=10.5)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
        self.save(4, "房源样本与双特征回归平面", fig)
        steps = self.config["figures"]["snapshots"]
        last = self.results["baseline"]["steps"]
        entries = [(self.beta("baseline", t), f"t={t}", color, "-")
                   for t, color in zip(steps+[last], [GRAY, ORANGE, GREEN, BLUE])]
        self.save(5, "梯度下降迭代中的条件回归直线", self.conditional(entries))
        fig, axes = two_axes()
        self.loss_curve(axes[0], "baseline", "梯度下降", BLUE)
        history = self.history("baseline")
        reference = np.array(self.results["baseline"]["reference"])
        distance = np.linalg.norm(history[:, 5:]-reference, axis=1)
        axes[1].semilogy(history[:, 0], distance/distance[0], color=BLUE, lw=1.7)
        axes[1].set_xlabel("更新次数")
        axes[1].set_ylabel("标准化参数的相对误差")
        self.save(6, "损失下降与参数误差", fig)

    def scaling_figures(self):
        fig, axes = two_axes()
        for name, label, color in [("raw_same", "未标准化", RED), ("baseline", "标准化", BLUE)]:
            self.loss_curve(axes[0], name, label, color)
        axes[0].set_title("相同数值步长", loc="left")
        stopped = self.results["raw_same"]["steps"]
        axes[0].annotate(f"第{stopped}次更新判为发散", (stopped, self.gap("raw_same")[1][-1]),
                         xytext=(13, 5e8), fontsize=9, color=RED,
                         arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
        for name, label, color in [("raw_adapted", "未标准化", ORANGE),
                                    ("center_adapted", "仅中心化", GREEN),
                                    ("baseline", "标准化", BLUE)]:
            self.loss_curve(axes[1], name, label, color)
        axes[1].set_xscale("symlog", linthresh=1)
        axes[1].set_xlim(0, self.results["raw_adapted"]["steps"] * 1.15)
        axes[1].set_title("各自采用稳定步长 0.2/L", loc="left")
        for ax in axes:
            ax.legend(frameon=False, fontsize=9, loc="lower left")
        self.save(7, "标准化前后的梯度下降收敛", fig)
        fig, axes = two_axes()
        self.contours(axes[0], "center_adapted")
        self.contours(axes[1], "baseline")
        axes[0].set_title("仅中心化，保留原尺度", loc="left")
        axes[1].set_title("中心化并标准化", loc="left")
        self.save(8, "特征尺度与损失曲面的几何关系", fig)
        budget = self.config["figures"]["scaling_budget"]
        entries = [(self.reference, "正规方程", GRAY, "--"),
                   (self.beta("raw_adapted", budget), "未标准化", ORANGE, "-"),
                   (self.beta("baseline", budget), "标准化", BLUE, "-")]
        self.save(9, f"最多{budget}次更新后的条件回归关系", self.conditional(entries))

    def learning_figures(self):
        fig, axes = two_axes()
        runs = [r for r in self.results.values() if r["group"] == "learning_rate" or r["name"] == "baseline"]
        stable = sorted([r for r in runs if r["status"] != "diverged"], key=lambda r: r["alpha"])
        L = self.config["base"]["step_factor"] / self.results["baseline"]["alpha"]
        for result, color in zip(stable[:4], [ORANGE, BLUE, GREEN, PURPLE]):
            label = f"c={result['alpha']*L:g}"
            self.loss_curve(axes[0], result["name"], label, color)
        axes[0].legend(frameon=False, fontsize=9)
        axes[0].set_title("稳定步长", loc="left")
        divergent = [r for r in runs if r["status"] == "diverged"]
        for result in divergent[:4]:
            self.loss_curve(axes[1], result["name"], f"c={result['alpha']*L:g}", RED)
        axes[1].set_ylim(bottom=0.5)
        axes[1].set_title("超过稳定范围", loc="left")
        if divergent:
            axes[1].legend(frameon=False)
        else:
            axes[1].text(0.5, 0.5, "本组没有发散实验", ha="center", transform=axes[1].transAxes)
        self.save(10, "不同学习率的收敛与发散", fig)
        t = self.config["figures"]["learning_rate_step"]
        entries = [(self.reference, "正规方程", GRAY, "--")]
        selected = list(dict.fromkeys([stable[0]["name"], "baseline", stable[-1]["name"]]))
        for name, color in zip(selected, [ORANGE, BLUE, PURPLE]):
            label = f"c={self.results[name]['alpha']*L:g}"
            entries.append((self.beta(name, t), label, color, "-"))
        self.save(11, f"第{t}次更新时不同学习率的模型", self.conditional(entries))

    def regularization_figures(self):
        strengths = self.config["experiments"]["regularization"]
        for number, kind, title in [(12, "l2", "L2"), (13, "l1", "L1"), (14, "elastic", "Elastic Net")]:
            entries = [(self.reference, "无正则", GRAY, "--")]
            for strength, color in zip(strengths, [BLUE, GREEN, ORANGE]):
                entries.append((self.beta(f"{kind}_{strength:g}"), f"r={strength:g}", color, "-"))
            self.save(number, f"不同{title}强度下的条件回归直线", self.conditional(entries))
        fig, axes = two_axes()
        middle = strengths[len(strengths)//2]
        groups = [("baseline", "无正则", GRAY), (f"l2_{middle:g}", "L2", BLUE),
                  (f"l1_{middle:g}", "L1", ORANGE), (f"elastic_{middle:g}", "Elastic Net", GREEN)]
        for name, label, color in groups:
            history = self.history(name)
            axes[0].plot(history[:, 0], 2*history[:, 1], label=label, color=color, lw=1.6)
            self.loss_curve(axes[1], name, label, color)
        axes[0].set_xlabel("更新次数")
        axes[0].set_ylabel("训练 MSE（元²/月²）")
        for ax in axes:
            ax.legend(frameon=False, fontsize=9)
        self.save(15, "中等正则强度下的拟合损失与目标收敛", fig)
        fig, axes = two_axes()
        for kind, label, color in [("l2", "L2", BLUE), ("l1", "L1", ORANGE), ("elastic", "Elastic Net", GREEN)]:
            betas = np.array([self.reference] + [self.beta(f"{kind}_{r:g}") for r in strengths])
            for j, ax in enumerate(axes):
                ax.plot([0]+strengths, betas[:, j+1], "o-", color=color, label=label, lw=1.6, ms=4)
                ax.set_xscale("symlog", linthresh=0.01)
                ax.set_xticks([0]+strengths, ["0"]+[f"{r:g}" for r in strengths])
                ax.set_xlabel("相对正则强度 r")
                ax.set_ylabel(["面积系数（元/月/平方米）", "房龄系数（元/月/年）"][j])
                ax.axhline(0, color=GRAY, ls=":", lw=0.7)
                ax.legend(frameon=False, fontsize=9)
        self.save(16, "正则强度与原单位回归系数", fig)


def draw_all(output):
    set_style()
    figures = Figures(output)
    figures.baseline_figures()
    if "raw_same" in figures.results:
        figures.scaling_figures()
    if any(r["group"] == "learning_rate" for r in figures.results.values()):
        figures.learning_figures()
    if any(r.get("kind") == "elastic" for r in figures.results.values()):
        figures.regularization_figures()
    (figures.folder / "manifest.json").write_text(json.dumps(figures.manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    draw_all(parser.parse_args().result_directory)
