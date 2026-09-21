"""从已保存的结果绘图，不重新训练或选择模型。"""

import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
import numpy as np
from model import LABELS, MODEL_NAMES


BASE = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#213A4B", "#647786", "#E6EDF1"
BLUE, TEAL, GOLD, PURPLE = "#477FA9", "#187F85", "#C58B43", "#81749E"
COLORS = ["#8A99A7", BLUE, PURPLE, TEAL]


def set_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.labelsize": 10.5, "axes.titlesize": 12,
        "axes.titleweight": "medium", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#BAC8D1", "axes.linewidth": .8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": GRID, "grid.linewidth": .7,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "path",
        "lines.solid_capstyle": "round"})


def canvas(title, subtitle, single=False):
    fig, axes = plt.subplots(1, 1 if single else 2,
                             figsize=(10.5, 8.3) if single else (12.2, 5.25))
    fig.subplots_adjust(left=.085, right=.97, bottom=.17, top=.75, wspace=.31)
    fig.text(.085, .935, title, fontsize=17, weight="bold", va="top")
    fig.text(.085, .858, subtitle, fontsize=10, color=MUTED, va="top")
    for ax in np.atleast_1d(axes):
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    return fig, axes


def panel(ax, text):
    ax.set_title(text, loc="left", pad=15)


def note(ax, text, xy=(.03, .97)):
    ax.text(*xy, text, transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=.55", fc="#F3F7F9", ec="none"))


def mean_bins(ax, x, residual, color=GOLD, count=10):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, count+1)))
    bins = np.digitize(x, edges[1:-1], right=True)
    centers, averages = [], []
    for index in range(len(edges)-1):
        mask = bins == index
        if mask.any():
            centers.append(x[mask].mean())
            averages.append(residual[mask].mean())
    ax.plot(centers, averages, "o-", color=color, lw=2.1, ms=5,
            markeredgecolor="white", markeredgewidth=.7,
            label="等频分箱残差均值", zorder=5)


class Figures:
    def __init__(self, output):
        self.output = output
        self.folder = output / "figures"
        self.folder.mkdir(exist_ok=True)
        self.data = np.load(output / "analysis.npz")
        self.x, self.y = self.data["x"], self.data["y"]
        self.xt, self.yt = self.data["x_test"], self.data["y_test"]
        self.comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
        self.model = json.loads((output / "final_model.json").read_text(encoding="utf-8"))
        self.groups = json.loads((output / "group_metrics.json").read_text(encoding="utf-8"))
        with open(output / "cv_summary.csv", encoding="utf-8-sig") as file:
            self.cv = list(csv.DictReader(file))
        self.selected = list(MODEL_NAMES).index(self.model["kind"])
        self.prediction = self.data["predictions"][:, self.selected]
        self.residual = self.yt-self.prediction

    def save(self, number, fig):
        stem = f"q2_fig{number:02d}"
        for extension in ["png", "svg"]:
            fig.savefig(self.folder / f"{stem}.{extension}", dpi=300)
        plt.close(fig)
        print(f"Saved {stem}", flush=True)

    def figure1(self):
        fig, axes = canvas("房源差异与租金分布", "训练集 · 7,500 条记录 · 全样本参与统计")
        ax = axes[0]
        panel(ax, "(a) 月租金分布")
        ax.hist(self.y, bins=38, color=BLUE, alpha=.85, edgecolor="white", linewidth=.7)
        median = np.median(self.y)
        ax.axvline(median, color=GOLD, lw=2, linestyle="--")
        ax.set(xlabel="月租金（元/月）", ylabel="房源数量")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        note(ax, f"中位数  {median:,.0f} 元/月\n四分位区间  {np.quantile(self.y,.25):,.0f}–{np.quantile(self.y,.75):,.0f}")
        ax = axes[1]
        panel(ax, "(b) 建筑面积与月租金")
        ax.scatter(self.x[:, 0], self.y, s=5, alpha=.12, color=BLUE,
                   edgecolors="none", rasterized=True)
        edges = np.quantile(self.x[:, 0], np.linspace(0, 1, 13))
        bins = np.digitize(self.x[:, 0], edges[1:-1])
        centers = [np.mean(self.x[bins == k, 0]) for k in range(12)]
        medians = [np.median(self.y[bins == k]) for k in range(12)]
        ax.plot(centers, medians, "o-", color=TEAL, lw=2, ms=4,
                label="等频分箱租金中位数")
        ax.set(xlabel="建筑面积（平方米）", ylabel="月租金（元/月）")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.legend(loc="upper left")
        self.save(1, fig)

    def figure2(self):
        fig, ax = canvas("训练特征的关联结构", "Pearson 相关系数 · 颜色表示方向，数值表示关联强度", single=True)
        fig.subplots_adjust(left=.18, right=.88, bottom=.15, top=.79)
        correlation = np.corrcoef(np.column_stack([self.x, self.y]).T)
        mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
        cmap = LinearSegmentedColormap.from_list("association", [GOLD, "#F8FAFB", TEAL])
        cmap.set_bad("white")
        picture = ax.imshow(np.ma.array(correlation, mask=mask), vmin=-1, vmax=1, cmap=cmap)
        names = LABELS + ["月租金"]
        ax.set_xticks(range(11), names, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(11), names)
        ax.tick_params(length=0, pad=9)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(11):
            for col in range(row+1):
                value = correlation[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8.8,
                        color="white" if abs(value) > .60 else INK)
        bar = fig.colorbar(picture, ax=ax, fraction=.032, pad=.04, shrink=.58)
        bar.outline.set_visible(False)
        bar.set_ticks([-1, -.5, 0, .5, 1])
        self.save(2, fig)

    def figure3(self):
        fig, axes = canvas("模型选择：新增特征带来主要改善", "固定五折验证 · 误差条为折间标准差，非置信区间 · 测试集不参与选择")
        ax = axes[0]
        panel(ax, "(a) 各组最优验证 RMSE")
        values = [r["rmse_mean"] for r in self.comparison]
        stds = [r["rmse_std"] for r in self.comparison]
        for i, (value, std, color) in enumerate(zip(values, stds, COLORS)):
            ax.hlines(i, 0, value, color=color, alpha=.23, lw=7)
            ax.errorbar(value, i, xerr=std, fmt="o", ms=7, color=color,
                        capsize=4, elinewidth=1.5)
            ax.text(value+16, i, f"{value:.2f}", va="center", fontsize=10, color=color)
        ax.set_yticks(range(4), [r["label"] for r in self.comparison])
        ax.set(xlim=(0, 555), ylim=(3.5, -.5), xlabel="验证 RMSE（元/月）")
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax = axes[1]
        panel(ax, "(b) 正则化强度敏感性")
        for kind, color, label in [("ridge", PURPLE, "完整 Ridge"),
                                    ("expanded", TEAL, "扩展回归")]:
            rows = [r for r in self.cv if r["model"] == kind and float(r["penalty"]) > 0]
            penalty = np.array([float(r["penalty"]) for r in rows])
            value = np.array([float(r["rmse_mean"]) for r in rows])
            std = np.array([float(r["rmse_std"]) for r in rows])
            ax.plot(penalty, value, "o-", color=color, lw=2, ms=4, label=label)
            ax.fill_between(penalty, value-std, value+std, color=color, alpha=.10)
        ax.set_xscale("log")
        ax.set(xlabel="正则化系数 λ", ylabel="验证 RMSE（元/月）", ylim=(140, 430))
        ax.legend(loc="upper left")
        ax.text(.03, .52, "弱惩罚区间表现接近\n较强惩罚使误差上升", transform=ax.transAxes,
                fontsize=9.5, color=MUTED)
        self.save(3, fig)

    def figure4(self):
        fig, axes = canvas("测试集预测：扩展模型更接近真实租金", "独立测试集 · 1,500 条房源 · 两图采用相同坐标与点大小")
        lo = min(self.yt.min(), self.data["predictions"][:, [1, self.selected]].min())-150
        hi = max(self.yt.max(), self.data["predictions"][:, [1, self.selected]].max())+150
        for ax, index, color, title in zip(axes, [1, self.selected], [BLUE, TEAL],
                                          ["(a) 完整 OLS", "(b) 最终扩展回归"]):
            prediction = self.data["predictions"][:, index]
            ax.scatter(self.yt, prediction, s=11, color=color, alpha=.32,
                       edgecolors="none", rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--", label="理想预测")
            panel(ax, title)
            score = self.comparison[index]["test"]
            note(ax, f'RMSE  {score["rmse"]:.2f} 元/月\nMAE    {score["mae"]:.2f} 元/月\nR²        {score["r2"]:.4f}')
            ax.set(xlabel="真实月租金（元/月）", ylabel="预测月租金（元/月）",
                   xlim=(lo, hi), ylim=(lo, hi))
            ax.xaxis.set_major_locator(MultipleLocator(1500))
            ax.yaxis.set_major_locator(MultipleLocator(1500))
            ax.set_aspect("equal", adjustable="box")
        self.save(4, fig)

    def figure5(self):
        fig, axes = canvas("残差诊断：整体精度之外仍有结构", "残差 = 真实租金 − 预测租金 · 正值表示低估 · 线条为等频分箱均值")
        limit = max(650, np.max(abs(self.residual))*1.10)
        for ax, x, title, label in zip(axes, [self.prediction, self.xt[:, 0]],
                                      ["(a) 残差与预测租金", "(b) 残差与建筑面积"],
                                      ["预测月租金（元/月）", "建筑面积（平方米）"]):
            panel(ax, title)
            ax.axhspan(-200, 200, color=TEAL, alpha=.045)
            ax.axhline(0, color=INK, lw=1, ls="--")
            ax.scatter(x, self.residual, s=10, color=BLUE, alpha=.22,
                       edgecolors="none", rasterized=True)
            mean_bins(ax, x, self.residual)
            ax.set(xlabel=label, ylabel="预测残差（元/月）", ylim=(-limit, limit))
        axes[1].legend(loc="upper right")
        self.save(5, fig)

    def figure6(self):
        fig, axes = canvas("分组检验：面积区间的误差并不一致", "分界使用训练集面积四分位数 · 在测试集上统计 · n 表示该组房源数")
        rows = [r for r in self.groups if r["group"] == "area"]
        labels = [f'Q{i+1}  ·  n={row["count"]}' for i, row in enumerate(rows)]
        y = np.arange(4)
        ax = axes[0]
        panel(ax, "(a) 各组预测 RMSE")
        values = [r["rmse"] for r in rows]
        bars = ax.barh(y, values, height=.52, color=[GOLD, BLUE, TEAL, PURPLE])
        ax.bar_label(bars, fmt="%.1f", padding=7, color=INK, fontsize=10)
        ax.axvline(self.comparison[self.selected]["test"]["rmse"], ls="--", lw=1.3,
                   color=MUTED, label="整体 RMSE")
        ax.set_yticks(y, labels)
        ax.set(xlim=(0, 220), xlabel="RMSE（元/月）", ylim=(3.55, -.55))
        ax.legend(loc="lower right")
        ax = axes[1]
        panel(ax, "(b) 各组平均残差")
        biases = [r["bias"] for r in rows]
        for index, value in enumerate(biases):
            color = GOLD if value < 0 else TEAL
            ax.hlines(index, 0, value, lw=5, color=color, alpha=.45)
            ax.plot(value, index, "o", color=color, ms=7)
            ax.text(value+(7 if value > 0 else -7), index, f"{value:+.1f}",
                    ha="left" if value > 0 else "right", va="center", color=color)
        ax.axvline(0, color=INK, lw=1)
        ax.set_yticks(y, [f"Q{i+1}" for i in range(4)])
        ax.set(xlim=(-125, 125), xlabel="平均残差（元/月）", ylim=(3.55, -.55))
        for ax in axes:
            ax.grid(axis="x")
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
        fig.text(.085, .045, "面积分界：66.11、96.92、125.81 平方米。负平均残差表示该组整体被高估。",
                 fontsize=9, color=MUTED)
        self.save(6, fig)

    def figure7(self):
        fig, axes = canvas("最终模型的条件响应", "由已拟合系数计算 · 固定其余输入 · 展示模型函数形状，不表示因果效应")
        beta = np.array(self.model["coefficients"])
        ax = axes[0]
        panel(ax, "(a) 地铁距离的合成响应")
        distance = np.linspace(self.x[:, 2].min()/1000, self.x[:, 2].max()/1000, 300)
        reference = np.median(self.x[:, 2])/1000
        effect = beta[2]*(distance-reference) + beta[10]*(np.log1p(distance)-np.log1p(reference))
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.plot(distance, effect, color=TEAL, lw=2.5)
        ax.plot(reference, 0, "o", color=GOLD, ms=6, zorder=5)
        ax.annotate(f"参照距离 {reference:.2f} km", (reference, 0),
                    (reference+.3, 160), fontsize=9, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
        ax.set(xlabel="距最近地铁站距离（千米）", ylabel="相对参照点的预测差异（元/月）")
        ax = axes[1]
        panel(ax, "(b) 楼层与电梯的交互作用")
        floor = np.arange(1, 29)
        for elevator, color, label in [(0, GOLD, "无电梯"), (1, TEAL, "有电梯")]:
            effect = beta[5]*(floor-1)+beta[8]*elevator+beta[12]*floor*elevator
            ax.plot(floor, effect, lw=2.5, color=color, label=label)
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set(xlabel="楼层（层）", ylabel="相对一层无电梯的预测差异（元/月）",
               xlim=(1, 28))
        ax.legend(loc="upper left")
        self.save(7, fig)

    def figure8(self):
        fig, axes = canvas("电梯配置变化下的预测表现", "样本构成与分组误差均在模型确定后描述 · 不用于重新调参")
        ax = axes[0]
        panel(ax, "(a) 训练集与测试集的配置比例")
        shares = [100*self.x[:, 8].mean(), 100*self.xt[:, 8].mean()]
        ax.barh([0, 1], shares, color=TEAL, height=.46, label="有电梯")
        ax.barh([0, 1], 100-np.array(shares), left=shares, color="#DCE5EB",
                height=.46, label="无电梯")
        for index, value in enumerate(shares):
            ax.text(value/2, index, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=12)
            ax.text(value+(100-value)/2, index, f"{100-value:.0f}%", ha="center", va="center", color=MUTED, fontsize=12)
        ax.set_yticks([0, 1], ["训练集\nn=7,500", "测试集\nn=1,500"])
        ax.set(xlim=(0, 100), ylim=(1.65, -.65), xlabel="占比（%）")
        ax.legend(loc="lower right", ncols=2)
        ax.grid(False)
        ax = axes[1]
        panel(ax, "(b) 最终模型在两类房源上的误差")
        rows = [r for r in self.groups if r["group"] == "elevator"]
        locations = np.arange(2)
        for offset, metric, label, color in [(-.16, "rmse", "RMSE", TEAL),
                                              (.16, "mae", "MAE", BLUE)]:
            bars = ax.bar(locations+offset, [r[metric] for r in rows], width=.28,
                          color=color, label=label)
            ax.bar_label(bars, fmt="%.1f", padding=5, color=color, fontsize=9.5)
        ax.set_xticks(locations, ["无电梯\nn=750", "有电梯\nn=750"])
        ax.set(ylabel="预测误差（元/月）", ylim=(0, 210))
        ax.legend(loc="upper right", ncols=2)
        self.save(8, fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/question2_complete")
    parser.add_argument("--figures", nargs="+", type=int, default=list(range(1, 9)))
    args = parser.parse_args()
    set_style()
    figures = Figures(BASE / args.output)
    for number in args.figures:
        getattr(figures, f"figure{number}")()
