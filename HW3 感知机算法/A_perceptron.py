import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


training_data = pd.read_csv(
    "/Users/tuyuchen/Downloads/HW3 感知机算法/data/A_train.csv"
)
test_data = pd.read_csv(
    "/Users/tuyuchen/Downloads/HW3 感知机算法/data/A_test.csv"
)

print("A_train 数据集形状：", training_data.shape)
print("A_test 数据集形状：", test_data.shape)

X = training_data[["x1", "x2"]].to_numpy(dtype=float)
y = training_data["y"].to_numpy(dtype=int)

test_X = test_data[["x1", "x2"]].to_numpy(dtype=float)
test_y = test_data["y"].to_numpy(dtype=int)

w = np.zeros(X.shape[1], dtype=float)
b = 0.0
eta = 1.0
k = 0
i = 0

history = [(k, w.copy(), b, None)]
update_records = []

while True:
    margin = y[i] * (np.dot(w, X[i]) + b)

    if margin <= 0:
        old_w = w.copy()
        old_b = b
        wrong_before = int(np.sum(y * (np.dot(X, w) + b) <= 0))
        test_before = np.mean(
            np.where(np.dot(test_X, w) + b >= 0, 1, -1) == test_y
        )

        w = w + eta * y[i] * X[i]
        b = b + eta * y[i]
        k = k + 1

        wrong_after = int(np.sum(y * (np.dot(X, w) + b) <= 0))
        test_after = np.mean(
            np.where(np.dot(test_X, w) + b >= 0, 1, -1) == test_y
        )
        update_records.append(
            {
                "step": k,
                "index": i,
                "id": training_data.iloc[i]["id"],
                "label": int(y[i]),
                "margin_before": margin,
                "w_before": old_w,
                "b_before": old_b,
                "w_after": w.copy(),
                "b_after": b,
                "wrong_before": wrong_before,
                "wrong_after": wrong_after,
                "test_before": test_before,
                "test_after": test_after,
            }
        )

        print(
            f"第 {k:02d} 次更新 | 样本 {i + 1} ({training_data.iloc[i]['id']})，"
            f"y={y[i]:+d}，更新前间隔={margin:.4f}"
        )
        print(
            f"    w: {np.array2string(old_w, precision=4)} "
            f"-> {np.array2string(w, precision=4)}；"
            f"b: {old_b:.4f} -> {b:.4f}"
        )
        print(
            f"    训练集误分类数：{wrong_before} -> {wrong_after}；"
            f"测试集准确率：{test_before:.2%} -> {test_after:.2%}"
        )

        history.append((k, w.copy(), b, i))

    i = i + 1

    if i == len(X):
        margins = y * (np.dot(X, w) + b)

        if np.all(margins > 0):
            break

        i = 0

train_prediction = np.where(np.dot(X, w) + b >= 0, 1, -1)
train_accuracy = np.mean(train_prediction == y)

test_prediction = np.where(np.dot(test_X, w) + b >= 0, 1, -1)
test_accuracy = np.mean(test_prediction == test_y)

print("训练完成")
print("更新次数 k：", k)
print("最终权重 w：", w)
print("最终偏置 b：", b)
print(f"训练集准确率：{train_accuracy:.2%}")
print(f"测试集准确率：{test_accuracy:.2%}")


# 可视化
colors = {-1: "#377eb8", 1: "#f28e2b"}

x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
x1_line = np.linspace(x1_min, x1_max, 200)

columns = 6
rows = int(np.ceil(len(history) / columns))
fig, axes = plt.subplots(rows, columns, figsize=(20, 3.8 * rows))
axes = np.atleast_1d(axes).ravel()

for ax, (step, w_state, b_state, updated_index) in zip(axes, history):
    # 绘制两类样本
    for label in [-1, 1]:
        ax.scatter(
            X[y == label, 0],
            X[y == label, 1],
            s=18,
            alpha=0.65,
            color=colors[label],
            label=f"y={label}",
        )

    scores = np.dot(X, w_state) + b_state
    wrong = y * scores <= 0

    # 标出当前仍然误分类的样本
    if np.any(wrong):
        ax.scatter(
            X[wrong, 0],
            X[wrong, 1],
            s=65,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.3,
        )

    # 标出本次更新所使用的样本
    if updated_index is not None:
        ax.scatter(
            X[updated_index, 0],
            X[updated_index, 1],
            s=120,
            marker="*",
            color="#d62728",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )

    # 绘制当前决策边界：w1*x1 + w2*x2 + b = 0
    if abs(w_state[1]) > 1e-12:
        x2_line = -(w_state[0] * x1_line + b_state) / w_state[1]
        ax.plot(x1_line, x2_line, color="#222222", linewidth=2)
    elif abs(w_state[0]) > 1e-12:
        ax.axvline(-b_state / w_state[0], color="#222222", linewidth=2)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.2)
    ax.set_title(f"更新 k={step}\n误分类样本数：{np.sum(wrong)}")

# 隐藏多余子图
for ax in axes[len(history):]:
    ax.axis("off")

legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[-1], markersize=8, label="y=-1"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], markersize=8, label="y=1"),
    Line2D([0], [0], marker="o", color="#d62728", markerfacecolor="none", markersize=8, label="误分类样本"),
    Line2D([0], [0], marker="*", color="#d62728", markeredgecolor="black", markersize=11, label="本次更新样本"),
    Line2D([0], [0], color="#222222", linewidth=2, label="决策边界"),
]

fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False)
fig.suptitle("感知机学习过程：决策边界随更新变化", fontsize=18, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.94])

output_path = "/Users/tuyuchen/Downloads/HW3 感知机算法/A_perceptron_learning_process.png"
fig.savefig(output_path, dpi=220, bbox_inches="tight")
print("学习过程图已保存到：", output_path)

plt.show()
