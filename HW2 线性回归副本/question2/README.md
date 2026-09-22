# 问题二：房源租金预测

[阅读实验论文](../docs/第二题_实验报告.md)。代码、数值输出与论文对应同一次已运行实验。

## 已完成的结果

四组候选模型使用相同的训练集五折划分。最终选择包含十个原始输入、两个距离对数项和楼层与电梯交互项的扩展回归，L2 系数为 $10^{-5}$。

| 测试指标 | 数值 |
| --- | ---: |
| RMSE（元/月） | 155.1210 |
| MAE（元/月） | 123.7959 |
| $R^2$ | 0.974739 |

弱正则化与无正则扩展模型的验证表现几乎相同；具体模型比较、残差局限和完整数学表达见论文。

## 文件入口

- [model.py](model.py)：固定特征构造、标准化、模型拟合及参数还原。
- [run.py](run.py)：五折交叉验证、选择模型、最终训练和测试评价。
- [plot.py](plot.py)：读取保存结果生成八张图，每张最多两个子图。
- [predict.py](predict.py)：为没有真实租金的新房源生成预测。
- [verify.py](verify.py)：独立求解和折外预测核验、文档静态检查。
- [最终测试预测](../results/question2_complete/test_predictions.csv)：1,500 条记录及残差。
- [精确模型参数](../results/question2_complete/final_model.json)：完整精度、标准化统计量和特征顺序。
- [独立核验结果](../results/question2_complete/verification.json)。

## 复现

从 `HW2 线性回归` 目录运行。在隔离 Python 环境中安装 [requirements.txt](requirements.txt)，再执行：

```bash
python -m pip install -r question2/requirements.txt
python question2/run.py --output results/question2_reproduce
python question2/plot.py --output results/question2_reproduce
python question2/verify.py --output results/question2_reproduce
```

训练命令要求指定的输出目录尚不存在。固定种子为 42，五折划分保存在结果目录。所有预处理统计量在每折训练部分拟合；代码先保存模型选择结果，再读取测试集。绘图程序不训练模型。

本机已成功运行的解释器为 `D:/software/Anaconda/envs/medsafety/python.exe`，其版本为 Python 3.10.20。本机 PowerShell 可用下列方式替代上面的 `python`：

```powershell
& 'D:/software/Anaconda/envs/medsafety/python.exe' -X utf8 question2/verify.py
```

PNG 图分辨率为 300 dpi，同目录 SVG 可用于缩放排版。默认中文字体为 Microsoft YaHei；其他系统应在 `plot.py` 的 `set_style()` 中设置可用中文字体。

## 新房源预测

输入 CSV 包含 `listing_id` 和原数据的十个特征字段即可，无需 `monthly_rent`。地铁距离沿用原数据的米单位。

```bash
python question2/predict.py --input data/new_listings.csv --output results/new_predictions.csv
```

输出文件必须尚不存在。预测采用保存的完整精度系数，不需要重新训练或手动标准化。

## 核验范围

已使用 NumPy 增广矩阵最小二乘独立核对四组最终模型和各组五折预测，验证数据哈希、指标与 CSV 对应关系；无标签预测入口也已通过检查。论文和八张图已人工检查，数学定界符与图片路径已静态检查。尚未在 Markpad 或 Obsidian 实际界面核验公式渲染。
