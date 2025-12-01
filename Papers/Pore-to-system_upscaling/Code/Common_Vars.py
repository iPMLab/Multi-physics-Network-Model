from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

fontsize = 14
figsize_x = 6
figsize_y = 5.5
rcparams = {}
rcparams["font.size"] = fontsize
rcparams["font.family"] = "Arial"
rcparams["savefig.dpi"] = 150
rcparams["figure.autolayout"] = True
rcparams["savefig.bbox"] = None  # 等效于 bbox_inches='tight'
rcparams["savefig.pad_inches"] = 0  # 等效于 pad_inches=0
matplotlib.rcParams.update(rcparams)
_Path_fig = Path(r"D:\yjp\Workdir\浙江大学\论文\2025-08-26英文论文\Figs")


def compute_percentage_weights(data, weight=None):
    """
    计算数据的权重，使得直方图各柱子的高度代表百分比。

    参数:
    - data: 要绘制直方图的数据数组。

    返回:
    - weights: 对应于data中每个元素的权重数组，用于hist函数中的weights参数。
    """
    data = np.asarray(data)
    if data.size == 0:
        # 如果数据为空，则返回空权重
        return np.array([])
    else:
        if weight is None:
            weights = np.ones_like(data) / data.size
        else:
            weight = np.asarray(weight)
            weights = weight / np.sum(weight)
            print(weights)

        return weights


def filter_data(x, y, relative_error=True):
    x = np.asarray(x)
    y = np.asarray(y)
    num_nodes_total = len(x)
    mask = np.zeros_like(x, dtype=bool)

    if relative_error:
        # 标记 x 或 y 为 0 的点（避免除零）
        zero_mask = np.isclose(x, 0, atol=1e-20) | np.isclose(y, 0, atol=1e-20)
        mask |= zero_mask
        rmse = np.abs(x - y)
        upper = np.percentile(rmse, 99.8, method="closest_observation")
        mask |= rmse > upper

    print(f"nodes removed: {mask.sum()} ({mask.sum() / num_nodes_total * 100:.2f}%)")
    return ~mask


def rmse(x, y):
    """
    计算归一化均方根误差（NRMSE）。

    参数:
    - x: 参考值数组。
    - y: 预测值数组。

    返回:
    - nrmse: 归一化均方根误差值。
    """
    if len(x) == 0 or len(y) == 0:
        # 如果数据为空，则返回NaN
        return np.nan
    else:
        rmse = np.sqrt(np.mean((x - y) ** 2))
        return rmse


def calculate_slope(x, y):
    """
    计算线性回归的斜率。

    参数:
    - x: 自变量数组。
    - y: 因变量数组。

    返回:
    - slope: 线性回归的斜率。
    """
    if len(x) == 0 or len(y) == 0:
        # 如果数据为空，则返回NaN
        return np.nan
    else:
        # 计算线性回归的斜率
        slope = np.dot(x, y) / np.dot(x, x)
        return slope
