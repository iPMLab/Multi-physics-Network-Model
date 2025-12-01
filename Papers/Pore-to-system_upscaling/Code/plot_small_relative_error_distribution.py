from sympy.physics.units import pressure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from addict import Dict
from Common_Vars import _Path_fig, plt, figsize_x, figsize_y

ed = Dict()


ed.relative_error.P.original = np.abs(
    pd.read_csv("./plot_relative_error_P_N5.000_original.csv").to_numpy()[:, 1:]
)
ed.relative_error.hf.original = np.abs(
    pd.read_csv("./plot_relative_error_hf_N5.000_original.csv").to_numpy()[:, 1:]
)
ed.relative_error.P.optimized = np.abs(
    pd.read_csv("./plot_relative_error_P_N5.000_optimized.csv").to_numpy()[:, 1:]
)
ed.relative_error.hf.optimized = np.abs(
    pd.read_csv("./plot_relative_error_hf_N5.000_optimized.csv").to_numpy()[:, 1:]
)


# 示例数据：3个工况点，每个工况点有5个样本的误差数据
cases = ["0.001", "0.005", "0.02", "0.1", "1"]  # 工况点名称

# 压降误差数据（每个工况点5个样本）
pressure_original = ed.relative_error.P.original.T
pressure_optimized = ed.relative_error.P.optimized.T
# 换热量误差数据（每个工况点5个样本）
heatflux_original = ed.relative_error.hf.original.T
heatflux_optimized = ed.relative_error.hf.optimized.T


# 创建图表
# 计算中位数
pressure_original_mean = np.mean(pressure_original, axis=0)
pressure_optimized_mean = np.mean(pressure_optimized, axis=0)
heatflux_original_mean = np.mean(heatflux_original, axis=0)
heatflux_optimized_mean = np.mean(heatflux_optimized, axis=0)

pressure_original_all_mean = np.mean(pressure_original_mean)
pressure_optimized_all_mean = np.mean(pressure_optimized_mean)
heatflux_original_all_mean = np.mean(heatflux_original_mean)
heatflux_optimized_all_mean = np.mean(heatflux_optimized_mean)
print("pressure_original_all_mean:", pressure_original_all_mean)
print("pressure_optimized_all_mean:", pressure_optimized_all_mean)
print("heatflux_original_all_mean:", heatflux_original_all_mean)
print("heatflux_optimized_all_mean:", heatflux_optimized_all_mean)

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize_x * 2, figsize_y))

# 设置X轴位置
x_pos = np.arange(len(cases))
width = 0.35  # 箱型图之间的间距

# 第一个子图：压降误差对比
# 原始数据箱型图
bp1 = ax1.boxplot(
    pressure_original,
    positions=x_pos - width / 2,
    widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="C0", alpha=0.5),
    medianprops=dict(color="darkblue", linewidth=1),
    whiskerprops=dict(color="C0"),
    capprops=dict(color="C0"),
    showfliers=False,
)

# 优化数据箱型图
bp2 = ax1.boxplot(
    pressure_optimized,
    positions=x_pos + width / 2,
    widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="C1", alpha=0.5),
    medianprops=dict(color="darkred", linewidth=1),
    whiskerprops=dict(color="C1"),
    capprops=dict(color="C1"),
    showfliers=False,
)

# 绘制中位数连线
(line1,) = ax1.plot(
    x_pos - width / 2,
    pressure_original_mean,
    "o-",
    color="navy",
    linewidth=1,
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="navy",
    label="Original DNM",
)

(line2,) = ax1.plot(
    x_pos + width / 2,
    pressure_optimized_mean,
    "o-",
    color="darkred",
    linewidth=1,
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="darkred",
    label="Calibrated DNM",
)


ax1.set_xlabel(
    "Re\n(a)",
)
ax1.set_ylabel(
    "Discrepancy",
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(
    cases,
)
ax1.legend(frameon=False, loc="upper left")
ax1.set_ylim([-0.01, 0.33])

# 第二个子图：换热量误差对比
# 原始数据箱型图
bp3 = ax2.boxplot(
    heatflux_original,
    positions=x_pos - width / 2,
    widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="C0", alpha=0.5),
    medianprops=dict(color="darkblue", linewidth=1),
    whiskerprops=dict(color="C0"),
    capprops=dict(color="C0"),
    showfliers=False,
)

# 优化数据箱型图
bp4 = ax2.boxplot(
    heatflux_optimized,
    positions=x_pos + width / 2,
    widths=0.25,
    patch_artist=True,
    boxprops=dict(facecolor="C1", alpha=0.5),
    medianprops=dict(color="darkred", linewidth=1),
    whiskerprops=dict(color="C1"),
    capprops=dict(color="C1"),
    showfliers=False,
)

# 绘制中位数连线
(line3,) = ax2.plot(
    x_pos - width / 2,
    heatflux_original_mean,
    "o-",
    color="navy",
    linewidth=1,
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="navy",
    label="Original DNM",
)

(line4,) = ax2.plot(
    x_pos + width / 2,
    heatflux_optimized_mean,
    "o-",
    color="darkred",
    linewidth=1,
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="darkred",
    label="Calibrated DNM",
)

# xlbl = ax.xaxis.get_label()

xlbl = ax2.set_xlabel(
    "Re\n(b)",
)
print(xlbl.get_position())
ax2.set_ylabel(
    "Discrepancy",
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(
    cases,
)
ax2.set_ylim([-0.01, 0.119])
ax2.legend(frameon=False, loc="upper left")


# ax1.text(
#     -0.1,
#     1.0,
#     "(a)",
#     transform=ax1.transAxes,
#     # fontweight="bold",
#     va="top",
#     ha="right",
# )
# 为(b)添加注释
# ax2.text(
#     0.5,
#     -0.2,
#     "(b)",
#     transform=ax2.transAxes,
#     # fontweight="bold",
#     va="top",
#     ha="center",
# )
# 调整布局
plt.tight_layout()
plt.savefig(
    _Path_fig / "small_relative_error_distribution.png",
)
plt.show()
