from copy import error
from sympy.physics.units import pressure
import pandas as pd
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from Common_Vars import (
    _Path_fig,
    plt,
    fontsize,
    figsize_x,
    figsize_y,
    filter_data,
    rmse,
    compute_percentage_weights,
    calculate_slope,
)

num_bins = 50
bins = np.linspace(-0.8, 0.8, num_bins)
# matplotlib.rcParams["font.size"] = 14
# matplotlib.rcParams["font.family"] = "Arial"

pore_data_optimized = pd.read_csv("plot_pore_N5.000_optimized_.csv")


# pore_data_original = pd.read_csv("plot_pore_N5.000_original.csv")
# pore_data_optimized = pd.read_csv("plot_pore_N5.000_optimized.csv")

# COMSOL 数据
P_comsol = pore_data_optimized["pore_P_comsol"].dropna().to_numpy()
P_dualn_optimized = pore_data_optimized["pore_P_dualn"].dropna().to_numpy()
volume_void_dualn_optimized = (
    pore_data_optimized["pore_volume_void_dualn"].dropna().to_numpy()
)
volume_solid_dualn_optimized = (
    pore_data_optimized["pore_volume_solid_dualn"].dropna().to_numpy()
)
# 温度数据
T_void_comsol = pore_data_optimized["pore_T_void_comsol"].dropna().to_numpy()
T_solid_comsol = pore_data_optimized["pore_T_solid_comsol"].dropna().to_numpy()
T_void_dualn_optimized = pore_data_optimized["pore_T_void_dualn"].dropna().to_numpy()
T_solid_dualn_optimized = pore_data_optimized["pore_T_solid_dualn"].dropna().to_numpy()

T_comsol = np.concatenate((T_void_comsol, T_solid_comsol))
T_dualn_optimized = np.concatenate((T_void_dualn_optimized, T_solid_dualn_optimized))
volume_dualn_optimized = np.concatenate(
    (volume_void_dualn_optimized, volume_solid_dualn_optimized)
)


# 过滤压力数据

P_mask_optimized = filter_data(P_comsol, P_dualn_optimized)
T_mask_optimized = filter_data(T_comsol, T_dualn_optimized)


P_comsol_optimized_filtered, P_dualn_optimized_filtered = (
    P_comsol[P_mask_optimized],
    P_dualn_optimized[P_mask_optimized],
)
volume_P_optimized_filtered = volume_void_dualn_optimized[P_mask_optimized]

T_comsol_optimized_filtered, T_dualn_optimized_filtered = (
    T_comsol[T_mask_optimized],
    T_dualn_optimized[T_mask_optimized],
)
volume_T_optimized_filtered = volume_dualn_optimized[T_mask_optimized]


np.random.seed(1)
num = min(2000, len(P_comsol_optimized_filtered))
num = len(P_comsol_optimized_filtered)
# 随机采样
P_index_optimized = np.random.choice(
    len(P_comsol_optimized_filtered), num, replace=False
)
T_index_optimized = np.random.choice(
    len(T_comsol_optimized_filtered), num, replace=False
)

P_comsol_optimized_sampled = P_comsol_optimized_filtered[P_index_optimized]
P_dualn_optimized_sampled = P_dualn_optimized_filtered[P_index_optimized]

# 计算R²值
r2_pressure_optimized = r2_score(P_comsol_optimized_sampled, P_dualn_optimized_sampled)
rmse_pressure_optimized = rmse(P_comsol_optimized_sampled, P_dualn_optimized_sampled)
slope_pressure_optimized = calculate_slope(
    P_comsol_optimized_sampled, P_dualn_optimized_sampled
)

temp = np.abs(
    (P_dualn_optimized_sampled - P_comsol_optimized_sampled)
    / P_comsol_optimized_sampled
)
mape_pressure_optimized = mean_absolute_percentage_error(
    P_comsol_optimized_sampled[temp < 0.5],
    P_dualn_optimized_sampled[temp < 0.5],
)


print("r2_pressure_optimized:", r2_pressure_optimized)
print("rmse_pressure_optimized:", rmse_pressure_optimized)
print("slope_pressure_optimized:", slope_pressure_optimized)
print("mape_pressure_optimized:", mape_pressure_optimized)

fig, axs = plt.subplots(2, 2, figsize=(figsize_x * 2, figsize_y * 2))
ax1, ax_hist1, ax2, ax_hist2 = axs.ravel()

c0_color = plt.cm.tab10(0)  # 获取C0颜色 (RGBA)
# 将RGB分量变淡一半，保持alpha=1
light_c0 = (
    (c0_color[0] + 1.0) / 2,  # R分量变淡
    (c0_color[1] + 1.0) / 2,  # G分量变淡
    (c0_color[2] + 1.0) / 2,  # B分量变淡
    1.0,
)  #

ax1.scatter(
    P_comsol_optimized_sampled,
    P_dualn_optimized_sampled,
    marker="o",
    color="C0",
    alpha=0.5,
    # label="Calibrated DNM",
)
# # 获取坐标轴范围，用于定位文本（避免遮挡数据）
# xlim = ax1.get_xlim()
# ylim = ax1.get_ylim()
# x_range = xlim[1] - xlim[0]
# y_range = ylim[1] - ylim[0]

# # 设置文本位置：比如放在右下角偏移一点
# text_x = xlim[0] + 0.08 * x_range
# text_y = ylim[1] - 0.215 * y_range

# # 构造多行文本
# text_str = (
#     f"a = {slope_pressure_optimized:.2f}\nMAPE = {mape_pressure_optimized * 100:.0f}%"
# )

# # 添加文本（使用深灰色，避免太突兀）
# ax1.text(
#     text_x,
#     text_y,
#     text_str,
#     fontsize=14,
#     verticalalignment="bottom",
#     horizontalalignment="left",
#     linespacing=1.5,
# )


ax1.axline((0, 0), slope=1, color="C3", linestyle="--", label="y = x")
ax1.set_xlabel("DNS (Pa)\n(a)")
ax1.set_ylabel("DNM (Pa)")
xy_max_P = (
    max(np.max(P_comsol_optimized_sampled), np.max(P_dualn_optimized_sampled)) * 1.02
)
ax1.set_xlim(0, xy_max_P)
ax1.set_ylim(0, xy_max_P)

# 添加R²值到图例
# ax1.text(
#     0.05,
#     0.95,
#     f"R² = {r2_pressure_original:.2f}",
#     transform=ax1.transAxes,
#     fontsize=12,
#     verticalalignment="top",
# )


# ax1.text(
#     0.03,
#     0.97,
#     f"$R^2$ = {r2_pressure_optimized:.2f}",
#     transform=ax1.transAxes,
#     verticalalignment="top",
# )
ax1.legend(frameon=False, edgecolor="none", loc="upper left")
# 添加误差直方图
# divider = make_axes_locatable(ax1)
# ax_hist1 = divider.append_axes("right", size=6.1, pad=0.8)


error_optimized = (
    P_dualn_optimized_sampled - P_comsol_optimized_sampled
) / P_comsol_optimized_sampled  # 百分比误差


mean_val_optimized = np.mean(error_optimized)
std_val_optimized = np.std(error_optimized)
ax_hist1.hist(
    error_optimized,
    bins=bins,
    orientation="vertical",
    alpha=0.5,
    color="C0",
    # density=True,
    weights=compute_percentage_weights(error_optimized),
)

error_optimized_P_min = np.min(error_optimized)
error_optimized_P_max = np.max(error_optimized)


mean_val_optimized_abs = np.abs(error_optimized).mean()
std_val_optimized_abs = np.abs(error_optimized).std()
print(
    f"Optimized DNM Pressure: mean = {mean_val_optimized_abs:.2f}%, std = {std_val_optimized_abs:.2f}%, R² = {r2_pressure_optimized:.4f}"
)

# ax_hist1.axvline(0, color="C3", linestyle="--", linewidth=1.5)
ax_hist1.set_xlabel("Relative error\n(b)")
ax_hist1.set_ylabel("Proportion")

# ax_hist1.set_ylim(-0.4,0.4)

# plt.tight_layout()
# plt.show()

T_comsol_optimized_filtered -= 293.15
T_dualn_optimized_filtered -= 293.15

# 随机采样
T_index_optimized = np.random.choice(
    len(T_comsol_optimized_filtered), num, replace=False
)
T_comsol_optimized_sampled = T_comsol_optimized_filtered[T_index_optimized]
T_dualn_optimized_sampled = T_dualn_optimized_filtered[T_index_optimized]
# 计算R²值
r2_T_optimized = r2_score(T_comsol_optimized_sampled, T_dualn_optimized_sampled)
rmse_T_optimized = rmse(T_comsol_optimized_sampled, T_dualn_optimized_sampled)
slope_T_optimized = calculate_slope(
    T_comsol_optimized_sampled, T_dualn_optimized_sampled
)
T_bool = (
    np.abs(
        (T_dualn_optimized_sampled - T_comsol_optimized_sampled)
        / T_comsol_optimized_sampled
    )
    < 0.5
)
mape_T_optimized = mean_absolute_percentage_error(
    T_comsol_optimized_sampled[T_bool], T_dualn_optimized_sampled[T_bool]
)
print("r2_T_optimized:", r2_T_optimized)
print("rmse_T_optimized:", rmse_T_optimized)
print("slope_T_optimized:", slope_T_optimized)
print("mape_T_optimized:", mape_T_optimized)

ax2.scatter(
    T_comsol_optimized_sampled,
    T_dualn_optimized_sampled,
    marker="o",
    color="C1",
    alpha=0.5,
)
# # 获取坐标轴范围，用于定位文本（避免遮挡数据）
# xlim = ax2.get_xlim()
# ylim = ax2.get_ylim()
# x_range = xlim[1] - xlim[0]
# y_range = ylim[1] - ylim[0]

# # 设置文本位置：比如放在右下角偏移一点
# text_x = xlim[0] + 0.08 * x_range
# text_y = ylim[1] - 0.215 * y_range

# # 构造多行文本
# text_str = f"a = {slope_T_optimized:.3f}\nMAPE = {mape_T_optimized * 100:.0f}%"

# # 添加文本（使用深灰色，避免太突兀）
# ax2.text(
#     text_x,
#     text_y,
#     text_str,
#     fontsize=14,
#     verticalalignment="bottom",
#     horizontalalignment="left",
#     linespacing=1.5,
# )
# origin = (min(T_comsol_original_sampled.min(), T_dualn_original_sampled.min()),) * 2
ax2.axline((0, 0), slope=1, color="C3", linestyle="--", label="y = x")
ax2.set_xlabel("DNS (K)\n(c)")
ax2.set_ylabel("DNM (K)")
xy_max_T = (
    max(np.max(T_comsol_optimized_sampled), np.max(T_dualn_optimized_sampled)) * 1.02
)

ax2.set_xlim(0, xy_max_T)
ax2.set_ylim(0, xy_max_T)

ax2.legend(
    frameon=False,
    edgecolor="none",
    loc="upper left",
)
# ax2.text(
#     0.03,
#     0.97,
#     f"$R^2$ = {r2_T_optimized:.2f}",
#     transform=ax2.transAxes,
#     verticalalignment="top",
# )


# 添加误差直方图
# divider = make_axes_locatable(ax2)
# ax_hist2 = divider.append_axes("right", size=2, pad=0.8)

error_optimized = (
    T_dualn_optimized_sampled - T_comsol_optimized_sampled
) / T_comsol_optimized_sampled  # 百分比误差


mean_val_optimized = np.mean(error_optimized)
std_val_optimized = np.std(error_optimized)

mean_val_optimized_abs = np.abs(error_optimized).mean()
std_val_optimized_abs = np.abs(error_optimized).std()
print(
    f"Optimized DNM Temperature: mean = {mean_val_optimized_abs:.2f}%, std = {std_val_optimized_abs:.2f}%, R² = {r2_T_optimized:.4f}"
)

ax_hist2.hist(
    error_optimized,
    bins=bins,
    orientation="vertical",
    alpha=0.5,
    color="C1",
    # density=True,
    weights=compute_percentage_weights(error_optimized),
)
ax_hist1.axvline(0, color="C3", linestyle="--", linewidth=1.5)
ax_hist2.axvline(0, color="C3", linestyle="--", linewidth=1.5)

# error_optimized_T_min = np.min(error_optimized)
# error_optimized_T_max = np.max(error_optimized)


# error_optimized_min = min(error_optimized_P_min, error_optimized_T_min)
# error_optimized_max = max(error_optimized_P_max, error_optimized_T_max)

# bins_norm = np.linspace(error_optimized_min, error_optimized_max, 300)
# y_norm = norm.pdf(bins_norm, mean_val_optimized, std_val_optimized)
# ax_hist1.plot(bins_norm, y_norm, color="C0", linestyle="-", linewidth=2)

# bins_norm = np.linspace(error_optimized_min, error_optimized_max, 300)
# y_norm = norm.pdf(bins_norm, mean_val_optimized, std_val_optimized)
# ax_hist2.plot(bins_norm, y_norm, color="C1", linestyle="-", linewidth=2)

# ax_hist1_xlim = ax_hist1.get_xlim()
# ax_hist2_xlim = ax_hist2.get_xlim()
# xlim_min = min(ax_hist1_xlim[0], ax_hist2_xlim[0])
# xlim_max = max(ax_hist1_xlim[1], ax_hist2_xlim[1])

# ax_hist1_ylim = ax_hist1.get_ylim()
# ax_hist2_ylim = ax_hist2.get_ylim()
# ylim_min = min(ax_hist1_ylim[0], ax_hist2_ylim[0])
# ylim_max = max(ax_hist1_ylim[1], ax_hist2_ylim[1])

# ax_hist1.set_xlim(xlim_min, xlim_max)
# ax_hist2.set_xlim(xlim_min, xlim_max)
# ax_hist1.set_ylim(ylim_min, ylim_max)
# ax_hist2.set_ylim(ylim_min, ylim_max)


ax_hist1.set_xlim(-0.5, 0.5)
ax_hist2.set_xlim(-0.5, 0.5)

# print(ax_hist1.get_xticklabels())
xticks = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
xlabels = [f"{x:.2f}" for x in xticks]

ax_hist1.set_xticks(xticks)
# ax_hist1.set_xticklabels(xlabels)

ax_hist2.set_xticks(xticks)
# ax_hist2.set_xticklabels(xlabels)
# ax_hist1.set_ylim(ax_hist2.get_ylim())
# ax_hist2.axvline(0, color="C3", linestyle="--", linewidth=1.5)
ax_hist2.set_xlabel("Relative error\n(d)")
ax_hist2.set_ylabel("Proportion")

plt.tight_layout()
plt.savefig(_Path_fig / "small_pore_distribution_N5.000.png")
plt.show()
