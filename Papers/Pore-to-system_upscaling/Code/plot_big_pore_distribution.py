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
    compute_percentage_weights,
    filter_data,
    rmse,
    fontsize,
    calculate_slope,
    figsize_x,
    figsize_y,
)


num_bins = 50
bins = np.linspace(-0.8, 0.8, num_bins)
pore_data_original = pd.read_csv("plot_pore_N10.000_original.csv")
pore_data_optimized = pd.read_csv("plot_pore_N10.000_optimized.csv")
pore_data_volume = pd.read_csv("_N10.000_sample0_volume.csv")
volume_void = pore_data_volume["volume_pore"].dropna().to_numpy()
volume_solid = pore_data_volume["volume_solid"].dropna().to_numpy()
volume_dual = np.concatenate((volume_solid, volume_void))
# pore_data_original = pd.read_csv("plot_pore_N5.000_original.csv")
# pore_data_optimized = pd.read_csv("plot_pore_N5.000_optimized.csv")

# COMSOL 数据
P_comsol = pore_data_optimized["pore_P_comsol"].dropna().to_numpy()
P_dualn_original = pore_data_original["pore_P_dualn"].dropna().to_numpy()
P_dualn_optimized = pore_data_optimized["pore_P_dualn"].dropna().to_numpy()

# 温度数据
T_void_comsol = pore_data_optimized["pore_T_void_comsol"].dropna().to_numpy()
T_solid_comsol = pore_data_optimized["pore_T_solid_comsol"].dropna().to_numpy()
T_void_dualn_original = pore_data_original["pore_T_void_dualn"].dropna().to_numpy()
T_solid_dualn_original = pore_data_original["pore_T_solid_dualn"].dropna().to_numpy()
T_void_dualn_optimized = pore_data_optimized["pore_T_void_dualn"].dropna().to_numpy()
T_solid_dualn_optimized = pore_data_optimized["pore_T_solid_dualn"].dropna().to_numpy()

T_comsol = np.concatenate((T_void_comsol, T_solid_comsol))
T_dualn_original = np.concatenate((T_void_dualn_original, T_solid_dualn_original))
T_dualn_optimized = np.concatenate((T_void_dualn_optimized, T_solid_dualn_optimized))


# 过滤压力数据

P_mask_original = filter_data(P_comsol, P_dualn_original)
P_mask_optimized = filter_data(P_comsol, P_dualn_optimized)
P_mask = P_mask_original & P_mask_optimized
P_comsol_original_filtered = P_comsol[P_mask]
P_dualn_original_filtered = P_dualn_original[P_mask]
P_comsol_optimized_filtered = P_comsol[P_mask]
P_dualn_optimized_filtered = P_dualn_optimized[P_mask]
volume_void_filtered = volume_void[P_mask]


T_mask_original = filter_data(T_comsol, T_dualn_original)
T_mask_optimized = filter_data(T_comsol, T_dualn_optimized)
T_mask = T_mask_original & T_mask_optimized
T_comsol_original_filtered = T_comsol[T_mask]
T_dualn_original_filtered = T_dualn_original[T_mask]
T_comsol_optimized_filtered = T_comsol[T_mask]
T_dualn_optimized_filtered = T_dualn_optimized[T_mask]
volume_dual_filtered = volume_dual[T_mask]


P_comsol_min = min(
    np.min(P_comsol_original_filtered), np.min(P_comsol_optimized_filtered)
)
P_comsol_max = max(
    np.max(P_comsol_original_filtered), np.max(P_comsol_optimized_filtered)
)


T_comsol_min = min(
    np.min(T_comsol_original_filtered), np.min(T_comsol_optimized_filtered)
)
T_comsol_max = max(
    np.max(T_comsol_original_filtered), np.max(T_comsol_optimized_filtered)
)


np.random.seed(1)
num_P = len(P_comsol_original_filtered)
num_T = len(T_comsol_original_filtered)
# 随机采样
# P_index_original = np.random.choice(len(P_comsol_original_filtered), num, replace=False)
# P_index_optimized = np.random.choice(
#     len(P_comsol_optimized_filtered), num, replace=False
# )
# T_index_original = np.random.choice(len(T_comsol_original_filtered), num, replace=False)
# T_index_optimized = np.random.choice(
#     len(T_comsol_optimized_filtered), num, replace=False
# )

P_comsol_original_sampled = P_comsol_original_filtered
P_comsol_optimized_sampled = P_comsol_optimized_filtered
P_dualn_original_sampled = P_dualn_original_filtered
P_dualn_optimized_sampled = P_dualn_optimized_filtered
# df = pd.DataFrame(
#     {
#         "" "P_comsol_original": P_comsol_original_sampled,
#         "P_dualn_original": P_dualn_original_sampled,
#         "P_comsol_optimized": P_comsol_optimized_sampled,
#         "P_dualn_optimized": P_dualn_optimized_sampled,
#     }
# )
# df.to_csv("plot_pore_N10.000.csv", index=False)


# 计算R²值
r2_pressure_original = r2_score(P_comsol_original_sampled, P_dualn_original_sampled)
r2_pressure_optimized = r2_score(P_comsol_optimized_sampled, P_dualn_optimized_sampled)
nrmse_P_original = rmse(P_comsol_original_sampled, P_dualn_original_sampled)
nrmse_P_optimized = rmse(P_comsol_optimized_sampled, P_dualn_optimized_sampled)

MAPE_P_original_bool = (
    np.abs(
        (P_comsol_original_sampled - P_dualn_original_sampled)
        / P_comsol_original_sampled
    )
    < 0.5
)
MAPE_P_original = mean_absolute_percentage_error(
    P_comsol_original_sampled[MAPE_P_original_bool],
    P_dualn_original_sampled[MAPE_P_original_bool],
)
MAPE_P_optimized_bool = (
    np.abs(
        (P_comsol_optimized_sampled - P_dualn_optimized_sampled)
        / P_comsol_optimized_sampled
    )
    < 0.5
)
MAPE_P_optimized = mean_absolute_percentage_error(
    P_comsol_optimized_sampled[MAPE_P_optimized_bool],
    P_dualn_optimized_sampled[MAPE_P_optimized_bool],
)
slope_P_original = calculate_slope(P_comsol_original_sampled, P_dualn_original_sampled)
slope_P_optimized = calculate_slope(
    P_comsol_optimized_sampled, P_dualn_optimized_sampled
)


print("r2_pressure_original:", r2_pressure_original)
print("r2_pressure_optimized:", r2_pressure_optimized)
print("nrmse_P_original:", nrmse_P_original)
print("nrmse_P_optimized:", nrmse_P_optimized)
print("MAPE_P_original:", MAPE_P_original)
print("MAPE_P_optimized:", MAPE_P_optimized)
print("slope_P_original:", slope_P_original)
print("slope_P_optimized:", slope_P_optimized)

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
    P_comsol_original_sampled,
    P_dualn_original_sampled,
    marker="o",
    # color="#8EBAD9",
    color="C0",
    alpha=0.5,
    label="Original DNM",
)
ax1.scatter(
    P_comsol_optimized_sampled,
    P_dualn_optimized_sampled,
    marker="o",
    color="C1",
    alpha=0.5,
    label="Calibrated DNM",
)

ax1.axline((0, 0), slope=1, color="C3", linestyle="--", label="y = x")
ax1.set_xlabel("DNS (Pa)\n(a)")
ax1.set_ylabel("DNM (Pa)")
ax1.legend(frameon=False, edgecolor="none", handletextpad=0.5, handlelength=1.5)
# 添加R²值到图例
# ax1.text(
#     0.05,
#     0.95,
#     f"R² = {r2_pressure_original:.2f}",
#     transform=ax1.transAxes,
#     fontsize=12,
#     verticalalignment="top",
# )

# ax1.text()

# ax1.text(
#     0.05,
#     0.98,
#     f"R² = {r2_pressure_optimized:.2f}",
#     transform=ax1.transAxes,
#     fontsize=12,
#     verticalalignment="top",
# )

# 添加误差直方图
# divider = make_axes_locatable(ax1)
# ax_hist1 = divider.append_axes("right", size=6.1, pad=0.8)


error_P_original = (
    P_dualn_original_sampled - P_comsol_original_sampled
) / P_comsol_original_sampled  # 百分比误差
error_P_optimized = (
    P_dualn_optimized_sampled - P_comsol_optimized_sampled
) / P_comsol_optimized_sampled  # 百分比误差


error_P_original_min, error_P_original_max = (
    np.min(error_P_original),
    np.max(error_P_original),
)
error_P_optimized_min, error_P_optimized_max = (
    np.min(error_P_optimized),
    np.max(error_P_optimized),
)
error_P_min, error_P_max = (
    min(error_P_original_min, error_P_optimized_min),
    max(error_P_original_max, error_P_optimized_max),
)

ax_hist1.hist(
    error_P_original,
    bins=bins,
    orientation="vertical",
    alpha=0.5,
    color="C0",
    weights=compute_percentage_weights(error_P_original),
    label="Original DNM",
)

print(compute_percentage_weights(error_P_original).sum())

# ax_hist1.axhline(mean_val_original, color="C0", linestyle="--", linewidth=1.5)
mean_val_original_abs = np.abs(error_P_original).mean()
std_val_original_abs = np.abs(error_P_original).std()

print(
    f"Original DNM Pressure: mean = {mean_val_original_abs:.2f}%, std = {std_val_original_abs:.2f}%, R² = {r2_pressure_original:.4f}"
)
mean_val_optimized = np.mean(error_P_optimized)
std_val_optimized = np.std(error_P_optimized)
ax_hist1.hist(
    error_P_optimized,
    bins=bins,
    orientation="vertical",
    alpha=0.5,
    color="C1",
    weights=compute_percentage_weights(error_P_optimized),
    label="Calibrated DNM",
)
ax_hist1.set_ylim(0, 0.28)
ax_hist1.legend(frameon=False, edgecolor="none", loc="upper left")

# ax_hist1.axhline(mean_val_optimized, color="C1", linestyle="--", linewidth=1.5)

mean_val_optimized_abs = np.abs(error_P_optimized).mean()
std_val_optimized_abs = np.abs(error_P_optimized).std()
print(
    f"Optimized DNM Pressure: mean = {mean_val_optimized_abs:.2f}%, std = {std_val_optimized_abs:.2f}%, R² = {r2_pressure_optimized:.4f}"
)

# ax_hist1.axvline(0, color="C3", linestyle="--", linewidth=1.5)
ax_hist1.set_xlabel("Relative error\n(b)")
ax_hist1.set_ylabel("Proportion")

# ax_hist1.set_ylim(-0.4,0.4)

# plt.tight_layout()
# plt.show()


T_comsol_original_filtered -= 293.15
T_dualn_original_filtered -= 293.15

T_comsol_optimized_filtered -= 293.15
T_dualn_optimized_filtered -= 293.15

# 随机采样
# T_index_original = np.random.choice(len(T_comsol_original_filtered), num, replace=False)
# T_index_optimized = np.random.choice(
#     len(T_comsol_optimized_filtered), num, replace=False
# )
T_comsol_original_sampled = T_comsol_original_filtered
T_comsol_optimized_sampled = T_comsol_optimized_filtered
T_dualn_original_sampled = T_dualn_original_filtered
T_dualn_optimized_sampled = T_dualn_optimized_filtered
# 计算R²值
r2_T_original = r2_score(T_comsol_original_sampled, T_dualn_original_sampled)
r2_T_optimized = r2_score(T_comsol_optimized_sampled, T_dualn_optimized_sampled)
nrmse_T_original = rmse(T_comsol_original_sampled, T_dualn_original_sampled)
nrmse_T_optimized = rmse(T_comsol_optimized_sampled, T_dualn_optimized_sampled)
MAPE_T_original_bool = (
    np.abs(
        (T_comsol_original_sampled - T_dualn_original_sampled)
        / T_comsol_original_sampled
    )
    < 0.5
)
MAPE_T_original = mean_absolute_percentage_error(
    T_comsol_original_sampled[MAPE_T_original_bool],
    T_dualn_original_sampled[MAPE_T_original_bool],
)
MAPE_T_optimized_bool = (
    np.abs(
        (T_comsol_optimized_sampled - T_dualn_optimized_sampled)
        / T_comsol_optimized_sampled
    )
    < 0.5
)
MAPE_T_optimized = mean_absolute_percentage_error(
    T_comsol_optimized_sampled[MAPE_T_optimized_bool],
    T_dualn_optimized_sampled[MAPE_T_optimized_bool],
)
slope_T_original = calculate_slope(T_comsol_original_sampled, T_dualn_original_sampled)
slope_T_optimized = calculate_slope(
    T_comsol_optimized_sampled, T_dualn_optimized_sampled
)


print("r2_T_original:", r2_T_original)
print("r2_T_optimized:", r2_T_optimized)
print("nrmse_T_original:", nrmse_T_original)
print("nrmse_T_optimized:", nrmse_T_optimized)
print("MAPE_T_original:", MAPE_T_original)
print("MAPE_T_optimized:", MAPE_T_optimized)
print("slope_T_original:", slope_T_original)
print("slope_T_optimized:", slope_T_optimized)


ax2.scatter(
    T_comsol_original_sampled,
    T_dualn_original_sampled,
    marker="o",
    color="C0",
    alpha=0.5,
    label="Original DNM",
)

ax2.scatter(
    T_comsol_optimized_sampled,
    T_dualn_optimized_sampled,
    marker="o",
    color="C1",
    alpha=0.5,
    label="Calibrated DNM",
)

# origin = (min(T_comsol_original_sampled.min(), T_dualn_original_sampled.min()),) * 2
ax2.axline((0, 0), slope=1, color="C3", linestyle="--", label="y = x")
ax2.set_xlabel("DNS (K)\n(c)")
ax2.set_ylabel("DNM (K)")
ax2.legend(frameon=False, edgecolor="none", handletextpad=0.5, handlelength=1.5)


# 添加误差直方图
# divider = make_axes_locatable(ax2)
# ax_hist2 = divider.append_axes("right", size=2, pad=0.8)

error_T_original = (
    T_dualn_original_sampled - T_comsol_original_sampled
) / T_comsol_original_sampled  # 百分比误差
error_T_optimized = (
    T_dualn_optimized_sampled - T_comsol_optimized_sampled
) / T_comsol_optimized_sampled  # 百分比误差


mean_val_original_abs = np.abs(error_T_original).mean()
std_val_original_abs = np.abs(error_T_original).std()

# ax_hist2.axhline(mean_val_original, color="C0", linestyle="--", linewidth=1.5)
print(
    f"Original DNM Temperature: mean = {mean_val_original_abs:.2f}%, std = {std_val_original_abs:.2f}%, R² = {r2_T_original:.4f}"
)


mean_val_optimized_abs = np.abs(error_T_optimized).mean()
std_val_optimized_abs = np.abs(error_T_optimized).std()
print(
    f"Optimized DNM Temperature: mean = {mean_val_optimized_abs:.2f}%, std = {std_val_optimized_abs:.2f}%, R² = {r2_T_optimized:.4f}"
)

error_T_original_min, error_T_original_max = (
    np.min(error_T_original),
    np.max(error_T_original),
)
error_T_optimized_min, error_T_optimized_max = (
    np.min(error_T_optimized),
    np.max(error_T_optimized),
)
error_T_min, error_T_max = (
    min(error_T_original_min, error_T_optimized_min),
    max(error_T_original_max, error_T_optimized_max),
)
# error_T_original = error_T_original[np.abs(error_T_original) < 0.8]

bins = np.linspace(-0.8, 0.8, num_bins)
ax_hist2.hist(
    error_T_original,
    bins=bins,
    orientation="vertical",
    alpha=0.5,
    color="C0",
    # density=True,
    weights=compute_percentage_weights(error_T_original),
    label="Original DNM",
)
# error_T_optimized = error_T_optimized[np.abs(error_T_optimized) < 0.8]

ax_hist2.hist(
    error_T_optimized,
    bins=bins,
    orientation="vertical",
    alpha=0.5,
    color="C1",
    # density=True,
    weights=compute_percentage_weights(error_T_optimized),
    label="Calibrated DNM",
)

ax_hist2.legend(frameon=False, edgecolor="none", loc="upper left")
ax_hist2.set_ylim(0, 0.6)
mean_P_original = np.mean(error_P_original)
std_P_original = np.std(error_P_original)
mean_P_optimized = np.mean(error_P_optimized)
std_P_optimized = np.std(error_P_optimized)


# error_min = min(error_T_min, error_P_min)
# error_max = max(error_T_max, error_P_max)
# bins_norm = np.linspace(error_min, error_max, 300)
# y_norm = norm.pdf(bins_norm, mean_P_original, std_P_original)
# ax_hist1.plot(bins_norm, y_norm, color="C0", linestyle="-", linewidth=2)
# y_norm = norm.pdf(bins_norm, mean_P_optimized, std_P_optimized)
# ax_hist1.plot(bins_norm, y_norm, color="C1", linestyle="-", linewidth=2)


# mean_T_original = np.mean(error_T_original)
# std_T_original = np.std(error_T_original)
# mean_T_optimized = np.mean(error_T_optimized)
# std_T_optimized = np.std(error_T_optimized)
# y_norm = norm.pdf(bins_norm, mean_T_original, std_T_original)
# ax_hist2.plot(bins_norm, y_norm, color="C0", linestyle="-", linewidth=2)
# y_norm = norm.pdf(bins_norm, mean_T_optimized, std_T_optimized)
# ax_hist2.plot(bins_norm, y_norm, color="C1", linestyle="-", linewidth=2)


ah1_xlim = ax_hist1.get_xlim()
ah2_xlim = ax_hist2.get_xlim()
# ah_xlim_min = min(ah1_xlim[0], ah2_xlim[0])
# ah_xlim_max = max(ah1_xlim[1], ah2_xlim[1])
# ax_hist1.set_xlim(ah_xlim_min, ah_xlim_max)
# ax_hist2.set_xlim(ah_xlim_min, ah_xlim_max)

# ax_hist1.set_ylim(ax_hist2.get_ylim())
# # ax_hist2.axvline(0, color="C3", linestyle="--", linewidth=1.5)
ax_hist2.set_xlabel("Relative error\n(d)")
ax_hist2.set_ylabel("Proportion")


ah1_xlim_abs_max = max(np.abs(ah1_xlim[0]), np.abs(ah1_xlim[1]))
ah2_xlim_abs_max = max(np.abs(ah2_xlim[0]), np.abs(ah2_xlim[1]))


ax_hist1.set_xlim(-0.5, 0.5)
ax_hist2.set_xlim(-0.5, 0.5)
xticks = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
ax_hist1.set_xticks(xticks)
ax_hist2.set_xticks(xticks)
ax1_xylim_min, ax1_xylim_max = (
    min(ax1.get_xlim()[1], ax1.get_ylim()[1]),
    max(ax1.get_xlim()[1], ax1.get_ylim()[1]),
)
ax2_xylim_min, ax2_xylim_max = (
    min(ax2.get_xlim()[1], ax2.get_ylim()[1]),
    max(ax2.get_xlim()[1], ax2.get_ylim()[1]),
)
ax1.set_xlim(0, ax1_xylim_max)
ax1.set_ylim(0, ax1_xylim_max)
ax2.set_xlim(0, ax2_xylim_max)
ax2.set_ylim(0, ax2_xylim_max)
ax_hist1.axvline(x=0.0, color="C3", linestyle="--", linewidth=1.5)
ax_hist2.axvline(x=0.0, color="C3", linestyle="--", linewidth=1.5)


plt.tight_layout()
plt.savefig(_Path_fig / "big_pore_distribution_N10.000.png")
plt.show()
