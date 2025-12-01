import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
from sklearn.metrics import r2_score

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["font.family"] = "Arial"

# 读取数据 - 假设DNM有两种版本的结果
pore_data_original = pd.read_csv("plot_pore_N10.000_original.csv")
pore_data_optimized = pd.read_csv("plot_pore_N10.000_optimized.csv")

print(pore_data_original)

# COMSOL 数据是相同的，只从其中一个文件读取
P_comsol = pore_data_original["pore_P_comsol"].dropna().to_numpy()
P_dualn_original = pore_data_original["pore_P_dualn"].dropna().to_numpy()
P_dualn_optimized = pore_data_optimized["pore_P_dualn"].dropna().to_numpy()

# 温度数据
T_void_comsol = pore_data_original["pore_T_void_comsol"].dropna().to_numpy()
T_solid_comsol = pore_data_original["pore_T_solid_comsol"].dropna().to_numpy()
T_void_dualn_original = pore_data_original["pore_T_void_dualn"].dropna().to_numpy()
T_solid_dualn_original = pore_data_original["pore_T_solid_dualn"].dropna().to_numpy()
T_void_dualn_optimized = pore_data_optimized["pore_T_void_dualn"].dropna().to_numpy()
T_solid_dualn_optimized = pore_data_optimized["pore_T_solid_dualn"].dropna().to_numpy()

T_comsol = np.concatenate((T_void_comsol, T_solid_comsol))
T_dualn_original = np.concatenate((T_void_dualn_original, T_solid_dualn_original))
T_dualn_optimized = np.concatenate((T_void_dualn_optimized, T_solid_dualn_optimized))


def filter_data(x, y, relative_error=True, border=True, min_max=True):
    mask = np.ones_like(x, dtype=bool)
    if relative_error:
        relative_error = np.abs((x - y) / x)
        relative_error[np.isnan(relative_error)] = 0
        upper = np.percentile(relative_error, 98)
        mask &= relative_error < upper
    if border:
        lower, upper = np.percentile(x, [1, 99])
        mask &= (x > lower) & (x < upper)

    return x[mask], y[mask]


# 过滤压力数据
P_comsol_original, P_dualn_original = filter_data(P_comsol, P_dualn_original)
P_comsol_optimized, P_dualn_optimized = filter_data(P_comsol, P_dualn_optimized)

np.random.seed(1)
num = 2000

# 随机采样
P_original_index = np.random.choice(len(P_comsol_original), num, replace=False)
P_comsol_original = P_comsol_original[P_original_index]
P_dualn_original = P_dualn_original[P_original_index]

P_optimized_index = np.random.choice(len(P_comsol_optimized), num, replace=False)
P_comsol_optimized = P_comsol_optimized[P_optimized_index]
P_dualn_optimized = P_dualn_optimized[P_optimized_index]

fig, ax = plt.subplots(figsize=(8, 6))

# 计算R²分数
r2_original = r2_score(P_comsol_original, P_dualn_original)
r2_optimized = r2_score(P_comsol_optimized, P_dualn_optimized)

# 主散点图
methods = {
    f"Original DNM (R²={r2_original:.4f})": (
        "o",
        "C0",
        P_comsol_original,
        P_dualn_original,
    ),
    f"Optimized DNM (R²={r2_optimized:.4f})": (
        "D",
        "C1",
        P_comsol_optimized,
        P_dualn_optimized,
    ),
}

for name, (marker, color, x, y) in methods.items():
    ax.scatter(
        x,
        y,
        marker=marker,
        color=color,
        alpha=0.5,
        label=name,
    )

ax.axline((0, 0), slope=1, color="red", linestyle="--", label="y = x")
ax.set_xlabel("DNS (Pa)")
ax.set_ylabel("DNM (Pa)")
ax.legend(frameon=False)

# 添加误差直方图
divider = make_axes_locatable(ax)
ax_hist = divider.append_axes("right", size=2.0, pad=0.7)

for name, (_, color, x, y) in methods.items():
    error = (y - x) / x * 100  # 百分比误差
    error = error[(-80 < error)]
    ax_hist.hist(
        error,
        bins=50,
        orientation="horizontal",
        alpha=0.5,
        color=color,
        density=True,
    )
    mean_val = np.mean(error)
    std_val = np.std(error)
    bins_norm = np.linspace(error.min(), error.max(), 300)
    y_norm = norm.pdf(bins_norm, mean_val, std_val)
    ax_hist.plot(y_norm, bins_norm, color=color, linestyle="-", linewidth=2)
    ax_hist.axhline(mean_val, color=color, linestyle="--", linewidth=1.5)
    print(f"{name}: mean = {mean_val:.2f}%, std = {std_val:.2f}%")

ax_hist.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax_hist.set_xlabel("Error Distribution Density")
ax_hist.set_ylabel("Relative Error (%)")

plt.tight_layout()
plt.show()

# 温度数据处理
T_comsol_original, T_dualn_original = filter_data(T_comsol, T_dualn_original)
T_comsol_optimized, T_dualn_optimized = filter_data(T_comsol, T_dualn_optimized)

T_comsol_original -= 293.15
T_dualn_original -= 293.15
T_comsol_optimized -= 293.15
T_dualn_optimized -= 293.15

# 随机采样
T_original_index = np.random.choice(len(T_comsol_original), num, replace=False)
T_comsol_original = T_comsol_original[T_original_index]
T_dualn_original = T_dualn_original[T_original_index]

T_optimized_index = np.random.choice(len(T_comsol_optimized), num, replace=False)
T_comsol_optimized = T_comsol_optimized[T_optimized_index]
T_dualn_optimized = T_dualn_optimized[T_optimized_index]

fig, ax = plt.subplots(figsize=(8, 6))

# 计算R²分数
r2_original_temp = r2_score(T_comsol_original, T_dualn_original)
r2_optimized_temp = r2_score(T_comsol_optimized, T_dualn_optimized)

# 主散点图
methods = {
    f"Original DNM (R²={r2_original_temp:.4f})": (
        "o",
        "C0",
        T_comsol_original,
        T_dualn_original,
    ),
    f"Optimized DNM (R²={r2_optimized_temp:.4f})": (
        "D",
        "C1",
        T_comsol_optimized,
        T_dualn_optimized,
    ),
}

for name, (marker, color, x, y) in methods.items():
    ax.scatter(
        x,
        y,
        marker=marker,
        color=color,
        alpha=0.5,
        label=name,
    )

origin = (min(T_comsol_original.min(), T_dualn_original.min()),) * 2
ax.axline(origin, slope=1, color="red", linestyle="--", label="y = x")
ax.set_xlabel("DNS (K)")
ax.set_ylabel("DNM (K)")
ax.legend(frameon=False)

# 添加误差直方图
divider = make_axes_locatable(ax)
ax_hist = divider.append_axes("right", size=2.0, pad=0.7)

for name, (_, color, x, y) in methods.items():
    error = (y - x) / x * 100  # 百分比误差
    error = error[(-60 < error) & (error < 60)]
    ax_hist.hist(
        error,
        bins=50,
        orientation="horizontal",
        alpha=0.5,
        color=color,
        density=True,
    )

    mean_val = np.mean(error)
    std_val = np.std(error)
    bins_norm = np.linspace(error.min(), error.max(), 300)
    y_norm = norm.pdf(bins_norm, mean_val, std_val)
    ax_hist.plot(y_norm, bins_norm, color=color, linestyle="-", linewidth=2)
    ax_hist.axhline(mean_val, color=color, linestyle="--", linewidth=1.5)
    print(f"{name}: mean = {mean_val:.2f}%, std = {std_val:.2f}%")

ax_hist.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax_hist.set_xlabel("Error Distribution Density")
ax_hist.set_ylabel("Relative Error (%)")

plt.tight_layout()
plt.show()
