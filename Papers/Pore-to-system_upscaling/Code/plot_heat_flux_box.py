import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import ticker

matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["font.family"] = "Arial"


relative_error_voxel = pd.read_csv("plot_relative_error_voxel.csv")
relative_error_marching_cubes = pd.read_csv("plot_relative_error_marching_cubes.csv")
relative_error_minkowski = pd.read_csv("plot_relative_error_minkowski.csv")
print(relative_error_voxel)


Res = ["0.001", "0.005", "0.02", "0.1", "1"]


data_voxel = relative_error_voxel.to_numpy()[:, 1:] * 100
data_marching_cubes = relative_error_marching_cubes.to_numpy()[:, 1:] * 100
data_minkowski = relative_error_minkowski.to_numpy()[:, 1:] * 100


fig, ax = plt.subplots()

# 设置每组箱线图的位置（避免重叠）
positions = np.arange(len(Res)) * 3  # 例如 Res=[1,2,3] 则位置为 [0,3,6]

# 绘制箱线图时指定位置和宽度
width = 0.6
bplot1 = ax.boxplot(
    data_voxel.T,
    positions=positions - width,
    widths=width,
    patch_artist=True,
    boxprops={"facecolor": "C2", "alpha": 0.5},
    medianprops={"color": "red"},
    showfliers=False,
)
bplot2 = ax.boxplot(
    data_marching_cubes.T,
    positions=positions,
    widths=width,
    patch_artist=True,
    showfliers=False,
    boxprops={"facecolor": "C1", "alpha": 0.5},
    medianprops={"color": "red"},
)
bplot3 = ax.boxplot(
    data_minkowski.T,
    positions=positions + width,
    widths=width,
    patch_artist=True,
    showfliers=False,
    boxprops={"facecolor": "C0", "alpha": 0.5},
    medianprops={"color": "red"},
)


# 设置x轴标签和标题
ax.set_xticks(positions)
ax.set_xticklabels(Res)
ax.set_xlabel("Re")
ax.set_ylabel("Relative Error (%)")

# 添加图例
ax.legend(
    [bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0]],
    ["Voxel Counting", "Marching Cubes", "Surface Density"],
    frameon=False,
)

plt.tight_layout()
plt.show()

voxel_means = np.mean(data_voxel, axis=1)
marching_means = np.mean(data_marching_cubes, axis=1)
minkowski_means = np.mean(data_minkowski, axis=1)
print("Voxel Means:", voxel_means)
print("Marching Means:", marching_means)
print("Minkowski Means:", minkowski_means)

voxel_mean = np.mean(voxel_means)
marching_mean = np.mean(marching_means)
minkowski_mean = np.mean(minkowski_means)
print("Voxel Mean:", voxel_mean)
print("Marching Mean:", marching_mean)
print("Minkowski Mean:", minkowski_mean)