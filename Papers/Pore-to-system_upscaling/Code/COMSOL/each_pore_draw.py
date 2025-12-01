import sys

sys.path.append(r"../../")
import numpy as np
import pickle
import multiprocessing
from tqdm import trange, tqdm
from mpnm_new import network

from pathlib import Path
import itertools
from joblib import Parallel, delayed
from Papers.P1.Code.COMSOL.comsol_params import ComsolParams_N10_000_sample0 as PARAM, extract_u_hf
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# matplotlib.use("Agg")
PARAM = PARAM()
Path_results = PARAM.Path_results
Path_comsol = PARAM.Path_comsol


def show_single_X(dualn_i, Comsol_Data_i, Path_Pore_Distribution, i, field):
    Comsol_Data_i_mean = np.mean(Comsol_Data_i)
    scale = np.abs(dualn_i / Comsol_Data_i_mean)
    fig, ax = plt.subplots()
    ax.hist(Comsol_Data_i, bins=100, density=True, alpha=0.5, label="COMSOL")
    ax.axline((dualn_i, 0), slope=np.inf, color="r", linestyle="--", label="MPNM")
    ax.axline(
        (Comsol_Data_i_mean, 0),
        slope=np.inf,
        color="b",
        linestyle="--",
        label="Mean",
    )
    ax.annotate(f"{scale:.2f}Mean", (dualn_i, 0.05), color="r")
    ax.legend(loc="upper right")
    ax.set_title(f"{field} distribution of pore {i}")
    ax.set_xlabel(f"{field}")
    ax.set_ylabel("Density")
    fig.savefig(Path_Pore_Distribution / f"{field}_distribution_pore_{i}.png")
    # plt.show()
    # ax.close()
    plt.close(fig)


def plot_percentile_distribution(dualn, Comsol_Data, field, save_path):
    """
    绘制dualn值在对应Comsol区域中的百分位分布

    参数:
        dualn_values: 所有孔隙的MPNM值数组 (形状: [n_pores])
        Comsol_Data_values: 对应的Comsol数据数组 (形状: [n_pores, n_samples_per_pore])
        field: 字段名称 ('P','U'或'T')
        save_path: 图片保存路径
    """
    # 计算每个dualn值在其对应Comsol数据中的百分位
    percentiles = []
    for i in range(len(dualn["pore.all"])):
        percentiles.append(
            stats.percentileofscore(Comsol_Data[i][field], dualn[f"pore.{field}"][i])
        )

    # 转换为numpy数组并移除无效值
    percentiles = np.array(percentiles)[dualn["pore.void"]]
    # print(percentiles)
    len_origin = len(percentiles)
    percentiles = percentiles[(5 <= percentiles) & (percentiles <= 95)]
    loss_rate = (len_origin - len(percentiles)) / len_origin * 100
    print(loss_rate)
    if len(percentiles) == 0:
        print(f"No valid data for {field}")
        return

    # 创建图形
    # plt.figure(figsize=(10, 6))

    # 绘制百分位分布直方图

    # 配色方案 (现代科技风)
    COLORS = {
        "background": "#F5F7FA",  # 浅灰背景
        "grid": "#E0E6ED",  # 浅网格线
        "hist": "#B3E0FF",  # 深蓝柱状图
        "fit_line": "#2A5A7B",  #
        "mean_line": "#0D47A1",  #
        "text": "#2F3E4E",  # 深灰文字
        "accent": "#6A8DBA",  # 辅助色
    }

    # 尝试拟合正态分布

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["background"])

    # hist, edges = np.histogram(percentiles, bins=30, density=True)
    # x_fit = np.linspace(0, 100, 500)
    # # pdf = 1/(21.1*np.sqrt(2*np.pi)) * np.exp(-(x_fit-49.6)​**​2/(2 * 21.1​**​2))

    # # 创建自定义渐变colormap
    # colors = ["#4A90E2", "#88B8F4"]  # 科技蓝 → 浅蓝
    # cmap = LinearSegmentedColormap.from_list("blue_gradient", colors, N=256)

    # # 绘图设置

    # # 柱状图渐变效果实现
    # heights = hist
    # left = edges[:-1]
    # for l, r, h in zip(left, edges[1:], heights):
    #     grad = np.linspace(0, 1, 100).reshape(100, 1)
    #     ax.imshow(grad, cmap=cmap, aspect='auto',
    #             extent=[l, r, 0, h], origin='lower',
    #             alpha=0.85, vmin=0.3, vmax=1.0)

    ax.grid(False)
    n, bins, patches = ax.hist(
        percentiles,
        bins=30,
        density=True,
        alpha=0.7,
        color=COLORS["hist"],
        edgecolor="#4682B4",
        linewidth=0.7,
    )
    mu, std = stats.norm.fit(percentiles)
    x = np.linspace(0, 100, 300)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(
        x,
        p,
        color=COLORS["fit_line"],
        linestyle="--",
        linewidth=1,
        label=f"Normal Fit\nμ={mu:.1f}, σ={std:.1f}",
    )

    # 标注均值线
    mean_val = np.mean(percentiles)
    ax.axvline(
        mean_val,
        color=COLORS["mean_line"],
        linestyle="-",
        linewidth=2,
        label=f"Mean = {mean_val:.1f}",
    )

    # 图形标注
    ax.set_title(f"{field} - Percentile Distribution of MPNM in COMSOL Data", pad=20)
    ax.set_xlabel("Percentile Position in Local COMSOL Data [%]")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right")
    # plt.grid(False, alpha=0.3)

    # 保存图形

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np


def plot_relative_error_distribution(dualn, Comsol_Data, field, save_path):
    """
    绘制dualn值与对应Comsol区域均值的相对误差分布（横坐标：万分比‱）

    参数:
        dualn_values: 所有孔隙的MPNM值数组 (形状: [n_pores])
        Comsol_Data_values: 对应的Comsol数据数组 (形状: [n_pores, n_samples_per_pore])
        field: 字段名称 ('P','U'或'T')
        save_path: 图片保存路径
    """
    # 计算每个Comsol数据集的均值
    comsol_means = []
    for i in range(len(dualn["pore.all"])):
        comsol_means.append(np.mean(Comsol_Data[i][field]))

    # 计算相对误差 (dualn - mean)/mean
    relative_errors = (dualn[f"pore.{field}"] - comsol_means) / comsol_means

    # 转换为numpy数组并移除无效值
    relative_errors = np.array(relative_errors)
    len_origin = len(relative_errors)

    # 过滤掉极端值（保留-1到1之间的值，即±100%误差）
    relative_errors = relative_errors[(-1 <= relative_errors) & (relative_errors <= 1)]
    loss_rate = (len_origin - len(relative_errors)) / len_origin * 100
    print(f"Filtered out {loss_rate:.1f}% of data points")

    if len(relative_errors) == 0:
        print(f"No valid data for {field}")
        return

    # 配色方案
    COLORS = {
        "background": "#F5F7FA",
        "grid": "#E0E6ED",
        "hist": "#B3E0FF",
        "fit_line": "#2A5A7B",
        "mean_line": "#0D47A1",
        "text": "#2F3E4E",
        "accent": "#6A8DBA",
    }

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["background"])

    # 绘制相对误差分布直方图（横坐标：万分比‱）
    n, bins, patches = ax.hist(
        relative_errors,  # 转换为万分比‱
        bins=30,
        density=True,
        alpha=0.7,
        color=COLORS["hist"],
        edgecolor="#4682B4",
        linewidth=0.7,
    )
    
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # 拟合正态分布
    mu, sigma = norm.fit(relative_errors)  # 注意：这里用万分比单位拟合

    # 生成拟合曲线
    x = np.linspace(min(relative_errors), max(relative_errors), 100)
    y = norm.pdf(x, mu, sigma)

    # 绘制拟合曲线
    ax.plot(
        x,
        y,
        color=COLORS["fit_line"],
        linestyle="--",
        linewidth=1,
        label=f"Normal Fit: μ={mu:.2e}, σ={sigma:.2e}",
    )

    # 标注均值线（万分比‱）
    mean_val = np.mean(relative_errors)
    ax.axvline(
        mean_val,
        color=COLORS["mean_line"],
        linestyle="-",
        linewidth=2,
        label=f"Mean = {mean_val:.2e}",
    )

    # 标注零误差线
    ax.axvline(
        0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Zero Error"
    )

    # 图形标注
    ax.set_title(f"{field} - Relative Error Distribution (MPNM vs COMSOL Mean)", pad=20)
    ax.set_xlabel(
        "Relative Error"
    )
    ax.set_ylabel("Probability Density")
    # ax.set_xlim(-10000, 10000)  # 横坐标范围：±10000‱（即±100%）

    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, linestyle="--", color=COLORS["grid"])

    # 保存图形
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


Res = PARAM.Res

_Re = 0.001
_hf = 10000
Path_Pore_Distribution = Path(Path_results / "Pore_Distribution")
Path_Pore_Distribution.mkdir(exist_ok=True)

key = f"_Re{_Re:.5f}_hf{_hf}"
Comsol_Data = pickle.load(
    open(
        Path_comsol / f"Poredata{key}.pkl",
        "rb",
    )
)
dualn = network.vtk2network(Path_results / f"dualn{key}.vtp")
dualn["pore.P"][dualn["pore.void"]] -= 1000

fig_args = itertools.product(range(dualn["pore.all"].size), ["p", "spf.U", "T"])
# 设置最大并发进程数

# pbar = tqdm()
# ,callback=lambda x: pbar.update(1)
# 使用 pool.apply_async 来异步执行任务
fig_args = list(fig_args)

# for i, field in tqdm(fig_args):
# show_single_X(dualn, Comsol_Data, Path_Pore_Distribution, i, field)

# max_processes = 60  # 你可以根据需要调整这个值
# Parallel(n_jobs=max_processes, backend="loky")(
#     delayed(show_single_X)(
#         dualn[f"pore.{field}"][i],
#         Comsol_Data[i][field],
#         Path_Pore_Distribution,
#         i,
#         field,
#     )
#     for i, field in tqdm(fig_args)
# )


plot_percentile_distribution(
    dualn=dualn,
    Comsol_Data=Comsol_Data,
    field="T",
    save_path=Path_results / "P_percentile_distribution.png",
)


plot_relative_error_distribution(
    dualn=dualn,
    Comsol_Data=Comsol_Data,
    field="T",
    save_path=Path_results / "T_relative_error_distribution.png",
)
