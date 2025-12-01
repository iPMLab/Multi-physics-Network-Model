import numpy as np
import sys
import copy
from itertools import zip_longest
import pandas as pd
import addict

sys.path.append("../../../")
from mpnm_new import topotool, algorithm, network as net, util
from Papers.P1.Code.COMSOL.comsol_params import (
    PARAMS_N2_500,
    PARAMS_N3_455,
    PARAMS_N4_353,
    PARAMS_N4_689,
    PARAMS_N4_869,
    PARAMS_N5_000,
    PARAMS_N10_000,
    PARAMS_N5_000_marching_cube,
    PARAMS_N5_000_voxel,
    PARAMS_N5_000_constrained_smooth,
)
import matplotlib
from Common_Vars import _Path_fig as Path_fig, plt, fontsize, figsize_x, figsize_y


Res = [0.001, 0.005, 0.02, 0.1, 1]
hf = [
    10000,
] * len(Res)

prefixes = ["N2.500", "N3.455", "N4.355", "N4.869", "N5.000"]
sample_indices = {
    "N2.500": 2,
    "N3.455": 4,
    "N4.355": 3,
    "N4.689": 3,
    "N4.869": 0,
    "N5.000": 0,
}

prefix = "N5.000"


ed = addict.Dict()
sample_index = sample_indices[prefix]

ed.P.original = pd.read_csv(f"./plot_relative_error_P_{prefix}_original.csv")
ed.P.optimized = pd.read_csv(f"./plot_relative_error_P_{prefix}_optimized.csv")
ed.hf.original = pd.read_csv(f"./plot_relative_error_hf_{prefix}_original.csv")
ed.hf.optimized = pd.read_csv(f"./plot_relative_error_hf_{prefix}_optimized.csv")
ed.P.original.sample = ed.P.original[f"Sample_{sample_index}_Relative_Error"].abs()
ed.P.optimized.sample = ed.P.optimized[f"Sample_{sample_index}_Relative_Error"].abs()
ed.hf.original.sample = ed.hf.original[f"Sample_{sample_index}_Relative_Error"].abs()
ed.hf.optimized.sample = ed.hf.optimized[f"Sample_{sample_index}_Relative_Error"].abs()

print("P original sample mean", ed.P.original.sample.mean())
print("P optimized sample mean", ed.P.optimized.sample.mean())
print("hf original sample mean", ed.hf.original.sample.mean())
print("hf optimized sample mean", ed.hf.optimized.sample.mean())


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize_x * 2, figsize_y))

bar_width = 0.35
x_indexes = np.arange(len(Res))
plt.yticks(np.arange(0, 0.021, 0.005))
bars_P_orig = ax1.bar(
    x_indexes - bar_width / 2,
    ed.P.original.sample,
    bar_width,
    label="Pressure drop",
    color="C0",
    alpha=1,
)
bars_P_opt = ax1.bar(
    x_indexes + bar_width / 2,
    ed.hf.original.sample,
    bar_width,
    label="Heat transfer",
    color="C1",
    alpha=1,
)

bars_hf_orig = ax2.bar(
    x_indexes - bar_width / 2,
    ed.P.optimized.sample,
    bar_width,
    label="Pressure drop",
    color="C0",
    alpha=1,
)
bars_hf_opt = ax2.bar(
    x_indexes + bar_width / 2,
    ed.hf.optimized.sample,
    bar_width,
    label="Heat transfer",
    color="C1",
    alpha=1,
)


# 设置图表标题和标签
ax1.set_title("Original DNM", fontsize=fontsize)
ax1.set_xlabel("Re\n(a)")
ax1.set_ylabel("Discrepancy")
ax1.set_xticks(x_indexes)
ax1.set_xticklabels(Res)
ax1.legend(frameon=False, loc="upper left")
# ax1.grid(True, alpha=0.3, axis="y")
ax2.set_title("Calibrated DNM", fontsize=fontsize)
ax2.set_xlabel("Re\n(b)")
ax2.set_ylabel("Discrepancy")
ax2.set_xticks(x_indexes)
ax2.set_xticklabels(Res)
ax2.legend(frameon=False, loc="upper left")
# ax2.grid(True, alpha=0.3, axis="y")


# 调整y轴范围以确保所有标签可见
# def adjust_y_lim(ax, values_orig, values_opt):
#     max_val = max(max(values_orig), max(values_opt))
#     ax.set_ylim(0, max_val * 1.25)


def adjust_y_lim(ax_0, ax_1, values_orig_0, values_opt_0, values_orig_1, values_opt_1):
    max_val = max(
        max(values_orig_0), max(values_opt_0), max(values_orig_1), max(values_opt_1)
    )
    ax_0.set_ylim(0, max_val * 1.25)
    ax_1.set_ylim(0, max_val * 1.25)


# adjust_y_lim(ax1, ed.P.original.sample, ed.P.optimized.sample)
# adjust_y_lim(ax2, ed.hf.original.sample, ed.hf.optimized.sample)

ax1.set_ylim(0, max(ed.P.original.sample) * 1.25)
ax2.set_ylim(0, 0.02)

# adjust_y_lim(
#     ax1,
#     ax2,
#     ed.P.original.sample,
#     ed.P.optimized.sample,
#     ed.hf.original.sample,
#     ed.hf.optimized.sample,
# )

# show 1% xline on y coordination left
# ax1.axhline(y=1, color="red", linestyle="--", linewidth=1)
# ax1.text(-0.83, 1, " 1", color="red", va="center")
# ax2.axhline(y=1, color="red", linestyle="--", linewidth=1)
# ax2.text(-0.83, 1, " 1", color="red", va="center")
# 为(a)添加注释

# 调整布局
plt.tight_layout()
plt.savefig(Path_fig / f"get_data_single_{prefix}.png")
# 显示图表
plt.show()
