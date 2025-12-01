import numpy as np
import sys
import pandas as pd

np.set_printoptions(legacy="1.13")
sys.path.append("../../../")
from mpnm_new import topotool, algorithm, network as net, util
from tools.COMSOL.comsol_params import (
    PARAMS_N2_500,
    PARAMS_N3_455,
    PARAMS_N4_353,
    PARAMS_N4_689,
    PARAMS_N4_869,
    PARAMS_N5_000,
    PARAMS_N10_000,
    PARAMS_N5_000_marching_cube,
    PARAMS_N5_000_voxel,
)
import matplotlib.pyplot as plt
import copy

PARAMS = PARAMS_N5_000

Res = [0.001, 0.005, 0.02, 0.1, 1]
hf = [
    10000,
] * len(Res)
mean_relative_errors = []


def r2_score(y_true, y_pred):
    """
    Calculate the R^2 score.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: R^2 score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    return r2


def draw_comsol_pnm_graph(
    x,
    y,
    title,
    Re,
    PARAM,
    save=True,
    xlabel="",
    ylabel="",
    percentile=95,
    show=True,
    color_Re=None,
):
    color_points = color_Re if color_Re is not None else "C0"
    color_line = color_Re if color_Re is not None else "r"
    abs_error = np.abs((y - x))
    abs_error_percentile = np.percentile(abs_error, percentile)
    valid_bool = np.ones_like(x, dtype=bool)
    valid_bool = abs_error < abs_error_percentile
    valid_bool[x.argmax()] = False
    valid_bool[x.argmin()] = False
    valid_bool[y.argmax()] = False
    valid_bool[y.argmin()] = False

    x = x[valid_bool]
    y = y[valid_bool]
    plt.title(title)
    plt.scatter(
        x=x,
        y=y,
        c=color_points,
        alpha=0.75,
    )

    origin = (min(x.min(), y.min()),) * 2
    plt.axline(origin, slope=1, color=color_Re, label="y = x")
    r2 = r2_score(x, y)

    plt.text(
        0.02,
        0.98,
        f"Re = {Re}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

    plt.text(
        0.02,
        0.94,
        f"$R^2$ = {r2:.2f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    plt.ticklabel_format(style="sci", scilimits=(-1, 3), axis="x")
    plt.ticklabel_format(style="sci", scilimits=(-1, 3), axis="y")
    # plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    # plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # if save:
    #     plt.savefig(Path_figs / f"{PARAM.prefix}_Re{Re}_{title}.png")
    if show:
        plt.legend()
        plt.show()


relative_error_dict = {Re_i: [] for Re_i in Res}
mean_relative_error_dict = copy.deepcopy(relative_error_dict)
pore_T_dict = copy.deepcopy(relative_error_dict)
pore_T_comsol_dict = copy.deepcopy(relative_error_dict)
pore_P_dict = copy.deepcopy(relative_error_dict)
pore_P_comsol_dict = copy.deepcopy(relative_error_dict)
for i in range(len(Res)):
    Re_i = Res[i]
    hf_i = hf[i]
    for j, PARAM in enumerate(PARAMS):
        dualn = net.vtk2network(
            PARAM.Path_results / f"dualn_Re{Re_i:.5f}_hf{hf_i:.0f}.vtp"
        )
        pore_T_dict[Re_i] = np.concatenate((pore_T_dict[Re_i], dualn["pore.T"]))
        pore_T_comsol_dict[Re_i] = np.concatenate(
            (pore_T_comsol_dict[Re_i], dualn["pore.T_ave_comsol"])
        )

        dualn_pore_P = dualn["pore.P"]
        dualn_pore_P[dualn["pore.solid"]] = 0
        pore_P_dict[Re_i] = np.concatenate((pore_P_dict[Re_i], dualn_pore_P))
        pore_P_comsol_dict[Re_i] = np.concatenate(
            (pore_P_comsol_dict[Re_i], dualn["pore.p_ave_comsol"])
        )

        P_pn = dualn["pore.P_pn"][0]
        P_comsol = PARAM.pressure_out[i]

        err_P = (P_pn - P_comsol) / P_comsol
        relative_error_dict[Re_i].append(err_P)

    mean_error = np.median(relative_error_dict[Re_i])
    print(f"Median index {np.argwhere(relative_error_dict[Re_i]==mean_error)}")
    mean_relative_error_dict[Re_i].append(mean_error)

    print(f"Re={Re_i:.5f}, Mean Relative Error={mean_error:.2%}")


print(
    "average relative error:",
    np.mean(np.abs([mean_relative_error_dict[Re_i] for Re_i in Res])) * 100,
    "%",
)

percentile = 95
color_Res = {0.001: "C0", 0.005: "C1", 0.02: "C2", 0.1: "C3", 1: "C4"}

labels = [f"{Re_i}" for Re_i in Res]
print(relative_error_dict)
plt.figure(figsize=(10, 6))
plt.boxplot(
    list(relative_error_dict.values()),
    tick_labels=labels,
    patch_artist=True,
    boxprops={"facecolor": "skyblue", "linewidth": 1.5},
    medianprops={"color": "red", "linewidth": 2},
    whiskerprops={"linewidth": 1.5},
    capprops={"linewidth": 1.5},
    flierprops={
        "marker": "o",
        "markersize": 5,
        "markerfacecolor": "gray",
        "markeredgecolor": "none",
    },
    showfliers=False,
)

# Customize the plot
# plt.title("Distribution of Relative Errors for Outlet Heat Flux", fontsize=14)
plt.xlabel("Re", fontsize=14)
plt.ylabel("Relative Error", fontsize=14)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


Relative_Errors = {}
for i in range(len(PARAMS)):
    Relative_Errors[f"Sample_{i}_Relative_Error"] = []
    for j in range(len(Res)):
        Relative_Errors[f"Sample_{i}_Relative_Error"].append(
            relative_error_dict[Res[j]][i]
        )
plot_data = pd.DataFrame({"Re": Res, **Relative_Errors})


plot_data.to_csv(
    f"plot_relative_error_P_{(PARAMS[0].prefix)[1:-8]}_optimized.csv", index=False
)


# plot_data.to_csv(f"plot_relative_error_P_N10.000_N4.869.csv", index=False)


# 绘制图形
# plt.figure(figsize=(8, 6))
# plt.semilogx(Res, mean_relative_errors, "C0o-", linewidth=2, markersize=8)
# # plt.plot(Res, mean_relative_errors, "C0", markersize=8)
# plt.xlabel("Reynolds number (Re)", fontsize=12)
# plt.ylabel("Mean relative error", fontsize=12)
# plt.title("Mean relative error of outlet heat flux", fontsize=14)
# plt.grid(True, linestyle="--", alpha=0.6)

# # 如果Re的范围很大，可以考虑使用对数坐标
# # plt.xscale('log')
# # plt.yscale('log')

# # plt.tight_layout()
# plt.show()
