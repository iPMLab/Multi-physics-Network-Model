import numpy as np
import sys
import copy
from itertools import zip_longest

sys.path.append("../../../")
from mpnm_new import topotool, algorithm, network as net, util
from tools.COMSOL.comsol_params import (
    PARAMS_N4_353,
    PARAMS_N4_869,
    PARAMS_N5_000,
    PARAMS_N10_000,
    PARAMS_N5_000_voxel,
    PARAMS_N5_000_constrained_smooth,
    PARAMS_N5_000_marching_cube,
)
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.family"] = "Arial"
PARAMS = PARAMS_N5_000

Res = [0.001, 0.005, 0.02, 0.1, 1]
# Res = [0.001]
hf = [
    10000,
] * len(Res)


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


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def clip_central_95(data):
    """保留中间95%数据，去除两端2.5%的极端值"""
    lower, upper = np.quantile(data, [0.025, 0.975])
    return data[(data >= lower) & (data <= upper)]


def draw_comsol_pnm_graph(
    x, y, title, Re, PARAM, save=True, xlabel="", ylabel="", percentile=95, show=False
):
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

    lower, upper = np.percentile(x, [0.01, 99.99])

    normal = (lower < x) & (x < upper)
    x = x[normal]
    y = y[normal]
    # plt.title(title)
    plt.scatter(
        x=x,
        y=y,
        c="C0",
        alpha=0.75,
    )

    origin = (min(x.min(), y.min()),) * 2
    plt.axline(origin, slope=1, color="r", label="y = x")
    r2 = r2_score(x, y)

    # plt.text(
    #     0.02,
    #     0.98,
    #     f"Re = {Re}",
    #     transform=plt.gca().transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=12,
    # )

    plt.text(
        0.02,
        0.98,
        f"$R^2$ = {r2:.2f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    plt.ticklabel_format(style="sci", scilimits=(-1, 2), axis="x")
    plt.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    # plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    # plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(PARAM.Path_figs / f"{PARAM.prefix}_Re{Re}_{title}.png")
    if show:
        plt.show()


plt_throat_flux_comsol = []
plt_throat_flux_dualn = []


plt_throat_heat_flux_conductive_comsol = []
plt_throat_heat_flux_conductive_dualn = []

plt_throat_heat_flux_conductive_void_comsol = []
plt_throat_heat_flux_conductive_void_dualn = []

plt_throat_heat_flux_conductive_solid_comsol = []
plt_throat_heat_flux_conductive_solid_dualn = []

plt_throat_heat_flux_conductive_connect_comsol = []
plt_throat_heat_flux_conductive_connect_dualn = []

plt_throat_heat_flux_convective_comsol = []
plt_throat_heat_flux_convective_dualn = []

plt_throat_heat_flux_total_comsol = []
plt_throat_heat_flux_total_dualn = []


plt_pore_P_comsol = []
plt_pore_P_dualn = []

plt_pore_T_void_comsol = []
plt_pore_T_void_dualn = []

plt_pore_T_solid_comsol = []
plt_pore_T_solid_dualn = []


for j, PARAM in enumerate(PARAMS):
    for i in range(len(Res)):
        Re_i = Res[i]
        hf_i = hf[i]
        dualn = net.vtk2network(
            PARAM.Path_results / f"dualn_Re{Re_i:.5f}_hf{hf_i:.0f}.vtp"
        )
        pn = net.vtk2network(PARAM.Path_results / f"pn_Re{Re_i:.5f}_hf{hf_i:.0f}.vtp")
        throat_flux = dualn["throat.flux"]
        throat_heat_flux_conductive = dualn["throat.heat_flux_conductive"]
        throat_heat_flux_convective = dualn["throat.heat_flux_convective"]
        throat_heat_flux_total = dualn["throat.heat_flux_total"]
        ### fluid flux ###
        plt_throat_flux_comsol.append(pn["throat.fluid_flux_comsol"])
        plt_throat_flux_dualn.append(throat_flux[dualn["throat.void"]])

        ### conductive heat flux ###
        plt_throat_heat_flux_conductive_comsol.append(
            dualn["throat.heat_flux_conductive_comsol"]
        )
        plt_throat_heat_flux_conductive_dualn.append(throat_heat_flux_conductive)
        ### void conductive heat flux ###
        plt_throat_heat_flux_conductive_void_comsol.append(
            dualn["throat.heat_flux_conductive_comsol"][dualn["throat.void"]]
        )
        plt_throat_heat_flux_conductive_void_dualn.append(
            throat_heat_flux_conductive[dualn["throat.void"]]
        )
        ### solid conductive heat flux ###
        plt_throat_heat_flux_conductive_solid_comsol.append(
            dualn["throat.heat_flux_conductive_comsol"][dualn["throat.solid"]]
        )
        plt_throat_heat_flux_conductive_solid_dualn.append(
            throat_heat_flux_conductive[dualn["throat.solid"]]
        )
        ### interfacial conductive heat flux ###
        plt_throat_heat_flux_conductive_connect_comsol.append(
            dualn["throat.heat_flux_conductive_comsol"][dualn["throat.connect"]]
        )
        plt_throat_heat_flux_conductive_connect_dualn.append(
            throat_heat_flux_conductive[dualn["throat.connect"]]
        )

        ### convective heat flux ###
        plt_throat_heat_flux_convective_comsol.append(
            dualn["throat.heat_flux_convective_comsol"]
        )
        plt_throat_heat_flux_convective_dualn.append(throat_heat_flux_convective)
        ### total heat flux ###
        plt_throat_heat_flux_total_comsol.append(dualn["throat.heat_flux_total_comsol"])
        plt_throat_heat_flux_total_dualn.append(throat_heat_flux_total)

        ### pore pressure ###
        plt_pore_P_comsol.append(dualn["pore.p_ave_comsol"][dualn["pore.void"]])
        plt_pore_P_dualn.append(dualn["pore.P"][dualn["pore.void"]] - 1000)

        ### void pore temperature ###
        plt_pore_T_void_comsol.append(dualn["pore.T_ave_comsol"][dualn["pore.void"]])
        plt_pore_T_void_dualn.append(dualn["pore.T"][dualn["pore.void"]])

        ### solid pore temperature ###
        plt_pore_T_solid_comsol.append(dualn["pore.T_ave_comsol"][dualn["pore.solid"]])
        plt_pore_T_solid_dualn.append(dualn["pore.T"][dualn["pore.solid"]])


###### concate #######
plt_throat_flux_comsol = np.concatenate(plt_throat_flux_comsol)
plt_throat_flux_dualn = np.concatenate(plt_throat_flux_dualn)
plt_throat_heat_flux_conductive_comsol = np.concatenate(
    plt_throat_heat_flux_conductive_comsol
)
plt_throat_heat_flux_conductive_dualn = np.concatenate(
    plt_throat_heat_flux_conductive_dualn
)
plt_throat_heat_flux_conductive_void_comsol = np.concatenate(
    plt_throat_heat_flux_conductive_void_comsol
)
plt_throat_heat_flux_conductive_void_dualn = np.concatenate(
    plt_throat_heat_flux_conductive_void_dualn
)
plt_throat_heat_flux_conductive_solid_comsol = np.concatenate(
    plt_throat_heat_flux_conductive_solid_comsol
)
plt_throat_heat_flux_conductive_solid_dualn = np.concatenate(
    plt_throat_heat_flux_conductive_solid_dualn
)
plt_throat_heat_flux_conductive_connect_comsol = np.concatenate(
    plt_throat_heat_flux_conductive_connect_comsol
)
plt_throat_heat_flux_conductive_connect_dualn = np.concatenate(
    plt_throat_heat_flux_conductive_connect_dualn
)
plt_throat_heat_flux_convective_comsol = np.concatenate(
    plt_throat_heat_flux_convective_comsol
)
plt_throat_heat_flux_convective_dualn = np.concatenate(
    plt_throat_heat_flux_convective_dualn
)
plt_throat_heat_flux_total_comsol = np.concatenate(plt_throat_heat_flux_total_comsol)
plt_throat_heat_flux_total_dualn = np.concatenate(plt_throat_heat_flux_total_dualn)

plt_pore_P_comsol = np.concatenate(plt_pore_P_comsol)
plt_pore_P_dualn = np.concatenate(plt_pore_P_dualn)
plt_pore_T_solid_comsol = np.concatenate(plt_pore_T_solid_comsol)
plt_pore_T_solid_dualn = np.concatenate(plt_pore_T_solid_dualn)
plt_pore_T_void_comsol = np.concatenate(plt_pore_T_void_comsol)
plt_pore_T_void_dualn = np.concatenate(plt_pore_T_void_dualn)


draw_comsol_pnm_graph(
    x=plt_throat_flux_comsol,
    y=plt_throat_flux_dualn,
    title="Throat flux",
    Re=Re_i,
    xlabel=r"$DNS\ (m^3/s)$",
    ylabel=r"$DNM\ (m^3/s)$",
    PARAM=PARAM,
)
plt.show()

draw_comsol_pnm_graph(
    x=plt_throat_heat_flux_conductive_solid_comsol,
    y=plt_throat_heat_flux_conductive_solid_dualn,
    title="Solid throat conductive heat flux",
    Re=Re_i,
    xlabel=r"$DNS\ (W)$",
    ylabel=r"$DNM\ (W)$",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_throat_heat_flux_conductive_void_comsol,
    y=plt_throat_heat_flux_conductive_void_dualn,
    title="Void throat conductive heat flux",
    Re=Re_i,
    xlabel=r"$DNS\ (W)$",
    ylabel=r"$DNM\ (W)$",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_throat_heat_flux_conductive_connect_comsol,
    y=plt_throat_heat_flux_conductive_connect_dualn,
    title="Interfacial throat conductive heat flux",
    Re=Re_i,
    xlabel=r"$DNS\ (W)$",
    ylabel=r"$DNM\ (W)$",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_throat_heat_flux_convective_comsol,
    y=plt_throat_heat_flux_convective_dualn,
    title="Void throat convective heat flux",
    Re=Re_i,
    xlabel=r"$DNS\ (W)$",
    ylabel=r"$DNM\ (W)$",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_throat_heat_flux_total_comsol,
    y=plt_throat_heat_flux_total_dualn,
    title="Throat total heat flux",
    Re=Re_i,
    xlabel=r"$DNS\ (W)$",
    ylabel=r"$DNM\ (W)$",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_pore_P_comsol,
    y=plt_pore_P_dualn,
    title="Pore pressure",
    Re=Re_i,
    xlabel=r"$P_{COMSOL}$ (Pa)",
    ylabel=r"$P_{DNM}$ (Pa)",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_pore_T_void_comsol,
    y=plt_pore_T_void_dualn,
    title="Void pore temperature",
    Re=Re_i,
    xlabel=r"$T_{COMSOL}$ (K)",
    ylabel=r"$T_{DNM}$ (K)",
    PARAM=PARAM,
    show=True,
)

draw_comsol_pnm_graph(
    x=plt_pore_T_solid_comsol,
    y=plt_pore_T_solid_dualn,
    title="Solid pore temperature",
    Re=Re_i,
    xlabel=r"$T_{COMSOL}$ (K)",
    ylabel=r"$T_{DNM}$ (K)",
    PARAM=PARAM,
    show=True,
)

# draw_comsol_pnm_graph(
#     x=dualn["pore.p_ave_comsol"][dualn["pore.void"]],
#     y=np.clip(dualn["pore.P"] - 1000, a_min=0, a_max=None)[dualn["pore.void"]],
#     title="Pore average pressure",
#     Re=Re_i,
#     xlabel=r"$DNS\ (Pa)$",
#     ylabel=r"$DNM\ (Pa)$",
#     PARAM=PARAM,
# )
# draw_comsol_pnm_graph(
#     x=dualn["pore.p_pore_center_comsol"][dualn["pore.void"]],
#     y=np.clip(dualn["pore.P"] - 1000, a_min=0, a_max=None)[dualn["pore.void"]],
#     title="Pore center pressure",
#     Re=Re_i,
#     xlabel=r"$DNS\ (Pa)$",
#     ylabel=r"$DNM\ (Pa)$",
#     PARAM=PARAM,
# )

# void_true_bool = dualn[
#     "pore.void"
# ]  # & ~dualn['pore.surface_x-'] & ~dualn['pore.surface_z-'] & ~dualn['pore.error_T'] & ~dualn['pore.error_U']
# solid_true_bool = dualn[
#     "pore.solid"
# ]  # & ~dualn['pore.surface_x-'] & ~dualn['pore.surface_z-']  & ~dualn['pore.error_T']

# draw_comsol_pnm_graph(
#     x=pn["pore.spf.U_ave_comsol"],
#     y=pn["pore.U"],
#     title="Pore velocity",
#     Re=Re_i,
#     xlabel=r"$DNS\ (m/s)$",
#     ylabel=r"$DNM\ (m/s)$",
#     PARAM=PARAM,
# )


# draw_comsol_pnm_graph(
#     x=dualn["pore.T_ave_comsol"],
#     y=dualn["pore.T"],
#     title="Solid pore temperature",
#     Re=Re_i,
#     xlabel=r"$T_{COMSOL}$ (K)",
#     ylabel=r"$T_{DNM}$ (K)",
#     PARAM=PARAM,
# )

# draw_comsol_pnm_graph(
#     x=dualn["pore.T_pore_center_comsol"],
#     y=dualn["pore.T"],
#     title="Pore temperature",
#     Re=Re_i,
#     xlabel=r"$T_{COMSOL}$ (K)",
#     ylabel=r"$T_{DNM}$ (K)",
#     PARAM=PARAM,
# )


import pandas as pd

output_throat = pd.DataFrame(
    data=list(
        zip_longest(
            plt_throat_flux_comsol,
            plt_throat_flux_dualn,
            plt_throat_heat_flux_conductive_comsol,
            plt_throat_heat_flux_conductive_dualn,
            plt_throat_heat_flux_conductive_void_comsol,
            plt_throat_heat_flux_conductive_void_dualn,
            plt_throat_heat_flux_conductive_solid_comsol,
            plt_throat_heat_flux_conductive_solid_dualn,
            plt_throat_heat_flux_conductive_connect_comsol,
            plt_throat_heat_flux_conductive_connect_dualn,
            plt_throat_heat_flux_convective_comsol,
            plt_throat_heat_flux_convective_dualn,
            plt_throat_heat_flux_total_comsol,
            plt_throat_heat_flux_total_dualn,
        )
    ),
    columns=(
        "throat_flux_comsol",
        "throat_flux_dualn",
        "throat_heat_flux_conductive_comsol",
        "throat_heat_flux_conductive_dualn",
        "throat_heat_flux_conductive_void_comsol",
        "throat_heat_flux_conductive_void_dualn",
        "throat_heat_flux_conductive_solid_comsol",
        "throat_heat_flux_conductive_solid_dualn",
        "throat_heat_flux_conductive_connect_comsol",
        "throat_heat_flux_conductive_connect_dualn",
        "throat_heat_flux_convective_comsol",
        "throat_heat_flux_convective_dualn",
        "throat_heat_flux_total_comsol",
        "throat_heat_flux_total_dualn",
    ),
)
output_pore = pd.DataFrame(
    data=list(
        zip_longest(
            plt_pore_P_comsol,
            plt_pore_P_dualn,
            plt_pore_T_solid_comsol,
            plt_pore_T_solid_dualn,
            plt_pore_T_void_comsol,
            plt_pore_T_void_dualn,
        )
    ),
    columns=(
        "pore_P_comsol",
        "pore_P_dualn",
        "pore_T_solid_comsol",
        "pore_T_solid_dualn",
        "pore_T_void_comsol",
        "pore_T_void_dualn",
    ),
)
output_throat.to_csv("output_throat_minkowski.csv")
output_pore.to_csv("output_pore_minkowski.csv")
