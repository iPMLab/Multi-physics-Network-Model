#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

sys.path.append("../../../")

# Papers\P1\Code\COMSOL
from mpnm_new import topotool, algorithm, network as net, util
from mpnm_new.enum import Boundary_Condition_Types as BC_Type
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

from Papers.P1.Code.COMSOL.comsol_params import extract_Re_hf
import pickle
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from Papers.P1.Code.COMSOL.comsol_params import (
    PARAMS_N2_500,
    PARAMS_N3_455,
    PARAMS_N4_353,
    PARAMS_N4_689,
    PARAMS_N4_869,
    PARAMS_N5_000,
    PARAMS_N10_000,
)

np.set_printoptions(legacy="1.21")
util.set_num_threads(4)


def draw_comsol_pnm_graph(
    x, y, title, Re, PARAM, save=True, xlabel="", ylabel="", percentile=100
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
    plt.title(title)
    plt.scatter(
        x=x,
        y=y,
        c="C0",
        alpha=0.75,
    )

    origin = (min(x.min(), y.min()),) * 2
    plt.axline(origin, slope=1, color="r", label="y = x")
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
    plt.ticklabel_format(style="sci", scilimits=(-1, 2), axis="x")
    plt.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    # plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    # plt.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save:
        plt.savefig(PARAM.Path_figs / f"{PARAM.prefix}_Re{Re}_{title}.png")
    plt.show()


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


def run_simulation(PARAM, dict_adjustments=None):
    PARAM = PARAM()
    # u_inlets = PARAM.u_inlets
    img_shape = PARAM.raw_shape
    heat_flux_out = PARAM.heat_flux_out
    area_inlet = PARAM.area_inlet

    Path_net_pore = PARAM.Path_net_pore
    Path_net_dual = PARAM.Path_net_dual

    Path_results = PARAM.Path_results
    Path_PNdata = PARAM.Path_PNdata

    Path_figs = PARAM.Path_results / "figs"
    Path_figs.mkdir(exist_ok=True, parents=True)

    # import matplotlib

    plt.rc("font", family="Times New Roman")
    plt.rcParams["font.size"] = 12

    adjust = bool(1)
    opt = {}
    opt = defaultdict(lambda: 1)

    if adjust:
        opt["pn['throat.radius']"] = dict_adjustments.get("pn['throat.radius']", 1.0)
        opt["dualn['throat.length'][dualn['throat.void']]"] = 1
        opt["dualn['throat.length'][dualn['throat.solid']]"] = 1.0
        opt["dualn['throat.length'][dualn['throat.connect']]"] = dict_adjustments.get(
            "dualn['throat.length'][dualn['throat.connect']]", 1.0
        )

        opt["alpha"] = 1  # 2.7

    imsize = np.array(img_shape)
    resolution = PARAM.resolution
    plot = bool(0)
    pn = net.vtk2network(Path_net_pore)
    dualn = net.vtk2network(Path_net_dual)

    dualn["throat.length"][dualn["throat.void"]] *= opt[
        "dualn['throat.length'][dualn['throat.void']]"
    ]
    dualn["throat.length"][dualn["throat.solid"]] *= opt[
        "dualn['throat.length'][dualn['throat.solid']]"
    ]
    dualn["throat.length"][dualn["throat.connect"]] *= opt[
        "dualn['throat.length'][dualn['throat.connect']]"
    ]

    dualn["throat.radius"][dualn["throat.void"]] *= 1.0
    dualn["throat.radius"][dualn["throat.solid"]] *= 1.0
    dualn["throat.radius"][dualn["throat.connect"]] *= 1.0
    dualn["pore.radius"][dualn["pore.void"]] *= 1.0

    num_solid_ball = np.count_nonzero(dualn["pore.solid"])
    num_void_ball = np.count_nonzero(dualn["pore.void"])
    print(num_void_ball, num_solid_ball)

    dualn["pore.U"] = np.zeros(len(dualn["pore.all"]))
    dualn["pore.flux"] = np.zeros(len(dualn["pore.all"]))

    ############ physical properties ############
    T_fluid = 293.15

    fluid = {}
    solid = {}
    fluid["density"] = PARAM.rho_void
    fluid["Cp"] = PARAM.Cp_void  # J/(kg*K)
    fluid["lambda"] = PARAM.k_void  # W/(m*K)
    fluid["viscosity"] = PARAM.mu_void  # Pa*s
    print(fluid["viscosity"])
    solid["density"] = PARAM.rho_solid  # kg/m^3 sandstone
    solid["Cp"] = PARAM.Cp_solid  # J/(kg*K)
    solid["lambda"] = PARAM.k_solid  # W/(m K)

    ###
    pn["throat.radius"] *= opt["pn['throat.radius']"]
    ###
    fluid["initial_temperature"] = T_fluid
    solid["initial_temperature"] = T_fluid
    # material = "copper"
    ############ physical properties ############
    pn["properties"] = {"resolution": resolution, "img_size": imsize}
    dualn["properties"] = pn["properties"]

    Params = ["p", "T", "spf.U"]
    specific_params = [
        f"pore.{p}_{s}_comsol" for p in Params for s in ["min", "max", "ave"]
    ]

    keys_index_map = {f"{p}": i for i, p in enumerate(specific_params)}
    print(keys_index_map)

    PNdata = pickle.load(Path_PNdata.open("rb"))

    Re_hf = [
        extract_Re_hf(key, _Re_astype=float, _hf_astype=float) for key in PNdata.keys()
    ]
    Re_hf_str = [(f"{Re_i:.5f}", f"{int(hf_i)}") for Re_i, hf_i in Re_hf]
    print(Re_hf_str)

    b_len = PARAM.raw_shape[0] * resolution
    De = 2 * b_len * b_len / (b_len + b_len)

    area_inlet = PARAM.area_inlet

    df_data_ml = pd.DataFrame()
    df_data_ml["Re"] = [Re_i for Re_i, _ in Re_hf]

    def single_condition(Re_hf_i):
        Re_i = Re_hf[Re_hf_i][0]
        hf_i = Re_hf[Re_hf_i][1]

        # dualn["properties"]["hf_comsol"] = heat_flux_dict[u_str]
        dualn["properties"]["hf_comsol"] = heat_flux_out[Re_hf_i]
        PNdata_i = PNdata[f"_Re{Re_i:.5f}_hf{int(hf_i)}"]
        PTU_min_max_ave_comsol = np.empty(
            (dualn["pore.all"].size, len(specific_params))
        )
        for i, specific_param in enumerate(specific_params):
            PTU_min_max_ave_comsol[:, i] = PNdata_i[specific_param]
        PTU_min_max_ave_comsol[PTU_min_max_ave_comsol < 0] = 0

        for key in keys_index_map:
            dualn[key] = PTU_min_max_ave_comsol[:, keys_index_map[key]]
            pn[key] = dualn[key][:num_void_ball]

        for param in ("p", "spf.U", "T"):
            pore_param_pore_center = PNdata_i[f"pore.{param}_pore_center_comsol"]
            pore_param_pore_center = np.where(
                pore_param_pore_center < 0, 0, pore_param_pore_center
            )
            dualn[f"pore.{param}_pore_center_comsol"] = pore_param_pore_center
            pn[f"pore.{param}_pore_center_comsol"] = pore_param_pore_center[
                :num_void_ball
            ]

        throat_fluid_flux_comsol = PNdata_i["throat.fluid_flux_comsol"]
        dualn["throat.fluid_flux_comsol"] = throat_fluid_flux_comsol
        pn["throat.fluid_flux_comsol"] = throat_fluid_flux_comsol[dualn["throat.void"]]

        throat_heat_flux_convective_comsol = PNdata_i[
            "throat.heat_flux_convective_comsol"
        ]
        dualn["throat.heat_flux_convective_comsol"] = throat_heat_flux_convective_comsol
        throat_heat_flux_conductive_comsol = PNdata_i[
            "throat.heat_flux_conductive_comsol"
        ]
        dualn["throat.heat_flux_conductive_comsol"] = throat_heat_flux_conductive_comsol

        dualn["throat.heat_flux_total_comsol"] = (
            throat_heat_flux_conductive_comsol + throat_heat_flux_convective_comsol
        )
        delta_T_comsol = dualn["pore.T_ave_comsol"][dualn["throat.conns"]]
        delta_T_comsol = delta_T_comsol[:, 0] - delta_T_comsol[:, 1]

        print("alpha=", opt["alpha"])

        inlet = ["x-", "left"][0]
        outlet = ["x+", "right"][0]

        # ------------------------boundary conditions-------------------------------#

        Boundary_condition_P_inlet = pd.DataFrame()
        Boundary_condition_P_inlet["ids"] = np.where(pn[f"pore.surface_{inlet}"])[0]
        Boundary_condition_P_inlet["types"] = BC_Type.neumann
        area = pn[f"pore.surface_area_{inlet}"][Boundary_condition_P_inlet["ids"]]
        Q_in_i = Re_i * PARAM.mu_void * b_len * b_len / (De * PARAM.rho_void)
        u_i = Q_in_i / area_inlet
        print("u_i=", u_i)
        Boundary_condition_P_inlet["values"] = u_i * area
        Boundary_condition_P_inlet["names"] = f"pore.surface_{inlet}"

        Boundary_condition_P_outlet = pd.DataFrame()
        Boundary_condition_P_outlet["ids"] = np.where(pn[f"pore.surface_{outlet}"])[0]
        Boundary_condition_P_outlet["types"] = BC_Type.dirichlet
        Boundary_condition_P_outlet["values"] = np.full(
            len(Boundary_condition_P_outlet["ids"]), 0, dtype=np.float64
        )
        Boundary_condition_P_outlet["names"] = f"pore.surface_{outlet}"

        Boundary_condition_P2 = util.dfs2df(
            [Boundary_condition_P_inlet, Boundary_condition_P_outlet],
            subset="ids",
            keep="first",
        )

        Boundary_condition_T_inlet = pd.DataFrame()
        Boundary_condition_T_inlet["ids"] = np.where(
            dualn[f"pore.surface_{inlet}"] & dualn["pore.void"]
        )[0]
        Boundary_condition_T_inlet["types"] = BC_Type.dirichlet
        Boundary_condition_T_inlet["values"] = 293.15
        Boundary_condition_T_inlet["names"] = "pore_inlet_T"

        Boundary_condition_T_outlet = pd.DataFrame()
        Boundary_condition_T_outlet["ids"] = np.where(dualn["pore.surface_z-"])[0]
        Boundary_condition_T_outlet["types"] = BC_Type.neumann
        area = dualn["pore.surface_area_z-"][Boundary_condition_T_outlet["ids"]]
        Boundary_condition_T_outlet["values"] = hf_i * area
        Boundary_condition_T_outlet["names"] = "pore_inlet_T1"

        Boundary_condition_T_outlet2 = pd.DataFrame()
        # Boundary_condition_T_outlet2["ids"] = np.where()[0]
        Boundary_condition_T_outlet2["ids"] = np.where(
            dualn["pore.surface_x+"] & dualn["pore.void"]
        )[0]
        Boundary_condition_T_outlet2["types"] = BC_Type.outflow
        Boundary_condition_T_outlet2["values"] = 0
        Boundary_condition_T_outlet2["names"] = "pore_outlet_T2"
        Boundary_condition_T2 = util.dfs2df(
            [
                Boundary_condition_T_inlet,
                Boundary_condition_T_outlet,
                Boundary_condition_T_outlet2,
            ],
            subset="ids",
            keep="first",
        )

        # ------------------------boundary conditions-------------------------------#
        Tem_c = dualn["pore.all"] * 293.15
        T_x0 = dualn["pore.all"] * 290.15
        P_profile = dualn["pore.all"] * 0.0
        P_profile_tem = np.copy(P_profile[dualn["pore.void"]])
        for j in fluid:
            pn["pore." + j] = fluid[j] * pn["pore.all"]
            pn["throat." + j] = fluid[j] * pn["throat.all"]
            dualn["pore." + j] = fluid[j] * dualn["pore.all"]
            dualn["throat." + j] = fluid[j] * dualn["throat.all"]
        for j in solid:
            if "pore." + j in dualn:
                dualn["pore." + j][dualn["pore.solid"]] = (
                    solid[j] * dualn["pore.all"][dualn["pore.solid"]]
                )
            if "throat." + j in dualn:
                dualn["throat." + j][dualn["throat.solid"]] = (
                    solid[j] * dualn["throat.all"][dualn["throat.solid"]]
                )

        T_res = []

        # u=np.array([0.10])[n]#
        dualn["pore.viscosity"][:] = fluid["viscosity"]
        pn["pore.viscosity"] = dualn["pore.viscosity"][dualn["pore.void"]]

        dualn["throat.viscosity"][:] = fluid["viscosity"]
        pn["throat.viscosity"] = dualn["throat.viscosity"][dualn["throat.void"]]

        T_x0[:] = Tem_c
        dualn["pore.viscosity"][:] = fluid["viscosity"]
        pn["pore.viscosity"] = dualn["pore.viscosity"][dualn["pore.void"]]

        pn["pore.P"] = np.zeros(pn["pore.all"].size)
        P_profile_tem, coe_A_P = algorithm.single_phase_steady_iteration_algorithm(
            pn,
            pn["pore.viscosity"],
            Boundary_condition_P2,
            tol=1e-6,
            boundary_len=pn["pore.radius"],
            n=0.296,
            m=1,
        )

        P_profile[dualn["pore.void"]] = P_profile_tem
        output = topotool.calculate_mass_flow2(
            pn, Boundary_condition_P2, coe_A_P, P_profile_tem
        )
        print(output)
        abs_perm = output[f"pore.surface_{inlet}"] / (
            np.max(P_profile_tem[pn[f"pore.surface_{inlet}"]])
            - np.min(P_profile_tem[pn[f"pore.surface_{outlet}"]])
        )

        abs_perm *= (
            np.average(pn["pore.viscosity"])
            * imsize[0]
            / (imsize[1] * imsize[2])
            / resolution
        )
        Perm = np.copy(abs_perm)
        Pressure_drop = np.mean(P_profile_tem[pn[f"pore.surface_{inlet}"]]) - np.mean(
            P_profile_tem[pn[f"pore.surface_{inlet}"]]
        )
        dualn["pore.P"] = P_profile
        pn["pore.P"] = P_profile_tem
        if len(T_res) == 0:
            output = topotool.calculate_mass_flow2(
                pn, Boundary_condition_P2, coe_A_P, P_profile_tem
            )
            abs_perm = output[f"pore.surface_{inlet}"] / Pressure_drop
            abs_perm *= (
                np.average(pn["pore.viscosity"])
                * imsize[0]
                / (imsize[1] * imsize[2])
                / resolution
            )
            Perm = abs_perm

        delta_p = (
            P_profile_tem[pn["throat.conns"][:, 1]]
            - P_profile_tem[pn["throat.conns"][:, 0]]
        )

        # ---------------temperature process---------------#

        # flux_Throat_profile = delta_p * coe_A_P
        Vel_Pore_profile = topotool.cal_pore_veloc(
            pn, coe_A_P, P_profile_tem, pn["pore._id"]
        )
        ###
        Vel_Pore_profile = Vel_Pore_profile / opt["alpha"]  # *100
        ###
        pn["pore.U"] = Vel_Pore_profile
        dualn["pore.U"] = np.zeros(len(dualn["pore.all"]))
        dualn["pore.U"][dualn["pore.void"]] = Vel_Pore_profile
        RE_po = np.zeros(len(dualn["pore.all"]))
        RE_po[dualn["pore.void"]] = (
            Vel_Pore_profile
            * 2
            * pn["pore.radius"]
            * fluid["density"]
            / pn["pore.viscosity"]
        )
        g_ij = np.zeros(len(dualn["throat._id"]))
        g_ij[dualn["throat.void"]] = coe_A_P

        # coe_A for convection heat transfer
        total_r = np.sum(dualn["pore.radius"][dualn["throat.conns"]], axis=1)
        pore0, pore1 = dualn["throat.conns"][:, 0], dualn["throat.conns"][:, 1]
        len_w0 = dualn["pore.radius"][pore0] / total_r
        len_w1 = dualn["pore.radius"][pore1] / total_r
        len0 = dualn["throat.length"] * len_w0
        len1 = dualn["throat.length"] * len_w1
        heat_s_f = dualn["throat.length"] / (
            len0 / fluid["lambda"] + len1 / solid["lambda"]
        )

        thermal_con_dual = (
            dualn["throat.solid"] * dualn["throat.lambda"]
            + dualn["throat.connect"] * heat_s_f
            + dualn["throat.void"] * dualn["throat.lambda"]
        )

        coe_B = (
            dualn["throat.radius"] ** 2
            * np.pi
            * thermal_con_dual
            / dualn["throat.length"]
        )

        Tem_c = algorithm.two_phase_steady_convection_algorithm2(
            dualn,
            g_ij,
            coe_B,
            Boundary_condition_T2,
            P_profile,
        )

        hf_output = topotool.calculate_heat_flow2(
            dualn, Boundary_condition_T2, g_ij, Tem_c, coe_B, P_profile
        )

        print(hf_output)
        dualn["pore.T"] = Tem_c
        # print(Tem_c)
        T_res.append(Tem_c)

        pore0 = dualn["throat.conns"][:, 0]
        pore1 = dualn["throat.conns"][:, 1]
        delta_p = dualn["pore.P"][pore0] - dualn["pore.P"][pore1]
        throat_flux = g_ij * delta_p

        delta_T = Tem_c[pore0] - Tem_c[pore1]
        throat_heat_flux_conductive = coe_B * delta_T

        throat_heat_flux_convective = (
            throat_flux
            * dualn["throat.Cp"]
            * dualn["throat.density"]
            * (np.where(delta_p > 0, Tem_c[pore0], Tem_c[pore1]) - 293.15)
        )

        throat_heat_flux_total = (
            throat_heat_flux_conductive + throat_heat_flux_convective
        )

        dualn["throat.heat_flux_conductive"] = throat_heat_flux_conductive
        dualn["throat.heat_flux_convective"] = throat_heat_flux_convective
        dualn["throat.heat_flux_total"] = throat_heat_flux_total
        dualn["throat.flux"] = throat_flux

        if plot:
            draw_comsol_pnm_graph(
                x=dualn["pore.p_ave_comsol"][dualn["pore.void"]],
                y=dualn["pore.P"][dualn["pore.void"]],
                title="Pore average pressure",
                Re=Re_i,
                xlabel=r"$DNS\ (Pa)$",
                ylabel=r"$DNM\ (Pa)$",
                PARAM=PARAM,
            )
            dualn["pore.error_T"] = np.abs(
                dualn["pore.T"] - dualn["pore.T_ave_comsol"]
            ) > np.percentile(
                np.abs(
                    dualn["pore.T"][dualn["pore.void"]]
                    - dualn["pore.T_ave_comsol"][dualn["pore.void"]]
                ),
                100,
            )

            void_true_bool = dualn["pore.void"]
            solid_true_bool = dualn["pore.solid"]

            draw_comsol_pnm_graph(
                x=pn["pore.spf.U_ave_comsol"],
                y=pn["pore.U"],
                title="Pore velocity",
                Re=Re_i,
                xlabel=r"$DNS\ (m/s)$",
                ylabel=r"$DNM\ (m/s)$",
                PARAM=PARAM,
            )

            draw_comsol_pnm_graph(
                x=dualn["pore.T_ave_comsol"][void_true_bool],
                y=dualn["pore.T"][void_true_bool],
                title="Void pore temperature",
                Re=Re_i,
                xlabel=r"$T_{COMSOL}$ (K)",
                ylabel=r"$T_{DNM}$ (K)",
                PARAM=PARAM,
            )

            draw_comsol_pnm_graph(
                x=dualn["pore.T_ave_comsol"][solid_true_bool],
                y=dualn["pore.T"][solid_true_bool],
                title="Solid pore temperature",
                Re=Re_i,
                xlabel=r"$T_{COMSOL}$ (K)",
                ylabel=r"$T_{DNM}$ (K)",
                PARAM=PARAM,
            )

            draw_comsol_pnm_graph(
                x=dualn["pore.T_ave_comsol"],
                y=dualn["pore.T"],
                title="Solid pore temperature",
                Re=Re_i,
                xlabel=r"$T_{COMSOL}$ (K)",
                ylabel=r"$T_{DNM}$ (K)",
                PARAM=PARAM,
            )

            draw_comsol_pnm_graph(
                x=dualn["pore.T_pore_center_comsol"],
                y=dualn["pore.T"],
                title="Pore temperature",
                Re=Re_i,
                xlabel=r"$T_{COMSOL}$ (K)",
                ylabel=r"$T_{DNM}$ (K)",
                PARAM=PARAM,
            )
        pn["pore.T"] = dualn["pore.T"][dualn["pore.void"]]
        pn["pore.Re"] = RE_po[dualn["pore.void"]]

        outlet_void_tem = np.average(
            dualn["pore.T"][dualn["pore.void"] & dualn["pore.surface_x+"]],
            weights=dualn["pore.volume"][dualn["pore.void"] & dualn["pore.surface_x+"]],
        )
        outlet_void_tem_comsol = np.average(
            dualn["pore.T_ave_comsol"][dualn["pore.void"] & dualn["pore.surface_x+"]],
            weights=dualn["pore.volume"][dualn["pore.void"] & dualn["pore.surface_x+"]],
        )
        print("PNM", outlet_void_tem)
        print("COMSOL", outlet_void_tem_comsol)

        pore_flux = topotool.cal_pore_flux(pn, coe_A_P, P_profile, pn["pore._id"])
        Q_flux_pn = np.sum(
            np.abs(pore_flux[pn["pore.surface_x+"]])
            * (dualn["pore.T"][dualn["pore.surface_x+"] & dualn["pore.void"]] - 293.15)
            * fluid["Cp"]
            * fluid["density"]
        )
        print("Q_flux_pn", Q_flux_pn)

        Q_flux_comsol = dualn["properties"]["hf_comsol"]
        dualn["pore.hf_pn"] = dualn["pore.all"] * Q_flux_pn
        dualn["pore.P_pn"] = dualn["pore.all"] * (
            np.average(
                pn["pore.P"][pn["pore.surface_x-"]],
                weights=pn["pore.volume"][pn["pore.surface_x-"]],
            )
        )
        # print(dualn["pore.P_pn"])
        print("Q_flux_comsol", Q_flux_comsol)
        relative_error = (Q_flux_pn - Q_flux_comsol) / Q_flux_comsol
        print("Q_flux_relative_error", relative_error * 100, "%")
        heat_flux_pn.append(Q_flux_pn)
        heat_flux_comsol.append(Q_flux_comsol)
        print(np.count_nonzero(dualn["pore.void"]))
        print(
            np.average(
                dualn["pore.T"][dualn["pore.void"] & dualn["pore.surface_x-"]],
                weights=dualn["pore.volume"][
                    dualn["pore.void"] & dualn["pore.surface_x-"]
                ],
            )
        )
        print(
            np.average(
                dualn["pore.T"][dualn["pore.void"] & dualn["pore.surface_z-"]],
                weights=dualn["pore.volume"][
                    dualn["pore.void"] & dualn["pore.surface_z-"]
                ],
            )
        )
        print(
            np.average(
                dualn["pore.T"][dualn["pore.solid"] & dualn["pore.surface_z-"]],
                weights=dualn["pore.volume"][
                    dualn["pore.solid"] & dualn["pore.surface_z-"]
                ],
            )
        )

        net.network2vtk(
            dualn, filename=str(Path_results / f"dualn_Re{Re_i:.5f}_hf{hf_i:.0f}")
        )
        net.network2vtk(
            pn, filename=str(Path_results / f"pn_Re{Re_i:.5f}_hf{hf_i:.0f}")
        )

    heat_flux_pn = []
    heat_flux_comsol = []

    Re_hf_len = len(Re_hf)
    print(Re_hf)

    for i in range(Re_hf_len):
        Re_i = Re_hf[i][0]
        hf_i = Re_hf[i][1]
        print(f"Re = {Re_i} hf = {hf_i}")
        single_condition(i)


if __name__ == "__main__":
    dict_adjustments = {
        "_N4.689": {
            "pn['throat.radius']": 0.93,
            "dualn['throat.length'][dualn['throat.connect']]": 1.29,
        },
        "_N2.500": {
            "pn['throat.radius']": 0.9,
            "dualn['throat.length'][dualn['throat.connect']]": 1.4,
        },
        "_N3.455": {
            "pn['throat.radius']": 0.942,
            "dualn['throat.length'][dualn['throat.connect']]": 1.3,
        },
        "_N4.353": {
            "pn['throat.radius']": 0.932,
            "dualn['throat.length'][dualn['throat.connect']]": 1.25,
        },
        "_N4.869": {
            "pn['throat.radius']": 0.9285,
            "dualn['throat.length'][dualn['throat.connect']]": 1.25,
        },
        "_N5.000": {
            "pn['throat.radius']": 0.932,
            "dualn['throat.length'][dualn['throat.connect']]": 1.15,
        },
        "_N10.000": {
            "pn['throat.radius']": 0.932,
            "dualn['throat.length'][dualn['throat.connect']]": 1.15,
        },
    }
    import time

    for PARAM in [
        *PARAMS_N4_689,
        *PARAMS_N2_500,
        *PARAMS_N3_455,
        *PARAMS_N4_353,
        *PARAMS_N4_869,
        *PARAMS_N5_000,
        *PARAMS_N10_000,
    ]:
        t0 = time.time()
        print(PARAM.prefix)
        dict_adjustments_key = "_".join(PARAM.prefix.split("_")[:2])
        run_simulation(
            PARAM=PARAM, dict_adjustments=dict_adjustments[dict_adjustments_key]
        )
        print(PARAM.prefix, "time:", time.time() - t0)

tend = time.time()
