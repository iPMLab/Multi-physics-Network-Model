from sparse_dot_mkl import pardisoinit as _pardisoinit, pardiso
from ..util import check_mpn, add_at, keys_in_dict
from ..topotool import H_P_fun, func_g, mass_balance_conv
from ..enum import Boundary_Condition_Types, Pore_Types, Throat_Types
from ..network import getting_zoom_value, network2vtk
from scipy.sparse import coo_array, csr_array
import numpy as np
from collections import OrderedDict
import numpy.linalg as LA


def pardisoinit(mtype, cache=OrderedDict()):
    if mtype in cache:
        return cache[mtype]
    else:
        pt, iparm = _pardisoinit(mtype=mtype)
        # iparm[3] = 61
        # iparm[10] = 1
        # iparm[12] = 1
        # iparm[8] = 25
        # iparm[9] = 25
        # iparm[23] = 10
        cache[mtype] = (pt, iparm)
        return pt, iparm


def spsolve(A, b, single_precision=False, mtype=11):
    pt, iparm = pardisoinit(mtype=mtype)
    if single_precision:
        A = A.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        iparm[27] = 1
    else:
        pass
    res = pardiso(A, b, pt, mtype, iparm)[0]
    return res


def calculate_zoom_factor(mpn, boundary_condition, img_size, resolution):
    """
    calculate zoom factor for boundary condition if align_boundary_area is True
    """
    scope_key = boundary_condition.scope_key
    if boundary_condition.axis == "x":
        real_img_size = img_size[1] * img_size[2] * resolution**2
        real_mpn_size = np.sum(mpn["pore.radius"][mpn[scope_key]] ** 2 * np.pi)
    elif boundary_condition.axis == "y":
        real_img_size = img_size[0] * img_size[2] * resolution**2
        real_mpn_size = np.sum(mpn["pore.radius"][mpn[scope_key]] ** 2 * np.pi)
    elif boundary_condition.axis == "z":
        real_img_size = img_size[0] * img_size[1] * resolution**2
        real_mpn_size = np.sum(mpn["pore.radius"][mpn[scope_key]] ** 2 * np.pi)
    else:
        raise ValueError(
            "scope_key should has 'x', 'y', or 'z', when align_boundary_area is True,e.g. 'pore.x-_surface'"
        )
    area_zoom_factor = real_img_size / real_mpn_size
    r_zoom_factor = np.sqrt(area_zoom_factor)
    return area_zoom_factor, r_zoom_factor


def single_phase_steady_iteration_algorithm(
    mpn,
    viscosity_pore,
    boundary_conditions,
    boundary_len=1e-20,
    max_iter=10000,
    tol=1e-6,
    C=26,
    E=27,
    n=0.296,
    m=1.0,
):
    if "pore.P" not in mpn:
        mpn["pore.P"] = np.zeros(mpn["pore._id"].size, dtype=np.float64)
    check_mpn(
        mpn,
        (
            "throat.conns",
            "throat.density",
            "throat.viscosity",
            "throat.radius",
            "throat.total_length",
        ),
    )
    mpn_pore_P = mpn["pore.P"]
    mpn_throat_conns = mpn["throat.conns"]
    mpn_throat_density = mpn["throat.density"]
    mpn_throat_viscosity = mpn["throat.viscosity"]
    mpn_throat_radius = mpn["throat.radius"]
    mpn_throat_length = mpn["throat.total_length"]
    mpn_throat_conns_0 = mpn_throat_conns[:, 0]
    mpn_throat_conns_1 = mpn_throat_conns[:, 1]
    mpn_pore_P_old = mpn_pore_P + np.inf
    num_pore = mpn_pore_P.size
    coefficient = H_P_fun(mpn_throat_radius, mpn_throat_length, mpn_throat_viscosity)
    for it in range(max_iter):
        mpn_pore_P = single_phase_steady_algorithm2(
            mpn=mpn,
            coefficient=coefficient,
            viscosity_pore=viscosity_pore,
            boundary_conditions=boundary_conditions,
            boundary_len=boundary_len,
        )
        if LA.norm(mpn_pore_P_old - mpn_pore_P) / np.sqrt(num_pore) < tol:
            break
        elif it == max_iter - 1:
            print("Warning: max iteration reached, but not converged")
        else:
            mpn_pore_P_old[:] = mpn_pore_P
            throat_flux = coefficient * (
                mpn_pore_P[mpn_throat_conns_1] - mpn_pore_P[mpn_throat_conns_0]
            )
            Re_throat = np.fabs(
                2
                / np.pi
                * mpn_throat_density
                * throat_flux
                / mpn_throat_viscosity
                / mpn_throat_radius
            )
            coefficient = func_g(mpn, throat_flux, Re_throat, C=C, E=E, n=n, m=m)

    return mpn_pore_P, coefficient

    """ """


def apply_bc_dirichlet(dig, b, group, conductivity):
    bc_ids = group["ids"].values
    bc_values = group["values"].values
    conductivity = conductivity[bc_ids]
    dig[bc_ids] += conductivity
    b[bc_ids] -= conductivity * bc_values


def apply_bc_neumann(b, group):
    # Consider bc_values Flux in pore is +, out is -
    bc_ids = group["ids"].values
    bc_values = group["values"].values
    b[bc_ids] -= bc_values


def single_phase_steady_algorithm2(
    mpn, coefficient, viscosity_pore, boundary_conditions, boundary_len=1e-20
):
    """
    area mode choose the area calculation method
    direction:for pore_i, flux_j in pore is +, out is -
    """
    check_mpn(mpn)
    # bc_names = np.asarray(boundary_conditions['names'])
    mpn_pore__id = mpn["pore._id"]
    num_pore = mpn_pore__id.size
    num_throat = mpn["throat.all"].size
    mpn_throat_conns = mpn["throat.conns"]

    num_nondig = num_throat * 2
    num_total = num_nondig + num_pore

    mpn_throat_conns_col_0 = np.ascontiguousarray(mpn_throat_conns[:, 0])
    mpn_throat_conns_col_1 = np.ascontiguousarray(mpn_throat_conns[:, 1])

    rows = np.empty(shape=num_total, dtype=np.int64)
    rows[:num_throat] = mpn_throat_conns_col_0
    rows[num_throat:num_nondig] = mpn_throat_conns_col_1
    rows[num_nondig:] = mpn_pore__id

    cols = np.empty(shape=num_total, dtype=np.int64)
    cols[:num_throat] = mpn_throat_conns_col_1
    cols[num_throat:num_nondig] = mpn_throat_conns_col_0
    cols[num_nondig:] = mpn_pore__id

    coefficients = np.empty(shape=num_total, dtype=np.float64)
    coefficients[:num_throat] = coefficient
    coefficients[num_throat:num_nondig] = coefficient

    dig = np.bincount(
        rows[:num_nondig], weights=coefficients[:num_nondig], minlength=num_pore
    )

    b = np.zeros(shape=num_pore, dtype=np.float64)

    conductivity_pore = H_P_fun(mpn["pore.radius"], boundary_len, viscosity_pore)
    bc_types_grouped = boundary_conditions.groupby("types")
    for bc_type, group in bc_types_grouped:
        if bc_type == Boundary_Condition_Types.dirichlet:
            apply_bc_dirichlet(dig, b, group, conductivity_pore)
        elif bc_type == Boundary_Condition_Types.neumann:
            apply_bc_neumann(b, group)

    coefficients[num_nondig:] = -dig

    coefficients_adjacency_matrix = csr_array(
        (coefficients, (rows, cols)), shape=(num_pore, num_pore), dtype=np.float64
    )

    Profile = spsolve(coefficients_adjacency_matrix, b)
    return Profile


def two_phase_steady_convection_algorithm2(
    mpn,
    coefficient_flow,
    coefficient_conduction,
    boundary_conditions,
    P_profile,
    solid_boundary_len=1e-20,
    void_boundary_len=1e-20,
):
    mpn = check_mpn(mpn)
    keys_in_dict(
        mpn,
        (
            "pore.solid",
            "pore.Cp",
            "pore.density",
            "pore.lambda",
            "throat.Cp",
            "throat.density",
            "throat.lambda",
        ),
    )
    # bc_names = np.asarray(boundary_conditions['names'])

    mpn_pore_solid = mpn["pore.solid"]
    mpn_pore_void = mpn["pore.void"]
    mpn_pore_radius = mpn["pore.radius"]
    mpn_pore__id = mpn["pore._id"]
    num_pore = mpn_pore__id.size
    num_throat = mpn["throat.all"].size

    mpn_throat_conns = mpn["throat.conns"]
    mpn_throat_conns_col_0 = np.ascontiguousarray(mpn_throat_conns[:, 0])
    mpn_throat_conns_col_1 = np.ascontiguousarray(mpn_throat_conns[:, 1])
    delta_p = np.where(
        mpn["throat.void"],
        P_profile[mpn_throat_conns_col_1] - P_profile[mpn_throat_conns_col_0],
        0,
    )  # consider pore at [0], other pores flow in is +, out is -

    throat_flux_abs = np.abs(
        coefficient_flow * mpn["throat.Cp"] * mpn["throat.density"] * delta_p
    )
    delta_p_pos = delta_p > 0
    coefficient_convection_in = np.where(delta_p_pos, throat_flux_abs, 0)
    coefficient_convection_out = np.where(~delta_p_pos, throat_flux_abs, 0)

    flux_pore = np.zeros(num_pore, dtype=np.float64)
    flux_pore[mpn_pore_void] = mass_balance_conv(
        mpn, coefficient_flow, P_profile, mpn["pore._id"][mpn_pore_void]
    )

    convention_fluid = mpn["pore.density"] * mpn["pore.Cp"] * flux_pore

    num_nondig = num_throat * 4
    num_total = num_nondig + num_pore

    rows = np.empty(shape=num_total, dtype=np.int64)
    rows[:num_throat] = mpn_throat_conns_col_0
    rows[num_throat : num_throat * 2] = mpn_throat_conns_col_1
    rows[num_throat * 2 : num_throat * 3] = mpn_throat_conns_col_0
    rows[num_throat * 3 : num_nondig] = mpn_throat_conns_col_1
    rows[num_nondig:] = mpn_pore__id

    cols = np.empty(shape=num_total, dtype=np.int64)
    cols[:num_throat] = mpn_throat_conns_col_1
    cols[num_throat : num_throat * 2] = mpn_throat_conns_col_0
    cols[num_throat * 2 : num_throat * 3] = mpn_throat_conns_col_1
    cols[num_throat * 3 : num_nondig] = mpn_throat_conns_col_0
    cols[num_nondig:] = mpn_pore__id

    coefficients = np.empty(shape=num_total, dtype=np.float64)
    coefficients[:num_throat] = coefficient_conduction
    coefficients[num_throat : num_throat * 2] = coefficient_conduction
    coefficients[num_throat * 2 : num_throat * 3] = coefficient_convection_in
    coefficients[num_throat * 3 : num_nondig] = coefficient_convection_out

    dig = np.bincount(
        cols[:num_nondig], weights=coefficients[:num_nondig], minlength=num_pore
    )
    dig -= convention_fluid

    b = np.zeros(shape=num_pore, dtype=np.float64)

    area = np.pi * mpn_pore_radius**2
    boundary_len = np.where(mpn_pore_solid, solid_boundary_len, void_boundary_len)
    conductivity_pore = area * mpn["pore.lambda"] / boundary_len
    bc_ids = boundary_conditions["ids"].to_numpy(copy=False)
    # Solid
    bc_types_solid_grouped = boundary_conditions[mpn_pore_solid[bc_ids]].groupby(
        "types"
    )
    for bc_type, group in bc_types_solid_grouped:
        if bc_type == Boundary_Condition_Types.dirichlet:
            apply_bc_dirichlet(dig, b, group, conductivity_pore)
        elif bc_type == Boundary_Condition_Types.neumann:
            apply_bc_neumann(b, group)
        elif bc_type == Boundary_Condition_Types.outflow:
            pass
        # TODO:Robin
    # Void
    bc_types_void_grouped = boundary_conditions[mpn_pore_void[bc_ids]].groupby("types")
    for bc_type, group in bc_types_void_grouped:
        if bc_type == Boundary_Condition_Types.dirichlet:
            apply_bc_dirichlet(dig, b, group, conductivity_pore)
        elif bc_type == Boundary_Condition_Types.neumann:
            apply_bc_neumann(b, group)
        elif bc_type == Boundary_Condition_Types.outflow:
            pass

    coefficients[num_nondig:] = -dig

    coefficients_adjacency_matrix = csr_array(
        (coefficients, (rows, cols)), shape=(num_pore, num_pore), dtype=np.float64
    )
    # dig += 1e-20
    Tem_c = spsolve(coefficients_adjacency_matrix, b)
    return Tem_c


def stead_stay_alg_multi(pn, fluid, coe_A, Boundary_condition, resolution, bound_cond):
    num_pore = len(pn["pore.all"])
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_As = np.empty(shape=len(coe_A) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    coe_As[: len(coe_A)] = coe_A
    coe_As[len(coe_A) :] = coe_A
    A = coo_array((coe_As, (rows, cols)), shape=(num_pore, num_pore), dtype=np.float64)
    # A = csr_matrix((coe_A, (pn['throat.conns'][:, 1], pn['throat.conns'][:, 0])),
    #                shape=(num_pore, num_pore), dtype=np.float64)
    # A = (A.T + A).tolil()
    dig = np.asarray(A.sum(axis=0)).reshape(num_pore)
    b = np.zeros(num_pore)

    # B=A.toarray()
    # mean_gil=np.max(coe_A)
    # condctivity=H_P_fun(pn['pore.radius'],resolution,fluid['visocity'])
    def diffusion_coe_SO3(r, T, M1, M2, p, L):
        F_l = 2e-10 * T / p
        coe1 = 0.97 * r * (T / M1) ** 0.5
        x = 0.0060894 * T
        omb = (
            0.0051 * x**6
            - 0.0917 * x**5
            + 0.6642 * x**4
            - 2.477 * x**3
            + 5.0798 * x**2
            - 5.684 * x
            + 3.941
        )
        coe3 = 2.5e-9 * T**1.5 / p / omb
        coe2 = 1 / (1 / coe1 + 1 / coe3)
        data_coe = np.copy(coe2)
        data_coe[r < 0.1 * F_l] = coe1[r < 0.1 * F_l]
        data_coe[r >= 100 * F_l] = coe3
        return data_coe / L

    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            condctivity = diffusion_coe_SO3(
                pn["pore.radius"], fluid["temperature"], 56, 28, 1, resolution
            )
            if not bound_cond:
                bound_cond = {}
                index = np.argwhere(pn[n])
                value = condctivity[index]
                bound_cond["throat_inlet_cond"] = np.column_stack(
                    (index.reshape(-1), value.reshape(-1))
                )
                bound_cond["throat_outlet_cond"] = np.column_stack(
                    (index.reshape(-1), value.reshape(-1))
                )
                throat_inlet_cond = bound_cond["throat_inlet_cond"]
                throat_outlet_cond = bound_cond["throat_outlet_cond"]
                bound_cond = False
            else:
                throat_inlet_cond = bound_cond["throat_inlet_cond"]

                throat_outlet_cond = bound_cond["throat_outlet_cond"]

            # area_i=(imsize*resolution)**2/sum(pn['pore.radius'][pn[n]]**2*np.pi)
            condctivity = diffusion_coe_SO3(
                pn["pore.radius"], fluid["temperature"], 56, 28, 1, resolution
            )  ##
            if "solid" in m:
                dig[pn[n]] += condctivity[pn[n]]
                b[pn[n]] -= Boundary_condition[m][n][0] * condctivity[pn[n]]
            elif "pore" in m:
                if "inlet" in m:
                    condctivity[throat_inlet_cond[:, 0].astype(int)] = (
                        throat_inlet_cond[:, 1]
                    )  # if bound_cond!=False else condctivity[throat_inlet_cond[:,0].astype(int)]
                    dig[throat_inlet_cond[:, 0].astype(int)] += condctivity[
                        throat_inlet_cond[:, 0].astype(int)
                    ]
                    b[throat_inlet_cond[:, 0].astype(int)] -= (
                        Boundary_condition[m][n][0]
                        * condctivity[throat_inlet_cond[:, 0].astype(int)]
                    )
                elif "outlet" in m:
                    condctivity[throat_outlet_cond[:, 0].astype(int)] = (
                        throat_outlet_cond[:, 1]
                    )  # if bound_cond!=False else condctivity[throat_outlet_cond[:,0].astype(int)]
                    dig[throat_outlet_cond[:, 0].astype(int)] += condctivity[
                        throat_outlet_cond[:, 0].astype(int)
                    ]
                    b[throat_outlet_cond[:, 0].astype(int)] -= (
                        Boundary_condition[m][n][0]
                        * condctivity[throat_outlet_cond[:, 0].astype(int)]
                    )

    A.setdiag(-dig, 0)
    A = A.tocsr()
    Profile = spsolve(A, b)
    # Profile,j=ssl.bicg(A,b,tol=1e-9)
    return Profile


def stead_stay_alg(
    pn,
    fluid,
    coe_A,
    Boundary_condition,
    resolution,
    bound_cond=False,
    area_mode: str = "radius",
):
    """
    area mode choose the area calculation method
    """
    # num_pore = len(pn['pore.all'])

    num_pore = len(pn["pore.all"])
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_As = np.empty(shape=len(coe_A) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    coe_As[: len(coe_A)] = coe_A
    coe_As[len(coe_A) :] = coe_A
    A = coo_array((coe_As, (rows, cols)), shape=(num_pore, num_pore), dtype=np.float64)
    # A = csr_matrix((coe_A, (pn['throat.conns'][:, 1], pn['throat.conns'][:, 0])),
    #                shape=(num_pore, num_pore), dtype=np.float64)
    # A = (A.T + A).tolil()
    # dig = np.array(A.sum(axis=0)).reshape(num_pore)
    dig = np.asarray(A.sum(axis=0)).reshape(num_pore)
    b = np.zeros(num_pore)
    # B=A.toarray()
    # mean_gil=np.max(coe_A)
    # condctivity=H_P_fun(pn['pore.radius'],resolution,fluid['viscosity'])
    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            condctivity = H_P_fun(
                pn["pore.radius"], resolution, pn["pore.viscosity"]
            )  ##
            if not bound_cond:
                bound_cond = {}
                index = np.argwhere(pn[n])
                value = condctivity[index]
                bound_cond["throat_inlet_cond"] = np.column_stack(
                    (index.reshape(-1), value.reshape(-1))
                )
                bound_cond["throat_outlet_cond"] = np.column_stack(
                    (index.reshape(-1), value.reshape(-1))
                )
                throat_inlet_cond = bound_cond["throat_inlet_cond"]
                throat_outlet_cond = bound_cond["throat_outlet_cond"]
                bound_cond = False
            else:
                throat_inlet_cond = bound_cond["throat_inlet_cond"]

                throat_outlet_cond = bound_cond["throat_outlet_cond"]

            # area_i=(imsize*resolution)**2/sum(pn['pore.radius'][pn[n]]**2*np.pi)

            if "solid" in m:
                dig[pn[n]] += condctivity[pn[n]]
                b[pn[n]] -= Boundary_condition[m][n][0] * condctivity[pn[n]]
            elif "pore" in m:
                if "inlet" in m:
                    if Boundary_condition[m][n][1] == "Dirichlet":
                        condctivity[throat_inlet_cond[:, 0].astype(int)] = (
                            throat_inlet_cond[:, 1]
                        )  # if bound_cond!=False else condctivity[throat_inlet_cond[:,0].astype(int)]
                        dig[throat_inlet_cond[:, 0].astype(int)] += condctivity[
                            throat_inlet_cond[:, 0].astype(int)
                        ]
                        b[throat_inlet_cond[:, 0].astype(int)] -= (
                            Boundary_condition[m][n][0]
                            * condctivity[throat_inlet_cond[:, 0].astype(int)]
                        )
                    elif Boundary_condition[m][n][1] == "Neumann":
                        dig[throat_inlet_cond[:, 0].astype(int)] += 0
                        which_surface = n.split(".")[1]
                        area = (
                            np.pi * pn["pore.radius"][pn[n]] ** 2
                            if area_mode == "radius"
                            else pn["pore." + which_surface + "_area"][pn[n]]
                        )
                        b[throat_inlet_cond[:, 0].astype(int)] -= (
                            Boundary_condition[m][n][0] * area
                        )

                elif "outlet" in m:
                    if Boundary_condition[m][n][1] == "Dirichlet":
                        condctivity[throat_outlet_cond[:, 0].astype(int)] = (
                            throat_outlet_cond[:, 1]
                        )  # if bound_cond!=False else condctivity[throat_outlet_cond[:,0].astype(int)]
                        dig[throat_outlet_cond[:, 0].astype(int)] += condctivity[
                            throat_outlet_cond[:, 0].astype(int)
                        ]
                        b[throat_outlet_cond[:, 0].astype(int)] -= (
                            Boundary_condition[m][n][0]
                            * condctivity[throat_outlet_cond[:, 0].astype(int)]
                        )
                    elif Boundary_condition[m][n][1] == "Neumann":
                        dig[throat_inlet_cond[:, 0].astype(int)] += 0
                        b[throat_inlet_cond[:, 0].astype(int)] = 0
    A.setdiag(-dig, 0)
    A = A.tocsr()
    Profile = spsolve(A, b)
    # Profile,j=ssl.bicg(A,b,tol=1e-9)
    return Profile


def correct_pressure(pn, coe_A, Boundary_condition, resolution, S_term=False):
    num_pore = len(pn["pore.all"])
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_As = np.empty(shape=len(coe_A) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    coe_As[: len(coe_A)] = coe_A
    coe_As[len(coe_A) :] = coe_A
    A = coo_array((coe_As, (rows, cols)), shape=(num_pore, num_pore), dtype=np.float64)
    # A = csr_matrix((coe_A, (pn['throat.conns'][:, 1], pn['throat.conns'][:, 0])),
    #                shape=(num_pore, num_pore), dtype=np.float64)
    # A = (A.T + A).tolil()
    dig = -np.asarray(A.sum(axis=0)).reshape(num_pore)
    b = np.zeros(num_pore)

    A.setdiag(dig, 0)
    b = b + S_term
    A = A.tocsr()
    Profile = spsolve(A, b)
    # Profile,j=ssl.bicg(A,b,tol=1e-9)
    return Profile


def stead_stay_alg_convection(
    pn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    g_ij,
    P_profile,
    fluid,
    solid,
    imsize,
    resolution,
    side,
):
    num_node = len(pn["pore.all"])
    # Num = num_node // 25000
    # Num = 2 if Num < 2 else Num

    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_Bs = np.empty(shape=len(coe_B) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    coe_Bs[: len(coe_B)] = coe_B
    coe_Bs[len(coe_B) :] = coe_B
    A0 = coo_array(
        (coe_Bs, (rows, cols)), shape=(num_node, num_node), dtype=np.float64
    ).tocsr()
    # B = csr_matrix((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B
    A = coo_array(
        (coe_A, (pn["throat.conns"][:, 0], pn["throat.conns"][:, 1])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    AH = coo_array(
        (coe_A_i, (pn["throat.conns"][:, 1], pn["throat.conns"][:, 0])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    A1 = (AH + A).tocoo()
    A = (A0 - A1).tocoo()

    dig = np.asarray(A.sum(axis=0)).reshape(num_node)
    b = np.zeros(num_node)
    # B=A.toarray()
    # resulation=np.average(pn['throat.length'])
    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * solid["lambda"] * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Neumann":
                    value = (
                        imsize[0]
                        * imsize[1]
                        * resolution**2
                        / np.sum(
                            pn["pore.radius"][pn["pore.boundary_" + side + "_surface"]]
                            ** 2
                            * np.pi
                        )
                    )
                    T_conductivity_s = np.pi * pn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]] * 0
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi
                        * pn["pore.radius"] ** 2
                        / (resolution / solid["lambda"] + resolution / fluid["lambda"])
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig

            elif "pore" in m:
                # P_conductivity=H_P_fun(pn['pore.radius'],resolution,fluid['viscosity'])##
                # mass_calcu=lambda i:tool.mass_balance_conv(pn,g_ij,P_profile,i)
                P_conductivity = mass_balance_conv(
                    pn, g_ij, P_profile, pn["pore._id"][pn[n] & pn["pore.void"]]
                )
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * fluid["lambda"] * pn["pore.radius"] ** 2 / resolution
                    )
                if "inlet" in m:
                    convection_term = (
                        fluid["density"] * fluid["Cp"] * abs(P_conductivity)
                    )  # *(Boundary_condition_P[m][n][0]-P_profile[pn[n]&pn['pore.void']])
                    diffusion_term = T_conductivity_f[pn[n] & pn["pore.void"]]
                    tem_b = Boundary_condition[m][n][0] * (
                        convection_term + diffusion_term
                    )

                    tem_dig = T_conductivity_f[pn[n] & pn["pore.void"]]
                else:
                    tem_b = 0

                    tem_dig = (
                        fluid["density"] * fluid["Cp"] * abs(P_conductivity)
                    )  # *(P_profile[pn[n]&pn['pore.void']]-Boundary_condition_P[m][n][0]))
                dig[pn[n] & pn["pore.void"]] -= tem_dig
                b[pn[n] & pn["pore.void"]] += tem_b
        # boundary condition set shuold be discussed
    # t0 = time.time()
    A_c = A  # copy A
    # T_res=[]
    # Phase= op.phases.Water(pn=pn)

    # _----------------------------steady-state-------------------------------#

    A_c.setdiag(-dig, 0)
    A_c = A_c.tocsr()
    # Tem_c,j=ssl.bicgstab(A_c,b,tol=1e-8)
    Tem_c = spsolve(A_c, b)
    return Tem_c


def transient_temperature(
    pn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    x0,
    g_ij,
    P_profile,
    fluid,
    solid,
    imsize,
    resolution,
    time_step,
    delta_t,
    side,
    Phase,
):
    num_node = len(pn["pore.all"])
    # Num = num_node // 25000
    # Num = 2 if Num < 2 else Num

    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_Bs = np.empty(shape=len(coe_B) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    coe_Bs[: len(coe_B)] = coe_B
    coe_Bs[len(coe_B) :] = coe_B
    A0 = coo_array((coe_Bs, (rows, cols)), shape=(num_node, num_node)).tocsr()
    # B = csr_matrix((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B
    A = coo_array(
        (coe_A, (pn["throat.conns"][:, 0], pn["throat.conns"][:, 1])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    AH = coo_array(
        (coe_A_i, (pn["throat.conns"][:, 1], pn["throat.conns"][:, 0])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    A1 = (AH + A).tocoo()
    A = (A0 - A1).tocoo()

    dig = np.asarray(A.sum(axis=0)).reshape(num_node)
    b = np.zeros(num_node)
    # B=A.toarray()
    # resulation=np.average(pn['throat.length'])

    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * solid["lambda"] * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Neumann":
                    value = (
                        imsize[0]
                        * imsize[1]
                        * resolution**2
                        / np.sum(
                            pn["pore.radius"][pn["pore.boundary_" + side + "_surface"]]
                            ** 2
                            * np.pi
                        )
                    )
                    T_conductivity_s = np.pi * pn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]] * 0
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi
                        * pn["pore.radius"] ** 2
                        / (resolution / solid["lambda"] + resolution / fluid["lambda"])
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
            elif "pore" in m:
                # P_conductivity=H_P_fun(pn['pore.radius'],resolution,fluid['viscosity'])##
                P_conductivity = mass_balance_conv(pn, g_ij, P_profile, pn["pore._id"])

                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * fluid["lambda"] * pn["pore.radius"] ** 2 / resolution
                    )
                if "inlet" in m:
                    convection_term = (
                        fluid["density"]
                        * fluid["Cp"]
                        * abs(P_conductivity[pn[n] & pn["pore.void"]])
                    )  # *(Boundary_condition_P[m][n][0]-P_profile[pn[n]&pn['pore.void']])
                    diffusion_term = T_conductivity_f[pn[n] & pn["pore.void"]]
                    tem_b = Boundary_condition[m][n][0] * (
                        convection_term + diffusion_term
                    )

                    tem_dig = T_conductivity_f[pn[n] & pn["pore.void"]]
                else:
                    tem_b = 0

                    tem_dig = (
                        fluid["density"]
                        * fluid["Cp"]
                        * abs(P_conductivity[pn[n] & pn["pore.void"]])
                    )  # *(P_profile[pn[n]&pn['pore.void']]-Boundary_condition_P[m][n][0]))
                dig[pn[n] & pn["pore.void"]] -= tem_dig
                b[pn[n] & pn["pore.void"]] += tem_b
        # boundary condition set shuold be discussed
    # t0 = time.time()
    A_c = A  # copy A
    # T_res=[]
    # Phase= op.phases.Water(pn=pn)

    # _----------------------------steady-state-------------------------------#
    T_res = []
    # Phase= op.phases.Water(pn=pn)

    for i in np.arange(time_step):
        delta_dig = (
            fluid["density"]
            * fluid["Cp"]
            * pn["pore.volume"]
            / delta_t
            * pn["pore.void"]
            + solid["density"]
            * solid["Cp"]
            * pn["pore.volume"]
            / delta_t
            * pn["pore.solid"]
        )
        dig_c = (
            dig - delta_dig
        )  # fluid_density*fluid_Cp*pn['pore.radius']**3*4/3/delta_t*pn['pore.void']-solid_density*solid_Cp*pn['pore.radius']**3*4/3/delta_t*pn['pore.solid']
        # diagonal should update for delta_t
        A_c.setdiag(-dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = b + delta_dig * x0
        # fluid_density*fluid_Cp*pn['pore.radius']**3*4/3*x0/delta_t #update b array for previous time step
        # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9) # calculate the temperature profile
        Tem_c = spsolve(A_c, b_c)
        x0 = Tem_c.astype(np.float16)  # update
        T_res.append(Tem_c)
        print(max(Tem_c), min(Tem_c))
        Phase["pore.temperature"] = Tem_c
        network2vtk(pn=pn, filename="./_{}".format(i))
        # op.io.VTK.export_data(pn=pn, phases=Phase, filename='./_{}'.format(i))
    return T_res


def transient_temperature_single(
    pn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    x0,
    g_ij,
    P_profile,
    fluid,
    solid,
    imsize,
    resolution,
    delta_t,
    side,
):
    num_node = len(pn["pore.all"])
    # Num = min((num_node // 25000) + 1, 10)
    # Num=2 if Num <2 else Num
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_Bs = np.empty(shape=len(coe_B) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    coe_Bs[: len(coe_B)] = coe_B
    coe_Bs[len(coe_B) :] = coe_B
    A0 = coo_array(
        (coe_Bs, (rows, cols)), shape=(num_node, num_node), dtype=np.float64
    ).tocsr()
    # B = coo_array((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B
    A = coo_array(
        (coe_A, (pn["throat.conns"][:, 0], pn["throat.conns"][:, 1])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    AH = coo_array(
        (coe_A_i, (pn["throat.conns"][:, 1], pn["throat.conns"][:, 0])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    A1 = (AH + A).tocoo()
    A = (A0 - A1).tocoo()

    dig = np.asarray(A.sum(axis=0)).reshape(num_node)
    b = np.zeros(num_node)
    # B=A.toarray()
    # resulation=np.average(pn['throat.length'])

    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * solid["lambda"] * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Neumann":
                    value = (
                        imsize[0]
                        * imsize[1]
                        * resolution**2
                        / np.sum(
                            pn["pore.radius"][pn["pore.boundary_" + side + "_surface"]]
                            ** 2
                            * np.pi
                        )
                    )
                    T_conductivity_s = np.pi * pn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]] * 0
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi
                        * pn["pore.radius"] ** 2
                        / (resolution / solid["lambda"] + resolution / fluid["lambda"])
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    b[pn[n] & pn["pore.solid"]] += b_dig
            elif "pore" in m:
                # P_conductivity=H_P_fun(pn['pore.radius'],resolution,fluid['viscosity'])##
                P_conductivity = mass_balance_conv(
                    pn, g_ij, P_profile, pn["pore._id"][pn[n] & pn["pore.void"]]
                )
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * fluid["lambda"] * pn["pore.radius"] ** 2 / resolution
                    )
                if "inlet" in m:
                    convection_term = (
                        fluid["density"] * fluid["Cp"] * abs(P_conductivity)
                    )  # *(Boundary_condition_P[m][n][0]-P_profile[pn[n]&pn['pore.void']])
                    diffusion_term = T_conductivity_f[pn[n] & pn["pore.void"]]
                    tem_b = Boundary_condition[m][n][0] * (
                        convection_term + diffusion_term
                    )

                    tem_dig = T_conductivity_f[pn[n] & pn["pore.void"]]
                else:
                    tem_b = 0

                    tem_dig = (
                        fluid["density"] * fluid["Cp"] * abs(P_conductivity)
                    )  # *(P_profile[pn[n]&pn['pore.void']]-Boundary_condition_P[m][n][0]))
                dig[pn[n] & pn["pore.void"]] -= tem_dig
                b[pn[n] & pn["pore.void"]] += tem_b
        # boundary condition set shuold be discussed
    # t0 = time.time()
    A_c = A  # .copy()  # copy A
    # T_res=[]
    # Phase= op.phases.Water(pn=pn)

    # _----------------------------steady-state-------------------------------#
    T_res = []
    # Phase= op.phases.Water(pn=pn)

    delta_dig = (
        fluid["density"] * fluid["Cp"] * pn["pore.volume"] / delta_t * pn["pore.void"]
        + solid["density"]
        * solid["Cp"]
        * pn["pore.volume"]
        / delta_t
        * pn["pore.solid"]
    )
    dig_c = (
        dig - delta_dig
    )  # fluid_density*fluid_Cp*pn['pore.radius']**3*4/3/delta_t*pn['pore.void']-solid_density*solid_Cp*pn['pore.radius']**3*4/3/delta_t*pn['pore.solid']
    # diagonal should update for delta_t
    A_c.setdiag(-dig_c, 0)  # add diagonal into A_c
    A_c = A_c.tocsr()  # we transfer A_c for next calculation

    b_c = b + delta_dig * x0
    # fluid_density*fluid_Cp*pn['pore.radius']**3*4/3*x0/delta_t #update b array for previous time step
    # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9) # calculate the temperature profile
    Tem_c = spsolve(A_c, b_c)
    x0 = Tem_c.astype(np.float16)  # update
    T_res.append(Tem_c)
    # print(max(Tem_c),min(Tem_c))

    return Tem_c


def transient_temperature_s(
    pn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    x0,
    g_ij,
    P_profile,
    fluid,
    solid,
    imsize,
    resolution,
    delta_t,
    side,
):
    num_node = len(pn["pore.all"])
    Num = 1
    # Num=2 if Num <2 else Num
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_Bs = np.empty(shape=len(coe_B) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    coe_Bs[: len(coe_B)] = coe_B
    coe_Bs[len(coe_B) :] = coe_B
    A0 = coo_array(
        (coe_Bs, (rows, cols)), shape=(num_node, num_node), dtype=np.float64
    ).tocsr()
    # B = csr_matrix((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B
    A = coo_array(
        (coe_A, (pn["throat.conns"][:, 0], pn["throat.conns"][:, 1])),
        shape=(num_node, num_node),
        dtype=np.float64,
    )
    AH = coo_array(
        (coe_A_i, (pn["throat.conns"][:, 1], pn["throat.conns"][:, 0])),
        shape=(num_node, num_node),
        dtype=np.float64,
    )
    A1 = (AH + A).tocoo()
    A = (A0 - A1).tocoo()

    dig = -np.asarray(A.sum(axis=0)).reshape(num_node)
    b = np.zeros(num_node)
    # B=A.toarray()
    # resulation=np.average(pn['throat.length'])
    dig, b = setting_Boundary_condition(
        pn,
        g_ij,
        P_profile,
        dig,
        b,
        Boundary_condition,
        resolution,
        imsize,
        Num,
        side,
        "heat",
    )

    # boundary condition set shuold be discussed
    # t0 = time.time()
    A_c = A  # .copy()  # copy A
    # T_res=[]
    # Phase= op.phases.Water(pn=pn)

    # _----------------------------steady-state-------------------------------#
    T_res = []
    # Phase= op.phases.Water(pn=pn)

    delta_dig = pn["pore.density"] * pn["pore.Cp"] * pn["pore.volume"] / delta_t
    dig_c = (
        dig + delta_dig
    )  # fluid_density*fluid_Cp*pn['pore.radius']**3*4/3/delta_t*pn['pore.void']-solid_density*solid_Cp*pn['pore.radius']**3*4/3/delta_t*pn['pore.solid']

    # diagonal should update for delta_t
    A_c.setdiag(dig_c, 0)  # add diagonal into A_c
    A_c = A_c.tocsr()  # we transfer A_c for next calculation

    b_c = b + delta_dig * x0
    # fluid_density*fluid_Cp*pn['pore.radius']**3*4/3*x0/delta_t #update b array for previous time step
    # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9) # calculate the temperature profile
    Tem_c = spsolve(A_c, b_c)

    T_res.append(Tem_c)
    # print(max(Tem_c),min(Tem_c))

    return Tem_c


def setting_Boundary_condition(
    pn,
    g_ij,
    P_profile,
    dig,
    b,
    Boundary_condition,
    resolution,
    imsize,
    Num,
    side,
    type_f,
):
    if type_f == "diffusion":
        alpha = 1
        cond = pn["pore.diffusivity"]
    elif type_f == "heat":
        alpha = pn["pore.density"] * pn["pore.Cp"]
        cond = pn["pore.lambda"]
    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )

                elif Boundary_condition[m][n][1] == "Neumann":
                    value = getting_zoom_value(pn, side, imsize, resolution)
                    T_conductivity_s = np.pi * pn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]] * 0
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                dig[pn[n] & pn["pore.solid"]] += tem_dig
                b[pn[n] & pn["pore.solid"]] += b_dig
            elif "pore" in m:
                # P_conductivity=H_P_fun(pn['pore.radius'],resolution,fluid['viscosity'])##
                P_conductivity = mass_balance_conv(
                    pn, g_ij, P_profile, pn["pore._id"][pn[n] & pn["pore.void"]]
                )
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_f = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )

                if "inlet" in m:
                    convection_term = abs(
                        alpha[pn[n] & pn["pore.void"]] * P_conductivity
                    )  # *(Boundary_condition_P[m][n][0]-P_profile[pn[n]&pn['pore.void']])
                    diffusion_term = T_conductivity_f[pn[n] & pn["pore.void"]]
                    tem_b = Boundary_condition[m][n][0] * (
                        convection_term + diffusion_term
                    )
                    tem_dig = T_conductivity_f[pn[n] & pn["pore.void"]]
                else:
                    tem_b = 0
                    tem_dig = abs(
                        alpha[pn[n] & pn["pore.void"]] * P_conductivity
                    )  # *(P_profile[pn[n]&pn['pore.void']]-Boundary_condition_P[m][n][0]))
                dig[pn[n] & pn["pore.void"]] += tem_dig
                b[pn[n] & pn["pore.void"]] += tem_b
    return dig, b


def transient_energy_s(
    pn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    x0,
    g_ij,
    P_profile,
    imsize,
    resolution,
    delta_t,
    side,
    type_f="heat",
):
    num_node = len(pn["pore.all"])
    # Num = max((num_node // 25000), 2)
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_Bs = np.empty(shape=len(coe_B) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    coe_Bs[: len(coe_B)] = coe_B
    coe_Bs[len(coe_B) :] = coe_B
    A0 = coo_array(
        (coe_Bs, (rows, cols)), shape=(num_node, num_node), dtype=np.float64
    ).tocsr()
    # B = csr_matrix((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B

    A = coo_array(
        (coe_A, (pn["throat.conns"][:, 0], pn["throat.conns"][:, 1])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    AH = coo_array(
        (coe_A_i, (pn["throat.conns"][:, 1], pn["throat.conns"][:, 0])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    A1 = (AH + A).tocoo()
    A = (A0 - A1).tocoo()

    alpha = pn["pore.density"] * pn["pore.Cp"]
    cond = pn["pore.lambda"]
    alpha_ = pn["pore.Cp"]
    # dig,b=algorithm.setting_Boundary_condition(pn,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
    dig = -np.asarray(A.sum(axis=0)).reshape(num_node)
    b = np.zeros(num_node)
    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    dig[pn[n] & pn["pore.solid"]] += tem_dig
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Neumann":
                    value = getting_zoom_value(pn, side, imsize, resolution)
                    T_conductivity_s = np.pi * pn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]] * 0
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi * pn["pore.radius"] ** 2 / (resolution / cond)
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    dig[pn[n] & pn["pore.solid"]] += tem_dig
                    b[pn[n] & pn["pore.solid"]] += b_dig
            elif "pore" in m:
                # P_conductivity=H_P_fun(pn['pore.radius'],resolution,fluid['viscosity'])##
                P_conductivity = mass_balance_conv(
                    pn, g_ij, P_profile, pn["pore._id"][pn[n] & pn["pore.void"]]
                )
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )

                    if "inlet" in m:
                        convection_term = abs(
                            alpha_[pn[n] & pn["pore.void"]] * P_conductivity
                        )  # *(Boundary_condition_P[m][n][0]-P_profile[pn[n]&pn['pore.void']])

                        diffusion_term = T_conductivity_f[pn[n] & pn["pore.void"]]
                        tem_b = Boundary_condition[m][n][0] * (
                            convection_term + diffusion_term
                        )
                        tem_dig = T_conductivity_f[pn[n] & pn["pore.void"]]
                    else:
                        tem_b = 0
                        tem_dig = abs(
                            alpha_[pn[n] & pn["pore.void"]] * P_conductivity
                        )  # *(P_profile[pn[n]&pn['pore.void']]-Boundary_condition_P[m][n][0]))
                elif Boundary_condition[m][n][1] == "Neumann":
                    if "inlet" in m:
                        tem_dig = 0
                        tem_b = (
                            Boundary_condition[m][n][0]
                            * pn["pore.radius"][pn[n] & pn["pore.void"]] ** 2
                            * np.pi
                        )
                    else:
                        tem_dig = abs(alpha_[pn[n] & pn["pore.void"]] * P_conductivity)
                        tem_b = 0
                dig[pn[n] & pn["pore.void"]] += tem_dig
                b[pn[n] & pn["pore.void"]] += tem_b

    A_c = A  # .copy()  # copy A

    # _----------------------------steady-state-------------------------------#

    # Var_c=np.copy(x0)

    delta_dig = alpha * pn["pore.volume"] / delta_t
    dig_c = dig + delta_dig

    # diagonal should update for delta_t
    A_c.setdiag(dig_c, 0)  # add diagonal into A_c
    A_c = A_c.tocsr()  # we transfer A_c for next calculation

    b_c = b + delta_dig * x0

    Var_c = spsolve(A_c, b_c)
    # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9)

    return Var_c


def transient_model_test(
    pn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    x0,
    g_ij,
    P_profile,
    imsize,
    resolution,
    delta_t,
    side,
    mass_flow=False,
    S_term=False,
    Bound_cond_P=False,
    type_f="species",
):
    # func_pv=lambda c,T:8.314*T*c
    # func_ps=lambda T:611.21*np.exp((18.678-(T-273.15)/234.5)*((T-273.15)/(257.14+(T-273.15))))
    num_node = len(pn["pore.all"])
    # Num = max((num_node // 25000), 2)
    rows = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    cols = np.empty(shape=len(pn["throat.conns"]) * 2, dtype=int)
    coe_Bs = np.empty(shape=len(coe_B) * 2, dtype=float)
    rows[: len(pn["throat.conns"])] = pn["throat.conns"][:, 0]
    rows[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 1]
    cols[: len(pn["throat.conns"])] = pn["throat.conns"][:, 1]
    cols[len(pn["throat.conns"]) :] = pn["throat.conns"][:, 0]
    coe_Bs[: len(coe_B)] = coe_B
    coe_Bs[len(coe_B) :] = coe_B
    A0 = coo_array(
        (coe_Bs, (rows, cols)), shape=(num_node, num_node), dtype=np.float64
    ).tocsr()
    # B = csr_matrix((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B
    A = coo_array(
        (coe_A, (pn["throat.conns"][:, 0], pn["throat.conns"][:, 1])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    AH = coo_array(
        (coe_A_i, (pn["throat.conns"][:, 1], pn["throat.conns"][:, 0])),
        shape=(num_node, num_node),
        dtype=np.float64,
    ).tocsr()
    A1 = (AH + A).tocoo()
    A = (A0 - A1).tocoo()
    # RH= func_pv(1.0,302)/func_ps(302)
    # Bound_c=HAProps('C', 'T', 302, 'P', (1e5+400)/1000,'R',RH)*1000*1.29
    if type_f == "species":
        alpha = pn["pore.density"] * 0 + 1
        cond = pn["pore.diffusivity"]
    elif type_f == "heat":
        alpha = pn["pore.density"] * pn["pore.Cp"]
        cond = pn["pore.lambda"]
    elif type_f == "momentum":
        alpha = pn["pore.density"]
        cond = pn["pore.viscosity"]
    elif type_f == "density":
        alpha = pn["pore.density"] * 0 + 1
        cond = pn["pore.diffusivity"] * 0
        # dig,b=algorithm.setting_Boundary_condition(pn,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
    dig = -np.asarray(A.sum(axis=0)).reshape(num_node)
    b = np.zeros(num_node)
    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    dig[pn[n] & pn["pore.solid"]] += tem_dig
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Neumann":
                    value = getting_zoom_value(pn, side, imsize, resolution)
                    T_conductivity_s = np.pi * pn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]] * 0
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    dig[pn[n] & pn["pore.solid"]] -= tem_dig
                    b[pn[n] & pn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi * pn["pore.radius"] ** 2 / (resolution / cond)
                    )
                    tem_dig = T_conductivity_s[pn[n] & pn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[pn[n] & pn["pore.solid"]]
                    )
                    dig[pn[n] & pn["pore.solid"]] += tem_dig
                    b[pn[n] & pn["pore.solid"]] += b_dig
            elif "pore" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * cond * pn["pore.radius"] ** 2 / resolution
                    )
                    if Bound_cond_P:
                        P_g_ij = H_P_fun(
                            pn["pore.radius"], resolution, pn["pore.viscosity"]
                        )  ##
                        P_conductivity = (
                            Bound_cond_P[m][n][0] - P_profile[pn[n] & pn["pore.void"]]
                        ) * P_g_ij[pn[n] & pn["pore.void"]]
                    else:
                        P_conductivity = mass_balance_conv(
                            pn,
                            g_ij,
                            P_profile,
                            pn["pore._id"][pn[n] & pn["pore.void"]],
                        )
                    if "inlet" in m:
                        convection_term = np.abs(
                            alpha[pn[n] & pn["pore.void"]] * P_conductivity
                        )  # *(Boundary_condition_P[m][n][0]-P_profile[pn[n]&pn['pore.void']])

                        diffusion_term = T_conductivity_f[pn[n] & pn["pore.void"]]
                        tem_b = Boundary_condition[m][n][0] * (
                            convection_term + diffusion_term
                        )
                        tem_dig = T_conductivity_f[pn[n] & pn["pore.void"]]
                    else:
                        tem_b = 0
                        tem_dig = abs(
                            alpha[pn[n] & pn["pore.void"]] * P_conductivity
                        )  # *(P_profile[pn[n]&pn['pore.void']]-Boundary_condition_P[m][n][0]))
                elif Boundary_condition[m][n][1] == "Neumann":
                    P_conductivity = mass_balance_conv(
                        pn, g_ij, P_profile, pn["pore._id"][pn[n] & pn["pore.void"]]
                    )
                    if "inlet" in m:
                        tem_dig = 0
                        tem_b = (
                            Boundary_condition[m][n][0]
                            * pn["pore.radius"][pn[n] & pn["pore.void"]] ** 2
                            * np.pi
                        )
                    else:
                        tem_dig = abs(alpha[pn[n] & pn["pore.void"]] * P_conductivity)
                        tem_b = 0
                dig[pn[n] & pn["pore.void"]] += tem_dig
                b[pn[n] & pn["pore.void"]] += tem_b

    A_c = A  # .copy()  # copy A

    # _----------------------------steady-state-------------------------------#

    # Var_c=np.copy(x0)

    delta_dig = alpha * pn["pore.volume"] / delta_t
    dig_c = dig + delta_dig

    # diagonal should update for delta_t
    A_c.setdiag(dig_c, 0)  # add diagonal into A_c
    A_c = A_c.tocsr()  # we transfer A_c for next calculation

    b_c = b + delta_dig * x0 + S_term

    Var_c = spsolve(A_c, b_c)
    # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9)
    if type_f == "momentum":
        return Var_c, dig_c
    else:
        return Var_c


def transient_model(
    mpn,
    coe_A,
    coe_A_i,
    coe_B,
    Boundary_condition,
    x0,
    g_ij,
    P_profile,
    imsize,
    resolution,
    delta_t,
    side,
    type_f="species",
):
    num_throat = mpn["throat.conns"].shape[0]
    mpn_throat_conns_col_0 = np.ascontiguousarray(mpn["throat.conns"][:, 0])
    mpn_throat_conns_col_1 = np.ascontiguousarray(mpn["throat.conns"][:, 1])
    mpn_pore__id = mpn["pore._id"]
    num_pore = mpn_pore__id.size

    num_nondig = num_throat * 4
    num_total = num_nondig + num_pore

    rows = np.empty(shape=num_total, dtype=np.int64)
    rows[:num_throat] = mpn_throat_conns_col_0
    rows[num_throat : num_throat * 2] = mpn_throat_conns_col_1
    rows[num_throat * 2 : num_throat * 3] = mpn_throat_conns_col_0
    rows[num_throat * 3 : num_nondig] = mpn_throat_conns_col_1
    rows[num_nondig:] = mpn_pore__id

    cols = np.empty(shape=num_total, dtype=np.int64)
    cols[:num_throat] = mpn_throat_conns_col_1
    cols[num_throat : num_throat * 2] = mpn_throat_conns_col_0
    cols[num_throat * 2 : num_throat * 3] = mpn_throat_conns_col_1
    cols[num_throat * 3 : num_nondig] = mpn_throat_conns_col_0
    cols[num_nondig:] = mpn_pore__id

    coefficients = np.empty(shape=num_total, dtype=np.float64)
    coefficients[:num_throat] = -coe_B
    coefficients[num_throat : num_throat * 2] = -coe_B
    coefficients[num_throat * 2 : num_throat * 3] = coe_A
    coefficients[num_throat * 3 : num_nondig] = coe_A_i

    # B = csr_matrix((coe_B, (pn['throat.conns'][:, 0], pn['throat.conns'][:, 1])),
    #                shape=(num_node, num_node), dtype=np.float64)
    # A0 = (B.T + B).tolil()
    # del B

    if type_f == "species":
        alpha = np.ones_like(mpn["pore.density"])
        beta = mpn["pore.density"]
        cond = mpn["pore.diffusivity"]
    elif type_f == "heat":
        alpha = mpn["pore.Cp"]
        beta = mpn["pore.density"] * mpn["pore.Cp"]
        cond = mpn["pore.lambda"]
    elif type_f == "density":
        alpha = np.ones_like(mpn["pore.density"])
        beta = np.ones_like(mpn["pore.density"])
        cond = mpn["pore.diffusivity"] * 0
    elif type_f == "fraction":
        alpha = np.ones_like(mpn["pore.density"])
        beta = np.ones_like(mpn["pore.density"])
        cond = mpn["pore.diffusivity"]
    elif type_f == "concentration":
        alpha = np.ones_like(mpn["pore.density"])  # *mpn['pore.volume']
        beta = np.ones_like(mpn["pore.density"])  # /mpn['pore.volume']
        cond = mpn["pore.diffusivity"]

        # dig,b=algorithm.setting_Boundary_condition(pn,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
    dig = np.zeros(num_pore)
    np.add.at(dig, cols[:num_nondig], coefficients[:num_nondig])
    b = np.zeros(num_pore)
    for m in Boundary_condition:
        for n in Boundary_condition[m]:
            if "solid" in m:
                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_s = (
                        np.pi * cond * mpn["pore.radius"] ** 2 / resolution
                    )
                    tem_dig = T_conductivity_s[mpn[n] & mpn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[mpn[n] & mpn["pore.solid"]]
                    )
                    dig[mpn[n] & mpn["pore.solid"]] += tem_dig
                    b[mpn[n] & mpn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Neumann":
                    value = getting_zoom_value(mpn, side, imsize, resolution)
                    T_conductivity_s = np.pi * mpn["pore.radius"] ** 2 * value
                    tem_dig = T_conductivity_s[mpn[n] & mpn["pore.solid"]] * 0
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[mpn[n] & mpn["pore.solid"]]
                    )
                    dig[mpn[n] & mpn["pore.solid"]] -= tem_dig
                    b[mpn[n] & mpn["pore.solid"]] += b_dig
                elif Boundary_condition[m][n][1] == "Robin":
                    T_conductivity_s = (
                        np.pi * mpn["pore.radius"] ** 2 / (resolution / cond)
                    )
                    tem_dig = T_conductivity_s[mpn[n] & mpn["pore.solid"]]
                    b_dig = (
                        Boundary_condition[m][n][0]
                        * T_conductivity_s[mpn[n] & mpn["pore.solid"]]
                    )
                    dig[mpn[n] & mpn["pore.solid"]] += tem_dig
                    b[mpn[n] & mpn["pore.solid"]] += b_dig
            elif "pore" in m:
                # P_conductivity=H_P_fun(mpn['pore.radius'],resolution,fluid['viscosity'])##
                P_conductivity = mass_balance_conv(
                    mpn, g_ij, P_profile, mpn["pore._id"][mpn[n] & mpn["pore.void"]]
                )

                if Boundary_condition[m][n][1] == "Dirichlet":
                    T_conductivity_f = (
                        np.pi * cond * mpn["pore.radius"] ** 2 / resolution
                    )
                    if "inlet" in m:
                        convection_term = abs(
                            alpha[mpn[n] & mpn["pore.void"]] * P_conductivity
                        )  # *(Boundary_condition_P[m][n][0]-P_profile[mpn[n]&mpn['pore.void']])
                        diffusion_term = T_conductivity_f[mpn[n] & mpn["pore.void"]]
                        tem_b = Boundary_condition[m][n][0] * (
                            convection_term + diffusion_term
                        )
                        tem_dig = T_conductivity_f[mpn[n] & mpn["pore.void"]]
                    else:
                        tem_b = 0
                        tem_dig = abs(
                            alpha[mpn[n] & mpn["pore.void"]] * P_conductivity
                        )  # *(P_profile[mpn[n]&mpn['pore.void']]-Boundary_condition_P[m][n][0]))
                elif Boundary_condition[m][n][1] == "Neumann":
                    if "inlet" in m:
                        tem_dig = 0
                        tem_b = (
                            Boundary_condition[m][n][0]
                            * mpn["pore.radius"][mpn[n] & mpn["pore.void"]] ** 2
                            * np.pi
                        )
                    else:
                        tem_dig = abs(alpha[mpn[n] & mpn["pore.void"]] * P_conductivity)
                        tem_b = 0
                dig[mpn[n] & mpn["pore.void"]] += tem_dig
                b[mpn[n] & mpn["pore.void"]] += tem_b

    # _----------------------------steady-state-------------------------------#

    # Var_c=np.copy(x0)

    delta_dig = beta * mpn["pore.volume"] / delta_t
    dig += delta_dig
    coefficients[num_nondig:] = dig

    A = csr_array(
        (coefficients, (rows, cols)), shape=(num_pore, num_pore), dtype=np.float64
    )

    b += delta_dig * x0

    Var = spsolve(A, b)
    # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9)

    return Var


def RK4(h, a, a_max, initial):
    def func_n(T):
        return 1 / (
            1 / 2.976 + 0.377 * (1 - 293.15 / T)
        )  # n0=1,alph=0 for Langmuir-Freundirch

    def func_b(T):
        return 4.002 * np.exp(
            51800 / 8.314 / 273.15 * (293.15 / T - 1)
        )  # b0=4.002, delta_E=65572,R=8.314,T0=273.15,for Langmuir-Freundirch

    def func_q_eq(c, T):
        return (
            19
            * ((func_b(T) * c * 8.314 * T) ** (1 / func_n(T)))
            / (1 + ((func_b(T) * c * 8.314 * T) ** (1 / func_n(T))))
        )  # for Langmuir-Freundirch

    def func_q_eq_pw(c, T):
        return (
            19
            * (func_b(T) * c * 8.314 * T) ** (1 / func_n(T) - 1)
            / (func_n(T) * (1 + ((func_b(T) * c * 8.314 * T) ** (1 / func_n(T)))) ** 2)
        )

    def func_k(c, T):
        return 7 / 1152 / 8.314 / T / func_q_eq_pw(c, T)

    def func_delta_H(T, q):
        return 65572 - 0.377 * 8.314 * 293.15 * func_n(T) ** 2 * np.log(q / (19 - q))

    def func_f(c, T, q):
        res = func_k(c, T) * (func_q_eq(c, T) - q)
        return res

    def func_g(c, T, q):
        res = -func_k(c, T) * (func_q_eq(c, T) - q) * 1152 * 18 / 0.58
        return res

    def func_r(c, T, q):
        res = func_f(c, T, q) * func_delta_H(T, q) / 880
        return res

    xarray = []
    yarray = []
    zarray = []
    array = []
    x, y, z = initial[0], initial[1], initial[2]
    while a < a_max:
        array.append(a)
        xarray.append(x)
        yarray.append(y)
        zarray.append(z)

        a += h
        f1, g1, r1 = func_f(x, y, z), func_g(x, y, z), func_r(x, y, z)
        m1, n1, l1 = x + f1 * h / 2, y + g1 * h / 2, z + r1 * h / 2

        f2, g2, r2 = func_f(m1, n1, l1), func_g(m1, n1, l1), func_r(m1, n1, l1)
        m2, n2, l2 = x + f2 * h / 2, y + g2 * h / 2, z + r2 * h / 2

        f3, g3, r3 = func_f(m2, n2, l2), func_g(m2, n2, l2), func_r(m2, n2, l2)
        m3, n3, l3 = x + f3 * h / 2, y + g3 * h / 2, z + r3 * h / 2

        f4, g4, r4 = func_f(m3, n3, l3), func_g(m3, n3, l3), func_r(m3, n3, l3)

        x = x + (f1 + 2 * f2 + 2 * f3 + f4) * h / 6
        y = y + (g1 + 2 * g2 + 2 * g3 + g4) * h / 6
        z = z + (r1 + 2 * r2 + 2 * r3 + r4) * h / 6
    return xarray, yarray, zarray, array
