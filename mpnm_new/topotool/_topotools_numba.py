import numba as nb
import numpy as np
from ..enum import Pore_Types, Throat_Types

none_array = np.array(((-1,),), dtype=np.int64)

Throat_Types_void = int(Throat_Types.void)
Throat_Types_solid = int(Throat_Types.solid)
Throat_Types_interface = int(Throat_Types.interface)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_find_neighbor_ball_info(inner_info, inner_start2end, _id):
    if inner_start2end[_id, 0] == -1:
        void = none_array
        solid = none_array
        interface = none_array
        return void, solid, interface
    else:
        inner_info = inner_info[inner_start2end[_id, 0] : inner_start2end[_id, 1]]
        inner_info_throat_type = inner_info[:, 3]  # np.ascontiguousarray()

        counts = np.bincount(inner_info_throat_type)
        split_indices = np.empty(counts.size + 1, dtype=np.int64)
        split_indices[0] = 0
        split_indices[1:] = np.cumsum(counts)

        void_start, void_end = split_indices[0], split_indices[1]
        solid_start, solid_end = split_indices[2], split_indices[3]
        interface_start, interface_end = split_indices[4], split_indices[5]

        void = inner_info[void_start:void_end]
        solid = inner_info[solid_start:solid_end]
        interface = inner_info[interface_start:interface_end]
        if void.size == 0:
            void = none_array
        if solid.size == 0:
            solid = none_array
        if interface.size == 0:
            interface = none_array
        return void, solid, interface


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_find_neighbor_ball(mpn_pore_void, mpn_pore_solid, mpn_throat_conns, ids):
    """
    Pore_type 0: void, 1: solid, 2: interface
    """
    throat_conns = np.argwhere(mpn_throat_conns == ids)
    if len(throat_conns) != 0:
        pore_neighbor_throat_type = np.zeros((len(throat_conns), 4), dtype=np.int64)
        for i in range(len(throat_conns)):
            current_throat = throat_conns[i, 0]
            if throat_conns[i, 1] == 0:
                pore_neighbor = mpn_throat_conns[current_throat, 1]
            else:
                pore_neighbor = mpn_throat_conns[current_throat, 0]
            if mpn_pore_void[ids]:
                if mpn_pore_void[pore_neighbor]:
                    pore_neighbor_throat_type[i, 3] = 0
                else:
                    pore_neighbor_throat_type[i, 3] = 2
            else:
                if mpn_pore_solid[pore_neighbor]:
                    pore_neighbor_throat_type[i, 3] = 1
                else:
                    pore_neighbor_throat_type[i, 3] = 2
            pore_neighbor_throat_type[i, 0] = ids
            pore_neighbor_throat_type[i, 1] = pore_neighbor
            pore_neighbor_throat_type[i, 2] = current_throat
        void = pore_neighbor_throat_type[pore_neighbor_throat_type[:, 3] == 0]
        solid = pore_neighbor_throat_type[pore_neighbor_throat_type[:, 3] == 1]
        interface = pore_neighbor_throat_type[pore_neighbor_throat_type[:, 3] == 2]
        if len(void) == 0:
            void = none_array
        if len(solid) == 0:
            solid = none_array
        if len(interface) == 0:
            interface = none_array
        return [void, solid, interface]
    else:
        void = none_array
        solid = none_array
        interface = none_array
        return [void, solid, interface]


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_species_balance_conv(
    inner_info,
    inner_start2end,
    mpn_throat_radius,
    mpn_throat_length,
    g_ij,
    Tem,
    thermal_con_dual,
    P_profile,
    ids,
):
    result = np.zeros((ids.size, 5), dtype=np.float64)
    # res=find_neighbor_ball(network,[a])
    for i in nb.prange(ids.size):
        pore_id = ids[i]
        void, solid, interface = nb_find_neighbor_ball_info(
            inner_info, inner_start2end, pore_id
        )
        coe_B = mpn_throat_radius**2 * np.pi / mpn_throat_length * thermal_con_dual
        # Void
        if void[0, 0] != -1:
            # 先提取需要的变量
            P0 = P_profile[void[:, 0]]
            P1 = P_profile[void[:, 1]]
            T0 = Tem[void[:, 0]]
            T1 = Tem[void[:, 1]]
            cond_f = g_ij[void[:, 2]]
            cond_h = coe_B[void[:, 2]]
            # 计算压差和流量
            delta_p = P0 - P1
            flux = delta_p * cond_f
            # 计算对流热量
            h_conv_f = flux * np.where(delta_p > 0, T0, T1)
            # 计算传导热量
            h_cond_f = np.sum((T0 - T1) * cond_h)
            # 对流总热量
            h_conv_f = np.sum(h_conv_f)
        else:
            h_conv_f = 0.0
            h_cond_f = 0.0
        # cp_data,density=np.array(cp_data),np.array(density)
        # Solid
        if solid[0, 0] != -1:
            # 提取需要的变量
            T0_s = Tem[solid[:, 0]]
            T1_s = Tem[solid[:, 1]]
            cond_h_s = coe_B[solid[:, 2]]
            # 计算传导热量
            h_cond_s = np.sum((T0_s - T1_s) * cond_h_s)
        else:
            h_cond_s = 0.0
        # Interface
        if interface[0, 0] != -1:
            # 提取需要的变量
            T0_sf = Tem[interface[:, 0]]
            T1_sf = Tem[interface[:, 1]]
            cond_hs = coe_B[interface[:, 2]]
            # 计算传导热量
            h_cond_sf = np.sum((T0_sf - T1_sf) * cond_hs)
        else:
            h_cond_sf = 0.0
        total = h_conv_f + h_cond_f + h_cond_sf + h_cond_s
        result[i, 0] = total
        result[i, 1] = h_conv_f
        result[i, 2] = h_cond_f
        result[i, 3] = h_cond_sf
        result[i, 4] = h_cond_s
    return result


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_energy_balance_conv(
    inner_info,
    inner_start2end,
    mpn_throat_Cp,
    mpn_throat_density,
    g_ij,
    Tem,
    coe_B,
    P_profile,
    ids,
):
    result = np.empty((ids.size, 5), dtype=np.float64)
    for i in nb.prange(ids.size):
        pore_id = ids[i]
        void, solid, interface = nb_find_neighbor_ball_info(
            inner_info, inner_start2end, pore_id
        )
        # Void
        if void[0, 0] != -1:
            throat_id = void[:, 2]
            pore_id_0 = void[:, 0]
            pore_id_1 = void[:, 1]

            cond_f = g_ij[throat_id]
            cond_h = coe_B[throat_id]
            cp_data = mpn_throat_Cp[throat_id]
            density = mpn_throat_density[throat_id]

            Temp_f_0 = Tem[pore_id_0]
            Temp_f_1 = Tem[pore_id_1]  # 分别提取 Temp_f 的两列
            delta_p = P_profile[pore_id_0] - P_profile[pore_id_1]

            flux = delta_p * cond_f
            h_conv_f = np.sum(
                flux * cp_data * density * np.where(delta_p > 0, Temp_f_0, Temp_f_1)
            )  # 使用 where 来处理温度
            h_cond_f = np.sum((Temp_f_0 - Temp_f_1) * cond_h)
        else:
            h_conv_f = 0.0
            h_cond_f = 0.0
        # Solid
        if solid[0, 0] != -1:
            cond_h_s = coe_B[solid[:, 2]]
            h_cond_s = np.sum((Tem[solid[:, 0]] - Tem[solid[:, 1]]) * cond_h_s)
        else:
            h_cond_s = 0.0
        # Interface
        if interface[0, 0] != -1:
            cond_hs = coe_B[interface[:, 2]]
            h_cond_sf = np.sum((Tem[interface[:, 0]] - Tem[interface[:, 1]]) * cond_hs)
        else:
            h_cond_sf = 0.0
        # Total heat transfer
        total = h_conv_f + h_cond_f + h_cond_sf + h_cond_s
        result[i, 0] = total
        result[i, 1] = h_conv_f
        result[i, 2] = h_cond_f
        result[i, 3] = h_cond_sf
        result[i, 4] = h_cond_s
    return result  # - inlet + outlet


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_mass_balance_conv(inner_info, inner_start2end, g_ij, P_profile, ids):
    result = np.zeros(ids.size, dtype=np.float64)
    for i in nb.prange(ids.size):
        pore_id = ids[i]
        void, solid, interface = nb_find_neighbor_ball_info(
            inner_info, inner_start2end, pore_id
        )
        if void[0, 0] == -1:
            result[i] = 0.0
        else:
            delta_p = P_profile[void[:, 0]] - P_profile[void[:, 1]]
            flux = delta_p * g_ij[void[:, 2]]
            result[i] = np.sum(flux)

    return result


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_cal_pore_veloc(
    inner_info, inner_start2end, mpn_pore_radius, g_ij, P_profile, ids
):
    result = np.zeros(ids.size, dtype=np.float64)
    for i in nb.prange(ids.size):
        pore_id = ids[i]
        void, solid, interface = nb_find_neighbor_ball_info(
            inner_info, inner_start2end, pore_id
        )
        if void[0, 0] == -1:
            result[i] = 0.0
        else:
            cond_f = g_ij[void[:, 2]]
            delta_p = P_profile[void[:, 0]] - P_profile[void[:, 1]]
            flux = delta_p * cond_f
            flux_pos = 0  # out
            flux_neg_abs = 0
            for f in flux:
                if f > 0:
                    flux_pos += f
                else:
                    flux_neg_abs -= f

            flux = max(flux_pos, flux_neg_abs)
            # vel_p_m = flux / (mpn_pore_radius[pore_id] ** 2 / (16 * mpn_pore_real_shape_factor[pore_id]))*0.2347
            vel_p_m = flux / (mpn_pore_radius[pore_id] ** 2 * np.pi)

            # velocity = flux / area_t
            # momentum = velocity * np.abs(velocity)
            # momentum=flux*np.abs(flux)/area_t
            # momentum = flux * np.abs(flux) / area_t

            # flux = max(abs(np.sum(flux[flux > 0])), abs(np.sum(flux[flux < 0])))
            # momentum=max(abs(np.sum(momentum[momentum>0]+delta_p[delta_p>0]/network['pore.density'][a])),
            #             abs(np.sum(momentum[momentum<0]+delta_p[delta_p<0]/network['pore.density'][a])))
            # momentum = abs(abs(np.sum(momentum[momentum > 0])) - abs(np.sum(momentum[momentum < 0])))

            # vel_p_m = np.sqrt(
            #     momentum / (mpn_pore_radius[pore_id] ** 2 / 4 / mpn_pore_real_shape_factor[pore_id]))
            # vel_p = flux / (mpn_pore_radius[pore_id] ** 2 / 4 / mpn_pore_real_shape_factor[pore_id])

            # result[i] = [abs(vel_p), abs(vel_p_m)][1]
            result[i] = vel_p_m
    return result


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_calculate_pore_flux(inner_info, inner_start2end, g_ij, P_profile, ids):
    result = np.zeros(ids.size, dtype=np.float64)
    for i in nb.prange(ids.size):
        pore_id = ids[i]
        void, solid, interface = nb_find_neighbor_ball_info(
            inner_info, inner_start2end, pore_id
        )
        if void[0, 0] == -1:
            result[i] = 0.0
        else:
            delta_p = P_profile[void[:, 0]] - P_profile[void[:, 1]]
            cond_f = g_ij[void[:, 2]]
            flux = delta_p * cond_f
            flux_pos = 0
            flux_neg_abs = 0
            for f in flux:
                if f > 0:
                    flux_pos += f
                else:
                    flux_neg_abs -= f
            result[i] = max(flux_pos, flux_neg_abs)

    return result


# @nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
# def nb_cal_pore_veloc_v2(inner_info, inner_start2end,mpn_throat_area, mpn_pore_radius, mpn_pore_real_shape_factor,mpn_pore_coords, g_ij, P_profile, ids):
#     result = np.zeros(ids.size, dtype=np.float64)
#     for i in nb.prange(ids.size):
#         pore_id = ids[i]
#         void, solid, interface = nb_find_neighbor_ball_info(inner_info, inner_start2end, pore_id)
#         if void[0, 0] == -1:
#             result[i] = 0
#         else:
#             delta_p = P_profile[void[:, 0]] - P_profile[void[:, 1]]
#             cond_f = g_ij[void[:, 2]]
#             flux = delta_p * cond_f
#             if np.count_nonzero(flux> 0):
#                 coords_i = mpn_pore_coords[void[:, 0]]
#                 coords_j = mpn_pore_coords[void[:, 1]]
#                 coords_i = coords_i[flux>0]
#                 coords_j = coords_j[flux>0]
#                 x_ij = coords_i[:, 0] - coords_j[:, 0]
#                 y_ij = coords_i[:, 1] - coords_j[:, 1]
#                 z_ij = coords_i[:, 2] - coords_j[:, 2]
#                 L_ij = np.sqrt(x_ij ** 2 + y_ij ** 2 + z_ij ** 2)
#                 flux = flux[flux>0]
#                 vel = np.sqrt(np.sum(flux**4/L_ij**2*(x_ij**2+y_ij**2+z_ij**2))/np.sum(flux**2))
#             else:
#                 vel = 0
#             result[i] = vel
#             # print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
#     return result
