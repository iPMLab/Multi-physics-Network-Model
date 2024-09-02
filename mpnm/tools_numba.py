import numpy as np
import numba as nb
import os
num_threads=os.environ.get('num_threads')
if num_threads==None:
    pass
else:
    nb.set_num_threads(int(num_threads))

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def find_neighbor_ball_nb(network_pore_void, network_pore_solid, network_throat_conns, a):
    '''
    Pore_type 0: void, 1: solid, 2: interface
    '''
    throat_conns = np.argwhere(network_throat_conns == a)
    if len(throat_conns) != 0:
        pore_neighbor_throat_type = np.zeros((len(throat_conns), 4), dtype=np.int64)
        for i in nb.prange(len(throat_conns)):
            current_throat = throat_conns[i, 0]
            if throat_conns[i, 1] == 0:
                pore_neighbor = network_throat_conns[current_throat, 1]
            else:
                pore_neighbor = network_throat_conns[current_throat, 0]
            if network_pore_void[a]:
                if network_pore_void[pore_neighbor]:
                    pore_neighbor_throat_type[i, 3] = 0
                else:
                    pore_neighbor_throat_type[i, 3] = 2
            else:
                if network_pore_solid[pore_neighbor]:
                    pore_neighbor_throat_type[i, 3] = 1
                else:
                    pore_neighbor_throat_type[i, 3] = 2
            pore_neighbor_throat_type[i, 0] = a
            pore_neighbor_throat_type[i, 1] = pore_neighbor
            pore_neighbor_throat_type[i, 2] = current_throat
        # total = throat_conns[:, 0]
        void = pore_neighbor_throat_type[pore_neighbor_throat_type[:, 3] == 0]
        solid = pore_neighbor_throat_type[pore_neighbor_throat_type[:, 3] == 1]
        interface = pore_neighbor_throat_type[pore_neighbor_throat_type[:, 3] == 2]
        if len(void) == 0:
            void = np.array([[-1]])
        if len(solid) == 0:
            solid = np.array([[-1]])
        if len(interface) == 0:
            interface = np.array([[-1]])
        return [void, solid, interface]
    else:
        void = np.array([[-1]])
        solid = np.array([[-1]])
        interface = np.array([[-1]])
        return [void, solid, interface]


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def find_neighbor_ball_information_nb(pore_throat_conns, inner_start2end, a):
    if inner_start2end[a,0] == -1:
        void = np.array([[-1]])
        solid = np.array([[-1]])
        interface = np.array([[-1]])
        return [void, solid, interface]
    else:
        inner_info = pore_throat_conns[inner_start2end[a,0]:inner_start2end[a,1]]
        void = inner_info[inner_info[:, 3] == 0]
        solid = inner_info[inner_info[:, 3] == 1]
        interface = inner_info[inner_info[:, 3] == 2]
        if len(void) == 0:
            void = np.array([[-1]])
        if len(solid) == 0:
            solid = np.array([[-1]])
        if len(interface) == 0:
            interface = np.array([[-1]])
        return [void, solid, interface]


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def mass_balance_conv_nb( g_ij, P_profile, ids,network_pore_void, network_pore_solid, network_throat_conns, pore_throat_conns, inner_start2end):
    result = np.zeros(len(ids), dtype=np.float64)
    for i in nb.prange(len(ids)):
        pore_id = ids[i]
        void, solid, interface = find_neighbor_ball_information_nb(pore_throat_conns, inner_start2end, pore_id)
        if void[0, 0] == -1:
            result[i] = 0
        else:
            pressure = [P_profile[void[:, 0]], P_profile[void[:, 1]]]
            delta_p = pressure[0] - pressure[1]
            if len(delta_p) >= 1:
                flux = delta_p * g_ij[void[:, 2]]
            else:
                flux = np.array([0.])
            result[i] = np.sum(flux)

    return result


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def cal_pore_veloc_nb(network_throat_area, network_pore_radius, network_pore_real_shape_factor, g_ij, P_profile, ids,
                      network_pore_void, network_pore_solid, network_throat_conns, pore_throat_conns, inner_start2end):
    # res=find_neighbor_ball(network,[a])
    result = np.zeros(len(ids), dtype=np.float64)
    for i in nb.prange(len(ids)):
        pore_id = ids[i]
        void, solid, interface = find_neighbor_ball_information_nb(pore_throat_conns, inner_start2end, pore_id)
        if void[0, 0] == -1:
            result[i] = 0
        else:
            # len_pore = len(pores)
            # pressure = np.zeros((len_pore, 2), dtype=np.float64)
            # cond_f = np.zeros((len_pore, 2), dtype=np.float64)
            # area_t = np.zeros((len_pore, 2), dtype=np.float64)
            # pressure = np.vstack((P_profile[void[:, 0]], P_profile[void[:, 1]]))
            cond_f = g_ij[void[:, 2]]
            area_t = network_throat_area[void[:, 2]]
            delta_p = P_profile[void[:, 0]] - P_profile[void[:, 1]]
            # throat_p=(pressure[:,0]+pressure[:,1])/2
            flux = delta_p * cond_f
            velocity = flux / area_t
            # momentum = velocity * np.abs(velocity)
            # momentum=flux*np.abs(flux)/area_t
            momentum = flux * np.abs(flux) / area_t

            flux = max(abs(np.sum(flux[flux > 0])), abs(np.sum(flux[flux < 0])))
            # momentum=max(abs(np.sum(momentum[momentum>0]+delta_p[delta_p>0]/network['pore.density'][a])),
            #             abs(np.sum(momentum[momentum<0]+delta_p[delta_p<0]/network['pore.density'][a])))
            momentum = abs(abs(np.sum(momentum[momentum > 0])) - abs(np.sum(momentum[momentum < 0])))

            vel_p_m = np.sqrt(
                momentum / (network_pore_radius[pore_id] ** 2 / 16 / network_pore_real_shape_factor[pore_id]))*2
            # vel_p = flux / (network_pore_radius[pore_id] ** 2 / 4 / network_pore_real_shape_factor[pore_id])

            # result[i] = [abs(vel_p), abs(vel_p_m)][1]
            result[i] = abs(vel_p_m)

            # print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
    return result


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def species_balance_conv_nb(network_throat_radius, network_throat_length, network_throat_Cp, network_throat_density, g_ij, Tem,
                            thermal_con_dual, P_profile, ids, network_pore_void,
                            network_pore_solid,network_throat_conns, pore_throat_conns, inner_start2end):
    result = np.zeros((len(ids), 5), dtype=np.float64)
    # res=find_neighbor_ball(network,[a])
    for i in nb.prange(len(ids)):
        pore_id = ids[i]
        void, solid, interface = find_neighbor_ball_information_nb(pore_throat_conns, inner_start2end, pore_id)
        # g_ij=H_P_fun(network['throat.radius'],network['throat.length'],fluid['viscosity'])

        # g_ij*=network['throat.void']
        # mean_gil=np.max(g_ij)

        # coe_A for convection heat transfer
        # _i for slecting direct of fluid
        # thermal_con_dual=network['throat.solid']*solid['lambda']+network['throat.connect']*(solid['lambda'])+network['throat.void']*fluid['lambda'] #solid_pore

        coe_B = network_throat_radius ** 2 * np.pi / network_throat_length * thermal_con_dual
        # Void
        if void[0, 0] != -1:
            pressure = np.empty((len(void), 2), dtype=np.float64)
            pressure[:, 0] = P_profile[void[:, 0]]
            pressure[:, 1] = P_profile[void[:, 1]]
            cond_f = g_ij[void[:, 2]]
            Temp_f = np.empty((len(void), 2), dtype=np.float64)
            Temp_f[:, 0] = Tem[void[:, 0]]
            Temp_f[:, 1] = Tem[void[:, 1]]
            cond_h = coe_B[void[:, 2]]
            # cp_data = network_throat_Cp[void[:, 2]]
            # density = network_throat_density[void[:, 2]]
            delta_p = pressure[:, 0] - pressure[:, 1]
            flux = delta_p * cond_f
            h_conv_f = flux
            h_conv_f[delta_p > 0] *= Temp_f[delta_p > 0][:, 0]
            h_conv_f[delta_p < 0] *= Temp_f[delta_p < 0][:, 1]
            h_conv_f = np.sum(h_conv_f)
            h_cond_f = np.sum((Temp_f[:, 0] - Temp_f[:, 1]) * cond_h)
        else:
            h_conv_f = 0.
            h_cond_f = 0.
        # cp_data,density=np.array(cp_data),np.array(density)
        # Solid
        if solid[0, 0] != -1:
            Temp_s = np.empty((len(solid), 2), dtype=np.float64)
            Temp_s[:, 0] = Tem[solid[:, 0]]
            Temp_s[:, 1] = Tem[solid[:, 1]]
            cond_h_s = coe_B[solid[:, 2]]
            h_cond_s = np.sum((Temp_s[:, 0] - Temp_s[:, 1]) * cond_h_s)
        else:
            h_cond_s = 0
        if interface[0, 0] != -1:
            # Interface
            Temp_sf = np.empty((len(interface), 2), dtype=np.float64)
            Temp_sf[:, 0] = Tem[interface[:, 0]]
            Temp_sf[:, 1] = Tem[interface[:, 1]]
            cond_hs = coe_B[interface[:, 2]]
            h_cond_sf = np.sum((Temp_sf[:, 0] - Temp_sf[:, 1]) * cond_hs)
        else:
            h_cond_sf = 0
            # mass_f=np.sum((pressure[:,0]-pressure[:,1])*cond_f) if len(pressure)>0 else 0
            # h_conv_f=np.sum((pressure[:,0]-pressure[:,1])*cond_f*fluid['Cp']) if len(pressure)>0 else 0
        total = h_conv_f + h_cond_f + h_cond_sf + h_cond_s
        result[i] = np.array([total, h_conv_f, h_cond_f, h_cond_sf, h_cond_s])
        # print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
    return result  # - inlet + outlet


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def energy_balance_conv_nb(network_throat_radius, network_throat_length, network_throat_Cp, network_throat_density, g_ij, Tem,
                           thermal_con_dual, P_profile, ids, network_pore_void, network_pore_solid, network_throat_conns,
                           pore_throat_conns, inner_start2end):
    result = np.zeros((len(ids), 5), dtype=np.float64)
    for i in nb.prange(len(ids)):
        pore_id = ids[i]
        void, solid, interface = find_neighbor_ball_information_nb(pore_throat_conns, inner_start2end, pore_id)
        # g_ij=H_P_fun(network['throat.radius'],network['throat.length'],fluid['viscosity'])

        # g_ij*=network['throat.void']
        # mean_gil=np.max(g_ij)

        # coe_A for convection heat transfer
        # _i for slecting direct of fluid
        # thermal_con_dual=network['throat.solid']*solid['lambda']+network['throat.connect']*(solid['lambda'])+network['throat.void']*fluid['lambda'] #solid_pore
        coe_B = network_throat_radius ** 2 * np.pi / network_throat_length * thermal_con_dual

        # Void
        if void[0, 0] != -1:
            pressure = np.empty((len(void), 2), dtype=np.float64)
            pressure[:, 0] = P_profile[void[:, 0]]
            pressure[:, 1] = P_profile[void[:, 1]]
            cond_f = g_ij[void[:, 2]]
            Temp_f = np.empty((len(void), 2), dtype=np.float64)
            Temp_f[:, 0] = Tem[void[:, 0]]
            Temp_f[:, 1] = Tem[void[:, 1]]
            cond_h = coe_B[void[:, 2]]
            cp_data = network_throat_Cp[void[:, 2]]
            density = network_throat_density[void[:, 2]]
            delta_p = pressure[:, 0] - pressure[:, 1]
            flux = delta_p * cond_f
            h_conv_f = flux * cp_data * density
            h_conv_f[delta_p > 0] *= Temp_f[delta_p > 0][:, 0]
            h_conv_f[delta_p < 0] *= Temp_f[delta_p < 0][:, 1]
            h_conv_f = np.sum(h_conv_f)
            h_cond_f = np.sum((Temp_f[:, 0] - Temp_f[:, 1]) * cond_h)
        else:
            h_conv_f = 0.
            h_cond_f = 0.

        # Solid
        if solid[0, 0] != -1:
            Temp_s = np.empty((len(solid), 2), dtype=np.float64)
            Temp_s[:, 0] = Tem[solid[:, 0]]
            Temp_s[:, 1] = Tem[solid[:, 1]]
            cond_h_s = coe_B[solid[:, 2]]
            h_cond_s = np.sum((Temp_s[:, 0] - Temp_s[:, 1]) * cond_h_s)
        else:
            h_cond_s = 0
        # Interface
        if interface[0, 0] != -1:
            Temp_sf = np.empty((len(interface), 2), dtype=np.float64)
            Temp_sf[:, 0] = Tem[interface[:, 0]]
            Temp_sf[:, 1] = Tem[interface[:, 1]]
            cond_hs = coe_B[interface[:, 2]]
            h_cond_sf = np.sum((Temp_sf[:, 0] - Temp_sf[:, 1]) * cond_hs)
        else:
            h_cond_sf = 0
        # mass_f=np.sum((pressure[:,0]-pressure[:,1])*cond_f) if len(pressure)>0 else 0
        # h_conv_f=np.sum((pressure[:,0]-pressure[:,1])*cond_f*fluid['Cp']) if len(pressure)>0 else 0

        total = h_conv_f + h_cond_f + h_cond_sf + h_cond_s
        result[i] = np.array([total, h_conv_f, h_cond_f, h_cond_sf, h_cond_s])
        # print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
    return result  # - inlet + outlet


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def update_pore_throat_conns_nb(network_pore_void, network_pore_solid, network_throat_conns, a, result, elements,
                                     total_information, inner_start2end):
    # information = np.zeros_like(total_information,dtype=np.int64)-1
    for i in nb.prange(len(elements)):
        pore_id = elements[i]
        total_information_i = total_information[inner_start2end[pore_id][0]:inner_start2end[pore_id][1]]
        if network_pore_void[pore_id]:
            bool_void = network_pore_void[total_information_i[:, 1]]
            total_information_i[:, 3] = np.where(bool_void, 0, 2)

        elif network_pore_solid[pore_id]:
            bool_solid = network_pore_solid[total_information_i[:, 1]]
            total_information_i[:, 3] = np.where(bool_solid, 1, 2)
        total_information[inner_start2end[pore_id][0]:inner_start2end[pore_id][1]] = total_information_i