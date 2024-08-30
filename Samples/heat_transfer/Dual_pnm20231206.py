#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 14:25:38 2021

@author: htmt
"""

import sys

from mpnm import topotools, algorithm, network as net
from scipy.sparse import coo_matrix
import numpy as np
import time
import pandas as pd
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import numba as nb

nb.set_num_threads(10)


def lambda_calc(network, j, heat_coe, fluid, solid):
    res = lambda_calc_nb(j, heat_coe, fluid['lambda'], solid['lambda'], network['throat.conns'],
                         network['throat.length'], network['pore.radius'])
    return res


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def lambda_calc_nb(j, heat_coe, fluid_lambda, solid_lambda, network_throat_conns, network_throat_length,
                   network_pore_radius):
    result = np.zeros((len(j), 3), dtype=np.float64)
    for id in nb.prange(len(j)):
        j_ = j[id]
        i = network_throat_conns[j_]
        l = network_throat_length[j_] / (network_pore_radius[i[0]] + network_pore_radius[i[1]])
        heat_s_f = (network_throat_length[j_] / (
                network_pore_radius[i[1]] * l * 3 / heat_coe[i[0]] + network_pore_radius[i[1]] * l / solid_lambda))
        heat_s_f_bronze = (network_throat_length[j_] / (
                network_pore_radius[i[1]] * l * 3 / heat_coe[i[0]] + network_pore_radius[i[1]] * l / 75.3))
        effect_lambda = (network_throat_length[j_] / (
                network_pore_radius[i[0]] * l / fluid_lambda + network_pore_radius[i[1]] * l / solid_lambda))
        result[id, 0], result[id, 1], result[id, 2] = heat_s_f, heat_s_f_bronze, effect_lambda
    return result


# def lambda_calc (network,j,heat_coe,fluid,solid):
#     i=network['throat.conns'][j]
#     l=network['throat.length'][j]/(network['pore.radius'][i[0]]+network['pore.radius'][i[1]])
#     heat_s_f=(network['throat.length'][j]/(network['pore.radius'][i[1]]*l*3/heat_coe[i[0]]+network['pore.radius'][i[1]]*l/solid['lambda']))
#     heat_s_f_bronze=(network['throat.length'][j]/(network['pore.radius'][i[1]]*l*3/heat_coe[i[0]]+network['pore.radius'][i[1]]*l/75.3))
#     #heat_s_f.append(network['throat.length'][j]*solid['lambda']/(network['pore.radius'][i[1]]*l)/(1+solid['lambda']/(network['pore.radius'][i[1]]*l)/heat_coe[i[0]]))
#     effect_lambda=(network['throat.length'][j]/(network['pore.radius'][i[0]]*l/fluid['lambda']+network['pore.radius'][i[1]]*l/solid['lambda']))
#     return [heat_s_f,heat_s_f_bronze,effect_lambda]


sample_data_root = '../../sample_data/Sphere_stacking_250_500_2800_20/'
path = sample_data_root + 'pore_network'
# project = op.io.Statoil.load(path=path, prefix='sphere_stacking_250_500_2800_20')

project = net.read_network(path=path, name='sphere_stacking_250_500_2800_20', remove_in_out_throats=True)
pn_o = project
# pn_o.name = 'pore'
# pn['pore._id']=np.arange(len(pn['pore._id']))
# pn['throat._id']=np.arange(len(pn['throat._id']))
# pn['pore.coords']=pn['pore.coords'].astype(np.float32)
# pn['throat.conns']=(pn['throat.conns']).astype(np.int32)

# water = op.phases.Water(network=pn)
imsize = np.array([2800, 500, 250])
resolution = 21.4e-6
connect_throat = pd.read_csv(
    sample_data_root + 'solid_network/dual_network_interface_sphere_stacking_250_500_2800_20.csv')
connect_throat = np.array(connect_throat)[:, 1:] - [1, 1, 0]
connect_throat[:, 2] = (connect_throat[:, 2] / np.pi) ** 0.5 * resolution
conn_t = np.array(connect_throat)
data_volume = pd.read_csv(sample_data_root + 'solid_network/dual_network_volume_sphere_stacking_250_500_2800_20.csv')

'''
path='./pore_center_500_500_2500_20.csv'
pore_im=pd.read_csv(path)
pore_im=np.array(pore_im)
'''
path = sample_data_root + 'solid_network/solid_center_sphere_stacking_250_500_2800_20.csv'
solid_im = pd.read_csv(path)
solid_im = np.array(solid_im)

'''
pore={}
pore['pore.all']=pore_im[:,1].astype(bool)
pore['pore.coords']=(pore_im[:,2:5]*resolution).astype(np.float64)
pore['pore.radius']=(pore_im[:,5]*resolution).astype(np.float64)
pore_num=len(pore['pore.all'])
pore['throat.conns']=conn_t[(conn_t[:,0]<pore_num)&(conn_t[:,1]<pore_num)]
pore['throat.conns']=pore['throat.conns'][pore['throat.conns'][:,0]<pore['throat.conns'][:,1]]
pore['pore.radius']=pn_o['pore.radius']
pore['throat.radius']=pn_o['throat.radius']###
pore['throat.conns']=pn_o['throat.conns']
pore['throat.shape_factor']=pn_o['throat.shape_factor']

#pore['throat.radius']=(pore['throat.conns'][:,2]*resolution).astype(np.float64)#
#pore['throat.conns']=(pore['throat.conns'][:,:2]).astype(np.int64)
pore['throat.all']=np.ones(len(pore['throat.conns'])).astype(bool)

pore['throat.length']=pn_o['throat.length']#
#pore['throat.length']=np.array([np.linalg.norm((pore['pore.coords'][k[0]]-pore['pore.coords'][k[1]])) for k in pore['throat.conns']])
pore['throat.void']=np.ones(pore['throat.all'].size).astype(bool)
'''
# pn = op.network.GenericNetwork(name='pn')
pn = {}
pn.update(pn_o)
# pn.update(pore)

pn['throat.real_shape_factor'] = pn['throat.shape_factor']
pn['throat.real_shape_factor'][
    (pn['throat.shape_factor'] > np.sqrt(3) / 36 + 1e-5) & (pn['throat.shape_factor'] < 0.07)] = 1 / 16
pn['throat.real_shape_factor'][(pn['throat.shape_factor'] > 0.07)] = 1 / 4 / np.pi
pn['pore.real_shape_factor'] = pn['pore.shape_factor']
pn['pore.real_shape_factor'][
    (pn['pore.shape_factor'] > np.sqrt(3) / 36 + 1e-5) & (pn['pore.shape_factor'] < 0.07)] = 1 / 16
pn['pore.real_shape_factor'][(pn['pore.shape_factor'] > 0.07)] = 1 / 4 / np.pi
pn['throat.real_k'] = pn['throat.all'] * 0.6
pn['throat.real_k'][(pn['throat.shape_factor'] > np.sqrt(3) / 36 + 1e-5) & (pn['throat.shape_factor'] < 0.07)] = 0.5623
pn['throat.real_k'][(pn['throat.shape_factor'] > 0.07)] = 0.5
pn['pore.real_k'] = pn['pore.all'] * 0.6
pn['pore.real_k'][(pn['pore.shape_factor'] > np.sqrt(3) / 36 + 1e-5) & (pn['pore.shape_factor'] < 0.07)] = 0.5623
pn['pore.real_k'][(pn['pore.shape_factor'] > 0.07)] = 0.5
pn['throat.void'] = pn['throat.all']
pn['pore.void'] = pn['pore.all']
# pn['throat.area']=pn['throat.radius']**2*np.pi

solid = {}
solid['pore.all'] = solid_im[:, 1].astype(bool)
solid['pore.coords'] = (solid_im[:, 2:5] * resolution).astype(np.float64)
solid['pore.radius'] = (solid_im[:, 5] * resolution).astype(np.float64)
solid['throat.conns'] = conn_t[(conn_t[:, 0] >= len(pn['pore._id'])) & (conn_t[:, 1] >= len(pn['pore._id']))]
solid['throat.conns'] = solid['throat.conns'][solid['throat.conns'][:, 0] < solid['throat.conns'][:, 1]]
solid['throat.radius'] = (solid['throat.conns'][:, 2]).astype(np.float64)
# solid['throat.radius']=((solid['throat.conns'][:,2]/np.pi)**0.5*resolution).astype(np.float64)
solid['throat.conns'] = (solid['throat.conns'][:, :2]).astype(np.int64) - len(pn['pore._id'])
solid['throat.all'] = np.ones(len(solid['throat.conns'])).astype(bool)
# print(np.unique(solid['throat.conns']))
solid['throat.length'] = np.array(
    [np.linalg.norm((solid['pore.coords'][k[0]] - solid['pore.coords'][k[1]])) for k in solid['throat.conns']])
solid['throat.void'] = np.ones(solid['throat.all'].size).astype(bool)

# sn = op.network.GenericNetwork(name='sn')
sn = {}
sn.update(solid)

sn['pore._id'] = np.arange(len(solid['pore.all']))
sn['throat._id'] = np.arange(len(sn['throat.all']))

# sn['pore._id'] = np.arange(len(sn['pore._id']))
# sn['throat._id'] = np.arange(len(sn['throat._id']))
pn['pore._id'] = np.arange(len(pn['pore._id']))
pn['throat._id'] = np.arange(len(pn['throat._id']))

del connect_throat
data_surface = np.array(data_volume)[:, 3]
data_volume = np.array(data_volume)[:, 2]

conn_t = conn_t[conn_t[:, 1] > len(pn['pore._id'])]
conn_t = conn_t[conn_t[:, 2] > 0]
conn_t = conn_t[conn_t[:, 0] < len(pn['pore._id'])]
conn_t_s = np.float64(conn_t[:, 2])
conn_t = conn_t[:, 0:2].astype(np.int64)

dualn = {}
# dualn = op.network.GenericNetwork(name='dual')
dualn['pore.all'] = np.ones(len(pn['pore.coords']) + len(sn['pore.coords']), dtype=bool)
dualn['pore._id'] = np.arange(len(dualn['pore.all']))
dualn['throat.all'] = np.ones(len(pn['throat.all']) + len(sn['throat.all']) + len(conn_t), dtype=bool)  ###
dualn['throat._id'] = np.arange(len(dualn['throat.all']))
dualn['pore.coords'] = np.vstack((pn['pore.coords'], sn['pore.coords'])).astype(
    np.float64)  # need keep the same type as cor
dualn['pore.label'] = np.arange(len(dualn['pore.coords']))
dualn['pore.void'] = dualn['pore.label'] * False
dualn['pore.void'][:len(pn['pore.all'])] = True
dualn['pore.void'] = dualn['pore.void'].astype(bool)
dualn['pore.solid'] = ~dualn['pore.void']

num_solid_ball = np.count_nonzero(dualn['pore.solid'])
num_void_ball = np.count_nonzero(dualn['pore.void'])
num_solid_throat = len(sn['throat.all'])
num_void_throat = len(pn['throat.all'])

dualn['throat.conns'] = np.vstack((pn['throat.conns'],
                                   sn['throat.conns'] + [num_void_ball, num_void_ball],
                                   conn_t)).astype(np.int64)  ###

dualn['throat.label'] = np.arange(len(dualn['throat.conns']))
popt = 'throat.length'
conn_t_l = np.array(
    [np.sqrt(np.sum((dualn['pore.coords'][k[0]] - dualn['pore.coords'][k[1]]) ** 2)) for k in conn_t])  ###
dualn[popt] = np.concatenate((pn['throat.total_length'], sn[popt], conn_t_l)).astype(np.float64)
popt = 'throat.radius'
dualn[popt] = np.concatenate((pn[popt], sn[popt], conn_t_s)).astype(np.float64)  ###
# dualn[popt]=np.concatenate((pn[popt],sn[popt],(conn_t_s/np.pi)**0.5*resolution)).astype(np.float64)####
popt = 'pore.radius'
dualn[popt] = np.concatenate((pn[popt], sn[popt])).astype(np.float64)
dualn['pore.volume'] = data_volume * resolution ** 3
dualn['pore.surface'] = data_surface * resolution ** 2
dualn['pore.surface'][dualn['pore.surface'] <= 0] = resolution
dualn['pore.hydraulic_diameter'] = 6 * dualn['pore.volume'] / dualn['pore.surface']

topotools().find_surface_KDTree(pn, 'x', imsize, resolution, label_1='left_surface', label_2='right_surface')
topotools().find_surface_KDTree(pn, 'y', imsize, resolution, label_1='back_surface', label_2='front_surface')
topotools().find_surface_KDTree(pn, 'z', imsize, resolution, label_1='bottom_surface', label_2='top_surface')
topotools().trim_surface(pn)

topotools().find_surface_KDTree(dualn, 'x', imsize, resolution, label_1='left_surface', label_2='right_surface')
topotools().find_surface_KDTree(dualn, 'y', imsize, resolution, label_1='back_surface', label_2='front_surface')
topotools().find_surface_KDTree(dualn, 'z', imsize, resolution, label_1='bottom_surface', label_2='top_surface')
topotools().trim_surface(dualn)

for i in ['left', 'right', 'top', 'bottom', 'front', 'back']:
    dualn['pore.' + i + '_surface'][dualn['pore.void']] += pn['pore.' + i + '_surface']
topotools().trim_surface(dualn)
for i in ['left', 'right', 'top', 'bottom', 'front', 'back']:
    dualn['pore.' + i + '_surface'][dualn['pore.void']] += pn['pore.' + i + '_surface']

dualn['throat.void'] = dualn['throat.label'] * False
dualn['throat.void'][
    (dualn['throat.conns'][:, 0] < num_void_ball) & (dualn['throat.conns'][:, 1] < num_void_ball)] = True
dualn['throat.void'] = dualn['throat.void'].astype(bool)
dualn['throat.solid'] = dualn['throat.label'] * False

dualn['throat.solid'][
    (dualn['throat.conns'][:, 0] >= num_void_ball) & (dualn['throat.conns'][:, 1] >= num_void_ball)] = True
dualn['throat.solid'] = dualn['throat.solid'].astype(bool)
dualn['throat.connect'] = np.zeros(dualn['throat.void'].size).astype(bool)
dualn['throat.connect'][~dualn['throat.void'] & ~dualn['throat.solid']] = True

dualn['throat.connect'] = dualn['throat.connect'].astype(bool)
backup_dualn = dict(dualn)
# dualn['pore.volume']=4/3*dualn['pore.radius']**3*np.pi

pn['pore._id'] = np.arange(pn['pore._id'].size)
pn['throat._id'] = np.arange(pn['throat._id'].size)
sn['pore._id'] = np.arange(sn['pore._id'].size)
sn['throat._id'] = np.arange(sn['throat._id'].size)
sn['pore.volume'] = dualn['pore.volume'][dualn['pore.solid']]
sn['throat.length'] = dualn['throat.length'][dualn['throat.solid']]
sn['throat.label'] = sn['throat._id']
sn['pore.label'] = sn['pore._id']
pn['throat.label'] = pn['throat._id']
pn['pore.label'] = pn['pore._id']
pn['pore.volume'] = dualn['pore.volume'][dualn['pore.void']]
# pn['throat.length']=dualn['throat.length'][dualn['throat.void']]
health_p = topotools().pore_health(pn)
backup = topotools().trim_pore(pn, health_p['single_pore'], health_p['single_throat'])
pn.update(backup)

health = topotools().pore_health(dualn)
backup = topotools().trim_pore(dualn, np.append(health['single_pore'], health_p['single_pore']).astype(int)
                               , np.append(health['single_throat'], health_p['single_throat']).astype(int))
# backup=trim_pore(dualn,health_p['single_pore'],health_p['single_throat'])
dualn.update(backup)

# water = op.phases.Water(network=pn)
water = {}
'''
op.io.VTK.export_data(network=pn,phases=water,filename='./pore_structure')
rock= op.phases.Water(network=sn)
op.io.VTK.export_data(network=sn,phases=rock,filename='./solid_structure')
'''
t0 = time.time()

miu = lambda T, P: 2.4055e-5 * np.exp(
    4.42e-4 * P / 1e5 + (4.753 - 9.565e-4 * P / 1e5) / 8.314e-3 / (T - 139.7 - 1.24e-2 * P / 1e5))

fluid = {}
solid = {}
fluid['density'] = 998  # kg/m^3
solid['density'] = 2600  # kg/m^3 sandstone
fluid['Cp'] = 4200  # J/(kg*K)
solid['Cp'] = 878  # J/(kg*K)
delta_t = 10  # s
time_step = 3
fluid['lambda'] = 0.679  # W/(m*K)
solid['lambda'] = 75.3  # 2.596#W/(m K)
fluid['viscosity'] = 0.842e-3  # Pa*s
fluid['initial_temperature'] = 300
solid['initial_temperature'] = 300
material = 'bronze'
backup_dualn = {}
backup_dualn = {}
backup_dualn = dict(dualn).copy()

# _----------------------------transient------------------------------#
# T_res=algorithm(). transient_temperature(dualn,coe_A,coe_A_i,coe_B, Boundary_condition_T,x0,g_ij,P_profile,fluid,solid,imsize,resolution,time_step,delta_t,'bottom')
# op.io.VTK.export_data(network=dualn,phases=Phase,filename='./whole_structure')

# op.io.VTK.export_data(network=dualn,phases=Phase,filename='./simulation_result20220107/whole20211213_{}'.format(i))
# B=A_c.toarray()
# T_res=transient_temperature(dualn, P_profile,fluid,solid,imsize,resolution,time_step,delta_t)
it_num = []
Perm = 1.012e-10
re_s = np.array([1249, 1650, 1918, 100, 300, 743])
for n in np.arange(len(re_s)):
    re = re_s[n]
    print(re)
    vel = re * fluid['viscosity'] / fluid['density'] / (
            2 * imsize[1] * imsize[2] / (imsize[1] + imsize[2])) / resolution
    u = np.around(vel, 6)  #
    P = u * imsize[0] * resolution / 1.0121e-10 * fluid['viscosity']

    inlet = ['back', 'left', 'left'][1]
    outlet = ['front', 'right', 'right'][1]
    path = os.getcwd()
    path += '/' + material + '20230526_Jiang_' + str(re) + '/'
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    for side in ['bottom', 'top']:
        dualn = {}
        for j in backup_dualn:
            dualn[j] = np.copy(backup_dualn[j])

        # dualn=copy.deepcopy(backup_dualn)
        value = imsize[0] * imsize[1] * resolution ** 2 / np.sum(
            dualn['pore.radius'][dualn['pore.' + side + '_surface']] ** 2 * np.pi)
        boundary_surface = topotools().merge_clone_pore(dualn, dualn['pore.' + side + '_surface'], dualn['pore.radius'][
            dualn['pore.' + side + '_surface']] * value ** 0.5,
                                                        resolution * 20, imsize, side,
                                                        label='pore.boundary_' + side + '_surface')

        boundary_surface['throat.boundary_connect'] = boundary_surface['throat.all']
        network = topotools().add_pores(dualn, boundary_surface, trail=False)
        # dualn = op.network.GenericNetwork(name='dual_' + side)
        dualn = {}
        dualn.update(network)
        dualn['pore.solid'] = dualn['pore.solid'] | dualn['pore.boundary_' + side + '_surface']
        num_solid_ball = np.count_nonzero(dualn['pore.solid'])
        num_void_ball = np.count_nonzero(dualn['pore.void'])

        dualn['throat.solid'][
            (dualn['throat.conns'][:, 0] >= num_void_ball) & (dualn['throat.conns'][:, 1] >= num_void_ball)] = True
        # dualn['throat.connect']=np.zeros(dualn['throat.void'].size).astype(bool)
        dualn['throat.connect'][~dualn['throat.void'] & ~dualn['throat.solid']] = True
        dualn['throat.solid'] = dualn['throat.solid'].astype(bool)
        dualn['throat.connect'] = dualn['throat.connect'].astype(bool)

        # P=  49975

        # ------------------------boundary_condition-------------------------------#

        Boundary_condition_P = {}
        Boundary_condition_P['pore_inlet'] = {'pore.' + inlet + '_surface': [u, 'Neumann']}
        Boundary_condition_P['pore_outlet'] = {'pore.' + outlet + '_surface': [1000, 'Dirichlet']}

        Boundary_condition_T = {}
        Boundary_condition_T['solid_inlet'] = {'pore.boundary_' + side + '_surface': [200000, 'Neumann'],
                                               'pore.right_surface': [300, 'Robin']}
        # Boundary_condition['solid_outlet']={'pore.left_surface':25}
        Boundary_condition_T['pore_inlet'] = {'pore.' + inlet + '_surface': [300, 'Dirichlet']}
        Boundary_condition_T['pore_outlet'] = {'pore.' + outlet + '_surface': [300, 'Neumann']}

        # ------------------------boundary_condition-------------------------------#    Tem_c=dualn['pore.all']*302
        Tem_c = dualn['pore.all'] * 302
        T_x0 = dualn['pore.all'] * 300
        P_profile = dualn['pore.all'] * 0
        P_profile_back = np.copy(P_profile[dualn['pore.void']]) + 1.0
        P_profile_tem = np.copy(P_profile[dualn['pore.void']])
        flux_Throat_profile = pn['throat.all'] * 0 + 1.0e-10
        for j in fluid:
            pn['pore.' + j] = fluid[j] * pn['pore.all']
            pn['throat.' + j] = fluid[j] * pn['throat.all']
            dualn['pore.' + j] = fluid[j] * dualn['pore.all']
            dualn['throat.' + j] = fluid[j] * dualn['throat.all']
        for j in solid:
            if 'pore.' + j in dualn:
                dualn['pore.' + j][dualn['pore.solid']] = solid[j] * dualn['pore.all'][dualn['pore.solid']]
            if 'throat.' + j in dualn:
                dualn['throat.' + j][dualn['throat.solid']] = solid[j] * dualn['throat.all'][dualn['throat.solid']]

        t1 = time.time()
        T_res = []
        tol = 1e-2
        while mean_squared_error(Tem_c, T_x0) ** 0.5 > tol:
            if len(T_res) >= 20:
                if mean_absolute_percentage_error(Tem_c, T_res[-3]) < tol / 10:
                    break

            # vel=np.array([743,1249,1650,1918,2264])*np.average(pn['pore.viscosity'])/fluid['density']/(2*imsize[1]*imsize[2]/(imsize[1]+imsize[2]))/resolution
            # u= np.around(vel[::-1][n],3)#
            P = int(u * imsize[0] * resolution / Perm * np.average(pn['pore.viscosity']))
            Boundary_condition_P['pore_inlet'] = {'pore.' + inlet + '_surface': [P + 1000, 'Dirichlet']}
            T_x0 = np.copy(Tem_c)
            # u=np.array([0.10])[n]#
            dualn['pore.viscosity'] = miu(T_x0, P_profile + 1.003e5)
            pn['pore.viscosity'] = dualn['pore.viscosity'][dualn['pore.void']]
            # Throat_tem = []
            # Throat_pre = []
            # for k in dualn['throat.conns']:
            #     Throat_tem.append((T_x0[k[1]] + T_x0[k[0]]) / 2)
            #     Throat_pre.append((P_profile[k[1]] + P_profile[k[0]]) / 2)
            # Throat_tem = np.array(Throat_tem)
            # Throat_pre = np.array(Throat_pre)
            Throat_tem = (T_x0[dualn['throat.conns'][:, 1]] + T_x0[dualn['throat.conns'][:, 0]]) / 2
            Throat_pre = (P_profile[dualn['throat.conns'][:, 1]] + P_profile[dualn['throat.conns'][:, 0]]) / 2

            dualn['throat.viscosity'] = miu(Throat_tem, Throat_pre + 1.013e5)
            pn['throat.viscosity'] = dualn['throat.viscosity'][dualn['throat.void']]

            # water = op.phases.Water(network=pn)
            water = {}


            # ---------------'''pressure process '''---------------#
            def func_g(pn, flux_Throat_profile, RE_th, C, E, m, n):
                index = (flux_Throat_profile > 0).astype(bool)
                i_pore_e = np.abs(1 - (pn['throat.radius'] / pn['pore.radius'][pn['throat.conns'][:, 0]]) ** 2) ** (
                        2 * m)
                i_pore_c = np.abs(
                    (1 - (pn['throat.radius'] / pn['pore.radius'][pn['throat.conns'][:, 0]]) ** 2) / 2) ** (n)
                j_pore_e = np.abs(1 - (pn['throat.radius'] / pn['pore.radius'][pn['throat.conns'][:, 1]]) ** 2) ** (
                        2 * m)
                j_pore_c = np.abs(
                    (1 - (pn['throat.radius'] / pn['pore.radius'][pn['throat.conns'][:, 1]]) ** 2) / 2) ** (n)

                tem0 = 1 / (np.array(topotools().Mass_conductivity(pn)) / pn['throat.total_length'])
                tem1 = pn['throat.density'] / 2 / (np.pi ** 2) * np.abs(flux_Throat_profile) / pn[
                    'throat.radius'] ** 4 * \
                       ((E / RE_th) ** m + (index * j_pore_e + (~index) * i_pore_e))
                tem2 = pn['throat.density'] / 2 / (np.pi ** 2) * np.abs(flux_Throat_profile) / pn[
                    'throat.radius'] ** 4 * \
                       ((C / RE_th) ** n + ((~index) * j_pore_c + index * i_pore_c))
                tem3_1 = (1 / (pn['pore.radius'][pn['throat.conns'][:, 1]] ** 4) - 1 / (
                        pn['pore.radius'][pn['throat.conns'][:, 0]] ** 4)) * index
                tem3_1 += (1 / (pn['pore.radius'][pn['throat.conns'][:, 0]] ** 4) - 1 / (
                        pn['pore.radius'][pn['throat.conns'][:, 1]] ** 4)) * (~index)

                tem3 = pn['throat.density'] * np.abs(flux_Throat_profile) / 2 / (np.pi ** 2) * tem3_1

                result = tem1 + tem2 - tem3
                result = np.nan_to_num(result)
                result[RE_th == 0] = 0
                result = 1 / (tem0 + result)
                return result
                # ---------------'''pressure process '''---------------#


            idx = 0
            while mean_squared_error(P_profile_back, P_profile_tem) ** 0.5 > 1e-1:
                # print(mean_absolute_percentage_error(P_profile_back, P_profile_tem))

                if idx > 30 and mean_absolute_percentage_error(P_profile_back, P_profile_tem) < 1e-3:
                    break
                else:
                    idx += 1

                coe_A_P = np.array(topotools.Mass_conductivity(pn)) / pn['throat.total_length']
                Vel_Throat_profile = (np.abs(flux_Throat_profile / pn['throat.area']) * 0.5 + np.abs(
                    flux_Throat_profile / pn['throat.radius'] ** 2 / np.pi * 0.5))
                RE_th = (Vel_Throat_profile * pn['throat.radius'] * 2 * fluid['density'] / pn[
                    'throat.viscosity']) ** 0.7

                coe_A_P = func_g(pn, flux_Throat_profile, RE_th, C=26, E=27, m=1.1, n=0.2)
                P_profile_back = np.copy(P_profile_tem)
                P_profile_tem = algorithm().stead_stay_alg(pn, fluid, coe_A_P, Boundary_condition_P, resolution, False)
                P_profile_tem = (P_profile_tem + P_profile_back) / 2
                delta_p = P_profile_tem[pn['throat.conns'][:, 1]] - P_profile_tem[pn['throat.conns'][:, 0]]
                flux_Throat_profile = delta_p * coe_A_P + 1.0e-12

            P_profile_tem = algorithm().stead_stay_alg(pn, fluid, coe_A_P, Boundary_condition_P, resolution, False)
            P_profile[dualn['pore.void']] = P_profile_tem
            output = topotools().calculate_mass_flow(pn, Boundary_condition_P, coe_A_P, P_profile_tem, 4)
            abs_perm = output['pore.' + inlet + '_surface'] / (np.max(P_profile_tem[pn['pore.' + inlet + '_surface']]) - \
                                                               np.min(P_profile_tem[pn['pore.' + outlet + '_surface']]))

            abs_perm *= np.average(pn['pore.viscosity']) * imsize[0] / (imsize[1] * imsize[2]) / resolution
            Perm = np.copy(abs_perm)
            Pressure_drop = np.max(P_profile_tem[pn['pore.' + inlet + '_surface']]) - np.min(
                P_profile_tem[pn['pore.' + outlet + '_surface']])

            if len(T_res) == 0:
                output = topotools().calculate_mass_flow(pn, Boundary_condition_P, coe_A_P,
                                                         P_profile[dualn['pore.void']], 4)
                abs_perm = output['pore.' + inlet + '_surface'] / (
                        Boundary_condition_P['pore_inlet']['pore.' + inlet + '_surface'][0] -
                        Boundary_condition_P['pore_outlet']['pore.' + outlet + '_surface'][0])
                abs_perm *= np.average(pn['pore.viscosity']) * imsize[0] / (imsize[1] * imsize[2]) / resolution
                Perm = abs_perm
            water['pore.pressure'] = P_profile[dualn['pore.void']]
            delta_p = P_profile_tem[pn['throat.conns'][:, 1]] - P_profile_tem[pn['throat.conns'][:, 0]]

            # ---------------'''temperature process '''---------------#

            flux_Throat_profile = delta_p * coe_A_P + 1.0e-12
            Vel_Throat_profile = flux_Throat_profile / pn['throat.radius'] ** 2 / 4 / pn['throat.real_shape_factor']
            RE_th = Vel_Throat_profile * pn['throat.radius'] * 2 * fluid['density'] / pn['throat.viscosity']
            # flux_Pore_profile=[cal_pore_veloc(pn,fluid,coe_A,P_profile,a) for a in pn['pore._id']]
            Vel_Pore_profile = topotools().cal_pore_veloc(pn, fluid, coe_A_P, P_profile[dualn['pore.void']],
                                                          pn['pore._id']) / 2
            RE_po = Vel_Pore_profile * pn['pore.radius'] * 2 * fluid['density'] / pn['pore.viscosity']
            # P_profile_o=np.copy(P_profile)
            RE_po_o = np.copy(RE_po)
            delta_p_o = np.copy(delta_p)
            # P_profile=np.append(P_profile_o,np.zeros(np.count_nonzero(dualn['pore.solid'])))
            RE_po = np.append(RE_po_o, np.zeros(np.count_nonzero(dualn['pore.solid'])))
            num_node = len(dualn['pore.all'])
            delta_p = np.append(delta_p_o, np.zeros(np.count_nonzero(~dualn['throat.void'])))
            x0 = dualn['pore.void'] * fluid['initial_temperature'] + dualn['pore.solid'] * solid['initial_temperature']

            u_direct = (delta_p >= 0).astype(int)
            u_direct_i = (delta_p <= 0).astype(int)
            g_ij = np.append(coe_A_P, np.zeros(len(dualn['throat._id']) - len(coe_A_P)))

            Pr_num = dualn['pore.Cp'] * dualn['pore.viscosity'] / dualn['pore.lambda']

            heat_coe = (2 + (0.4 * RE_po ** 0.5 + 0.06 * RE_po ** 0.667) * Pr_num ** 0.4) * dualn['pore.lambda']
            # heat_coe=(2+1.3*Pr_num**0.15+(0.66*RE_po**0.5)*Pr_num**0.31)*dualn['pore.lambda']
            # heat_coe=(3.212*RE_po**0.335*Pr_num**0.438)*dualn['pore.lambda']
            '''
            heat_s_f=[]
            heat_s_f_bronze=[]
            length=[]
            effect_lambda=[]
            for j in dualn['throat._id']:
                i=dualn['throat.conns'][j]
                l=dualn['throat.length'][j]/(dualn['pore.radius'][i[0]]+dualn['pore.radius'][i[1]])
                length.append(l)
                heat_s_f.append(dualn['throat.length'][j]/(dualn['pore.radius'][i[1]]*l*5/heat_coe[i[0]]+dualn['pore.radius'][i[1]]*l/solid['lambda']))
                heat_s_f_bronze.append(dualn['throat.length'][j]/(dualn['pore.radius'][i[1]]*l*5/heat_coe[i[0]]+dualn['pore.radius'][i[1]]*l/75.3))
                #heat_s_f.append(dualn['throat.length'][j]*solid['lambda']/(dualn['pore.radius'][i[1]]*l)/(1+solid['lambda']/(dualn['pore.radius'][i[1]]*l)/heat_coe[i[0]]))
                effect_lambda.append(dualn['throat.length'][j]/(dualn['pore.radius'][i[0]]*l/fluid['lambda']+dualn['pore.radius'][i[1]]*l/solid['lambda']))
            '''
            # lambda_cal = lambda i: lambda_calc(dualn, i, heat_coe, fluid, solid)
            # lambda_map = Parallel(n_jobs=5, prefer='threads', require='sharedmem')(
            #     delayed(lambda_cal)(f) for f in dualn['throat._id'])
            # lambda_map = np.array(lambda_map)
            lambda_map = lambda_calc(dualn, dualn['throat._id'], heat_coe, fluid, solid)
            heat_s_f = lambda_map[:, 0]
            heat_s_f_bronze = lambda_map[:, 0]
            coe_A = g_ij * dualn['throat.Cp'] * dualn['throat.density'] * delta_p * (u_direct)
            coe_A_i = g_ij * dualn['throat.Cp'] * dualn['throat.density'] * delta_p * (-u_direct_i)

            # coe_A for convection heat transfer
            # _i for slecting direct of fluid
            # thermal_con_dual=dualn['throat.solid']*solid['lambda']+dualn['throat.connect']*solid['lambda']+dualn['throat.void']*fluid['lambda'] #solid_pore
            thermal_con_dual = dualn['throat.solid'] * dualn['throat.lambda'] + dualn['throat.connect'] * heat_s_f + \
                               dualn['throat.void'] * dualn['throat.lambda']  # solid_pore
            thermal_con_dual = np.array(thermal_con_dual)

            thermal_con_dual[dualn['throat.solid'] & dualn['throat.boundary_connect']] = 75.3
            thermal_con_dual[dualn['throat.connect'] & dualn['throat.boundary_connect']] = np.array(heat_s_f_bronze)[
                dualn['throat.connect'] & dualn['throat.boundary_connect']]

            coe_B = -dualn['throat.radius'] ** 2 * np.pi / dualn['throat.length'] * thermal_con_dual
            # coe_B for thermal conductivity matrix calculating

            # _----------------------------steady-state-------------------------------#

            Tem_c = algorithm().stead_stay_alg_convection(dualn, coe_A, coe_A_i, coe_B, Boundary_condition_T, g_ij,
                                                          P_profile, fluid, solid, imsize, resolution, side)
            # Phase = op.phases.Water(network=dualn)
            Tem_c = (Tem_c + T_x0) / 2
            T_res.append(Tem_c)
            print('finish 1 cycle')
            # T_res=algorithm().transient_temperature(dualn,coe_A,coe_A_i,coe_B, Boundary_condition_T,x0,g_ij,P_profile,fluid,solid,imsize,resolution,time_step,delta_t,side)

            # _----------------------------transient------------------------------#

        output = topotools().calculate_mass_flow(pn, Boundary_condition_P, coe_A_P, P_profile, 4)
        abs_perm = output['pore.' + inlet + '_surface'] / (
                Boundary_condition_P['pore_inlet']['pore.' + inlet + '_surface'][0] -
                Boundary_condition_P['pore_outlet']['pore.' + outlet + '_surface'][0])
        abs_perm *= np.average(pn['pore.viscosity']) * imsize[0] / (imsize[1] * imsize[2]) / resolution
        print(output)
        '''
        print (abs_perm)
        print(output['pore.'+inlet+'_surface']/(imsize[1]*imsize[2]*resolution**2))   
        print(np.average(pn['pore.viscosity']))
        '''
        print('number of iteration=', len(T_res))
        it_num.append(len(T_res))
        water['pore.velocity'] = Vel_Pore_profile
        # op.io.VTK.export_data(network=pn, phases=water, filename=path + '/pore_structure')
        # net.network2vtk(pn=pn, filename=path + '/pore_structure')

        # Phase = op.phases.Water(network=dualn)
        # op.io.VTK.export_data(network=dualn, phases=Phase, filename=path + '/' + side + '_flow')
        # net.network2vtk(pn=pn, filename=path + '/' + side + '_flow')

        layer_num = 14
        layer_way = 0  # 1,back front;0,left right;2,bottom top
        layer_way_1 = 2 if side == 'bottom' or side == 'top' else 1
        layer = topotools().devide_layer(dualn, [layer_num, 5, 5], imsize, resolution)

        model_res_void = []
        model_res_solid = []
        model_res_media = []
        model_res_pressure = []
        local_wall_tem = []

        vel_prof = np.zeros([layer_num, 5])
        tem_prof = np.zeros([layer_num, 5])
        tem_prof_v = np.zeros([layer_num, 5])
        tem_prof_s = np.zeros([layer_num, 5])
        for i in np.arange(layer_num):
            # tem_wall=T_res[-1][layer[layer_way][i]&dualn['pore.bottom_surface']&dualn['pore.solid']]
            tem_wall = (T_res[-1])[layer[layer_way][i] & (dualn['pore.boundary_' + side + '_surface'])]
            tem_wall[tem_wall == max(tem_wall)] = np.median(tem_wall)
            tem_wall *= dualn['pore.volume'][layer[layer_way][i] & (dualn['pore.boundary_' + side + '_surface'])]
            local_wall_tem.append(np.sum(tem_wall) / np.sum(
                (dualn['pore.volume'])[layer[layer_way][i] & (dualn['pore.boundary_' + side + '_surface'])]))
            tem_res_void = T_res[-1][layer[layer_way][i] & dualn['pore.void']]
            model_res_void.append(np.average(tem_res_void))
            tem_res_solid = T_res[-1][layer[layer_way][i] & dualn['pore.solid']]
            model_res_solid.append(np.average(tem_res_solid))
            tem_res_media = T_res[-1][layer[layer_way][i] & layer[layer_way_1][4] & dualn['pore.void']]
            model_res_media.append(np.average(tem_res_media))
            for j in np.arange(5):
                tem_wall = (T_res[-1] * dualn['pore.volume'])[layer[layer_way][i] & layer[layer_way_1][j]]
                tem_prof[i, j] = (np.sum(tem_wall) / np.sum(
                    (dualn['pore.volume'])[layer[layer_way][i] & layer[layer_way_1][j]]))
                tem_wall = (T_res[-1] * dualn['pore.volume'])[
                    layer[layer_way][i] & layer[layer_way_1][j] & dualn['pore.void']]
                tem_prof_v[i, j] = (np.sum(tem_wall) / np.sum(
                    (dualn['pore.volume'])[layer[layer_way][i] & layer[layer_way_1][j] & dualn['pore.void']]))
                tem_wall = (T_res[-1] * dualn['pore.volume'])[
                    layer[layer_way][i] & layer[layer_way_1][j] & dualn['pore.solid']]
                tem_prof_s[i, j] = (np.sum(tem_wall) / np.sum(
                    (dualn['pore.volume'])[layer[layer_way][i] & layer[layer_way_1][j] & dualn['pore.solid']]))
                vel_point = (Vel_Pore_profile * pn['pore.volume'])[
                    layer[layer_way][i][dualn['pore.void']] & layer[layer_way_1][j][dualn['pore.void']]]
                vel_prof[i, j] = (np.sum(vel_point) / np.sum((pn['pore.volume'])[
                                                                 layer[layer_way][i][dualn['pore.void']] &
                                                                 layer[layer_way_1][j][dualn['pore.void']]]))
        model_res_void = np.array(model_res_void)
        model_res_solid = np.array(model_res_solid)
        model_res_media = np.array(model_res_media)
        # model_res_pressure=np.array(model_res_pressure)
        local_wall_tem = np.array(local_wall_tem) - 200000 / 75.3 * resolution * 20
        hx = 200000 / (local_wall_tem - model_res_media)
        hx = pd.DataFrame(np.array(hx).T)
        local_wall_tem = pd.DataFrame(local_wall_tem.T)
        model_res_void = pd.DataFrame(model_res_void)
        model_res_solid = pd.DataFrame(model_res_solid)
        model_res_pressure = pd.DataFrame(model_res_pressure)
        local_wall_tem.to_csv(path + '/local_tem_' + side + '_' + material + '_' + str(re) + '.csv')
        model_res_void.to_csv(path + '/void_tem_' + side + '_' + material + '_' + str(re) + '.csv')
        model_res_solid.to_csv(path + '/solid_tem_' + side + '_' + material + '_' + str(re) + '.csv')
        # model_res_pressure.to_csv(path+'/pressure_tem_'+material+'_'+str(re)+'.csv')

        tem_prof = pd.DataFrame(tem_prof)

        tem_prof.to_csv(path + '/local_wall_tem_' + side + '_' + material + '_' + str(re) + '.csv')
        tem_prof_v = pd.DataFrame(tem_prof_v)
        tem_prof_v.to_csv(path + '/local_wall_tem_v' + side + '_' + material + '_' + str(re) + '.csv')
        tem_prof_s = pd.DataFrame(tem_prof_s)
        tem_prof_s.to_csv(path + '/local_wall_tem_s' + side + '_' + material + '_' + str(re) + '.csv')
        vel_prof = pd.DataFrame(vel_prof)
        vel_prof.to_csv(path + '/local_wall_vel_' + side + '_' + material + '_' + str(re) + '.csv')

        hx.to_csv(path + '/local_heat_transf_' + material + '_' + str(re) + side + '.csv')
        # local_wall_tem.to_csv(path+'/local_wall_tem_'+material+'_'+str(re)+side+'.csv')
        # model_res_pressure.append(np.average(P_profile[layer[1][i]&dualn['pore.void']]))
        tend = time.time()
        # Phase['pore.temperature'] = T_res[-1]
        num_node = len(dualn['pore.all'])
        B = coo_matrix(
            (coe_B / dualn['throat.radius'] ** 2 / np.pi, (dualn['throat.conns'][:, 0], dualn['throat.conns'][:, 1])),
            shape=(num_node, num_node), dtype=np.float64).tolil()
        A0 = (B.T + B).toarray()
        pore_h_coe = np.sum(A0, axis=0)
        # Phase['pore.h_coe'] = -pore_h_coe

        # net.network2vtk(pn=dualn, filename=path + '/' + side + '_flow')
        # op.io.VTK.export_data(network=dualn, phases=Phase, filename=path + '/' + side + '_flow')

        output = topotools().calculate_heat_flow(dualn, Boundary_condition_T, g_ij, T_res[-1], thermal_con_dual,
                                                 P_profile, 4)
        Thermal_out = np.sum(
            (T_res[-1][dualn['pore.left_surface']] - 300) / resolution * np.pi * solid['lambda'] * dualn['pore.radius'][
                dualn['pore.left_surface']] ** 2)
        print(output)
        print('thermal_conduction', Thermal_out)
        print('time cost' + '_' + material + '_' + str(re) + '_' + side + '：%.6fs' % (tend - t1))
        '''
        fig2=plt.figure(figsize=(8,6))
        #fig2=plt.hist(Phase['pore.h_coe'][dualn['pore.solid']],color=sns.color_palette("bright",10)[1],alpha=0.4,rwidth=0.8,bins=40,label=r'Void node $\ Re_D =$'+str(re))
        fig2=plt.hist(Phase['pore.h_coe'][dualn['pore.void']],color=sns.color_palette("bright",10)[2],density=True,alpha=0.9,rwidth=0.8,bins=40,label=r'Void node $\ Re_D =$'+str(re))
        fig2=plt.xlim(0,1.2e6)
        fig2=plt.tick_params(labelsize=12)
        fron1={'family':'Times New Roman','weight':'normal','size':12,}
        fig2=plt.xlabel(r'Heat transfer coefficient $\ (W/m^2) $',fron1)
        #fig2=plt.tick_params(labelsize=10)
        fig2=plt.ylabel(r'Frequency',fron1)

        fig2=plt.legend(prop=fron1,frameon=False, loc='upper right')
        fig2=plt.subplots_adjust(top=0.8,bottom=0.2,right=0.8,left=0.2,hspace=0,wspace=0)

        fig2=plt.savefig('./'+material+'_fig_h_c_v_1_'+str(n)+'.png',dpi = 600)

        fig2=plt.figure(figsize=(8,6))
        fig2=plt.hist(Phase['pore.h_coe'][dualn['pore.solid']],color=sns.color_palette("deep",10)[3],density=True,alpha=0.4,rwidth=0.8,bins=20,label=r'Solid node $\ Re_D =$'+str(re))
        #fig2=plt.hist(Phase['pore.h_coe'][dualn['pore.void']],color=sns.color_palette("bright",10)[3],alpha=0.9,rwidth=0.8,bins=40,label=r'Solid node $\ Re_D =$'+str(re))
        fig2=plt.xlim(0,3e6)
        fig2=plt.tick_params(labelsize=12)
        #fron1={'family':'Times New Roman','weight':'normal','size':12,}
        fig2=plt.xlabel(r'Heat transfer coefficient $\ (W/m^2) $',fron1)
        #fig2=plt.tick_params(labelsize=10)
        fig2=plt.ylabel(r'Frequency',fron1)

        fig2=plt.legend(prop=fron1,frameon=False, loc='upper right')
        fig2=plt.subplots_adjust(top=0.8,bottom=0.2,right=0.8,left=0.2,hspace=0,wspace=0)

        fig2=plt.savefig('./'+material+'_fig_h_c_s_1_'+str(n)+'.png',dpi = 600)

        fig2=plt.figure(figsize=(8,6))
        fig2=plt.hist(Vel_Pore_profile,color=sns.color_palette("bright",10)[4],density=True,alpha=0.9,rwidth=0.8,bins=50,label=r'$\ Re_D =$'+str(re))
        fig2=plt.tick_params(labelsize=12)
        #fron1={'family':'Times New Roman','weight':'normal','size':12,}
        fig2=plt.xlabel(r'Velocity $\ (m/s) $',fron1)
        #fig2=plt.tick_params(labelsize=10)
        fig2=plt.ylabel(r'Frequency',fron1)

        fig2=plt.legend(prop=fron1,frameon=False, loc='upper right')
        fig2=plt.subplots_adjust(top=0.8,bottom=0.2,right=0.8,left=0.2,hspace=0,wspace=0)

        fig2=plt.savefig('./'+material+'_fig_vel_1_'+str(n)+'.png',dpi = 600)
        '''
print(it_num)
it_num = pd.DataFrame(it_num)
# it_num.to_csv('/media/htmt/Data/Bead_packing/sphere_stacking_2800/it_num_20220903.csv')
it_num.to_csv('./it_num_20220903.csv')
tend = time.time()
print('total time cost：%.6fs' % (tend - t0))
