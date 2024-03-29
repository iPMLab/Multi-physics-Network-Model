#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:48:35 2022

@author: htmt
"""

from scipy.sparse import coo_matrix
import scipy.sparse.linalg as ssl
import pypardiso as pp
import openpnm as op
from MpNM.Base import *
from MpNM.topotools import topotools as tool
from MpNM.network import network as net


class algorithm(Base):
    def stead_stay_alg_multi(self, network, fluid, coe_A, Boundary_condition, resolution, bound_cond):
        num_pore = len(network.pores())
        A = coo_matrix((coe_A, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                       shape=(num_pore, num_pore), dtype=np.float64).tocsr()
        A = (A.T + A).tolil()
        dig = np.array(A.sum(axis=0)).reshape(num_pore)
        b = np.zeros(num_pore)

        # B=A.toarray()
        # mean_gil=np.max(coe_A)
        # condctivity=H_P_fun(network['pore.radius'],resolution,fluid['visocity'])
        def diffusion_coe_SO3(r, T, M1, M2, p, l):
            F_l = 2e-10 * T / p
            coe1 = 0.97 * r * (T / M1) ** 0.5
            x = 0.0060894 * T
            omb = 0.0051 * x ** 6 - 0.0917 * x ** 5 + 0.6642 * x ** 4 - 2.477 * x ** 3 + 5.0798 * x ** 2 - 5.684 * x + 3.941
            coe3 = 2.5e-9 * T ** 1.5 / p / omb
            coe2 = 1 / (1 / coe1 + 1 / coe3)
            data_coe = np.copy(coe2)
            data_coe[r < 0.1 * F_l] = coe1[r < 0.1 * F_l]
            data_coe[r >= 100 * F_l] = coe3
            return data_coe / l

        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                condctivity = diffusion_coe_SO3(network['pore.radius'], fluid['temperature'], 56, 28, 1, resolution)
                if bound_cond == False:
                    bound_cond = {}
                    index = np.argwhere(network[n] == True)
                    value = condctivity[index]
                    bound_cond['throat_inlet_cond'] = np.stack([index.flatten(), value.flatten()]).T
                    bound_cond['throat_outlet_cond'] = np.stack([index.flatten(), value.flatten()]).T
                    throat_inlet_cond = bound_cond['throat_inlet_cond']

                    throat_outlet_cond = bound_cond['throat_outlet_cond']
                    bound_cond = False
                else:
                    throat_inlet_cond = bound_cond['throat_inlet_cond']

                    throat_outlet_cond = bound_cond['throat_outlet_cond']

                # area_i=(imsize*resolution)**2/sum(network['pore.radius'][network[n]]**2*np.pi)
                condctivity = diffusion_coe_SO3(network['pore.radius'], fluid['temperature'], 56, 28, 1, resolution)  ##
                if 'solid' in m:
                    dig[network[n]] += condctivity[network[n]]
                    b[network[n]] -= Boundary_condition[m][n][0] * condctivity[network[n]]
                elif 'pore' in m:
                    if 'inlet' in m:
                        condctivity[throat_inlet_cond[:, 0].astype(int)] = throat_inlet_cond[:,
                                                                           1]  # if bound_cond!=False else condctivity[throat_inlet_cond[:,0].astype(int)]
                        dig[throat_inlet_cond[:, 0].astype(int)] += condctivity[throat_inlet_cond[:, 0].astype(int)]
                        b[throat_inlet_cond[:, 0].astype(int)] -= Boundary_condition[m][n][0] * condctivity[
                            throat_inlet_cond[:, 0].astype(int)]
                    elif 'outlet' in m:
                        condctivity[throat_outlet_cond[:, 0].astype(int)] = throat_outlet_cond[:,
                                                                            1]  # if bound_cond!=False else condctivity[throat_outlet_cond[:,0].astype(int)]
                        dig[throat_outlet_cond[:, 0].astype(int)] += condctivity[throat_outlet_cond[:, 0].astype(int)]
                        b[throat_outlet_cond[:, 0].astype(int)] -= Boundary_condition[m][n][0] * condctivity[
                            throat_outlet_cond[:, 0].astype(int)]

        A.setdiag(-dig, 0)
        A = A.tocsr()
        Profile = pp.spsolve(A, b)
        # Profile,j=ssl.bicg(A,b,tol=1e-9)
        return Profile

    def stead_stay_alg(self, network, fluid, coe_A, Boundary_condition, resolution, bound_cond):
        num_pore = len(network['pore.all'])
        A = coo_matrix((coe_A, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                       shape=(num_pore, num_pore), dtype=np.float64).tocsr()
        A = (A.T + A).tolil()
        dig = np.array(A.sum(axis=0)).reshape(num_pore)
        b = np.zeros(num_pore)
        # B=A.toarray()
        # mean_gil=np.max(coe_A)
        # condctivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                condctivity = tool().H_P_fun(network['pore.radius'], resolution, network['pore.viscosity'])  ##
                if bound_cond == False:
                    bound_cond = {}
                    index = np.argwhere(network[n] == True)
                    value = condctivity[index]
                    bound_cond['throat_inlet_cond'] = np.stack([index.flatten(), value.flatten()]).T
                    bound_cond['throat_outlet_cond'] = np.stack([index.flatten(), value.flatten()]).T
                    throat_inlet_cond = bound_cond['throat_inlet_cond']
                    throat_outlet_cond = bound_cond['throat_outlet_cond']
                    bound_cond = False
                else:
                    throat_inlet_cond = bound_cond['throat_inlet_cond']

                    throat_outlet_cond = bound_cond['throat_outlet_cond']

                # area_i=(imsize*resolution)**2/sum(network['pore.radius'][network[n]]**2*np.pi)

                if 'solid' in m:
                    dig[network[n]] += condctivity[network[n]]
                    b[network[n]] -= Boundary_condition[m][n][0] * condctivity[network[n]]
                elif 'pore' in m:
                    if 'inlet' in m:
                        if Boundary_condition[m][n][1] == 'Dirichlet':
                            condctivity[throat_inlet_cond[:, 0].astype(int)] = throat_inlet_cond[:,
                                                                               1]  # if bound_cond!=False else condctivity[throat_inlet_cond[:,0].astype(int)]
                            dig[throat_inlet_cond[:, 0].astype(int)] += condctivity[throat_inlet_cond[:, 0].astype(int)]
                            b[throat_inlet_cond[:, 0].astype(int)] -= Boundary_condition[m][n][0] * condctivity[
                                throat_inlet_cond[:, 0].astype(int)]
                        elif Boundary_condition[m][n][1] == 'Neumann':
                            dig[throat_inlet_cond[:, 0].astype(int)] += 0
                            b[throat_inlet_cond[:, 0].astype(int)] -= Boundary_condition[m][n][0] * \
                                                                      network['pore.radius'][network[n]] ** 2 * np.pi

                    elif 'outlet' in m:
                        if Boundary_condition[m][n][1] == 'Dirichlet':
                            condctivity[throat_outlet_cond[:, 0].astype(int)] = throat_outlet_cond[:,
                                                                                1]  # if bound_cond!=False else condctivity[throat_outlet_cond[:,0].astype(int)]
                            dig[throat_outlet_cond[:, 0].astype(int)] += condctivity[
                                throat_outlet_cond[:, 0].astype(int)]
                            b[throat_outlet_cond[:, 0].astype(int)] -= Boundary_condition[m][n][0] * condctivity[
                                throat_outlet_cond[:, 0].astype(int)]
                        elif Boundary_condition[m][n][1] == 'Neumann':
                            dig[throat_inlet_cond[:, 0].astype(int)] += 0
                            b[throat_inlet_cond[:, 0].astype(int)] = 0
        A.setdiag(-dig, 0)
        A = A.tocsr()
        Profile = pp.spsolve(A, b)
        # Profile,j=ssl.bicg(A,b,tol=1e-9)
        return Profile

    def correct_pressure(self, network, coe_A, Boundary_condition, resolution, S_term=False):
        num_pore = len(network['pore.all'])
        A = coo_matrix((coe_A, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                       shape=(num_pore, num_pore), dtype=np.float64).tocsr()
        A = (A.T + A).tolil()
        dig = -np.array(A.sum(axis=0)).reshape(num_pore)
        b = np.zeros(num_pore)

        A.setdiag(dig, 0)
        b = b + S_term
        A = A.tocsr()
        Profile = pp.spsolve(A, b)
        # Profile,j=ssl.bicg(A,b,tol=1e-9)
        return Profile

    def stead_stay_alg_convection(self, network, coe_A, coe_A_i, coe_B, Boundary_condition, g_ij, P_profile, fluid,
                                  solid, imsize, resolution, side):

        num_node = len(network['pore.all'])
        Num = num_node // 25000
        Num = 2 if Num < 2 else Num
        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B

        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        dig = np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        # B=A.toarray()
        # resulation=np.average(network['throat.length'])
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * solid['lambda'] * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = imsize[0] * imsize[1] * resolution ** 2 / np.sum(
                            network['pore.radius'][network['pore.boundary_' + side + '_surface']] ** 2 * np.pi)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (
                                resolution / solid['lambda'] + resolution / fluid['lambda'])
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig

                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    # mass_calcu=lambda i:tool().mass_balance_conv(network,g_ij,P_profile,i)
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile,
                                                              network['pore._id'][network[n] & network['pore.void']])
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * fluid['lambda'] * network['pore.radius'] ** 2 / resolution
                    if 'inlet' in m:
                        convection_term = fluid['density'] * fluid['Cp'] * abs(
                            P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])
                        diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                        tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)

                        tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                    else:
                        tem_b = 0

                        tem_dig = fluid['density'] * fluid['Cp'] * abs(
                            P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    dig[network[n] & network['pore.void']] -= tem_dig
                    b[network[n] & network['pore.void']] += tem_b
            # boundary condition set shuold be discussed
        # t0 = time.time()
        A_c = A  # copy A
        # T_res=[]
        # Phase= op.phases.Water(network=network)

        # _----------------------------steady-state-------------------------------#

        A_c.setdiag(-dig, 0)
        A_c = A_c.tocsr()
        # Tem_c,j=ssl.bicgstab(A_c,b,tol=1e-8)
        Tem_c = pp.spsolve(A_c, b)
        return Tem_c

    def stead_stay_model_tes(self, network, coe_A, coe_A_i, coe_B,
                             Boundary_condition, x0, g_ij, P_profile,
                             imsize, resolution, side, S_term=False,
                             type_f='species'):

        num_node = len(network['pore.all'])
        Num = max((num_node // 25000), 2)

        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B
        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        if type_f == 'species':
            alpha = network['pore.density'] * 0 + 1
            cond = network['pore.diffusivity']
        elif type_f == 'heat':
            alpha = network['pore.density'] * network['pore.Cp']
            cond = network['pore.lambda']
        elif type_f == 'momentum':
            alpha = network['pore.density']
            cond = network['pore.viscosity']
        elif type_f == 'density':
            alpha = network['pore.density'] * 0 + 1
            cond = network['pore.diffusivity'] * 0
            # dig,b=self.setting_Boundary_condition(network,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
        dig = -np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = net().getting_zoom_value(network, side, imsize, resolution)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (resolution / cond)
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile,
                                                              network['pore._id'][network[n] & network['pore.void']])
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * cond * network['pore.radius'] ** 2 / resolution

                        if 'inlet' in m:
                            convection_term = abs(alpha[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])

                            diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                            tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)
                            tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                        else:
                            tem_b = 0
                            tem_dig = abs(alpha[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        if 'inlet' in m:
                            tem_dig = 0
                            tem_b = Boundary_condition[m][n][0] * network['pore.radius'][
                                network[n] & network['pore.void']] ** 2 * np.pi
                        else:
                            tem_dig = abs(alpha[network[n] & network['pore.void']] * P_conductivity)
                            tem_b = 0
                    dig[network[n] & network['pore.void']] += tem_dig
                    b[network[n] & network['pore.void']] += tem_b

        A_c = A.copy()  # copy A

        # _----------------------------steady-state-------------------------------#

        # Var_c=np.copy(x0)

        # delta_dig=alpha*network['pore.volume']/delta_t
        dig_c = np.copy(dig)

        # diagonal should update for delta_t
        A_c.setdiag(dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = np.copy(b) + S_term
        test = A.sum(axis=0)

        if np.count_nonzero(test) == len(network['pore._id']):
            Var_c = pp.spsolve(A_c, b_c)
        else:
            Var_c, j = ssl.bicg(A_c, b_c, tol=1e-9)
            print('Matrix A is singular, because it contains empty row(s)')

        return Var_c

    def transient_temperature(self, network, coe_A, coe_A_i, coe_B, Boundary_condition, x0, g_ij, P_profile, fluid,
                              solid, imsize, resolution, time_step, delta_t, side, Phase):
        num_node = len(network['pore.all'])
        Num = num_node // 25000
        Num = 2 if Num < 2 else Num
        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B

        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        dig = np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        # B=A.toarray()
        # resulation=np.average(network['throat.length'])

        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * solid['lambda'] * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = imsize[0] * imsize[1] * resolution ** 2 / np.sum(
                            network['pore.radius'][network['pore.boundary_' + side + '_surface']] ** 2 * np.pi)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (
                                resolution / solid['lambda'] + resolution / fluid['lambda'])
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile, network['pore._id'])

                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * fluid['lambda'] * network['pore.radius'] ** 2 / resolution
                    if 'inlet' in m:
                        convection_term = fluid['density'] * fluid['Cp'] * abs(P_conductivity[network[n] & network[
                            'pore.void']])  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])
                        diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                        tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)

                        tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                    else:
                        tem_b = 0

                        tem_dig = fluid['density'] * fluid['Cp'] * abs(P_conductivity[network[n] & network[
                            'pore.void']])  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    dig[network[n] & network['pore.void']] -= tem_dig
                    b[network[n] & network['pore.void']] += tem_b
            # boundary condition set shuold be discussed
        # t0 = time.time()
        A_c = A  # copy A
        # T_res=[]
        # Phase= op.phases.Water(network=network)

        # _----------------------------steady-state-------------------------------#
        T_res = []
        # Phase= op.phases.Water(network=network)

        for i in np.arange(time_step):
            delta_dig = fluid['density'] * fluid['Cp'] * network['pore.volume'] / delta_t * network['pore.void'] + \
                        solid['density'] * solid['Cp'] * network['pore.volume'] / delta_t * network['pore.solid']
            dig_c = dig - delta_dig  # fluid_density*fluid_Cp*network['pore.radius']**3*4/3/delta_t*network['pore.void']-solid_density*solid_Cp*network['pore.radius']**3*4/3/delta_t*network['pore.solid']
            # diagonal should update for delta_t
            A_c.setdiag(-dig_c, 0)  # add diagonal into A_c
            A_c = A_c.tocsr()  # we transfer A_c for next calculation

            b_c = b + delta_dig * x0
            # fluid_density*fluid_Cp*network['pore.radius']**3*4/3*x0/delta_t #update b array for previous time step
            # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9) # calculate the temperature profile
            Tem_c = pp.spsolve(A_c, b_c)
            x0 = Tem_c.astype(np.float16)  # update
            T_res.append(Tem_c)
            print(max(Tem_c), min(Tem_c))
            Phase['pore.temperature'] = Tem_c
            op.io.VTK.export_data(network=network, phases=Phase, filename='./_{}'.format(i))
        return T_res

    def transient_temperature_single(self, network, coe_A, coe_A_i, coe_B, Boundary_condition, x0, g_ij, P_profile,
                                     fluid, solid, imsize, resolution, delta_t, side):
        num_node = len(network['pore.all'])
        Num = min((num_node // 25000) + 1, 10)
        # Num=2 if Num <2 else Num
        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B

        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        dig = np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        # B=A.toarray()
        # resulation=np.average(network['throat.length'])

        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * solid['lambda'] * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = imsize[0] * imsize[1] * resolution ** 2 / np.sum(
                            network['pore.radius'][network['pore.boundary_' + side + '_surface']] ** 2 * np.pi)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (
                                resolution / solid['lambda'] + resolution / fluid['lambda'])
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile,
                                                              network['pore._id'][network[n] & network['pore.void']])
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * fluid['lambda'] * network['pore.radius'] ** 2 / resolution
                    if 'inlet' in m:
                        convection_term = fluid['density'] * fluid['Cp'] * abs(
                            P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])
                        diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                        tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)

                        tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                    else:
                        tem_b = 0

                        tem_dig = fluid['density'] * fluid['Cp'] * abs(
                            P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    dig[network[n] & network['pore.void']] -= tem_dig
                    b[network[n] & network['pore.void']] += tem_b
            # boundary condition set shuold be discussed
        # t0 = time.time()
        A_c = A.copy()  # copy A
        # T_res=[]
        # Phase= op.phases.Water(network=network)

        # _----------------------------steady-state-------------------------------#
        T_res = []
        # Phase= op.phases.Water(network=network)

        delta_dig = fluid['density'] * fluid['Cp'] * network['pore.volume'] / delta_t * network['pore.void'] + solid[
            'density'] * solid['Cp'] * network['pore.volume'] / delta_t * network['pore.solid']
        dig_c = dig - delta_dig  # fluid_density*fluid_Cp*network['pore.radius']**3*4/3/delta_t*network['pore.void']-solid_density*solid_Cp*network['pore.radius']**3*4/3/delta_t*network['pore.solid']
        # diagonal should update for delta_t
        A_c.setdiag(-dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = b + delta_dig * x0
        # fluid_density*fluid_Cp*network['pore.radius']**3*4/3*x0/delta_t #update b array for previous time step
        # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9) # calculate the temperature profile
        Tem_c = pp.spsolve(A_c, b_c)
        x0 = Tem_c.astype(np.float16)  # update
        T_res.append(Tem_c)
        # print(max(Tem_c),min(Tem_c))

        return Tem_c

    def transient_temperature_s(self, network, coe_A, coe_A_i, coe_B, Boundary_condition, x0, g_ij, P_profile, fluid,
                                solid, imsize, resolution, delta_t, side):
        num_node = len(network['pore.all'])
        Num = (num_node // 25000) + 1
        # Num=2 if Num <2 else Num
        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B
        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        dig = -np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        # B=A.toarray()
        # resulation=np.average(network['throat.length'])
        dig, b = self.setting_Boundary_condition(network, g_ij, P_profile, dig, b, Boundary_condition, resolution,
                                                 imsize, Num, side, 'heat')

        # boundary condition set shuold be discussed
        # t0 = time.time()
        A_c = A  # copy A
        # T_res=[]
        # Phase= op.phases.Water(network=network)

        # _----------------------------steady-state-------------------------------#
        T_res = []
        # Phase= op.phases.Water(network=network)

        delta_dig = network['pore.density'] * network['pore.Cp'] * network['pore.volume'] / delta_t
        dig_c = dig + delta_dig  # fluid_density*fluid_Cp*network['pore.radius']**3*4/3/delta_t*network['pore.void']-solid_density*solid_Cp*network['pore.radius']**3*4/3/delta_t*network['pore.solid']

        # diagonal should update for delta_t
        A_c.setdiag(dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = b + delta_dig * x0
        # fluid_density*fluid_Cp*network['pore.radius']**3*4/3*x0/delta_t #update b array for previous time step
        # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9) # calculate the temperature profile
        Tem_c = pp.spsolve(A_c, b_c)

        T_res.append(Tem_c)
        # print(max(Tem_c),min(Tem_c))

        return Tem_c

    def setting_Boundary_condition(self, network, g_ij, P_profile, dig, b, Boundary_condition, resolution, imsize, Num,
                                   side, type_f):

        if type_f == 'diffusion':
            alpha = 1
            cond = network['pore.diffusivity']
        elif type_f == 'heat':
            alpha = network['pore.density'] * network['pore.Cp']
            cond = network['pore.lambda']
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]

                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = net().getting_zoom_value(network, side, imsize, resolution)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                    dig[network[n] & network['pore.solid']] += tem_dig
                    b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile,
                                                              network['pore._id'][network[n] & network['pore.void']])
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * cond * network['pore.radius'] ** 2 / resolution
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_f = np.pi * cond * network['pore.radius'] ** 2 / resolution

                    if 'inlet' in m:

                        convection_term = abs(alpha[network[n] & network[
                            'pore.void']] * P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])
                        diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                        tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)
                        tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                    else:
                        tem_b = 0
                        tem_dig = abs(alpha[network[n] & network[
                            'pore.void']] * P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    dig[network[n] & network['pore.void']] += tem_dig
                    b[network[n] & network['pore.void']] += tem_b
        return dig, b

    def transient_energy_s(self, network, coe_A, coe_A_i, coe_B,
                           Boundary_condition, x0, g_ij,
                           P_profile, imsize, resolution,
                           delta_t, side, type_f='heat'):
        num_node = len(network['pore.all'])
        Num = max((num_node // 25000), 2)

        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B
        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        alpha = network['pore.density'] * network['pore.Cp']
        cond = network['pore.lambda']
        alpha_ = network['pore.Cp']
        # dig,b=self.setting_Boundary_condition(network,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
        dig = -np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = net().getting_zoom_value(network, side, imsize, resolution)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (resolution / cond)
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile,
                                                              network['pore._id'][network[n] & network['pore.void']])
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * cond * network['pore.radius'] ** 2 / resolution

                        if 'inlet' in m:
                            convection_term = abs(alpha_[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])

                            diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                            tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)
                            tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                        else:
                            tem_b = 0
                            tem_dig = abs(alpha_[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    elif Boundary_condition[m][n][1] == 'Neumann':

                        if 'inlet' in m:
                            tem_dig = 0
                            tem_b = Boundary_condition[m][n][0] * network['pore.radius'][
                                network[n] & network['pore.void']] ** 2 * np.pi
                        else:
                            tem_dig = abs(alpha_[network[n] & network['pore.void']] * P_conductivity)
                            tem_b = 0
                    dig[network[n] & network['pore.void']] += tem_dig
                    b[network[n] & network['pore.void']] += tem_b

        A_c = A.copy()  # copy A

        # _----------------------------steady-state-------------------------------#

        # Var_c=np.copy(x0)

        delta_dig = alpha * network['pore.volume'] / delta_t
        dig_c = dig + delta_dig

        # diagonal should update for delta_t
        A_c.setdiag(dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = b + delta_dig * x0

        Var_c = pp.spsolve(A_c, b_c)
        # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9)

        return Var_c

    def transient_model_test(self, network, coe_A, coe_A_i, coe_B,
                             Boundary_condition, x0, g_ij, P_profile,
                             imsize, resolution, delta_t, side, mass_flow=False, S_term=False,
                             Bound_cond_P=False, type_f='species'):
        # func_pv=lambda c,T:8.314*T*c
        # func_ps=lambda T:611.21*np.exp((18.678-(T-273.15)/234.5)*((T-273.15)/(257.14+(T-273.15))))
        num_node = len(network['pore.all'])
        Num = max((num_node // 25000), 2)

        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B
        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()
        # RH= func_pv(1.0,302)/func_ps(302)
        # Bound_c=HAProps('C', 'T', 302, 'P', (1e5+400)/1000,'R',RH)*1000*1.29
        if type_f == 'species':
            alpha = network['pore.density'] * 0 + 1
            cond = network['pore.diffusivity']
        elif type_f == 'heat':
            alpha = network['pore.density'] * network['pore.Cp']
            cond = network['pore.lambda']
        elif type_f == 'momentum':
            alpha = network['pore.density']
            cond = network['pore.viscosity']
        elif type_f == 'density':
            alpha = network['pore.density'] * 0 + 1
            cond = network['pore.diffusivity'] * 0
            # dig,b=self.setting_Boundary_condition(network,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
        dig = -np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = net().getting_zoom_value(network, side, imsize, resolution)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (resolution / cond)
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:

                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        if Bound_cond_P:
                            P_g_ij = tool().H_P_fun(network['pore.radius'], resolution, network['pore.viscosity'])  ##
                            P_conductivity = (Bound_cond_P[m][n][0] - P_profile[network[n] & network['pore.void']]) * \
                                             P_g_ij[network[n] & network['pore.void']]
                        else:
                            P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile, network['pore._id'][
                                network[n] & network['pore.void']])
                        if 'inlet' in m:

                            convection_term = np.abs(alpha[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])

                            diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                            tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)
                            tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                        else:
                            tem_b = 0
                            tem_dig = abs(alpha[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile, network['pore._id'][
                            network[n] & network['pore.void']])
                        if 'inlet' in m:
                            tem_dig = 0
                            tem_b = Boundary_condition[m][n][0] * network['pore.radius'][
                                network[n] & network['pore.void']] ** 2 * np.pi
                        else:
                            tem_dig = abs(alpha[network[n] & network['pore.void']] * P_conductivity)
                            tem_b = 0
                    dig[network[n] & network['pore.void']] += tem_dig
                    b[network[n] & network['pore.void']] += tem_b

        A_c = A.copy()  # copy A

        # _----------------------------steady-state-------------------------------#

        # Var_c=np.copy(x0)

        delta_dig = alpha * network['pore.volume'] / delta_t
        dig_c = dig + delta_dig

        # diagonal should update for delta_t
        A_c.setdiag(dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = b + delta_dig * x0 + S_term

        Var_c = pp.spsolve(A_c, b_c)
        # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9)
        if type_f == 'momentum':
            return Var_c, dig_c
        else:
            return Var_c

    def transient_model(self, network, coe_A, coe_A_i, coe_B,
                        Boundary_condition, x0, g_ij, P_profile,
                        imsize, resolution, delta_t, side, type_f='species'):

        num_node = len(network['pore.all'])
        Num = max((num_node // 25000), 2)

        B = coo_matrix((coe_B, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        A0 = (B.T + B).tolil()
        del B
        A = coo_matrix((coe_A, (network['throat.conns'][:, 0], network['throat.conns'][:, 1])),
                       shape=(num_node, num_node), dtype=np.float64).tocsr()
        AH = coo_matrix((coe_A_i, (network['throat.conns'][:, 1], network['throat.conns'][:, 0])),
                        shape=(num_node, num_node), dtype=np.float64).tocsr()
        A1 = (AH + A).tolil()
        A = (A0 - A1).tolil()

        if type_f == 'species':
            alpha = network['pore.density'] * 0 + 1
            cond = network['pore.diffusivity']
        elif type_f == 'heat':
            alpha = network['pore.density'] * network['pore.Cp']
            cond = network['pore.lambda']
        elif type_f == 'density':
            alpha = network['pore.density'] * 0 + 1
            cond = network['pore.diffusivity']

            # dig,b=self.setting_Boundary_condition(network,g_ij,P_profile,dig,b,Boundary_condition,resolution,imsize,Num,side,'diffusion')
        dig = -np.array(A.sum(axis=0)).reshape(num_node)
        b = np.zeros(num_node)
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                if 'solid' in m:
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_s = np.pi * cond * network['pore.radius'] ** 2 / resolution
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Neumann':
                        value = net().getting_zoom_value(network, side, imsize, resolution)
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 * value
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']] * 0
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] -= tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                    elif Boundary_condition[m][n][1] == 'Robin':
                        T_conductivity_s = np.pi * network['pore.radius'] ** 2 / (resolution / cond)
                        tem_dig = T_conductivity_s[network[n] & network['pore.solid']]
                        b_dig = Boundary_condition[m][n][0] * T_conductivity_s[network[n] & network['pore.solid']]
                        dig[network[n] & network['pore.solid']] += tem_dig
                        b[network[n] & network['pore.solid']] += b_dig
                elif 'pore' in m:
                    # P_conductivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])##
                    P_conductivity = tool().mass_balance_conv(network, g_ij, P_profile,
                                                              network['pore._id'][network[n] & network['pore.void']])
                    if Boundary_condition[m][n][1] == 'Dirichlet':
                        T_conductivity_f = np.pi * cond * network['pore.radius'] ** 2 / resolution

                        if 'inlet' in m:
                            convection_term = abs(alpha[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(Boundary_condition_P[m][n][0]-P_profile[network[n]&network['pore.void']])

                            diffusion_term = T_conductivity_f[network[n] & network['pore.void']]
                            tem_b = Boundary_condition[m][n][0] * (convection_term + diffusion_term)
                            tem_dig = T_conductivity_f[network[n] & network['pore.void']]
                        else:
                            tem_b = 0
                            tem_dig = abs(alpha[network[n] & network[
                                'pore.void']] * P_conductivity)  # *(P_profile[network[n]&network['pore.void']]-Boundary_condition_P[m][n][0]))
                    elif Boundary_condition[m][n][1] == 'Neumann':

                        if 'inlet' in m:
                            tem_dig = 0
                            tem_b = Boundary_condition[m][n][0] * network['pore.radius'][
                                network[n] & network['pore.void']] ** 2 * np.pi
                        else:
                            tem_dig = abs(alpha[network[n] & network['pore.void']] * P_conductivity)
                            tem_b = 0
                    dig[network[n] & network['pore.void']] += tem_dig
                    b[network[n] & network['pore.void']] += tem_b

        A_c = A.copy()  # copy A

        # _----------------------------steady-state-------------------------------#

        # Var_c=np.copy(x0)

        delta_dig = alpha * network['pore.volume'] / delta_t
        dig_c = dig + delta_dig

        # diagonal should update for delta_t
        A_c.setdiag(dig_c, 0)  # add diagonal into A_c
        A_c = A_c.tocsr()  # we transfer A_c for next calculation

        b_c = b + delta_dig * x0

        Var_c = pp.spsolve(A_c, b_c)
        # Tem_c,j=ssl.bicg(A_c,b_c,tol=1e-9)

        return Var_c

    def RK4(self, h, a, a_max, initial):
        func_n = lambda T: 1 / (1 / 2.976 + 0.377 * (1 - 293.15 / T))  # n0=1,alph=0 for Langmuir-Freundirch
        func_b = lambda T: 4.002 * np.exp(51800 / 8.314 / 273.15 * (
                293.15 / T - 1))  # b0=4.002, delta_E=65572,R=8.314,T0=273.15,for Langmuir-Freundirch

        func_q_eq = lambda c, T: 19 * ((func_b(T) * c * 8.314 * T) ** (1 / func_n(T))) / (
                1 + ((func_b(T) * c * 8.314 * T) ** (1 / func_n(T))))  # for Langmuir-Freundirch
        func_q_eq_pw = lambda c, T: 19 * (func_b(T) * c * 8.314 * T) ** (1 / func_n(T) - 1) / (
                func_n(T) * (1 + ((func_b(T) * c * 8.314 * T) ** (1 / func_n(T)))) ** 2)

        func_k = lambda c, T: 7 / 1152 / 8.314 / T / func_q_eq_pw(c, T)
        func_delta_H = lambda T, q: 65572 - 0.377 * 8.314 * 293.15 * func_n(T) ** 2 * np.log(q / (19 - q))

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
