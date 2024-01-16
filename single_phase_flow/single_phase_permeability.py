#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:22:24 2022

@author: htmt
"""

from MpNM.network import network
from MpNM.topotools import topotools
from MpNM.algorithm import algorithm
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--Networkfile', default='../sample_data/Bead_packing', help='input network path')
parser.add_argument('--Networkname', default='Bead_packing_1000_25', help='network file name')
parser.add_argument('--out_dir_model', default='../mod_out_new', help='output file for model')
parser.add_argument('--imsize', type=int, default=[1000, 1000, 1000], help='network size')
parser.add_argument('--resolution', default=25e-6, help='network resolution')
parser.add_argument('--fluid_viscosity', default=1e-3, help='fluid viscosity')
parser.add_argument('--fluid_density', default=998, help='fluid density')
opt = parser.parse_args()

t0 = time.time()
path = opt.Networkfile
name = opt.Networkname
pn_o = network().read_network(path=path, name=name)

health = topotools().pore_health(pn_o)

back = topotools().trim_pore(pn_o, health['single_pore'], health['single_throat'])

# pn=op.network.GenericNetwork(name='pn')
# pn.update(back)
pn = {}
pn.update(back)

imsize = np.array(opt.imsize)
resolution = opt.resolution
fluid = {}
fluid['density'] = opt.fluid_density  # kg/m^3

fluid['viscosity'] = opt.fluid_viscosity  # Pa*s

for j in fluid:
    pn['pore.' + j] = fluid[j] * pn['pore.all']
    pn['throat.' + j] = fluid[j] * pn['throat.all']

pn['throat.void'] = pn['throat.all']
pn['pore.void'] = pn['pore.all']
pn['pore._id'] = np.arange(len(pn['pore._id']))
pn['throat._id'] = np.arange(len(pn['throat._id']))
Pores = np.loadtxt(path + '/' + name + '_node2.dat')
Pores1 = np.loadtxt(path + '/' + name + '_node1.dat', skiprows=1, usecols=(0, 1, 2, 3, 4))
Pores = Pores[Pores1[:, 4] > 0, :]
nonIsolatedPores = Pores[:, 0]
newPore = np.zeros(len(Pores1[:, 0]))
temp = np.arange(len(nonIsolatedPores))

newPore[Pores1[:, 4] > 0] = temp
Throats1 = np.loadtxt(path + '/' + name + '_link1.dat', skiprows=1)
Throats2 = np.loadtxt(path + '/' + name + '_link2.dat')
throat_inlet2 = Throats2[Throats1[:, 1] == -1]
throat_inlet1 = Throats1[Throats1[:, 1] == -1]
throat_outlet2 = Throats2[Throats1[:, 1] == 0]
throat_outlet1 = Throats1[Throats1[:, 1] == 0]

throat_inlet_cond = topotools().Boundary_cond_cal(pn, throat_inlet1, throat_inlet2, fluid, newPore, Pores)
throat_outlet_cond = topotools().Boundary_cond_cal(pn, throat_outlet1, throat_outlet2, fluid, newPore, Pores)
bound_cond = {}
bound_cond['throat_inlet_cond'] = throat_inlet_cond
bound_cond['throat_outlet_cond'] = throat_outlet_cond

model_res = []
Boundary_condition_P = {}
Boundary_condition_P['pore_inlet'] = {'pore.inlets': [1, 'Dirichlet']}
Boundary_condition_P['pore_outlet'] = {'pore.outlets': [0, 'Dirichlet']}

coe_A = np.array(topotools().Mass_conductivity(pn)) / pn['throat.total_length']
coe_A_P = coe_A
Profile = algorithm().stead_stay_alg(pn, fluid, coe_A, Boundary_condition_P, resolution, bound_cond)
delta_p = np.array([Profile[k[1]] - Profile[k[0]] for k in pn['throat.conns']])
flux_Throat_profile = delta_p * coe_A
Vel_Throat_profile = flux_Throat_profile / pn['throat.radius'] ** 2 * 4 * pn['throat.real_shape_factor']
output = topotools().calculate_mass_flow(pn, Boundary_condition_P, fluid, coe_A_P, Profile, 8)
abs_perm = output['pore.inlets'] / (
        Boundary_condition_P['pore_inlet']['pore.inlets'][0] - Boundary_condition_P['pore_outlet']['pore.outlets'][
    0])
abs_perm *= fluid['viscosity'] * imsize[0] / (imsize[1] * imsize[2]) / resolution
print('Mass Balance', output)
print('Absolute permeability', abs_perm)
print('inlet velocity', output['pore.inlets'] / (imsize[1] * imsize[2] * resolution ** 2))
t1 = time.time()
print('using time', t1 - t0, 's')
