import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calculate_physical_properties import *
import sympy as sp

epsilon,mu_g,u,d_p,rho,alpha,beta,H=sp.symbols('epsilon mu_g u d_p rho alpha beta H')

delta_P=(alpha*(1-epsilon)**2*mu_g/(epsilon**3*d_p**2)*u+beta*(1-epsilon)/(epsilon**3*d_p)*rho*u**2)*H
print(delta_P)
alpha = 150
beta = 1.75
a = 0.05
b = 0.05
De = 2*a*b/(a+b)
T_fluid = 303.15
rho_fluid = calculate_rho(T_fluid)
mu_fluid = calculate_mu(T_fluid)
porosity_big_swell2 = 4.469630963046977E-5/1.25E-4
print(porosity_big_swell2)
mu_g=mu_fluid
rho=rho_fluid
def calculate_block_Re(a, b,Q,rho,mu,porosity=None,return_v=False):
    result = []
    De = 2*a*b/(a+b)
    if porosity is None:
        v = Q / (a * b)
        Re = De*v*rho/mu
    else:
        v = Q / (a * b * porosity)
        Re = De*v*rho/mu
    result.append(Re)
    if return_v:
        result.append(v)
    return result if len(result)>1 else result[0]



# Read the data from the CSV files
comsol_data = pd.read_csv('./3D_1000_1000_1000_results/3D_1000_1000_1000_results.csv',index_col=0)

# pnm_origin = pd.read_csv('./3D_500_results/res_PNM_origin.csv',index_col=0)
# pnm_adjust = pd.read_csv('./3D_500_results/res_PNM_adjust.csv',index_col=0)
#3.5597165282751318E-6

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(comsol_data['vel'],comsol_data['inlet_pressure'],label='Comsol')
# ax[0].plot(pnm_origin['vel'],pnm_origin['inlet_pressure'],label='PNM origin')
# ax[0].plot(pnm_adjust['vel'],pnm_adjust['inlet_pressure'],label='PNM adjust')
ax[0].scatter(comsol_data['vel'],comsol_data['inlet_pressure'],label='Comsol')
# ax[0].scatter(pnm_origin['vel'],pnm_origin['inlet_pressure'],label='PNM origin')
# ax[0].scatter(pnm_adjust['vel'],pnm_adjust['inlet_pressure'],label='PNM adjust')
ax[0].set_xlabel('Velocity (m/s)')
ax[0].set_ylabel('Inlet pressure (Pa)')
ax[0].legend()

# ax[1].plot(comsol_data['vel'],comsol_data['heat_flux'],label='Comsol')
# ax[1].plot(pnm_origin['vel'],pnm_origin['heat_flux'],label='PNM origin')
# ax[1].plot(pnm_adjust['vel'],pnm_adjust['heat_flux'],label='PNM adjust')
# ax[1].scatter(comsol_data['vel'],comsol_data['heat_flux'],label='Comsol')
# ax[1].scatter(pnm_origin['vel'],pnm_origin['heat_flux'],label='PNM origin')
# ax[1].scatter(pnm_adjust['vel'],pnm_adjust['heat_flux'],label='PNM adjust')
# ax[1].set_xlabel('Velocity (m/s)')
# ax[1].set_ylabel('Heat flux (W)')
# ax[1].legend()
plt.show()

Q = comsol_data['volume_flux']
Re,v = calculate_block_Re(a, b, Q, rho_fluid, mu_fluid,return_v=True)
comsol_data['Re']=Re
# print(comsol_data)
# print(Re)
delta_P_ergun = []
for Q_i in Q:
    delta_P_subs={'epsilon':porosity_big_swell2,
                  'mu_g':mu_g,
                  'u':Q_i/0.05**2,
                  'd_p':0.00005*52*2,
                  'rho':rho,
                  'alpha':alpha,
                  'beta':beta,
                  'H':0.05}
    delta_P_ergun.append(delta_P.subs(delta_P_subs))
comsol_data['delta_P_ergun'] = delta_P_ergun
print(comsol_data)
fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(Re,comsol_data['inlet_pressure'],label='Comsol')
# ax[0].plot(Re,pnm_origin['inlet_pressure'],label='PNM origin')
# ax[0].plot(Re,pnm_adjust['inlet_pressure'],label='PNM adjust')
ax.scatter(Re,comsol_data['inlet_pressure'],label='Comsol')
# ax[0].scatter(Re,pnm_origin['inlet_pressure'],label='PNM origin')
# ax[0].scatter(Re,pnm_adjust['inlet_pressure'],label='PNM adjust')

ax.scatter(Re,comsol_data['delta_P_ergun'],label='delta_P_ergun')
ax.plot(Re,comsol_data['delta_P_ergun'],label='delta_P_ergun')
ax.set_xlabel('Re')
ax.set_ylabel('Inlet pressure (Pa)')
ax.legend()

# ax[1].plot(Re,comsol_data['heat_flux'],label='Comsol')
# ax[1].plot(Re,pnm_origin['heat_flux'],label='PNM origin')
# ax[1].plot(Re,pnm_adjust['heat_flux'],label='PNM adjust')
# ax[1].scatter(Re,comsol_data['heat_flux'],label='Comsol')
# ax[1].scatter(Re,pnm_origin['heat_flux'],label='PNM origin')
# ax[1].scatter(Re,pnm_adjust['heat_flux'],label='PNM adjust')
# ax[1].set_xlabel('Re')
# ax[1].set_ylabel('Heat flux (W)')
# ax[1].legend()
plt.show()
comsol_data.to_csv('3D_1000_1000_1000_results/3D_1000_1000_1000_results_Re.csv')