import numpy as np
from calculate_physical_properties import *
from typing import Union, Literal

# from CoolProp.CoolProp import PropsSI
#
# rho_Nitrogen = PropsSI('D', 'T', 298.15, 'P', 101325, 'Nitrogen')
# mu_Nitrogen = PropsSI('V', 'T', 298.15, 'P', 101325, 'Nitrogen')
# print(rho_Nitrogen)
# print(mu_Nitrogen)
# mu_water = PropsSI('V', 'T', 303.15, 'P', 101325, 'water')
# print(mu_water)
# np.set_printoptions(precision=50, suppress=True)
# a = 0.001
# b = 0.001


a = 0.0005
b = 0.0005
r = 5e-5
d = r * 2

porosity = 0.36
# real_inlet_area = 0.001088871610779609
real_inlet_area = a * b * porosity
real_inlet_area = 8.980615111747202E-8
De = 2 * a * b / (a + b)

T_fluid = 293.15
rho_fluid = calculate_rho(T_fluid)
mu_fluid = calculate_mu(T_fluid)

print("rho_fluid = ", rho_fluid)
print("mu_fluid = ", mu_fluid)
print("porosity = ", porosity)
Re = np.array([0.001,0.005,0.02,0.1,1])


def calculate_block_velocity(
    Re,
    a,
    b,
    rho,
    mu,
    real_area=None,
):
    result = []
    De = 2 * a * b / (a + b)
    Q = Re * mu * a * b / (De * rho)
    v_s = Q / (a * b)
    result.append(v_s)
    if real_area is not None:
        v_real = Q / real_area
        result.append(v_real)

    result = np.array(result)
    return result if len(result) > 1 else result[0]


def calculate_block_velocity_D(
    Re,
    a,
    b,
    rho,
    mu,
    real_area=None,
):
    result = []
    v_s = Re * mu / rho / d
    Q = v_s * a * b
    result.append(v_s)
    if real_area is not None:
        v_real = Q / real_area
        result.append(v_real)

    result = np.array(result)
    return result if len(result) > 1 else result[0]


v_s, v_real = calculate_block_velocity(
    Re, a, b, rho_fluid, mu_fluid, real_area=real_inlet_area
)
# v_real = np.round(v_real, 10)
print("Re = ", Re)
print("v_s = ", v_s)
print("v_real = ", v_real)