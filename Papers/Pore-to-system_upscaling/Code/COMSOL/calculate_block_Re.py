import numpy as np
from calculate_physical_properties import *

vel = 0.01
vel = 0.0437
a = 0.05
b = 0.05
r = 5e-5
d = r * 2
De = 2 * a * b / (a + b)
Q = 9.15059111838387e-4 * vel
print(Q)
T_fluid = 303.15
rho_fluid = calculate_rho(T_fluid)
mu_fluid = calculate_mu(T_fluid)

print(mu_fluid)
porosity = 4.4696309775898035e-5 / 1.250000000000003e-4
print("porosity = ", porosity)


def calculate_block_Re(a, b, Q, rho, mu, return_v=False):
    result = []
    De = 2 * a * b / (a + b)
    v = Q / (a * b)
    Re = De * v * rho / mu
    result.append(Re)
    if return_v:
        result.append(v)
    return result if return_v else result[0]


def calculate_diameter_Re(rho, mu, return_v=False):
    result = []
    De = 2 * r


Re, v = calculate_block_Re(a, b, Q, rho_fluid, mu_fluid, return_v=True)
print("Re = ", Re)
print("v = ", v)
print("v_real = ", v / porosity)
