import numpy as np


@np.vectorize
def calculate_mu(T):
    if 273.15 <= T < 413.15:
        mu = 1.3799566804 - 0.021224019151 * T ** 1 + 1.3604562827E-4 * T ** 2 - 4.6454090319E-7 * T ** 3 + 8.9042735735E-10 * T ** 4 - 9.0790692686E-13 * T ** 5 + 3.8457331488E-16 * T ** 6
    elif 413.15 <= T <= 553.75:
        mu = 0.00401235783 - 2.10746715E-5 * T ** 1 + 3.85772275E-8 * T ** 2 - 2.39730284E-11 * T ** 3
    else:
        raise ValueError("T should in range 273.15<=T<=553.75")
    return mu


@np.vectorize
def calculate_Cp(T):
    if 273.15 <= T <= 553.75:
        Cp = 12010.1471 - 80.4072879 * T ** 1 + 0.309866854 * T ** 2 - 5.38186884E-4 * T ** 3 + 3.62536437E-7 * T ** 4
    else:
        raise ValueError("T should in range 273.15<=T<=553.75")
    return Cp


@np.vectorize
def calculate_rho(T):
    if 273.15 <= T < 293.15:
        rho = 0.000063092789034 * T ** 3 - 0.060367639882855 * T ** 2 + 18.9229382407066 * T - 950.704055329848
    elif 293.15 <= T <= 373.15:
        rho = 0.000010335053319 * T ** 3 - 0.013395065634452 * T ** 2 + 4.969288832655160 * T + 432.257114008512
    else:
        raise ValueError("T should in range 273.15<=T<=373.15")

    return rho


@np.vectorize
def calculate_k(T):
    if 273.15 <= T <= 553.75:
        k = -0.869083936 + 0.00894880345 * T ** 1 - 1.58366345E-5 * T ** 2 + 7.97543259E-9 * T ** 3
    else:
        raise ValueError("T should in range 273.15<=T<=553.75")
    return k


if __name__ == "__main__":
    T = 293.15
    print(calculate_mu(T))
    print(calculate_rho(T))
    print(calculate_Cp(T))
    print(calculate_k(T))
    print(calculate_mu(T) / calculate_rho(T))
