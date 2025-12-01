import mph
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

vel_list=np.array([4.37081819e-04,4.37081819e-03,8.74163637e-03,1.31124546e-02,1.74832727e-02,2.18540909e-02,2.62249091e-02,3.05957273e-02,3.49665455e-02,3.93373637e-02,4.37081819e-02])
vel_list = np.round(vel_list,4)
client = mph.start()

df = pd.DataFrame(columns=['vel','volume_flux', 'inlet_pressure','heat_flux'])
inlet_surface_list = [1, 7, 74, 268, 333, 519, 620,627]
outlet_surface_list = [10841, 10844, 10867, 10897, 10898, 10903, 10905, 10906]
for i in range(len(vel_list)):
    vel = vel_list[i]
    model = client.load(f'./3D_1000_1000_1000_results/3D_1000_1000_1000_swell_slip_{vel_list[i]:.4f}.mph')
    model.parameter('rho_water','995.6205500412177[kg/m^3]')
    model.parameter('Cp_water','4179.657056812472[J/(K*kg)]')

    # pressure at inlet
    if (model / 'evaluations/Inlet_pressure').exists():
        (model / 'evaluations/Inlet_pressure').remove()
    Surface_Average_Inlet_pressure = (model / 'evaluations').create("AvSurface", name='Inlet_pressure')
    Surface_Average_Inlet_pressure.property('expr', 'p')
    Surface_Average_Inlet_pressure.select(inlet_surface_list)
    Inlet_pressure_i=np.array(Surface_Average_Inlet_pressure.java.getReal()).flatten()[0]

    # heat flux at outlet
    if (model / 'evaluations/Outlet_heat_flux').exists():
        (model / 'evaluations/Outlet_heat_flux').remove()
    Surface_Integral_Outlet_heat_flux = (model / 'evaluations').create("IntSurface", name='Outlet_heat_flux')
    Surface_Integral_Outlet_heat_flux.property('expr', ['(nx*u+ny*v+nz*w)*(T-293.15)*995.6205500412177[kg/m^3]*4179.657056812472[J/(K*kg)]'])
    Surface_Integral_Outlet_heat_flux.select(outlet_surface_list)
    Outlet_heat_flux_i=np.array(Surface_Integral_Outlet_heat_flux.java.getReal()).flatten()[0]

    # volume flux at outlet
    # if (model / 'evaluations/Outlet_volume_flux').exists():
    #     (model / 'evaluations/Outlet_volume_flux').remove()
    # Surface_Integral_Outlet_volume_flux = (model / 'evaluations').create("IntSurface", name='Outlet_volume_flux')
    # Surface_Integral_Outlet_volume_flux.property('expr', ['nx*u+ny*v+nz*w'])
    # Surface_Integral_Outlet_volume_flux.select(outlet_surface_list)
    # Outlet_volume_flux_i=np.array(Surface_Integral_Outlet_volume_flux.java.getReal()).flatten()[0]

    # volume flux at inlet
    if (model / 'evaluations/Inlet_volume_flux').exists():
        (model / 'evaluations/Inlet_volume_flux').remove()
    Surface_Integral_Inlet_volume_flux = (model / 'evaluations').create("AvSurface", name='Inlet_volume_flux')
    Surface_Integral_Inlet_volume_flux.property('expr', ['spf.out1.volumeFlowRate'])
    Surface_Integral_Inlet_volume_flux.select(inlet_surface_list)
    Inlet_volume_flux_i=np.array(Surface_Integral_Inlet_volume_flux.java.getReal()).flatten()[0]
    print(vel, Inlet_volume_flux_i, Inlet_pressure_i, Outlet_heat_flux_i)
    res = [vel, Inlet_volume_flux_i, Inlet_pressure_i, Outlet_heat_flux_i]
    df.loc[i] = res
    # model.save()

df.to_csv('3D_1000_1000_1000_results/3D_1000_1000_1000_results.csv')