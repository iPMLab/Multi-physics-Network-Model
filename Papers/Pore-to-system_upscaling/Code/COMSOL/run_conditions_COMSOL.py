import mph
import numpy as np
from joblib import Parallel, delayed
import os

# vel_list = np.linspace(0.0001,0.01,20)
vel_list=np.array([4.37081819e-05,4.37081819e-04,4.37081819e-03,8.74163637e-03,1.31124546e-02,1.74832727e-02,2.18540909e-02,2.62249091e-02,3.05957273e-02,3.49665455e-02,3.93373637e-02,4.37081819e-02])
vel_list = np.round(vel_list,4)
root_path = './3D_1000_1000_1000_results/'
model_name = '3D_1000_1000_1000_swell_slip_0.01.mph'
client = mph.start(cores=64)
model = client.load(root_path+model_name)
inlet_vel_node = model / 'physics/Laminar Flow/Inlet 1'

skip_list=np.array([0.000])
for vel in vel_list:
    if np.isin(vel,skip_list):
        print(f'Skipping {vel:.4f} m/s')
        continue
    print(f'Running simulation for {vel:.4f} m/s')
    inlet_vel_node.property('U0in', vel)
    model.solve()
    model.save(root_path+f'3D_1000_1000_1000_swell_slip_{vel:.4f}.mph')



# def worker(vel):
#     client = mph.start(cores=16)
#     model = client.load('./3D_500_results/3D_500_0.001.mph')
#     inlet_vel_node=model/'physics/Laminar Flow/Inlet 1'
#     inlet_vel_node.property('Uavfdf', vel)
#     model.solve()
#     model.save(f'./3D_500_results/3D_500_{vel:.4f}.mph')
#     del client
#
# Parallel(n_jobs=4)(delayed(worker)(vel) for vel in vel_list)