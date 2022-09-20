#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:48:35 2022

@author: Mingliang qu
"""

from scipy.sparse import coo_matrix
import scipy.sparse.linalg as ssl
import pypardiso as pp
import numpy as np 
from joblib import Parallel, delayed
from _Base import *
from _topotools import topotools as tool


class algorithm(Base):

    def stead_stay_alg(self,network,fluid,coe_A,Boundary_condition,resolution,bound_cond):
        num_pore=len(network['pore.all'])
        A=coo_matrix((coe_A,(network['throat.conns'][:,1],network['throat.conns'][:,0])),
                   shape=(num_pore,num_pore),dtype=np.float64).tolil()
        A=(A.getH()+A).tolil()
        dig=np.array(A.sum(axis=0)).reshape(num_pore)
        b=np.zeros(num_pore)
        #B=A.toarray()
        #mean_gil=np.max(coe_A)
        #condctivity=H_P_fun(network['pore.radius'],resolution,fluid['viscosity'])          
        for m in Boundary_condition:
            for n in  Boundary_condition[m]:
                condctivity=tool().H_P_fun(network['pore.radius'],resolution,network['pore.viscosity'])##  
                if bound_cond==False:
                    bound_cond={}
                    index=np.argwhere(network[n]==True)
                    value=condctivity[index]
                    bound_cond['throat_inlet_cond']=np.stack([index.flatten(),value.flatten()]).T 
                    bound_cond['throat_outlet_cond']=np.stack([index.flatten(),value.flatten()]).T 
                    throat_inlet_cond= bound_cond['throat_inlet_cond']
                   
                    
                    throat_outlet_cond= bound_cond['throat_outlet_cond']
                    bound_cond=False
                else:
                    throat_inlet_cond= bound_cond['throat_inlet_cond']
                   
                    
                    throat_outlet_cond= bound_cond['throat_outlet_cond']
                    
                #area_i=(imsize*resolution)**2/sum(network['pore.radius'][network[n]]**2*np.pi)
                condctivity=tool().H_P_fun(network['pore.radius'],resolution,network['pore.viscosity'])##
                if 'solid' in m:
                    dig[network[n]]+=condctivity[network[n]]
                    b[network[n]]-=Boundary_condition[m][n][0]*condctivity[network[n]]
                elif 'pore' in m:
                    if 'inlet' in m:
                        condctivity[throat_inlet_cond[:,0].astype(int)]=throat_inlet_cond[:,1] #if bound_cond!=False else condctivity[throat_inlet_cond[:,0].astype(int)]
                        dig[throat_inlet_cond[:,0].astype(int)]+=condctivity[throat_inlet_cond[:,0].astype(int)]
                        b[throat_inlet_cond[:,0].astype(int)]-=Boundary_condition[m][n][0]*condctivity[throat_inlet_cond[:,0].astype(int)]
                    elif 'outlet' in m:
                        condctivity[throat_outlet_cond[:,0].astype(int)]=throat_outlet_cond[:,1] #if bound_cond!=False else condctivity[throat_outlet_cond[:,0].astype(int)]
                        dig[throat_outlet_cond[:,0].astype(int)]+=condctivity[throat_outlet_cond[:,0].astype(int)]
                        b[throat_outlet_cond[:,0].astype(int)]-=Boundary_condition[m][n][0]*condctivity[throat_outlet_cond[:,0].astype(int)]

        A.setdiag(-dig,0)
        A=A.tocsc()
        Profile=pp.spsolve(A, b)
        #Profile,j=ssl.bicg(A,b,tol=1e-9)
        return Profile

