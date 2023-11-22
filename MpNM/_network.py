#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:00:02 2022

@author: htmt
"""
from _Base import *
import numpy as np

class network(Base):
    def __init__(self):
        pass
    def read_network(self,path=None,name=None):
        lines=[]
        with open(path+'/'+name+'_node1.dat', "r") as f:
            first_line = f.readline()
        f.close()
        a = (first_line.split())  
        #L = float(a[1])
        #Area = float(a[2])*float(a[3])
        
        #exit()
         

        
        Throats1 = np.loadtxt(path+'/'+name+"_link1.dat", skiprows=1)
        Throats2 = np.loadtxt(path+'/'+name+"_link2.dat")
        Pores    = np.loadtxt(path+'/'+name+"_node2.dat")
        Pores1   = np.loadtxt(path+'/'+name+'_node1.dat',skiprows=1, usecols=(0,1,2,3,4))
        '''
        Throats1 = np.loadtxt("test1D_link1.dat", skiprows=1)
        Throats2 = np.loadtxt("test1D_link2.dat")
        Pores   = np.loadtxt("test1D_node2.dat")
        Pores1   = np.loadtxt('test1D_node1.dat',skiprows=1, usecols=(0,1,2,3,4))
        #'''
        
        
        Pores    = Pores[Pores1[:,4]>0,:] #
        Pores1    = Pores1[Pores1[:,4]>0,:] #  
        nonIsolatedPores = Pores[:,0]
        newPore = np.zeros(len(Pores1[:,0]))
        temp = np.arange(len(nonIsolatedPores))

        newPore[Pores1[:,4]>0] = temp
        inThroats = np.array(Throats1[:,1] * Throats1[:,2])<0
        outThroats = np.abs(Throats1[:,1] * Throats1[:,2])<1 
        network={}
        nP=len(nonIsolatedPores)
        nT=len(Throats1)
        network['pore._id']=temp
        network['pore.label']=nonIsolatedPores-1
        network['pore.all']=np.ones(nP).astype(bool)
        network['pore.volume']=Pores[:,1]
        network['pore.radius']=Pores[:,2]
        
        network['pore.shape_factor']=Pores[:,3]
        network['pore.clay_volume']=Pores[:,3]
        network['pore.coords']=Pores1[:,1:4]
        network['throat._id']=Throats1[:,0]
        network['throat.label']=Throats1[:,0]
        network['throat.all']=np.ones(nT).astype(bool)
        #network['throat.inlets']=inThroats
        #network['throat.outlets']=outThroats

        network['throat.inside']=~(inThroats|outThroats)
        network['throat.shape_factor']=Throats1[:,0]        
        throat_conns=Throats1[:,1:3]-[1,1]
        throat_conns[throat_conns[:,0]>throat_conns[:,1]]=(throat_conns[:,[1,0]])[throat_conns[:,0]>throat_conns[:,1]]
        throat_out_in=throat_conns[(throat_conns[:,0]<0)]
        index=np.digitize(throat_out_in[:,1],network['pore.label'])-1
        throat_out_in[:,1]=network['pore._id'][index]
        throat_internal=throat_conns[(throat_conns[:,0]>=0)]
        index=np.digitize(throat_internal[:,0],network['pore.label'])-1
        throat_internal[:,0]=network['pore._id'][index]
        index=np.digitize(throat_internal[:,1],network['pore.label'])-1
        throat_internal[:,1]=network['pore._id'][index]    
        network['throat.conns']=np.concatenate((throat_out_in,throat_internal),axis=0).astype(np.int64)
        network['pore.label']=temp
        network['pore.inlets']=~np.copy(network['pore.all'])
        network['pore.inlets'][np.array(network['throat.conns'][network['throat.conns'][:,0]==-2][:,1]).astype(int)]=True
        network['pore.outlets']=~np.copy(network['pore.all'])
        network['pore.outlets'][np.array(network['throat.conns'][network['throat.conns'][:,0]==-1][:,1]).astype(int)]=True
        network['throat.radius']=Throats1[:,3]
        network['throat.shape_factor']=Throats1[:,4]
        network['throat.length']=Throats2[:,5]
        network['throat.total_length']=Throats1[:,5]
        network['throat.conduit_lengths_pore1']=Throats2[:,3]
        
        network['throat.conduit_lengths_pore2']=Throats2[:,4]
        
        network['throat.conduit_lengths_throat']=Throats2[:,5]
        
        BndG1 = (np.sqrt(3)/36+0.00001)
        BndG2 = 0.07
        network['throat.real_shape_factor']=network['throat.shape_factor']
        network['throat.real_shape_factor'][(network['throat.shape_factor']>BndG1)&(network['throat.shape_factor']<=BndG2)]=1/16
        network['throat.real_shape_factor'][(network['throat.shape_factor']>BndG2)]=1/4/np.pi
        network['pore.real_shape_factor']=network['pore.shape_factor']
        network['pore.real_shape_factor'][(network['pore.shape_factor']>BndG1)&(network['pore.shape_factor']<=BndG2)]=1/16
        network['pore.real_shape_factor'][(network['pore.shape_factor']>BndG2)]=1/4/np.pi
        network['throat.real_k']=network['throat.all']*0.6
        network['throat.real_k'][(network['throat.shape_factor']>BndG1)&(network['throat.shape_factor']<=BndG2)]=0.5623
        network['throat.real_k'][(network['throat.shape_factor']>BndG2)]=0.5
        network['pore.real_k']=network['pore.all']*0.6
        network['pore.real_k'][(network['pore.shape_factor']>BndG1)&(network['pore.shape_factor']<=BndG2)]=0.5623
        network['pore.real_k'][(network['pore.shape_factor']>BndG2)]=0.5
        return network
        
        
        
        
        
        
