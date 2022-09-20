#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:02:52 2022

@author: htmt
"""
from _Base import *
import numpy as np

class phase(Base):
    def __init__(self,network,settings=None,properties=None):
        self['pore.all']=network['pore.all']
        self['throat.all']=network['throat.all']
        for i in properties:
            self['pore'+'.'+i]=properties[i]*self['pore.all']
            self['throat'+'.'+i]=properties[i]*self['throat.all']
    def H_P_fun(self,radius,length,visocity):
        return np.pi*radius**4/8/visocity/length
    def set_cond_t(self,network):
        g_ij=[]
        for i in network['throat._id']:
            k=network['throat.conns'][i]
            ri=network['pore.radius'][k[0]]
            rj=network['pore.radius'][k[1]]
            rt=network['throat.radius'][i]
            lt=network['throat.length'][i]
            li=lt*(1-0.6*rt/ri)
            lj=lt*(1-0.6*rt/rj)
            
           
            g_i=self.H_P_fun(ri,li,self['throat.visocity'][i])
            g_j=self.H_P_fun(rj,lj,self['throat.visocity'][i])
            g_t=self.H_P_fun(rt,lt,self['throat.visocity'][i])
            g_ij.append((li+lj+lt)/(li/g_i+lj/g_j+lt/g_t))

        self['throat.cond']=g_ij
        #print(self)
        return g_ij
    def Mass_conductivity(self,network,fluid):
    
        g_ij=[]
        for i in network['throat._id']:
            k=network['throat.conns'][i]
            ri=network['pore.radius'][k[0]]
            rj=network['pore.radius'][k[1]]
            rt=network['throat.radius'][i]
            lt=network['throat.length'][i]
            #lt=network['throat.length'][i]/(3-0.6*(rt*(ri+rj)/ri/rj))
            #li=lt*(1-0.5*rt/ri)
            #lj=lt*(1-0.5*rt/rj)
            li=network['throat.conduit_lengths.pore1'][i]
            lj=network['throat.conduit_lengths.pore2'][i]
            lt=network['throat.conduit_lengths.throat'][i]

            cond=lambda r,G,k,v: (r**4)/16/G*k/v
            g_i=cond(ri,network['pore.real_shape_factor'][k[0]],network['pore.real_k'][k[0]],fluid['visocity'])
            g_j=cond(rj,network['pore.real_shape_factor'][k[1]],network['pore.real_k'][k[1]],fluid['visocity'])
            g_t=cond(rt,network['throat.real_shape_factor'][i],network['throat.real_k'][i],fluid['visocity'])
            
            g_ij.append((li+lj+lt)/(li/g_i+lj/g_j+lt/g_t))
            
            #g_ij.append((1)/(1/g_i+1/g_j+1/g_t))
        
        return g_ij
