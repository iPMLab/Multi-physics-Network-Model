#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:02:52 2022

@author: htmt
"""
from MpNM.Base import *
import numpy as np

class phase(Base):
    def __init__(self,pn,settings=None,properties=None):
        self['pore.all']=pn['pore.all']
        self['throat.all']=pn['throat.all']
        for i in properties:
            self['pore'+'.'+i]=properties[i]*self['pore.all']
            self['throat'+'.'+i]=properties[i]*self['throat.all']

    @staticmethod
    def H_P_fun(radius,length,visocity):
        return np.pi*radius**4/8/visocity/length

    def set_cond_t(self,pn):
        g_ij=[]
        for i in pn['throat._id']:
            k=pn['throat.conns'][i]
            ri=pn['pore.radius'][k[0]]
            rj=pn['pore.radius'][k[1]]
            rt=pn['throat.radius'][i]
            lt=pn['throat.length'][i]
            li=lt*(1-0.6*rt/ri)
            lj=lt*(1-0.6*rt/rj)
            g_i=self.H_P_fun(ri,li,self['throat.visocity'][i])
            g_j=self.H_P_fun(rj,lj,self['throat.visocity'][i])
            g_t=self.H_P_fun(rt,lt,self['throat.visocity'][i])
            g_ij.append((li+lj+lt)/(li/g_i+lj/g_j+lt/g_t))
        self['throat.cond']=g_ij
        #print(self)
        return g_ij

    @staticmethod
    def Mass_conductivity(pn,fluid):
    
        g_ij=[]
        for i in pn['throat._id']:
            k=pn['throat.conns'][i]
            ri=pn['pore.radius'][k[0]]
            rj=pn['pore.radius'][k[1]]
            rt=pn['throat.radius'][i]
            lt=pn['throat.length'][i]
            #lt=pn['throat.length'][i]/(3-0.6*(rt*(ri+rj)/ri/rj))
            #li=lt*(1-0.5*rt/ri)
            #lj=lt*(1-0.5*rt/rj)
            li=pn['throat.conduit_lengths.pore1'][i]
            lj=pn['throat.conduit_lengths.pore2'][i]
            lt=pn['throat.conduit_lengths.throat'][i]
            cond=lambda r,G,k,v: (r**4)/16/G*k/v
            g_i=cond(ri,pn['pore.real_shape_factor'][k[0]],pn['pore.real_k'][k[0]],fluid['visocity'])
            g_j=cond(rj,pn['pore.real_shape_factor'][k[1]],pn['pore.real_k'][k[1]],fluid['visocity'])
            g_t=cond(rt,pn['throat.real_shape_factor'][i],pn['throat.real_k'][i],fluid['visocity'])
            g_ij.append((li+lj+lt)/(li/g_i+lj/g_j+lt/g_t))
            #g_ij.append((1)/(1/g_i+1/g_j+1/g_t))
        
        return g_ij
