#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:02:15 2022

@author: htmt
"""

import numpy as np 

class Base(dict):
    def __init__(self,Np=0,Nt=0,network=None,name=None,project=None,settings=None):
        super().__init__()
        
        self.update({'pore.all':np.ones(shape=(Np,),dtype=bool)})
        self.update({'throat.all':np.ones(shape=(Nt,),dtype=bool)})