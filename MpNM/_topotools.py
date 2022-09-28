#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:55:27 2022
# Part of the code reference openpnm
@author: Mingliang qu
"""
import numpy as np 
import openpnm as op
from joblib import Parallel, delayed
from _Base import *


class topotools(_Base):
    def find_surface(self,network,status,imsize,resolution,label_1='surface',label_2='surface'):
        size=12
        marker=np.ones((size,size,3))
        for i in np.arange(size):
            for j in np.arange(size):
                a=np.linspace(0.15,0.9,size)[i]
                b=np.linspace(0.15,0.9,size)[j]
                marker[i,j,0]=1.1
                marker[i,j,1]=a
                marker[i,j,2]=b
        if   status=='x':        
            marker_i=marker.reshape(size**2,3)
            marker_j=np.copy(marker_i)
            marker_j[:,0]=-0.1
        elif status =='y':
            marker_i=marker.reshape(size**2,3)
            marker_i[:,[0,1]]=marker_i[:,[1,0]]
            marker_j=np.copy(marker_i)        
            marker_j[:,1]=-0.1
        elif status =='z':
            marker_i=marker.reshape(size**2,3)
            marker_i[:,[0,2]]=marker_i[:,[2,0]]
            marker_j=np.copy(marker_i)
            marker_j[:,2]=-0.1
        markers_0=marker_j*imsize*resolution   
        markers_1=marker_i*imsize*resolution
        op.topotools.find_surface_pores(network,markers_0,label=label_1)    
        op.topotools.find_surface_pores(network,markers_1,label=label_2)

    def trim_surface(self,network):
        for i in ['left','right']:
            for j in ['back','front']:
                for k in ['bottom','top']:
                    
                    back=np.copy(network['pore.'+i+'_surface']*network['pore.'+j+'_surface'])
                    network['pore.'+i+'_surface'][back]=False
                    network['pore.'+j+'_surface'][back]=False
                    back=np.copy(network['pore.'+i+'_surface']*network['pore.'+k+'_surface'])
                    network['pore.'+i+'_surface'][back]=False
                    network['pore.'+k+'_surface'][back]=False
                    back=np.copy(network['pore.'+j+'_surface']*network['pore.'+k+'_surface'])
                    network['pore.'+j+'_surface'][back]=False
                    network['pore.'+k+'_surface'][back]=False    
                
    def devide_layer(self,network,n,size,resolution ):
        layer={}
        #step=size*resulation/n
        layer[0]={}#left right 
        layer[1]={}#back front
        layer[2]={}#bottom top
        for i in np.arange(3):
            index=min(network['pore.coords'][:,i])
            step=size[i]*resolution/n[i]
            for j in np.arange(n[i]):
                layer[i][j]=np.copy(network['pore.all'])
                layer[i][j][(network['pore.coords'][:,i]-index)<j*step]=False
                layer[i][j][(network['pore.coords'][:,i]-index)>(j+1)*step]=False
        return layer
    
    def pore_health(self,network):
        number=len(network['pore.all'])
        conns=np.copy(network['throat.conns'])
        health={}
        health['single_pore']=[]
        health['single_throat']=[]
    
        for i in np.arange(number):
            val0=len(conns[:,0][conns[:,0]==i])
            val1=len(conns[:,1][conns[:,1]==i])
            
            if val1+val0<1:
               
                health['single_pore'].append(i)
                ind0=np.argwhere(conns[:,0]==i)
                ind1=np.argwhere(conns[:,1]==i)
                if len(ind0)>0 or len(ind1)>0 :
                    health['single_throat'].append(np.concatenate((ind0,ind1))[0][0])

        return health
    

    def trim_pore(self,network,pores,throat):
        #count=len(network)
        backup={}
        for i in network:
            if 'pore' in i and 'throat' not in i:
                
                backup[i]=np.delete(network[i],pores,axis=0) 
            elif 'throat' in i:
                backup[i]=np.delete(network[i],throat,axis=0) 
        backup['pore._id']=np.arange(len(backup['pore.all']))

        conns=[]
        isothroat=[]
        for j in np.arange(len(backup['throat.conns'])):
            i=backup['throat.conns'][j]
            if np.argwhere(backup['pore.label']==i[0]).size>0 and np.argwhere(backup['pore.label']==i[1]).size>0:
                ind0=np.argwhere(backup['pore.label']==i[0])[0][0]
                ind1=np.argwhere(backup['pore.label']==i[1])[0][0]
                conns.append([ind0,ind1])
            else:
                isothroat.append(j)
        for i in network:
            if 'throat' in i:
                backup[i]=np.delete(backup[i],isothroat,axis=0)   

        backup['throat._id']=np.arange(len(backup['throat.all']))
        backup['throat.conns']=np.array(conns)
        backup['pore.label']=np.arange(len(backup['pore.all']))
        return backup
    
    def trim_phase(self,network,pores,throat):
        #count=len(network)
        backup={}
        for i in network:
            if 'pore' in i:
                
                backup[i]=np.delete(network[i],pores,axis=0) #if len(network[i].shape)>1 else np.delete(network[i],pores,axis=0)
            elif 'throat' in i:
                backup[i]=np.delete(network[i],throat,axis=0) #if len(network[i].shape)>1 else np.delete(network[i],throat,axis=0)
        backup['pore._id']=np.arange(len(backup['pore.all']))
        backup['throat._id']=np.arange(len(backup['throat.all']))
        return backup
    
    def find_if_surface(self,network,index):
        res=[]
        for j in index:
            b=0
            for i in ['right','left','back','front','top','bottom']:
                if network['pore.'+i+'_surface'][j]:
                    b='pore.'+i+'_surface'
                    res.append(b)
        return res 
    
    def find_whereis_pore(self,network,parameter,index):
        index1=np.sort(parameter)[index]
        index2=np.argwhere(parameter==index1)[0]
        index3=self.find_if_surface(network,index2)  
        return [index1,index2,index3] 
      
    def find_throat(self,network,a):
        ind1=np.argwhere(network['throat.conns'][:,0]==a)
        ind2=np.argwhere(network['throat.conns'][:,1]==a)
        res=np.append(ind1,ind2)
        return res
    
    def find_neighbor_ball(self,network,a):
        
        res={}
        res['total']=self.find_throat(network,a)
        res['solid']=[]
        res['pore']=[]
        res['interface']=[]
        num_pore=np.count_nonzero(network['pore.void']) if 'pore.void' in network else 0
        if res['total'].size>0:
            for i in res['total']:
                index=network['throat.conns'][i]
                tem=index if index[0]==a else [a,index[0]]
                if a <num_pore:
                    if tem[1]<num_pore:
                        res['pore'].append(np.append(tem,i))
                    else:
                        res['interface'].append(np.append(tem,i))
                else:
                    if tem[1]>=num_pore:
                        res['solid'].append(np.append(tem,i))
                    else:
                        res['interface'].append(np.append(tem,i))
    
                    
            res['pore']=np.array(res['pore'],dtype=np.int64)
            res['solid']=np.array(res['solid'],dtype=np.int64)
            res['interface']=np.array(res['interface'],dtype=np.int64)
            
            return res
        else:
            return 0
    def H_P_fun(self,r,l,vis):
        g=np.pi*r**4/8/vis/l
        return g
    
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
            li=network['throat.conduit_lengths_pore1'][i]
            lj=network['throat.conduit_lengths_pore2'][i]
            lt=network['throat.conduit_lengths_throat'][i]
            
            '''
            g_i=H_P_fun(ri,li,fluid['viscosity'])
            g_j=H_P_fun(rj,lj,fluid['viscosity'])
            g_t=H_P_fun(rt,lt,fluid['viscosity'])
            
            '''
            cond=lambda r,G,k,v: (r**4)/16/G*k/v
            g_i=cond(ri,network['pore.real_shape_factor'][k[0]],network['pore.real_k'][k[0]],network['pore.viscosity'][k[0]])
            g_j=cond(rj,network['pore.real_shape_factor'][k[1]],network['pore.real_k'][k[1]],network['pore.viscosity'][k[1]])
            g_t=cond(rt,network['throat.real_shape_factor'][i],network['throat.real_k'][i],network['throat.viscosity'][i])
            
            g_ij.append((li+lj+lt)/(li/g_i+lj/g_j+lt/g_t))
            
            #g_ij.append((1)/(1/g_i+1/g_j+1/g_t))
        
        return g_ij
    # this function the viscosity has not change
    def Boundary_cond_cal(self,network,throat_inlet1,throat_inlet2,fluid,newPore,Pores):
        throat_inlet_cond=[]
        BndG1 = (np.sqrt(3)/36+0.00001)
        BndG2 = 0.07
        for i in np.arange(len(throat_inlet1)):
            indP2 = newPore[throat_inlet1[i,2].astype(int)-1].astype(int)
            GT = 1/fluid['viscosity']*(throat_inlet1[i,3]**2/4/(throat_inlet1[i,4]))**2*throat_inlet1[i,4]*(throat_inlet1[i,4]<BndG1)*0.6
            GT += 1/fluid['viscosity']*throat_inlet1[i,3]**2*throat_inlet1[i,3]**2/4/8/(1/4/np.pi) *(throat_inlet1[i,4]>BndG2)
            GT += 1/fluid['viscosity']*(throat_inlet1[i,3]**2/4/(1/16))**2*(1/16)*((throat_inlet1[i,4]>=BndG1) * (throat_inlet1[i,4]<=BndG2))*0.5623
            GP2 = 1/fluid['viscosity']*(Pores[indP2,2]**2/4/(Pores[indP2,3]))**2*Pores[indP2,3]*(Pores[indP2,3]<BndG1)*0.6
            GP2 += 1/fluid['viscosity']*Pores[indP2,2]**2*Pores[indP2,2]**2/4/8/(1/4/np.pi) *(Pores[indP2,3]>BndG2)
            GP2 += 1/fluid['viscosity']*(Pores[indP2,2]**2/4/(1/16))**2*(1/16)*((Pores[indP2,3]>=BndG1) * (Pores[indP2,3]<=BndG2))*0.5623
            #LP1 = throat_inlet2[i,3]
            LP2 = throat_inlet2[i,4]
            LT  = throat_inlet2[i,5]
            throat_inlet_cond.append([indP2,1/(LT/GT+LP2/GP2)])
        
        return np.array(throat_inlet_cond)
    def energy_balance_conv(self,network,fluid,solid,g_ij,Tem,thermal_con_dual,P_profile,a):
        #res=find_neighbor_ball(network,[a])
        res=self.find_neighbor_ball(network,a)
        
        pressure=[]
        cond_f=[]
        cond_h=[]
        Temp_f=[]
        cp_data=[]
        density=[]
        #g_ij=H_P_fun(network['throat.radius'],network['throat.length'],fluid['viscosity'])
        
        #g_ij*=network['throat.void']
        #mean_gil=np.max(g_ij)
     
        #coe_A for convection heat transfer
        #_i for slecting direct of fluid  
        #thermal_con_dual=network['throat.solid']*solid['lambda']+network['throat.connect']*(solid['lambda'])+network['throat.void']*fluid['lambda'] #solid_pore
        coe_B =network['throat.radius']**2*np.pi/network['throat.length']*thermal_con_dual
        
        for i in res['pore']:
            pressure.append([P_profile[i[0]],P_profile[i[1]]])
            cond_f.append(g_ij[i[2]])
            Temp_f.append([Tem[i[0]],Tem[i[1]]])
            cond_h.append(coe_B[i[2]])
            cp_data.append(network['throat.Cp'][i[2]])
            density.append(network['throat.density'][i[2]])
        pressure,Temp_f=np.array(pressure),np.array(Temp_f)
        cp_data,density=np.array(cp_data),np.array(density)
        Temp_s=[]
        cond_h_s=[]
        for j in res['solid']:
            Temp_s.append([Tem[j[0]],Tem[j[1]]])
            cond_h_s.append(coe_B[j[2]])
        Temp_s=np.array(Temp_s)
        Temp_sf=[]
        cond_hs=[]
        for k in res['interface']:
            Temp_sf.append([Tem[k[0]],Tem[k[1]]])
            cond_hs.append(coe_B[k[2]])
        Temp_sf=np.array(Temp_sf)
        #mass_f=np.sum((pressure[:,0]-pressure[:,1])*cond_f) if len(pressure)>0 else 0
        #h_conv_f=np.sum((pressure[:,0]-pressure[:,1])*cond_f*fluid['Cp']) if len(pressure)>0 else 0
        if len(pressure)>0:
            h_conv_f=[]

            delta_p=pressure[:,0]-pressure[:,1]
            flux=delta_p*cond_f
            h_conv_f=flux*cp_data*density
            h_conv_f[delta_p>0]*=Temp_f[delta_p>0][:,0]
            h_conv_f[delta_p<0]*=Temp_f[delta_p<0][:,1]
            
        else:
            h_conv_f=0
        h_conv_f=np.sum( h_conv_f)     
        h_cond_f=np.sum((Temp_f[:,0]-Temp_f[:,1])*cond_h) if len(Temp_f)>0 else 0
        h_cond_sf=np.sum((Temp_sf[:,0]-Temp_sf[:,1])*cond_hs) if len(Temp_sf)>0 else 0
        h_cond_s=np.sum((Temp_s[:,0]-Temp_s[:,1])*cond_h_s) if len(Temp_s)>0 else 0
        result=h_conv_f+h_cond_f+h_cond_sf+h_cond_s
        #print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
        return result,h_conv_f,h_cond_f,h_cond_sf,h_cond_s

    
    def calculate_heat_flow(self,network,Boundary_condition,fluid,solid,g_ij,Tem_c,thermal_con_dual,P_profile,Num):
        result=[]
        energy_calcu=lambda i:self.energy_balance_conv(network,fluid,solid,g_ij,Tem_c,thermal_con_dual,P_profile,i)[0]
        result=Parallel(n_jobs=Num)(delayed(energy_calcu)(j) for j in network['pore._id'])
        '''
        for i in np.arange(len(network['pore.all'])):
            result.append(energy_balance_conv(network,fluid,solid,Tem_c,thermal_con_dual,P_profile,i)[0])
        '''
        result=np.array(result)
        output={}
        total=0
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                output.update({n:np.sum(result[network[n]])})
        for i in  output:   
            total+=output[i]
        output.update({'total':np.sum(total)})
        return output
    
    def mass_balance_conv(self,network,fluid,g_ij,P_profile,a):
        #res=find_neighbor_ball(network,[a])
        res=self.find_neighbor_ball(network,a)
        if res==0:
            result=0 
        elif res['pore'].size>=1:
            
            delta_p=np.array(P_profile[res['pore'][:,0].T]-P_profile[res['pore'][:,1].T])
            cond_f=g_ij[res['pore'][:,2].T]
            if len(delta_p)>=1:            
                flux=delta_p*cond_f
            else:
                flux=0
            result=np.sum(flux)
            
        else:
            result=0
        #print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
        return result

    
    
    def calculate_mass_flow(self,network,Boundary_condition,fluid,g_ij,P_profile,Num):
        mass_calcu=lambda i:self.mass_balance_conv(network,fluid,g_ij,P_profile,i)
        result=Parallel(n_jobs=Num)(delayed(mass_calcu)(j) for j in network['pore._id'])
        
        result=np.array(result)
        output={}
        total=0
        for m in Boundary_condition:
            for n in Boundary_condition[m]:
                output.update({n:np.sum(result[network[n]])})
                #abs_perm=np.sum(result[network[n]])
        for i in  output:   
            total+=output[i]
        output.update({'total':np.sum(total)})
        return output
    def cal_pore_veloc(self,network,fluid,g_ij,P_profile,a):
        #res=find_neighbor_ball(network,[a])
        res=self.find_neighbor_ball(network,a)
        pressure=[]
        cond_f=[]
        area_t=[]
        if res==0:
            result=0
        else:
            for i in res['pore']:
                pressure.append([P_profile[i[0]],P_profile[i[1]]])
                cond_f.append(g_ij[i[2]])
                area_t.append(network['throat.radius'][i[2]]**2/4/network['throat.real_shape_factor'][i[2]])
            pressure=np.array(pressure)
            cond_f=np.array(cond_f)
            area_t=np.array(area_t)
            if len(pressure)>1:
                delta_p=pressure[:,0]-pressure[:,1]
                #throat_p=(pressure[:,0]+pressure[:,1])/2
                flux=delta_p*cond_f
                velocity=flux/area_t
                momentum=velocity*abs(velocity)
                #momentum=flux*np.abs(flux)/area_t
                momentum=flux*np.abs(flux)/area_t
                
                flux= max(abs(np.sum(flux[flux>0])),abs(np.sum(flux[flux<0])))
                #momentum=max(abs(np.sum(momentum[momentum>0]+delta_p[delta_p>0]/network['pore.density'][a])),
                #             abs(np.sum(momentum[momentum<0]+delta_p[delta_p<0]/network['pore.density'][a])))
                momentum=max(abs(np.sum(momentum[momentum>0])),
                             abs(np.sum(momentum[momentum<0])))
                #vel_p=np.sqrt(momentum/(network['pore.radius'][a]**2/4/network['pore.real_shape_factor'][a]))
                vel_p=flux/(network['pore.radius'][a]**2/4/network['pore.real_shape_factor'][a])
            else:
                vel_p=0
            result=abs(vel_p)
        #print('h_conv_f=%f,h_cond_f=%f, h_cond_sf=%f,h_cond_s=%f'%(h_conv_f,h_cond_f, h_cond_sf,h_cond_s))
        return result

    def add_pores(self,network1,network2,trail=True):#network=network1,network2
        num_p=len(network1['pore.all'])
        num_t=len(network1['throat.all'])
        network2['throat._id']+=num_t
        network2['pore.label']+=num_p
        
        network={}
        if trail:
            network2['throat.conns']+=num_p
        for i in network2:
            if i not in network1:
                network[i]=np.zeros(num_p).astype(bool) if 'pore' in i else np.zeros(num_t).astype(bool)
                network[i]=np.concatenate((network[i],network2[i])) 
                
            else:
                network[i]=np.concatenate((network1[i],network2[i])) 
        return network
    def clone_pores(self,network,pores,label='clone_p'):
        clone={}
        num=len(network['pore.all'][pores])
        for i in network:
            if 'pore' in i:
                if '_id' in i:
                    clone[i]=np.arange(num)
    
                elif network[i].dtype==bool:
                    clone[i]=np.zeros(num).astype(bool)                
                else: 
                    clone[i]=network[i][pores]
    
                    
            elif 'throat' in i:
                clone[i]=np.array([])
        clone['pore.all']=np.ones(num).astype(bool)
        clone[label]=np.ones(num).astype(bool)
        return clone
    def merge_clone_pore(self,network,pores,radius,resolution,imsize,side,label='clone_p'):
        network['pore.coords']+=[resolution,resolution,resolution]
        clone=self.clone_pores(network,pores,label=label)
        org_coords=np.copy(clone['pore.coords'])
        if side=='left':
            index=np.min(clone['pore.coords'][:,0])
            clone['pore.coords']*=[0,1,1]
            
            clone['pore.coords']+=[index-resolution,0,0]
        elif side == 'right':
            index=np.max(clone['pore.coords'][:,0])
            clone['pore.coords']*=[0,1,1]
            
            clone['pore.coords']+=[index+resolution,0,0]
        elif side == 'back':
            index=np.min(clone['pore.coords'][:,1])
            clone['pore.coords']*=[1,0,1]
            
            clone['pore.coords']+=[0,index-resolution,0]
        elif side == 'front':
            index=np.max(clone['pore.coords'][:,1])
            clone['pore.coords']*=[1,0,1]
            
            clone['pore.coords']+=[0,index+resolution,0]
        elif side == 'bottom':
            index=np.min(clone['pore.coords'][:,2])
            clone['pore.coords']*=[1,1,0]
            
            clone['pore.coords']+=[0,0,index-resolution]
        elif side == 'top':
            index=np.max(clone['pore.coords'][:,2])
            clone['pore.coords']*=[1,1,0]
            
            clone['pore.coords']+=[0,0,index+resolution]
        num=network['pore.all'].size
        num_pore=np.count_nonzero(network['pore.void'])
        clone['throat.conns']=np.vstack((clone['pore.label'],clone['pore._id']+num)).T
        clone['pore.label']=clone['pore._id']
        num_T=len(clone['throat.conns'])
        clone['throat.solid']=np.zeros(num_T).astype(bool)
        clone['throat.solid'][(clone['throat.conns'][:,0]>=num_pore)&(clone['throat.conns'][:,1]>=num_pore)]=True
        clone['throat.void']=np.zeros(num_T).astype(bool)
        clone['throat.void'][(clone['throat.conns'][:,0]<num_pore)&(clone['throat.conns'][:,1]<num_pore)]=True
        clone['throat.connect']=np.zeros(num_T).astype(bool)
        clone['throat.connect']=~(clone['throat.solid']|clone['throat.void'])
        clone['throat.label']=np.arange(num_T)
        clone['throat._id']=np.arange(num_T)
        clone['throat.all']=np.ones(num_T).astype(bool)
        clone['throat.radius']=radius#clone['throat.all']*radius
        clone['throat.length']=np.abs(np.linalg.norm(org_coords- clone['pore.coords'],axis=1))
        return clone
