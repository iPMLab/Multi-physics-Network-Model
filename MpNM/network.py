#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:00:02 2022

@author: htmt
"""
from Base import *
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

from vtkmodules.util.numpy_support import vtk_to_numpy



#import vtkmodules.all as vtk
#from vtkmodules.util.numpy_support import numpy_to_vtk
from xml.etree import ElementTree as ET
import logging
from joblib import Parallel, delayed
logger = logging.getLogger(__name__)


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
    def getting_zoom_value(self,network,side,imsize,resolution):
        if side in ['left','right']:
            value=imsize[0]*imsize[1]*resolution**2/np.sum(network['pore.radius'][network['pore.boundary_'+side+'_surface']]**2*np.pi)
        elif side in ['top','bottom']:
            value=imsize[1]*imsize[2]*resolution**2/np.sum(network['pore.radius'][network['pore.boundary_'+side+'_surface']]**2*np.pi)
        elif side in ['front','back']:   
            value=imsize[0]*imsize[2]*resolution**2/np.sum(network['pore.radius'][network['pore.boundary_'+side+'_surface']]**2*np.pi)
        return value
    
    
    def get_program_parameters(self,path):
        import argparse
        description = 'Read a VTK XML PolyData file.'
        epilogue = ''''''
        parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--filename',default=path)
        args = parser.parse_args()
        return args.filename
    
    def vtp2network(self,path):
        network={}
        filename = self.get_program_parameters(path)
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        rawdata=reader
        output = rawdata.GetOutput()
        network['pore._id'] = np.arange(output.GetNumberOfPoints()).astype(np.int64)

        #--------------#
        
        #Mapper = vtk.vtkPolyDataMapper()
        #Mapper.SetInputConnection(reader_coordinate.GetOutputPort())
        
        def get_point(i):
            p=[0,0,0]
            output.GetPoint(i, p)
            return p
        Num=output.GetNumberOfPoints()//200000
        
        network['pore.coords']=Parallel(n_jobs=max(Num,1),prefer='threads')(delayed(get_point)(f) for f in range(output.GetNumberOfPoints()))
        network['pore.coords']=np.array(network['pore.coords'])

        #--------------#
        
        get_throat=lambda i:[output.GetCell(i).GetPointIds().GetId(0),output.GetCell(i).GetPointIds().GetId(1)]
        Num=output.GetNumberOfCells()//200000
        network['throat.conns']=Parallel(n_jobs=max(Num,1),prefer='threads')(delayed(get_throat)(f) for f in range(output.GetNumberOfCells()))
        network['throat.conns']=np.array(network['throat.conns'])

        number_cellarray=rawdata.GetNumberOfCellArrays()
        number_pointarray=rawdata.GetNumberOfPointArrays()
        for i in range(number_cellarray):
            n = 0
            str1=rawdata.GetCellArrayName(i)
            for j in str1[::-1]:
                if j == ' ':
                    break
                n = n + 1
            str2 = str1[len(str1) - n:]
            network[str2]=vtk_to_numpy(output.GetCellData().GetArray(i))
            if max(network[str2])==1:
                network[str2]=network[str2].astype(bool)
    
        for i in range(number_pointarray):
            n=0
            str1=rawdata.GetPointArrayName(i)
            for j in str1[::-1]:
                if j == ' ':
                    break
                n = n + 1
            str2 = str1[len(str1) - n:]
            network[str2]=vtk_to_numpy(output.GetPointData().GetArray(i))
            if max(network[str2])==1:
                network[str2]=network[str2].astype(bool)
        network['pore.label']=network['pore.label'].astype(np.int64) if 'pore.label' in network.keys() else np.arange(len(network['pore._id']))
        network['throat._id']=network['throat._id'].astype(np.int64)
        return network
    
    
    def array_to_element(self, name, array, n=1):
        dtype_map = {
            "int8": "Int8",
            "int16": "Int16",
            "int32": "Int32",
            "int64": "Int64",
            "uint8": "UInt8",
            "uint16": "UInt16",
            "uint32": "UInt32",
            "uint64": "UInt64",
            "float32": "Float32",
            "float64": "Float64",
            "str": "String",
        }
        element = None
        if str(array.dtype) in dtype_map.keys():
            element = ET.Element("DataArray")
            element.set("Name", name)
            element.set("NumberOfComponents", str(n))
            element.set("type", dtype_map[str(array.dtype)])
            element.text = "\t".join(map(str, array.ravel()))
        return element    











    def network2vtk(self,network, filename="test.vtp",
                       fill_nans=None, fill_infs=None):
        _TEMPLATE = """
        <?xml version="1.0" ?>
        <VTKFile byte_order="LittleEndian" type="PolyData" version="0.1">
            <PolyData>
                <Piece NumberOfLines="0" NumberOfPoints="0">
                    <Points>
                    </Points>
                    <Lines>
                    </Lines>
                    <PointData>
                    </PointData>
                    <CellData>
                    </CellData>
                </Piece>
            </PolyData>
        </VTKFile>
        """.strip()
        d=globals()
        network_name=''
        for key in d:
            if type(d[key]) is type(network) and d[key] == network:
                network_name=key
    
    
        key_list = list(network.keys())
    
        points = network["pore.coords"]
        pairs = network["throat.conns"]
        num_points = np.shape(points)[0]
        num_throats = np.shape(pairs)[0]
    
    
        root = ET.fromstring(_TEMPLATE)
        piece_node = root.find("PolyData").find("Piece")
        piece_node.set("NumberOfPoints", str(num_points))
        piece_node.set("NumberOfLines", str(num_throats))
        points_node = piece_node.find("Points")
        coords = self.array_to_element("coords", points.T.ravel("F"), n=3)
        points_node.append(coords)
        lines_node = piece_node.find("Lines")
        connectivity = self.array_to_element("connectivity", pairs)
        lines_node.append(connectivity)
        offsets = self.array_to_element("offsets", 2 * np.arange(len(pairs)) + 2)
        lines_node.append(offsets)
    
        point_data_node = piece_node.find("PointData")
        cell_data_node = piece_node.find("CellData")
        for key in key_list:
            array = network[key]
            if array.dtype == "O":
                logger.warning(key + " has dtype object," + " will not write to file")
            else:
                if array.dtype == np.bool_:
                    array = array.astype(int)
                    key=str(network_name)+' network| label || '+key
                elif array.dtype == np.int32 or np.int64:
                    key = str(network_name) + 'network| properties|| ' + key
                if np.any(np.isnan(array)):
                    if fill_nans is None:
                        logger.warning(key + " has nans," + " will not write to file")
                        continue
                    else:
                        array[np.isnan(array)] = fill_nans
                if np.any(np.isinf(array)):
                    if fill_infs is None:
                        logger.warning(key + " has infs," + " will not write to file")
                        continue
                    else:
                        array[np.isinf(array)] = fill_infs
                #print(array)
                element = self.array_to_element(key, array)
                if array.size == num_points:
                    point_data_node.append(element)
                elif array.size == num_throats:
                    cell_data_node.append(element)
    
        tree = ET.ElementTree(root)
        tree.write(filename)
    
        with open(filename, "r+") as f:
            string = f.read()
            string = string.replace("</DataArray>", "</DataArray>\n\t\t\t")
            f.seek(0)
            # consider adding header: '<?xml version="1.0"?>\n'+
            f.write(string)