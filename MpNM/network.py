#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:00:02 2022

@author: htmt
"""
from MpNM.Base import *
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util import numpy_support
from vtkmodules.util.numpy_support import vtk_to_numpy
import pandas as pd
# import vtkmodules.all as vtk
# from vtkmodules.util.numpy_support import numpy_to_vtk
from xml.etree import ElementTree as ET
import logging

# from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class network(Base):
    def __init__(self):
        pass

    @staticmethod
    def read_network(path=None, name=None):
        # conclude the throats and pores
        Throats1=pd.read_csv(path + '/' + name + "_link1.dat", skiprows=1,names=['index', 'pore_1_index', 'pore_2_index', 'radius', 'shape_factor',
                                         'total_length'],delim_whitespace=True)
        Throats2=pd.read_csv(path + '/' + name + "_link2.dat", names=['index', 'pore_1_index', 'pore_2_index', 'conduit_lengths_pore1',
                                         'conduit_lengths_pore2', 'length', 'volume', 'clay_volume'],delim_whitespace=True)

        Throats = pd.concat((Throats1, Throats2.loc[:, np.isin(Throats2.columns, Throats1.columns) == False]), axis=1)
        Pores1=pd.read_csv(path + '/' + name + '_node1.dat', skiprows=1, usecols=(0, 1, 2, 3, 4),delim_whitespace=True,names=['index', 'x', 'y', 'z', 'connection_number'])
        Pores2=pd.read_csv(path + '/' + name + "_node2.dat",names=['index', 'volume', 'radius', 'shape_factor', 'clay_volume'],delim_whitespace=True)
        # Pores2 = pd.DataFrame(data=Pores2, columns=['index', 'volume', 'radius', 'shape_factor', 'clay_volume'])
        Pores = pd.concat((Pores1, Pores2.loc[:, np.isin(Pores2.columns, Pores1.columns) == False]), axis=1)
        '''
        Throats1 = np.loadtxt("test1D_link1.dat", skiprows=1)
        Throats2 = np.loadtxt("test1D_link2.dat")
        Pores   = np.loadtxt("test1D_node2.dat")
        Pores1   = np.loadtxt('test1D_node1.dat',skiprows=1, usecols=(0,1,2,3,4))
        #'''

        # Pores = Pores[Pores1[:, 4] >= 0, :]  #
        # Pores1 = Pores1[Pores1[:, 4] >= 0, :]  #
        Pores = Pores[Pores.loc[:, 'connection_number'] >= 0]
        nonIsolatedPores = Pores.loc[:, 'index'].to_numpy()
        newPores = np.arange(len(Pores.loc[:, 'index']))
        # inThroats = np.array(Throats1[:, 1] * Throats1[:, 2]) < 0
        # outThroats = np.abs(Throats1[:, 1] * Throats1[:, 2]) < 1
        inThroats = (Throats.loc[:, 'pore_1_index'] * Throats.loc[:, 'pore_2_index'] == -1).to_numpy()
        outThroats = (Throats.loc[:, 'pore_1_index'] * Throats.loc[:, 'pore_2_index'] == 0).to_numpy()
        pn = {}
        nP = len(nonIsolatedPores)
        nT = len(Throats1)
        pn['pore._id'] = newPores
        pn['pore.label'] = newPores
        oldPores = nonIsolatedPores - 1
        pn['pore.all'] = np.ones(nP, dtype=bool)
        pn['pore.volume'] = Pores.loc[:, 'volume'].to_numpy()
        pn['pore.radius'] = Pores.loc[:, 'radius'].to_numpy()

        pn['pore.shape_factor'] = Pores.loc[:, 'shape_factor'].to_numpy()
        pn['pore.clay_volume'] = Pores.loc[:, 'clay_volume'].to_numpy()
        pn['pore.coords'] = Pores.loc[:, ['x', 'y', 'z']].to_numpy()
        pn['throat._id'] = Throats.loc[:, 'index'].to_numpy()
        pn['throat.label'] = Throats.loc[:, 'index'].to_numpy()
        pn['throat.all'] = np.ones(nT, dtype=bool)
        # pn['throat.inlets']=inThroats
        # pn['throat.outlets']=outThroats

        pn['throat.inside'] = ~(inThroats | outThroats)
        # pn['throat.shape_factor'] = Throats1[:, 0]
        # pn['throat.shape_factor']=Throats.loc[:,'shape_factor'].to_numpy()
        throat_conns = (Throats.loc[:, ['pore_1_index', 'pore_2_index']] - [1, 1]).to_numpy()
        throat_conns[throat_conns[:, 0] > throat_conns[:, 1]] = throat_conns[:, [1, 0]][
            throat_conns[:, 0] > throat_conns[:, 1]]  # make throat_conns first < second
        throat_out_in = throat_conns[(throat_conns[:, 0] < 0)]
        index = np.digitize(throat_out_in[:, 1], oldPores) - 1
        throat_out_in[:, 1] = pn['pore._id'][index]
        throat_internal = throat_conns[(throat_conns[:, 0] >= 0)]
        index = np.digitize(throat_internal[:, 0], oldPores) - 1
        throat_internal[:, 0] = pn['pore._id'][index]
        index = np.digitize(throat_internal[:, 1], oldPores) - 1
        throat_internal[:, 1] = pn['pore._id'][index]
        pn['throat.conns'] = np.concatenate((throat_out_in, throat_internal), axis=0).astype(np.int64)

        pn['pore.inlets'] = ~np.copy(pn['pore.all'])
        pn['pore.inlets'][
            np.array(pn['throat.conns'][pn['throat.conns'][:, 0] == -2][:, 1]).astype(int)] = True
        pn['pore.outlets'] = ~np.copy(pn['pore.all'])
        pn['pore.outlets'][
            np.array(pn['throat.conns'][pn['throat.conns'][:, 0] == -1][:, 1]).astype(int)] = True
        pn['throat.radius'] = Throats.loc[:, 'radius'].to_numpy()
        pn['throat.shape_factor'] = Throats.loc[:, 'shape_factor'].to_numpy()
        pn['throat.length'] = Throats.loc[:, 'length'].to_numpy()
        pn['throat.total_length'] = Throats.loc[:, 'total_length'].to_numpy()
        pn['throat.conduit_lengths_pore1'] = Throats2.loc[:, 'conduit_lengths_pore1'].to_numpy()
        pn['throat.conduit_lengths_pore2'] = Throats2.loc[:, 'conduit_lengths_pore2'].to_numpy()
        pn['throat.conduit_lengths_throat'] = Throats2.loc[:, 'length'].to_numpy()

        BndG1 = (np.sqrt(3) / 36 + 0.00001)
        BndG2 = 0.07
        pn['throat.real_shape_factor'] = pn['throat.shape_factor']
        pn['throat.real_shape_factor'][
            (pn['throat.shape_factor'] > BndG1) & (pn['throat.shape_factor'] <= BndG2)] = 1 / 16
        pn['throat.real_shape_factor'][(pn['throat.shape_factor'] > BndG2)] = 1 / 4 / np.pi
        pn['pore.real_shape_factor'] = pn['pore.shape_factor']
        pn['pore.real_shape_factor'][
            (pn['pore.shape_factor'] > BndG1) & (pn['pore.shape_factor'] <= BndG2)] = 1 / 16
        pn['pore.real_shape_factor'][(pn['pore.shape_factor'] > BndG2)] = 1 / 4 / np.pi
        pn['throat.real_k'] = pn['throat.all'] * 0.6
        pn['throat.real_k'][
            (pn['throat.shape_factor'] > BndG1) & (pn['throat.shape_factor'] <= BndG2)] = 0.5623
        pn['throat.real_k'][(pn['throat.shape_factor'] > BndG2)] = 0.5
        pn['pore.real_k'] = pn['pore.all'] * 0.6
        pn['pore.real_k'][
            (pn['pore.shape_factor'] > BndG1) & (pn['pore.shape_factor'] <= BndG2)] = 0.5623
        pn['pore.real_k'][(pn['pore.shape_factor'] > BndG2)] = 0.5
        pn['throat.area'] = ((pn['throat.radius'] ** 2)
                             / (4.0 * pn['throat.shape_factor']))
        pn['pore.area'] = ((pn['pore.radius'] ** 2)
                           / (4.0 * pn['pore.shape_factor']))
        pn['pore.solid']=np.zeros(nP,dtype=bool)
        return pn

    @staticmethod
    def getting_zoom_value(pn, side, imsize, resolution):
        if side in ['left', 'right']:
            value = imsize[0] * imsize[1] * resolution ** 2 / np.sum(
                pn['pore.radius'][pn['pore.boundary_' + side + '_surface']] ** 2 * np.pi)
        elif side in ['top', 'bottom']:
            value = imsize[1] * imsize[2] * resolution ** 2 / np.sum(
                pn['pore.radius'][pn['pore.boundary_' + side + '_surface']] ** 2 * np.pi)
        elif side in ['front', 'back']:
            value = imsize[0] * imsize[2] * resolution ** 2 / np.sum(
                pn['pore.radius'][pn['pore.boundary_' + side + '_surface']] ** 2 * np.pi)
        else:
            raise ValueError('Wrong side name')
        return value

    @staticmethod
    def get_program_parameters(path):
        import argparse
        description = 'Read a VTK XML PolyData file.'
        epilogue = ''''''
        parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--filename', default=path)
        args = parser.parse_args()
        return args.filename

    @staticmethod
    def vtk2network(path):
        vtk_type = path.split(".")[-1]
        if vtk_type == "vtu":
            import meshio
            mesh = meshio.read(path)
            # cell data
            cell_keys = mesh.cell_data.keys()
            cell_values = mesh.cell_data.values()
            cell_data = dict(zip(cell_keys, [i[0] for i in cell_values]))

            # point data
            point_keys = mesh.point_data
            point_values = mesh.point_data.values()
            point_data = dict(zip(point_keys, [i[0] for i in point_values]))

            all_data = {}
            all_data.update(point_data)
            all_data.update(cell_data)
            return all_data

        elif vtk_type == "vtp":
            import vtk
            from vtkmodules.util import numpy_support
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            point_data = reader.GetOutput().GetPointData()
            point_data_count = point_data.GetNumberOfArrays()
            point_data = {point_data.GetArrayName(i): numpy_support.vtk_to_numpy(point_data.GetArray(i)) for
                          i in range(point_data_count)}

            cell_data = reader.GetOutput().GetCellData()
            cell_data_count = cell_data.GetNumberOfArrays()
            cell_data = {cell_data.GetArrayName(i): numpy_support.vtk_to_numpy(cell_data.GetArray(i)) for i
                         in range(cell_data_count)}

            all_data = {}
            all_data.update(point_data)
            all_data.update(cell_data)
            return all_data

    @staticmethod
    def array_to_element(name, array, n=1):
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

    @staticmethod
    def network2vtk(pn, filename="test.vtp",
                    fill_nans=None, fill_infs=None):
        '''
        from openpnm
        '''
        if filename.split('.')[-1] != 'vtp':
            filename += '.vtp'
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
        d = globals()
        network_name = ''
        for key in d:
            if type(d[key]) is type(pn) and d[key] == pn:
                network_name = key

        key_list = list(pn.keys())

        points = pn["pore.coords"]
        pairs = pn["throat.conns"]
        num_points = np.shape(points)[0]
        num_throats = np.shape(pairs)[0]

        root = ET.fromstring(_TEMPLATE)
        piece_node = root.find("PolyData").find("Piece")
        piece_node.set("NumberOfPoints", str(num_points))
        piece_node.set("NumberOfLines", str(num_throats))
        points_node = piece_node.find("Points")
        coords = network.array_to_element("coords", points.T.ravel("F"), n=3)
        points_node.append(coords)
        lines_node = piece_node.find("Lines")
        connectivity = network.array_to_element("connectivity", pairs)
        lines_node.append(connectivity)
        offsets = network.array_to_element("offsets", 2 * np.arange(len(pairs)) + 2)
        lines_node.append(offsets)

        point_data_node = piece_node.find("PointData")
        cell_data_node = piece_node.find("CellData")
        for key in key_list:
            array = pn[key]
            if array.dtype == "O":
                logger.warning(key + " has dtype object will not write to file")
            else:
                if array.dtype == bool:
                    array = array.astype(int)
                    key = str(network_name) + ' network| label || ' + key
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
                # print(array)
                element = network.array_to_element(key, array)
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


if __name__ == '__main__':
    path = '../sample_data/Sphere_stacking_500_500_2000_60/pore_network'
    network().read_network(path=path, name='sphere_stacking_500_500_2000_60')
