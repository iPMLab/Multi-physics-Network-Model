#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 09:43:07 2021

@author: htmt
"""
import os
import fileinput as fi
import numpy as np
import shutil
import time
import sys

sys.path.append("../")
from mpnm_new import extraction
import platform
from pathlib import Path
import textwrap


def pore_network_extraction(Path_dict, resolution, size):
    print(size)
    print("\npore extraction starting\n")
    Images_dict = Path_dict["Images"]
    t0 = time.time()
    images_pore = np.fromfile(Path_dict["Path_input"], dtype=np.uint8).reshape(
        size[2], size[1], size[0]
    )
    # data[data>0]=1
    images_pore = np.where(images_pore > 0, 1, 0)
    # if Images_dict.get('pore') is not None:
    images_pore.astype(np.uint8).tofile(Images_dict["pore"])
    pf = platform.system()
    mhd_content = textwrap.dedent(
        f"""
    ObjectType =  Image
    NDims =       3
    ElementType = MET_UCHAR
    ElementByteOrderMSB = False
    HeaderSize = 0
    DimSize =    	{size[0]}	{size[1]}	{size[2]}
    ElementSize = 	{resolution}   {resolution}   {resolution}
    Offset =      	0   	0   	0
    ElementDataFile = {Images_dict['pore']}
    DefaultImageFormat = .raw
    //direction z
    pore 0 0""".strip(
            "\n"
        )
    )
    Path_dict["mhd"]["target"].unlink(missing_ok=True)
    with open(f"{Path_dict['mhd']['target']}", "w+") as mhd:
        mhd.write(mhd_content)
    # Address of the network extraction code
    Path_dict["pnextract"]["target"].unlink(missing_ok=True)
    shutil.copy(
        str(Path_dict["pnextract"]["origin"]), str(Path_dict["pnextract"]["target"])
    )
    path_work = os.getcwd()
    Path_dict["pnextract"]["target"].chmod(0o777)
    os.chdir(Path_dict["pnextract"]["target"].parent)
    if pf == "Windows":
        os.system(
            f"{Path_dict['pnextract']['target'].name} {Path_dict['mhd']['target']}"
        )
    else:
        os.system(
            f"./{Path_dict['pnextract']['target'].name} {Path_dict['mhd']['target']}"
        )
    os.chdir(path_work)
    (Path_dict["pnextract"]["target"]).unlink(missing_ok=True)

    Path_pore_VElems_origin = (
        Path_dict["pore_network"] / Images_dict["pore_VElems"].name
    )
    Path_pore_VElems_origin.replace(Images_dict["pore_VElems_origin"])
    # size+=2 # please note the order has change after pnextract[Num,High,Width]
    array_pore_VElems = np.fromfile(
        Images_dict["pore_VElems_origin"], dtype=np.int32
    ).reshape(size[::-1] + 2)
    """
    0 represents solid pixels, (was 3)
    -2 represents the ignored pore pixels
    array_pore_pore.raw only include segmented pore pixels (>=1)
    """
    array_pore_VElems = (
        array_pore_VElems[1 : size[2] + 1, 1 : size[1] + 1, 1 : size[0] + 1] - 1
    )

    array_pore_VElems = extraction.check_pnextract_result(
        img=array_pore_VElems,
        path=Path_dict["pore_network"],
        name=f"{Images_dict['pore'].stem}",
    )

    # if len(pore_index)>0:
    #     print('it needs to be modified')

    #     tool.trim_pore(pn,pore_index-1)
    #     network.network2vtk(pn,filename=path_pore_network+file_name)
    #     VElems_map={i:j for i,j in zip(pore_number,range(1,len(pore_number)+1))}
    #     array_pore_VElems=fr.remap(array_pore_VElems, VElems_map,preserve_missing_labels=True)
    #     # for i in pore_index:
    #     #     index=np.round(pn['pore.coords'][i]/resolution).astype(int)
    #     #     index=np.where(index>size-1,size-1,index)
    #     #     array_pore_VElems[round(index[0]),round(index[1]),round(index[2])]=i
    # else:
    #     print('it does not need to be modified')
    #     network.network2vtk(pn,filename=path_pore_network+file_name)
    array_pore_VElems = np.where(array_pore_VElems == -3, 0, array_pore_VElems)
    if Images_dict.get("pore_VElems") is not None:
        array_pore_VElems.astype(np.int32).tofile(Images_dict["pore_VElems"])

    if Images_dict.get("pore_pore") is not None:
        array_pore_pore = np.where(array_pore_VElems < 1, 0, array_pore_VElems)
        array_pore_pore.astype(np.int32).tofile(Images_dict["pore_pore"])

    tend = time.time()
    print(
        "\n========================================\nfinish pore network extraction\ntime cost：%.6fs\n=========================================="
        % (tend - t0)
    )
    return array_pore_VElems.astype(np.int32)


if __name__ == "__main__":
    path_raw = "/home/htmt/Disk2/Documents/two_layers/560-960-6000/4mm/"
    path_pore_network = (
        "/home/htmt/Disk2/Documents/two_layers/560-960-6000/4mm/pore_network/"
    )
    path_images = "/home/htmt/Disk2/Documents/two_layers/560-960-6000/4mm/images/"
    path_root = "./"
    file_name = "4mm"
    resolution = 0.5e-4
    size = np.array([560, 960, 6000])
    pore_network_extraction(
        path_raw,
        path_pore_network,
        path_images,
        path_root,
        file_name,
        resolution,
        size,
    )
