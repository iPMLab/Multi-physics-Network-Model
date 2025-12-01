# r"""Compute the Minkowski functionals and functions

# This module can compute both the Minkowski functionals and functions for 2D and
# 3D Numpy [1]_ arrays. These computations can handle both isotropic and anisotropic
# image resolutions.

# Notes
# ----------
# More information about the used algorithm can be found in the book "Statistical
# analysis of microstructures in materials science" by Joachim Ohser and Frank
# Mücklich [2]_.
# """

# import os
# os.environ["NUMBA_OPT"] = "max"
# os.environ["NUMBA_SLP_VECTORIZE"] = "1"
# os.environ["NUMBA_ENABLE_AVX"] = "1"
# os.environ["NUMBA_FUNCTION_CACHE_SIZE"] = "1024"
# import numba as nb
# import numpy as np
# nb.config.reload_config()
# from numba_minkowski import c_functionals_3d,c_functionals_2d

# ###############################################################################
# # {{{ functionals


# def functionals(image:np.ndarray, res = None, norm=False):
#     r"""Compute the Minkowski functionals in 2D or 3D.

#     This function computes the Minkowski functionals for the Numpy array `image`. Both
#     2D and 3D arrays are supported. Optionally, the (anisotropic) resolution of the
#     array can be provided using the Numpy array `res`. When a resolution array is
#     provided it needs to be of the same dimension as the image array.

#     Parameters
#     ----------
#     image : ndarray, bool
#         Image can be either a 2D or 3D array of data type `bool`.
#     res : ndarray, {int, float}, optional
#         By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
#         If a resolution is provided it needs to be of the same dimension as the
#         image array.
#     norm : bool, defaults to False
#         When norm=True the functionals are normalized with the total area or
#         volume of the image. Defaults to norm=False.

#     Returns
#     -------
#     out : tuple, float
#         In the case of a 2D image this function returns a tuple of the area,
#         length, and the Euler characteristic. In the case of a 3D image this
#         function returns a tuple of the volume, surface, curvature, and the
#         Euler characteristic. The return data type is `float`.

#     See Also
#     --------
#     ~quantimpy.minkowski.functions_open
#     ~quantimpy.minkowski.functions_close

#     Notes
#     -----

#     The definition of the Minkowski functionals follows the convention in the
#     physics literature [3]_.

#     Considering a 2D body, :math:`X`, with a smooth boundary, :math:`\delta X`,
#     the following functionals are computed:

#     .. math:: M_{0} (X) &= \int_{X} d s, \\
#               M_{1} (X) &= \frac{1}{2 \pi} \int_{\delta X} d c, \text{ and } \\
#               M_{2} (X) &= \frac{1}{2 \pi^{2}} \int_{\delta X} \left[\frac{1}{R} \right] d c,

#     where :math:`d s` is a surface element and :math:`d c` is a circumference
#     element. :math:`R` is the radius of the local curvature. This results in the
#     following definitions for the surface area, :math:`S = M_{0} (X)`,
#     circumference, :math:`C = 2 \pi M_{1} (X)`, and the 2D Euler characteristic,
#     :math:`\chi (X) = \pi M_{2} (X)`.

#     Considering a 3D body, :math:`X`, with a smooth boundary surface, :math:`\delta
#     X`, the following functionals are computed:

#     .. math:: M_{0} (X) &= V = \int_{X} d v, \\
#               M_{1} (X) &= \frac{1}{8} \int_{\delta X} d s, \\
#               M_{2} (X) &= \frac{1}{2 \pi^{2}} \int_{\delta X}  \frac{1}{2} \left[\frac{1}{R_{1}} + \frac{1}{R_{2}}\right] d s, \text{ and } \\
#               M_{3} (X) &= \frac{3}{(4 \pi)^{2}} \int_{\delta X} \left[\frac{1}{R_{1} R_{2}}\right] d s,

#     where :math:`d v` is a volume element and :math:`d s` is a surface element.
#     :math:`R_{1}` and :math:`R_{2}` are the principal radii of curvature of
#     surface element :math:`d s`. This results in the following definitions for
#     the volume, :math:`V = M_{0} (X)`, surface area, :math:`S = 8 M_{1} (X)`,
#     integral mean curvature, :math:`H = 2 \pi^{2} M_{2} (X)`, and the 3D Euler
#     characteristic, :math:`\chi (X) = 4 \pi/3 M_{3} (X)`.

#     Examples
#     --------
#     These examples use the scikit-image Python package [4]_ and the Matplotlib Python
#     package [5]_. For a 2D image the Minkowski functionals can be computed using
#     the following example:

#     .. code-block:: python

#         import numpy as np
#         import matplotlib.pyplot as plt
#         from skimage.morphology import (disk)
#         from quantimpy import minkowski as mk

#         image = np.zeros([128,128],dtype=bool)
#         image[16:113,16:113] = disk(48,dtype=bool)

#         plt.gray()
#         plt.imshow(image[:,:])
#         plt.show()

#         minkowski = mk.functionals(image)
#         print(minkowski)

#         # Compute Minkowski functionals for image with anisotropic resolution
#         res = np.array([2, 1])
#         minkowski = mk.functionals(image,res)
#         print(minkowski)

#     For a 3D image the Minkowski functionals can be computed using the following
#     example:

#     .. code-block:: python

#         import numpy as np
#         import matplotlib.pyplot as plt
#         from skimage.morphology import (ball)
#         from quantimpy import minkowski as mk

#         image = np.zeros([128,128,128],dtype=bool)
#         image[16:113,16:113,16:113] = ball(48,dtype=bool)

#         plt.gray()
#         plt.imshow(image[:,:,64])
#         plt.show()

#         minkowski = mk.functionals(image)
#         print(minkowski)

#         # Compute Minkowski functionals for image with anisotropic resolution
#         res = np.array([2, 1, 3])
#         minkowski = mk.functionals(image,res)
#         print(minkowski)

#     """
# # Decompose resolution in number larger than one and a pre-factor
#     factor = 1.0
#     if res is not None:
#         factor = np.amin(res)
#         res = res/factor

#     if (image.dtype == 'bool'):
#         pass
#     else:
#         raise ValueError('Input image needs to be binary (data type bool)')

#     if (image.ndim == 2):
# # Set default resolution (length/voxel)
#         if (res is None):
#             res0 = 1.0
#             res1 = 1.0
#         elif (res.size == 2):
#             res = res.astype(np.float64,copy=False)
#             res0 = res[0]
#             res1 = res[1]
#         else:
#             raise ValueError('Input image and resolution need to be the same dimension')

#         return _functionals_2d(image, res0, res1, factor, norm)
#     elif (image.ndim == 3):
# # Set default resolution (length/voxel)
#         if res is None:
#             res0 = 1.0
#             res1 = 1.0
#             res2 = 1.0
#         elif (res.size == 3):
#             res = res.astype(np.float64,copy=False)
#             res0 = res[0]
#             res1 = res[1]
#             res2 = res[2]
#         else:
#             raise ValueError('Input image and resolution need to be the same dimension')

#         return _functionals_3d(image, res0, res1, res2, factor, norm)
#     else:
#         raise ValueError('Can only handle 2D or 3D images')



import os
os.environ["NUMBA_OPT"] = "max"
os.environ["NUMBA_SLP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
os.environ["NUMBA_FUNCTION_CACHE_SIZE"] = "1024"
import numba as nb
import numpy as np
# nb.config.reload_config()
from numba_minkowski import c_functionals_2d,c_functionals_3d

def mk_functionals_2d(
        image:np.ndarray[bool],
        resolution = None,
        norm:bool=False,
        return_area:bool=True,
        return_length:bool=True,
        return_euler8:bool=True,
        ):
    if resolution is not None:
        resolution.astype(np.float64,copy=False)
        factor = np.max(resolution)
        resolution = resolution/factor
    else:
        resolution = np.array([1.0,1.0],dtype=np.float64)
        factor = 1.0

    res0,res1 = resolution
    image = np.ascontiguousarray(image)
    area,length,euler8 = c_functionals_2d(image, res0, res1,return_area,return_length,return_euler8)
    if norm:
        total_area = image.shape[0]*image.shape[1]*res0*res1
        normed_area = area/total_area
        normed_length = length/(total_area*factor)
        normed_euler8 = euler8/(total_area*factor**2)
    else:
        normed_area = area*factor**2
        normed_length = length*factor
        normed_euler8 = euler8
    res = []
    if return_area:
        res.append(normed_area)
    if return_length:
        res.append(normed_length)
    if return_euler8:
        res.append(normed_euler8)
    return tuple(res) if len(res)>1 else res[0]


def mk_functionals_3d(
        image:np.ndarray[bool],
        resolution = None,
        norm:bool=False,
        return_volume:bool=True,
        return_surface:bool=True,
        return_curvature:bool=True,
        return_euler26:bool=True,):
    if resolution is not None:
        resolution.astype(np.float64,copy=False)
        res0,res1,res2 = resolution
        factor = np.max(resolution)
        resolution = resolution/factor
    else:
        resolution = np.array([1.0,1.0,1.0],dtype=np.float64)
        res0,res1,res2 = resolution
        factor = 1.0
        
    res0,res1,res2 = resolution
    image = np.ascontiguousarray(image)
    volume,surface,curvature,euler26 = c_functionals_3d(image, res0,res1,res2,return_volume,return_surface,return_curvature,return_euler26)
    if norm:
        total_volume = image.shape[0]*image.shape[1]*image.shape[2]*res0*res1*res2 
        normed_volume = volume/total_volume
        normed_surface = surface/(total_volume*factor)
        normed_curvature = curvature/(total_volume*factor**2)
        normed_euler26 = euler26/(total_volume*factor**3)
    else:
        normed_volume = volume
        normed_surface = surface
        normed_curvature = curvature
        normed_euler26 = euler26
    
    res = []
    if return_volume:
        res.append(normed_volume)
    if return_surface:
        res.append(normed_surface)
    if return_curvature:
        res.append(normed_curvature)
    if return_euler26:
        res.append(normed_euler26)
    return tuple(res) if len(res)>1 else res[0]


if __name__ == '__main__':
    import numpy as np
    nb.set_num_threads(6)

    # os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = "omp tbb workqueue"
    import matplotlib.pyplot as plt
    from skimage.morphology import disk
    from skimage.measure import marching_cubes
    from skimage.measure import mesh_surface_area
    from math import prod

    image0 = np.zeros([64, 64], dtype=bool)
    image0[8:57, 8:57] = disk(24, dtype=bool)
    res0 = np.array([2.0, 2.0])
    ext0 = [0, image0.shape[0] * res0[0], 0, image0.shape[1] * res0[1]]


    # plt.gray()
    # plt.imshow(image0[:, :], extent=ext0)
    # plt.show()


    minkowski0 = mk_functionals_2d(image0, res0)
    import quantimpy.minkowski as mk
    print(minkowski0)
    minkowski1 = mk.functionals(image0, res0)
    print(minkowski1)


    from skimage.morphology import ball
    image0 = np.zeros([100, 100,100], dtype=bool)
    image0[8:57, 8:57, 8:57] = ball(24, dtype=bool)

    res0 = np.array([1.0, 1.0, 1.0])
    ext0 = [0, image0.shape[0] * res0[0], 0, image0.shape[1] * res0[1]]

    minkowski0 = mk_functionals_3d(image0, res0)
    minkowski1 = mk.functionals(image0, res0)
    print(minkowski0)
    print(minkowski1)
    import time
    t0 = time.time()
    for i in range(100):
        mesh = marching_cubes(image0, 0.5)
        vertices = mesh[0]
        faces = mesh[1]
        area=mesh_surface_area(vertices, faces)
    t1 = time.time()
    print(t1-t0)
    print(area)

    t0 = time.time()
    for i in range(100):
        minkowski0 = mk_functionals_3d(image0, res0)
    t1 = time.time()
    for i in range(100):
        minkowski1 = mk.functionals(image0, res0)
    t2 = time.time()
    print(t1-t0, t2-t1)
    print(minkowski0[1]*8)