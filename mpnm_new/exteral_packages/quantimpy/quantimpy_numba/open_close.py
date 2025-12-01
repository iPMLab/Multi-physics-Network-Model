#
# def functions_open(opening:np.ndarray, res=None, norm:bool=False):
#     r"""
#     Compute the Minkowski functions in 2D or 3D.
#
#     This function computes the Minkowski functionals as function of the
#     grayscale values in the Numpy array `opening`. Both 2D and 3D arrays are supported.
#     Optionally, the (anisotropic) resolution of the array can be provided using
#     the Numpy array `res`. When a resolution array is provided it needs to be of
#     the same dimension as the 'opening' array.
#
#     The algorithm iterates over all grayscale values present in the array,
#     starting at the smallest value (black). For every grayscale value the array
#     is converted into a binary image where values larger than the grayscale
#     value become one (white) and all other values become zero (black). For each
#     of these binary images the minkowski functionals are computed according to
#     the function :func:`~quantimpy.minkowski.functionals`.
#
#     This function can be used in combination with the
#     :func:`~quantimpy.morphology` module to compute the Minkowski functions of
#     different morphological distance maps.
#
#     Parameters
#     ----------
#     opening : ndarray, float
#         Opening can be either a 2D or 3D array of data type `float`.
#     res : ndarray, {int, float}, optional
#         By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
#         If a resolution is provided it needs to be of the same dimension as the
#         image array.
#     norm : bool, defaults to False
#         When norm=True the functions are normalized with the total area or
#         volume of the image. Defaults to norm=False.
#
#     Returns
#     -------
#     out : tuple, ndarray, float
#         In the case of a 2D image this function returns a tuple of Numpy arrays
#         consisting of the distance (assuming one grayscale value is used per
#         unit of length), area, length, and the Euler characteristic. In the
#         case of a 3D image this function returns a tuple of Numpy arrays
#         consistenting of the distance, volume, surface, curvature, and the Euler
#         characteristic. The return data type is `float`.
#
#     See Also
#     --------
#     ~quantimpy.minkowski.functionals
#     ~quantimpy.minkowski.functions_close
#     ~quantimpy.morphology
#
#     Examples
#     --------
#     These examples use the Skimage Python package [4]_ and the Matplotlib Python
#     package [5]_. For a 2D image the Minkowski functions can be computed using
#     the following example:
#
#     .. code-block:: python
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from skimage.morphology import (disk)
#         from quantimpy import morphology as mp
#         from quantimpy import minkowski as mk
#
#         image = np.zeros([128,128],dtype=bool)
#         image[16:113,16:113] = disk(48,dtype=bool)
#
#         erosion_map = mp.erode_map(image)
#
#         plt.gray()
#         plt.imshow(image[:,:])
#         plt.show()
#
#         plt.gray()
#         plt.imshow(erosion_map[:,:])
#         plt.show()
#
#         dist, area, length, euler = mk.functions_open(erosion_map)
#
#         plt.plot(dist,area)
#         plt.show()
#
#         plt.plot(dist,length)
#         plt.show()
#
#         plt.plot(dist,euler)
#         plt.show()
#
#     For a 3D image the Minkowski functionals can be computed using the following
#     example:
#
#     .. code-block:: python
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from skimage.morphology import (ball)
#         from quantimpy import morphology as mp
#         from quantimpy import minkowski as mk
#
#         image = np.zeros([128,128,128],dtype=bool)
#         image[16:113,16:113,16:113] = ball(48,dtype=bool)
#
#         erosion_map = mp.erode_map(image)
#
#         plt.gray()
#         plt.imshow(image[:,:,64])
#         plt.show()
#
#         plt.gray()
#         plt.imshow(erosion_map[:,:,64])
#         plt.show()
#
#         dist, volume, surface, curvature, euler = mk.functions_open(erosion_map)
#
#         plt.plot(dist,volume)
#         plt.show()
#
#         plt.plot(dist,surface)
#         plt.show()
#
#         plt.plot(dist,curvature)
#         plt.show()
#
#         plt.plot(dist,euler)
#         plt.show()
#
#     References
#     ----------
#     .. [1] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt et al.,
#         "Array programming with NumPy", Nature, vol. 585, pp 357-362, 2020,
#         doi:`10.1038/s41586-020-2649-2`_
#
#     .. _10.1038/s41586-020-2649-2: https://doi.org/10.1038/s41586-020-2649-2
#
#     .. [2] Joachim Ohser and Frank Mücklich, "Statistical analysis of
#         microstructures in materials science", Wiley and Sons, New York (2000) ISBN:
#         0471974862
#
#     .. [3] Klaus R. Mecke, "Additivity, convexity, and beyond: applications of
#         Minkowski Functionals in statistical physics" in "Statistical Physics
#         and Spatial Statistics", pp 111–184, Springer (2000) doi:
#         `10.1007/3-540-45043-2_6`_
#
#     .. _10.1007/3-540-45043-2_6: https://doi.org/10.1007/3-540-45043-2_6
#
#     .. [4] Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
#         François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle Gouillart,
#         Tony Yu and the scikit-image contributors. "scikit-image: Image
#         processing in Python." PeerJ 2:e453 (2014) doi: `10.7717/peerj.453`_
#
#     .. _10.7717/peerj.453: https://doi.org/10.7717/peerj.453
#
#     .. [5] John D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in
#         Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
#         doi:`10.1109/MCSE.2007.55`_
#
#     .. _10.1109/MCSE.2007.55: https://doi.org/10.1109/MCSE.2007.55
#     """
# # Decompose resolution in number larger than one and a pre-factor
#     factor = 1.0
#     if not (res is None):
#         factor = np.amin(res)
#         res = res/factor
#
#     if not (opening.dtype == 'bool'):
#         opening = opening.astype('bool')
#
#     if (opening.ndim == 2):
# # Set default resolution (length/voxel)
#         if (res is None):
#             res0 = 1.0
#             res1 = 1.0
#         elif (res.size == 2):
#             res = res.astype(np.float64)
#             res0 = res[0]
#             res1 = res[1]
#         else:
#             raise ValueError('Input image and resolution need to be the same dimension')
#
#         return _functions_open_2d(opening, res0, res1, factor, norm)
#     elif (opening.ndim == 3):
# # Set default resolution (length/voxel)
#         if (res is None):
#             res0 = 1.0
#             res1 = 1.0
#             res2 = 1.0
#         elif (res.size == 3):
#             res = res.astype(np.float64)
#             res0 = res[0]
#             res1 = res[1]
#             res2 = res[2]
#         else:
#             raise ValueError('Input image and resolution need to be the same dimension')
#
#         return _functions_open_3d(opening, res0, res1, res2, factor, norm)
#     else:
#         raise ValueError('Can only handle 2D or 3D openings')
#
# def _functions_open_2d(
#         opening:np.ndarray[bool],
#         res0:np.float64,
#         res1:np.float64,
#         factor:np.float64,
#         norm:bool):
#     opening = np.ascontiguousarray(opening)
#     dim3 = np.amax(opening) - np.amin(opening)
#     dist = np.empty(dim3,dtype=np.float64)
#     area = np.empty(dim3,dtype=np.float64)
#     length = np.empty(dim3,dtype=np.float64)
#     euler4 = np.empty(dim3,dtype=np.float64)
#     euler8 = np.empty(dim3,dtype=np.float64)
#     status = c_functions_open_2d(opening, opening.shape[0], opening.shape[1], res0, res1, dist, area, length, euler4, euler8)
#     assert status == 0
#     if norm:
#         total_area = opening.shape[0]*opening.shape[1]*res0*res1
#         return dist*factor, area/(total_area), length/(total_area*factor), euler8/(total_area*factor**2)
#     else:
#         return dist*factor, area*factor**2, length*factor, euler8
#
#
# def _functions_open_3d(
#         opening:np.ndarray[bool],
#         res0:np.float64,
#         res1:np.float64,
#         res2:np.float64,
#         factor:np.float64,
#         norm:bool):
#     opening = np.ascontiguousarray(opening)
#     dim3 = np.amax(opening) - np.amin(opening)
#     dist = np.empty(dim3,dtype=np.float64)
#     volume = np.empty(dim3,dtype=np.float64)
#     surface = np.empty(dim3,dtype=np.float64)
#     curvature = np.empty(dim3,dtype=np.float64)
#     euler6 = np.empty(dim3,dtype=np.float64)
#     euler26 = np.empty(dim3,dtype=np.float64)
#     status = c_functions_open_3d(opening.reshape(-1), opening.shape[0], opening.shape[1], opening.shape[2], res0, res1, res2, dist, volume, surface, curvature, euler6, euler26)
#     assert status == 0
#     if norm:
#         total_volume = opening.shape[0]*opening.shape[1]*opening.shape[2]*res0*res1*res2
#         return dist*factor, volume/(total_volume), surface/(total_volume*factor), curvature/(total_volume*factor**2), euler26/(total_volume*factor**3)
#     else:
#         return dist*factor, volume*factor**3, surface*factor**2, curvature*factor, euler26
#
# # }}}
# ###############################################################################
# # {{{ functions_close
#
#
# def functions_close(closing:np.ndarray, res = None, norm:bool=False):
#     r"""
#     Compute the Minkowski functions in 2D or 3D.
#
#     This function computes the Minkowski functionals as function of the
#     grayscale values in the Numpy array `closing`. Both 2D and 3D arrays are supported.
#     Optionally, the (anisotropic) resolution of the array can be provided using
#     the Numpy array `res`. When a resolution array is provided it needs to be of
#     the same dimension as the 'closing' array.
#
#     The algorithm iterates over all grayscale values present in the array,
#     starting at the largest value (white). For every grayscale value the array
#     is converted into a binary image where values larger than the grayscale
#     value become one (white) and all other values become zero (black). For each
#     of these binary images the minkowski functionals are computed according to
#     the function :func:`~quantimpy.minkowski.functionals`.
#
#     This function can be used in combination with the
#     :func:`~quantimpy.morphology` module to compute the Minkowski functions of
#     different morphological distance maps.
#
#     Parameters
#     ----------
#     closing : ndarray, float
#         Closing can be either a 2D or 3D array of data type `float`.
#     res : ndarray, {int, float}, optional
#         By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
#         If a resolution is provided it needs to be of the same dimension as the
#         image array.
#     norm : bool, defaults to False
#         When norm=True the functions are normalized with the total area or
#         volume of the image. Defaults to norm=False.
#
#     Returns
#     -------
#     out : tuple, ndarray, float
#         In the case of a 2D image this function returns a tuple of Numpy arrays
#         consisting of the distance (assuming one grayscale value is used per
#         unit of length), area, length, and the Euler characteristic. In the
#         case of a 3D image this function returns a tuple of Numpy arrays
#         consistenting of the distance, volume, surface, curvature, and the Euler
#         characteristic. The return data type is `float`.
#
#     See Also
#     --------
#     ~quantimpy.minkowski.functionals
#     ~quantimpy.minkowski.functions_open
#     ~quantimpy.morphology
#
#     Examples
#     --------
#     These examples use the scikit-image Python package [4]_ and the Matplotlib Python
#     package [5]_. For a 2D image the Minkowski functions can be computed using
#     the following example:
#
#     .. code-block:: python
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from skimage.morphology import (disk)
#         from quantimpy import morphology as mp
#         from quantimpy import minkowski as mk
#
#         image = np.zeros([128,128],dtype=bool)
#         image[16:113,16:113] = disk(48,dtype=bool)
#
#         erosion_map = mp.erode_map(image)
#
#         plt.gray()
#         plt.imshow(image[:,:])
#         plt.show()
#
#         plt.gray()
#         plt.imshow(erosion_map[:,:])
#         plt.show()
#
#         dist, area, length, euler = mk.functions_close(erosion_map)
#
#         plt.plot(dist,area)
#         plt.show()
#
#         plt.plot(dist,length)
#         plt.show()
#
#         plt.plot(dist,euler)
#         plt.show()
#
#     For a 3D image the Minkowski functionals can be computed using the following
#     example:
#
#     .. code-block:: python
#
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from skimage.morphology import (ball)
#         from quantimpy import morphology as mp
#         from quantimpy import minkowski as mk
#
#         image = np.zeros([128,128,128],dtype=bool)
#         image[16:113,16:113,16:113] = ball(48,dtype=bool)
#
#         erosion_map = mp.erode_map(image)
#
#         plt.gray()
#         plt.imshow(image[:,:,64])
#         plt.show()
#
#         plt.gray()
#         plt.imshow(erosion_map[:,:,64])
#         plt.show()
#
#         dist, volume, surface, curvature, euler = mk.functions_close(erosion_map)
#
#         plt.plot(dist,volume)
#         plt.show()
#
#         plt.plot(dist,surface)
#         plt.show()
#
#         plt.plot(dist,curvature)
#         plt.show()
#
#         plt.plot(dist,euler)
#         plt.show()
#
#     """
# # Decompose resolution in number larger than one and a pre-factor
#     factor = 1.0
#     if not (res is None):
#         factor = np.amin(res)
#         res = res/factor
#
#     if not (closing.dtype == 'bool'):
#         closing = closing.astype('bool')
#
#     if (closing.ndim == 2):
# # Set default resolution (length/voxel)
#         if (res is None):
#             res0 = 1.0
#             res1 = 1.0
#         elif (res.size == 2):
#             res = res.astype(np.float64)
#             res0 = res[0]
#             res1 = res[1]
#         else:
#             raise ValueError('Input image and resolution need to be the same dimension')
#
#         return _functions_close_2d(closing, res0, res1, factor, norm)
#     elif (closing.ndim == 3):
# # Set default resolution (length/voxel)
#         if (res is None):
#             res0 = 1.0
#             res1 = 1.0
#             res2 = 1.0
#         elif (res.size == 3):
#             res = res.astype(np.float64)
#             res0 = res[0]
#             res1 = res[1]
#             res2 = res[2]
#         else:
#             raise ValueError('Input image and resolution need to be the same dimension')
#
#         return _functions_close_3d(closing, res0, res1, res2, factor, norm)
#     else:
#         raise ValueError('Can only handle 2D or 3D closings')
#
# def _functions_close_2d(
#         closing:np.ndarray[bool],
#         res0:np.float64,
#         res1:np.float64,
#         factor:np.float64,
#         norm:bool):
#     closing = np.ascontiguousarray(closing)
#     dim3 = np.amax(closing) - np.amin(closing)
#     dist = np.empty(dim3,dtype=np.float64)
#     area = np.empty(dim3,dtype=np.float64)
#     length = np.empty(dim3,dtype=np.float64)
#     euler4 = np.empty(dim3,dtype=np.float64)
#     euler8 = np.empty(dim3,dtype=np.float64)
#     status = c_functions_close_2d(closing.reshape(-1), closing.shape[0], closing.shape[1], res0, res1, dist, area, length, euler4, euler8)
#     assert status == 0
#     if norm:
#         total_area = closing.shape[0]*closing.shape[1]*res0*res1
#         return dist*factor, area/(total_area), length/(total_area*factor), euler8/(total_area*factor**2)
#     else:
#         return dist*factor, area*factor**2, length*factor, euler8
#
# def _functions_close_3d(
#         closing:np.ndarray[bool],
#         res0:np.float64,
#         res1:np.float64,
#         res2:np.float64,
#         factor:np.float64,
#         norm:bool):
#     closing = np.ascontiguousarray(closing)
#     dim3 = np.amax(closing) - np.amin(closing)
#     dist = np.empty(dim3,dtype=np.float64)
#     volume = np.empty(dim3,dtype=np.float64)
#     surface = np.empty(dim3,dtype=np.float64)
#     curvature = np.empty(dim3,dtype=np.float64)
#     euler6 = np.empty(dim3,dtype=np.float64)
#     euler26 = np.empty(dim3,dtype=np.float64)
#
#     status = c_functions_close_3d(closing, closing.shape[0], closing.shape[1], closing.shape[2], res0, res1, res2, dist, volume, surface, curvature, euler6, euler26)
#     assert status == 0
#     if norm:
#         total_volume = closing.shape[0]*closing.shape[1]*closing.shape[2]*res0*res1*res2
#         return dist*factor, volume/(total_volume), surface/(total_volume*factor), curvature/(total_volume*factor**2), euler26/(total_volume*factor**3)
#     else:
#         return dist*factor, volume*factor**3, surface*factor**2, curvature*factor, euler26
#
# # }}}
# ###############################################################################