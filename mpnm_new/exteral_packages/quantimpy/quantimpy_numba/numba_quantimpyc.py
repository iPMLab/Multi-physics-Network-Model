# import numba as nb
# import numpy as np
# from constants import USHRT_MAX



# unsigned short r_pixel_2d(int x, int y, unsigned short* image, int dim1) {
#     int i;
#
#     i = x*dim1 + y;
#
#     return image[i];
# }


# unsigned short r_pixel_3d(int x, int y, int z, unsigned short* image, int dim1, int dim2) {
#     int i;
#
#     i = (x*dim1 + y)*dim2 + z;
#
#     return image[i];
# }



# void w_pixel_2d(int x, int y, unsigned short* image, int dim1, unsigned short value) {
#     int i;
#
#     i = x*dim1 + y;
#
#     image[i] = value;
# }
#
# /******************************************************************************/
#
# void w_pixel_3d(int x, int y, int z, unsigned short* image, int dim1, int dim2, unsigned short value) {
#     int i;
#
#     i = (x*dim1 + y)*dim2 + z;
#
#     image[i] = value;
# }
#
# // }}}
# /******************************************************************************/
# // {{{ bin
#
# void bin_2d(int low, int value1, int value2, unsigned short* image, int dim0, int dim1) {
#     int x, y, val;
#
#     val = value2>USHRT_MAX? -1 : value2;
#
#     for (y = 0; y < dim1; y++)
#         for (x = 0; x < dim0; x++) {
#             if (r_pixel_2d(x, y, image, dim1) <= low)
#                 w_pixel_2d(x, y, image, dim1, value1);
#             else if (val != -1)
#                 w_pixel_2d(x, y, image, dim1, val);
#     }
# }
#
# /******************************************************************************/
#
# void bin_3d(int low, int value1, int value2, unsigned short* image, int dim0, int dim1, int dim2) {
#     int x, y, z, val;
#
#     val = value2>USHRT_MAX? -1 : value2;
#
#     for (z = 0; z < dim2; z++)
#         for (y = 0; y < dim1; y++)
#             for (x = 0; x < dim0; x++) {
#                 if (r_pixel_3d(x, y, z, image, dim1, dim2) <= low)
#                     w_pixel_3d(x, y, z, image, dim1, dim2, value1);
#                 else if (val != -1)
#                     w_pixel_3d(x, y, z, image, dim1, dim2, val);
#     }
# }

# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def r_pixel_2d(x:int, y:int, image :np.ndarray[np.uint16], dim1:int)->np.uint16:
#     i = x*dim1 + y
#     return image[i]

# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def r_pixel_3d(x:int, y:int, z:int, image :np.ndarray[np.uint16], dim1:int, dim2:int)->np.uint16:
#     i = (x*dim1 + y)*dim2 + z
#     return image[i]

# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def w_pixel_2d(x:int, y:int, image :np.ndarray[np.uint16], dim1:int, value:np.uint16)->None:
#     i = x*dim1 + y
#     image[i] = value

# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def w_pixel_3d(x:int, y:int, z:int, image :np.ndarray[np.uint16], dim1:int, dim2:int, value:np.uint16)->None:
#     i = (x*dim1 + y)*dim2 + z
#     image[i] = value

# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def bin_2d(low:int, value1:np.uint16, value2:np.uint16, image :np.ndarray[np.uint16], dim0:int, dim1:int)->None:
#     val = value2 if value2<=USHRT_MAX else -1
#     for y in range(dim1):
#         for x in range(dim0):
#             if r_pixel_2d(x, y, image, dim1) <= low:
#                 w_pixel_2d(x, y, image, dim1, value1)
#             elif val != -1:
#                 w_pixel_2d(x, y, image, dim1, val)

# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def bin_3d(low:int, value1:np.uint16, value2:np.uint16, image :np.ndarray[np.uint16], dim0:int, dim1:int, dim2:int)->None:
#     val = value2 if value2<=USHRT_MAX else -1
#     for z in range(dim2):
#         for y in range(dim1):
#             for x in range(dim0):
#                 if r_pixel_3d(x, y, z, image, dim1, dim2) <= low:
#                     w_pixel_3d(x, y, z, image, dim1, dim2, value1)
#                 elif val != -1:
#                     w_pixel_3d(x, y, z, image, dim1, dim2, val)