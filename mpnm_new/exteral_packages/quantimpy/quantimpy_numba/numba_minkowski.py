import numba as nb
import numpy as np
from constants import *


# int c_functionals_2d(unsigned short* image, int dim0, int dim1, double res0, double res1, double* area, double* length, double* euler4, double* euler8) {
#     double norm;
#     long int* h;
#
#     norm = (double)(dim0-1) * (dim1-1) * res0 * res1;
#
#     h = quant_2d(image, dim0, dim1);
#
#     *area   = norm * area_dens_2d(h);
#     *length = norm * leng_dens_2d(h,res0,res1);
#     *euler4 = norm * eul4_dens_2d(h,res0,res1);
#     *euler8 = norm * eul8_dens_2d(h,res0,res1);
#
#     free(h);
#
#     return 0;
# }
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def c_functionals_2d(image: np.ndarray[bool],res0: np.float64, res1: np.float64,return_area,return_length,return_euler8):
    dim0,dim1 = image.shape
    norm = (dim0 - 1) * (dim1 - 1) * res0 * res1
    h = quant_2d(image)
    area = norm * area_dens_2d(h) if return_area else 0.
    length = norm * leng_dens_2d(h, res0, res1) if return_length else 0.
    # euler4 = norm * eul4_dens_2d(h, res0, res1) if return_euler4 else 0.
    euler8 = norm * eul8_dens_2d(h, res0, res1) if return_euler8 else 0.
    return area, length, euler8


# /******************************************************************************/
#
# int c_functionals_3d(unsigned short* image, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* volume, double* surface, double* curvature, double* euler6, double* euler26) {
#     double norm;
#     long int* h;
#
#     norm = (double)(dim0-1) * (dim1-1) * (dim2-1) * res0 * res1 * res2;
#
#     h = quant_3d(image, dim0, dim1, dim2);
#
#     *volume    = norm * volu_dens_3d(h);
#     *surface   = norm * surf_dens_3d(h, res0, res1, res2);
#     *curvature = norm * curv_dens_3d(h, res0, res1, res2);
#     *euler6    = norm * eul6_dens_3d(h, res0, res1, res2);
#     *euler26   = norm * eu26_dens_3d(h, res0, res1, res2);
#
#     free(h);
#
#     return 0;
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def c_functionals_3d(image: np.ndarray[bool], res0: np.float64, res1: np.float64,
                     res2: np.float64,return_volume,return_surface,return_curvature,return_euler26):
    dim0, dim1, dim2 = image.shape
    norm = (dim0 - 1) * (dim1 - 1) * (dim2 - 1) * res0 * res1 * res2
    h = quant_3d(image)
    volume = norm * volu_dens_3d(h) if return_volume else 0.
    surface = norm * surf_dens_3d(h, res0, res1, res2) if return_surface else 0.
    curvature = norm * curv_dens_3d(h, res0, res1, res2) if return_curvature else 0.
    # euler6 = norm * eul6_dens_3d(h, res0, res1, res2)
    euler26 = norm * eu26_dens_3d(h, res0, res1, res2) if return_euler26 else 0.
    return volume, surface, curvature, euler26


# /******************************************************************************/
# // {{{ c_functions_open
#
# int c_functions_open_2d(unsigned short* opening, int dim0, int dim1, double res0, double res1, double* dist, double* area, double* length, double* euler4, double* euler8) {
#     int i, j, k;
#     double norm;
#     long int* h;
#     unsigned short* image;
#     unsigned short min, max;
#
#     image = (unsigned short *)malloc(dim0*dim1*sizeof(unsigned short));
#
#     min = USHRT_MAX;
#     max = 0;
#
#     for (k = 0; k < dim0*dim1; k++) {
#         if (min > opening[k]) min = opening[k];
#         if (max < opening[k]) max = opening[k];
#     }
#
#     norm = (double)(dim0-1) * (dim1-1) * res0 * res1;
#
#     for (i = 0, j = min; j < max; i++, j++) {
#         printf("\rFunctions open step: %d \n",i);
#
#         for (k = 0; k < dim0*dim1; k++) {
#             image[k] = opening[k];
#         }
#
#         bin_2d(j, 0, USHRT_MAX, image, dim0, dim1);
#
#         h = quant_2d(image, dim0, dim1);
#
#         dist[i]   = (double)(i);
#         area[i]   = norm * area_dens_2d(h);
#         length[i] = norm * leng_dens_2d(h,res0,res1);
#         euler4[i] = norm * eul4_dens_2d(h,res0,res1);
#         euler8[i] = norm * eul8_dens_2d(h,res0,res1);
#     }
#
#     free(h);
#     free(image);
#
#     return 0;
# }
# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def c_functions_open_2d(opening: np.ndarray[np.uint16], dim0: int, dim1: int, res0: float, res1: float,
#                         dist: nb.float64[:], area: nb.float64[:], length: nb.float64[:], euler4: nb.float64[:],
#                         euler8: nb.float64[:]) -> int:
#     min = USHRT_MAX
#     max = 0
#     for k in range(dim0 * dim1):
#         if min > opening[k]:
#             min = opening[k]
#         if max < opening[k]:
#             max = opening[k]
#     image = opening.copy()
#     norm = (dim0 - 1) * (dim1 - 1) * res0 * res1
#     for i, j in zip(range(0, max - min), range(min, max)):
#         print("\rFunctions open step: %d \n", i)
#         bin_2d(j, np.uint16(0), USHRT_MAX, image, dim0, dim1)
#         h = quant_2d(image, dim0, dim1)
#         dist[i] = np.float64(i)
#         area[i] = norm * area_dens_2d(h)
#         length[i] = norm * leng_dens_2d(h, res0, res1)
#         euler4[i] = norm * eul4_dens_2d(h, res0, res1)
#         euler8[i] = norm * eul8_dens_2d(h, res0, res1)

#     return 0


# /******************************************************************************/
#
# int c_functions_open_3d(unsigned short* opening, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* dist, double* volume, double* surface, double* curvature, double* euler6, double* euler26) {
#     int i, j, k;
#     double norm;
#     long int* h;
#     unsigned short* image;
#     unsigned short min, max;
#
#     image = (unsigned short *)malloc(dim0*dim1*dim2*sizeof(unsigned short));
#
#     min = USHRT_MAX;
#     max = 0;
#
#     for (k = 0; k < dim0*dim1*dim2; k++) {
#         if (min > opening[k]) min = opening[k];
#         if (max < opening[k]) max = opening[k];
#     }
#
#     norm = (double)(dim0-1) * (dim1-1) * (dim2-1) * res0 * res1 * res2;
#
#     for (i = 0, j = min; j < max; i++, j++) {
#         printf("\rFunctions open step: %d \n",i);
#
#         for (k = 0; k < dim0*dim1*dim2; k++) {
#             image[k] = opening[k];
#         }
#
#         bin_3d(j, 0, USHRT_MAX, image, dim0, dim1, dim2);
#
#         h = quant_3d(image, dim0, dim1, dim2);
#
#         dist[i]      = (double)(i);
#         volume[i]    = norm * volu_dens_3d(h);
#         surface[i]   = norm * surf_dens_3d(h, res0, res1, res2);
#         curvature[i] = norm * curv_dens_3d(h, res0, res1, res2);
#         euler6[i]    = norm * eul6_dens_3d(h, res0, res1, res2);
#         euler26[i]   = norm * eu26_dens_3d(h, res0, res1, res2);
#     }
#
#     free(h);
#     free(image);
#
#     return 0;
# }
#
# // }}}

def c_functions_open_3d(opening: np.ndarray[np.uint16], dim0: int, dim1: int, dim2: int, res0: float, res1: float,
                        res2: float, dist: nb.float64[:], volume: nb.float64[:], surface: nb.float64[:],
                        curvature: nb.float64[:], euler6: nb.float64[:], euler26: nb.float64[:]) -> int:
    image = opening.copy()
    min = USHRT_MAX
    max = 0
    for k in range(dim0 * dim1 * dim2):
        if min > opening[k]:
            min = opening[k]
        if max < opening[k]:
            max = opening[k]
    norm = (dim0 - 1) * (dim1 - 1) * (dim2 - 1) * res0 * res1 * res2
    for i, j in zip(range(0, max - min), range(min, max)):
        print("\rFunctions open step: %d \n", i)
        bin_3d(j, np.uint16(0), USHRT_MAX, image, dim0, dim1, dim2)
        h = quant_3d(image)
        dist[i] = np.float64(i)
        volume[i] = norm * volu_dens_3d(h)
        surface[i] = norm * surf_dens_3d(h, res0, res1, res2)
        curvature[i] = norm * curv_dens_3d(h, res0, res1, res2)
        euler6[i] = norm * eul6_dens_3d(h, res0, res1, res2)
        euler26[i] = norm * eu26_dens_3d(h, res0, res1, res2)

    return 0


# /******************************************************************************/
# // {{{ c_functions_close
#
# int c_functions_close_2d(unsigned short* closing, int dim0, int dim1, double res0, double res1, double* dist, double* area, double* length, double* euler4, double* euler8) {
#     int i, j, k;
#     double norm;
#     long int* h;
#     unsigned short* image;
#     unsigned short min, max;
#
#     image = (unsigned short *)malloc(dim0*dim1*sizeof(unsigned short));
#
#     min = USHRT_MAX;
#     max = 0;
#
#     for (k = 0; k < dim0*dim1; k++) {
#         if (min > closing[k]) min = closing[k];
#         if (max < closing[k]) max = closing[k];
#     }
#
#     norm = (double)(dim0-1) * (dim1-1) * res0 * res1;
#
#     for (i = 0, j = max-1; j > min-1; i++, j--) {
#         printf("\rFunctions close step: %d \n",i+1);
#
#         for (k = 0; k < dim0*dim1; k++) {
#             image[k] = closing[k];
#         }
#
#         bin_2d(j, 0, USHRT_MAX, image, dim0, dim1);
#
#         h = quant_2d(image, dim0, dim1);
#
#         dist[i]   = (double)(i+1.0);
#         area[i]   = norm * area_dens_2d(h);
#         length[i] = norm * leng_dens_2d(h,res0,res1);
#         euler4[i] = norm * eul4_dens_2d(h,res0,res1);
#         euler8[i] = norm * eul8_dens_2d(h,res0,res1);
#     }
#
#     free(h);
#     free(image);
#
#     return 0;
# }
def c_functions_close_2d(closing: np.ndarray[np.uint16], dim0: int, dim1: int, res0: float, res1: float,
                         dist: nb.float64[:], area: nb.float64[:], length: nb.float64[:], euler4: nb.float64[:],
                         euler8: nb.float64[:]) -> int:
    image = closing.copy()
    min = USHRT_MAX
    max = 0
    for k in range(dim0 * dim1):
        if min > closing[k]:
            min = closing[k]
        if max < closing[k]:
            max = closing[k]

    norm = (dim0 - 1) * (dim1 - 1) * res0 * res1
    for i, j in zip(range(0, max - min), range(max - 1, min - 1, -1)):
        print("\rFunctions close step: ", i + 1)
        bin_2d(j, np.uint16(0), USHRT_MAX, image, dim0, dim1)
        h = quant_2d(image, dim0, dim1)
        dist[i] = np.float64(i) + 1.0
        area[i] = norm * area_dens_2d(h)
        length[i] = norm * leng_dens_2d(h, res0, res1)
        euler4[i] = norm * eul4_dens_2d(h, res0, res1)
        euler8[i] = norm * eul8_dens_2d(h, res0, res1)

    return 0


# /******************************************************************************/
# int c_functions_close_3d(unsigned short* closing, int dim0, int dim1, int dim2, double res0, double res1, double res2, double* dist, double* volume, double* surface, double* curvature, double* euler6, double* euler26) {
#     int i, j, k;
#     double norm;
#     long int* h;
#     unsigned short* image;
#     unsigned short min, max;
#
#     image = (unsigned short *)malloc(dim0*dim1*dim2*sizeof(unsigned short));
#
#     min = USHRT_MAX;
#     max = 0;
#
#     for (k = 0; k < dim0*dim1*dim2; k++) {
#         if (min > closing[k]) min = closing[k];
#         if (max < closing[k]) max = closing[k];
#     }
#
#     norm = (double)(dim0-1) * (dim1-1) * (dim2-1) * res0 * res1 * res2;
#
#     for (i = 0, j = max-1; j > min-1; i++, j--) {
#         printf("\rFunctions close step: %d \n",i+1);
#
#         for (k = 0; k < dim0*dim1*dim2; k++) {
#             image[k] = closing[k];
#         }
#
#         bin_3d(j, 0, USHRT_MAX, image, dim0, dim1, dim2);
#
#         h = quant_3d(image, dim0, dim1, dim2);
#
#         dist[i]      = (double)(i+1.0);
#         volume[i]    = norm * volu_dens_3d(h);
#         surface[i]   = norm * surf_dens_3d(h, res0, res1, res2);
#         curvature[i] = norm * curv_dens_3d(h, res0, res1, res2);
#         euler6[i]    = norm * eul6_dens_3d(h, res0, res1, res2);
#         euler26[i]   = norm * eu26_dens_3d(h, res0, res1, res2);
#     }
#
#     free(h);
#     free(image);
#
#     return 0;
# }
#
# // }}}
# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def c_functions_close_3d(closing: np.ndarray[np.uint16], dim0: int, dim1: int, dim2: int, res0: np.float64,
#                          res1: np.float64, res2: np.float64, dist: nb.float64[:], volume: nb.float64[:],
#                          surface: nb.float64[:], curvature: nb.float64[:], euler6: nb.float64[:],
#                          euler26: nb.float64[:]) -> int:
#     image = closing.copy()
#     min = USHRT_MAX
#     max = 0
#     for k in range(dim0 * dim1 * dim2):
#         if min > closing[k]:
#             min = closing[k]
#         if max < closing[k]:
#             max = closing[k]

#     norm = (dim0 - 1) * (dim1 - 1) * (dim2 - 1) * res0 * res1 * res2
#     for i, j in zip(range(0, max - min), range(max - 1, min - 1, -1)):
#         print("\rFunctions close step: ", i + 1)
#         bin_3d(j, np.uint16(0), USHRT_MAX, image, dim0, dim1, dim2)
#         h = quant_3d(image)
#         dist[i] = np.float64(i) + 1.0
#         volume[i] = norm * volu_dens_3d(h)
#         surface[i] = norm * surf_dens_3d(h, res0, res1, res2)
#         curvature[i] = norm * curv_dens_3d(h, res0, res1, res2)
#         euler6[i] = norm * eul6_dens_3d(h, res0, res1, res2)
#         euler26[i] = norm * eu26_dens_3d(h, res0, res1, res2)

#     return 0


# /******************************************************************************/
# // {{{ quant
#
# long int* quant_2d(unsigned short* image, int dim0, int dim1) {
#     int x, y, i;
#     int mask;
# 	long int *h;
#
#     h = (long int*)malloc(16*sizeof(long int));
#
#  	for (i = 0; i < 16; i++) h[i] = 0;
#
#     for (x = 0; x < dim0 - 1; x++) {
#         mask =  (r_pixel_2d(x  ,0,image,dim1) == USHRT_MAX)
#              + ((r_pixel_2d(x+1,0,image,dim1) == USHRT_MAX) << 1);
#         for (y = 1; y < dim1; y++) {
#             mask += ((r_pixel_2d(x  ,y,image,dim1) == USHRT_MAX) << 2)
#                   + ((r_pixel_2d(x+1,y,image,dim1) == USHRT_MAX) << 3);
# 		    h[mask]++;
# 		    mask >>= 2;
#         }
#  	}
#
#     return h;
# }


# @nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
# def quant_2d(image: np.ndarray[bool], dim0: int, dim1: int) -> np.ndarray[np.int64]:
#     h = np.zeros(16, dtype=np.int64)
#     for x in range(dim0 - 1):
#         mask = image[x, 0] | (image[x + 1, 0] << 1)
#         for y in range(1, dim1):
#             mask |= (image[x, y] << 2) | (image[x + 1, y] << 3)
#             h[mask] += 1
#             mask >>= 2
#     return h

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def quant_2d(image: np.ndarray[bool]) -> np.ndarray[np.int64]:
    dim0, dim1 = image.shape
    h = np.zeros(16, dtype=np.int64)
    for x in range(dim0 - 1):
        row_current = image[x]
        row_next = image[x + 1]
        mask = (row_next[0] << 1) | row_current[0]
        for y in range(1, dim1):
            mask |= ((row_next[y] << 1) | row_current[y]) << 2
            h[mask] += 1
            mask >>= 2
    return h

    #     mask = (r_pixel_2d(x, 0, image, dim1) == USHRT_MAX) + ((r_pixel_2d(x + 1, 0, image, dim1) == USHRT_MAX) << 1)
    #     for y in range(1, dim1):
    #         mask += ((r_pixel_2d(x, y, image, dim1) == USHRT_MAX) << 2) + (
    #                 (r_pixel_2d(x + 1, y, image, dim1) == USHRT_MAX) << 3)
    #         h[mask] += 1
    #         mask >>= 2
    # return h


# /******************************************************************************/
#
# long int* quant_3d(unsigned short* image, int dim0, int dim1, int dim2) {
# 	int x, y, z, i;
#     int mask;
# 	long int *h;
#
# 	h = (long int*)malloc(256*sizeof(long int));
#
#  	for (i = 0; i < 256; i++) h[i] = 0;
#
#     for (x = 0; x < dim0-1; x++)
#         for (y = 0; y < dim1-1; y++) {
# 		    mask =  (r_pixel_3d(x  ,y  ,0,image,dim1,dim2) == USHRT_MAX)
#                  + ((r_pixel_3d(x+1,y  ,0,image,dim1,dim2) == USHRT_MAX) << 1)
#                  + ((r_pixel_3d(x  ,y+1,0,image,dim1,dim2) == USHRT_MAX) << 2)
#                  + ((r_pixel_3d(x+1,y+1,0,image,dim1,dim2) == USHRT_MAX) << 3);
#     		for (z = 1; z < dim2; z++) {
#                 mask += ((r_pixel_3d(x  ,y  ,z,image,dim1,dim2) == USHRT_MAX) << 4)
#                       + ((r_pixel_3d(x+1,y  ,z,image,dim1,dim2) == USHRT_MAX) << 5)
#                       + ((r_pixel_3d(x  ,y+1,z,image,dim1,dim2) == USHRT_MAX) << 6)
#                       + ((r_pixel_3d(x+1,y+1,z,image,dim1,dim2) == USHRT_MAX) << 7);
#     		    h[mask]++;
# 		        mask >>= 4;
# 		    }
#     }
#
#     return h;
# }

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def quant_3d(image: np.ndarray[bool]) -> np.ndarray[np.int64]:
    dim0, dim1, dim2 = image.shape
    h = np.zeros(256, dtype=np.int64)
    for x in range(dim0 - 1):
        for y in range(dim1 - 1):
            px0y0 = image[x,y]
            px1y0 = image[x+1,y]
            px0y1 = image[x,y+1]
            px1y1 = image[x+1,y+1]
            # 初始化低四位掩码
            mask = px0y0[0] | (px1y0[0] << 1) | (px0y1[0] << 2) | (px1y1[0] << 3)
            for z in range(1, dim2):
                # 预加载当前z层的四个相邻点
                mask |= (px0y0[z] << 4) | (px1y0[z] << 5) | (px0y1[z] << 6) | (px1y1[z] << 7)
                h[mask] += 1
                mask >>= 4
    return h

    # for x in range(dim0 - 1):
    #     for y in range(dim1 - 1):
    #         mask = (r_pixel_3d(x, y, 0, image, dim1, dim2) == USHRT_MAX) + (
    #                 (r_pixel_3d(x + 1, y, 0, image, dim1, dim2) == USHRT_MAX) << 1) + (
    #                        (r_pixel_3d(x, y + 1, 0, image, dim1, dim2) == USHRT_MAX) << 2) + (
    #                        (r_pixel_3d(x + 1, y + 1, 0, image, dim1, dim2) == USHRT_MAX) << 3)
    #         for z in range(1, dim2):
    #             mask += ((r_pixel_3d(x, y, z, image, dim1, dim2) == USHRT_MAX) << 4) + (
    #                     (r_pixel_3d(x + 1, y, z, image, dim1, dim2) == USHRT_MAX) << 5) + (
    #                             (r_pixel_3d(x, y + 1, z, image, dim1, dim2) == USHRT_MAX) << 6) + (
    #                             (r_pixel_3d(x + 1, y + 1, z, image, dim1, dim2) == USHRT_MAX) << 7)
    #
    #             h[mask] += 1
    #
    #             mask >>= 4
    # return h


# /******************************************************************************/
# // {{{ area_dens_2d
#
# double area_dens_2d(long int *h) {
#     int i;
# 	unsigned long int iChi = 0, iVol = 0;
#
#     for (i = 0; i < 16; i++) {
#         iChi += (i&1)*h[i];
# 		iVol += h[i];
# 	}
#
# 	if(!iVol) return 0;
#  	else return (double)iChi/iVol;
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def area_dens_2d(h: np.ndarray[np.int64]) -> np.float64:
    iChi = 0
    iVol = 0
    for i in range(16):
        iChi += (i & 1) * h[i]
        iVol += h[i]
    if iVol == 0:
        return np.float64(0.)
    else:
        return np.float64(iChi) / np.float64(iVol)


# /******************************************************************************/
# // {{{ leng_dens_2d

# double leng_dens_2d(long int *h, double res0, double res1) {
# 	unsigned int i, l;
# 	long int  numpix=0, ii;
# 	double II=0, LI=0, w[4], r[4];
# 	int kl[4][2] = {{1,2},{1,4},{1,8},{2,4}};
#
# 	r[0] = res0;
# 	r[1] = res1;
# 	r[2] = r[3] = sqrt(r[0]*r[0] + r[1]*r[1]);
#
# 	w[0] = atan(res1/res0)/M_PI;
# 	w[1] = atan(res0/res1)/M_PI;
# 	w[2] = w[3] = (1 - w[0] - w[1])/2;
#
# 	for (l = 0; l < 16; l++) numpix += h[l];
#
# 	for (i = 0; i < 4; i++) {
#         ii = 0;
#  	    for (l = 0; l < 16; l++) {
# 	        ii += h[l] * (l == (l|kl[i][0])) * (0 == (l&kl[i][1]));
# 		    ii += h[l] * (l == (l|kl[i][1])) * (0 == (l&kl[i][0]));
# 	    }
# 	    II += (double)w[i]*ii;
# 	    LI += (double)w[i]*numpix*r[i];
# 	}
#
# 	if(!LI) return 0;
# 	else return 0.25 * II/LI;
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def leng_dens_2d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64) -> np.float64:
    II = 0.
    LI = 0.
    w = np.empty(4, dtype=np.float64)
    r = np.empty(4, dtype=np.float64)
    kl = leng_dens_2d_kl

    r[0] = res0
    r[1] = res1
    r[2] = r[3] = np.sqrt(res0 ** 2 + res1 ** 2)

    w[0] = np.arctan(res1 / res0) / np.pi
    w[1] = np.arctan(res0 / res1) / np.pi
    w[2] = (1 - w[0] - w[1]) / 2
    w[3] = w[2]
    numpix = np.sum(h)
    for i in range(4):
        ii = 0
        for l in range(16):
            ii += (h[l] * (l == (l | kl[i][0])) * (0 == (l & kl[i][1])) + h[l] * (l == (l | kl[i][1])) * (
                    0 == (l & kl[i][0])))
        II += w[i] * ii
        LI += w[i] * numpix * r[i]
    if LI == 0.:
        return np.float64(0.)
    else:
        return 0.25 * II / LI


# /******************************************************************************/
# // {{{ eule_dens_2d
#
# double eul4_dens_2d(long int *h, double res0, double res1) {
# 	int i;
# 	long int iChi = 0, iVol = 0;
# 	int iu[16] = {0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0};
#
#  	for (i = 0; i < 16; i++) {
# 		iChi += iu[i]*h[i];
# 		iVol += h[i];
# 	}
#
#  	return (double)iChi/((double)iVol*res0*res1)/M_PI;
# }
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eul4_dens_2d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64) -> np.float64:
    iu = eul4_dens_2d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    return np.float64(iChi) / (np.float64(iVol) * res0 * res1) / np.pi


# /******************************************************************************/
#
# double eul8_dens_2d(long int *h, double res0, double res1) {
# 	int i;
# 	long int iChi = 0, iVol = 0;
# 	int iu[16] = {0, 3, 3, 0, 3, 0,  6, -3, 3, 6, 0, -3, 0, -3, -3, 0};
#
#  	for(i = 0; i < 16; i++) {
# 		iChi += iu[i]*h[i];
# 		iVol += h[i];
# 	}
#
#     return (double)iChi/((double)iVol*res0*res1)/(12*M_PI);
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eul8_dens_2d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64) -> np.float64:
    iu = eul8_dens_2d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    return np.float64(iChi) / (np.float64(iVol) * res0 * res1) / (12 * np.pi)


# /******************************************************************************/
# // {{{ volu_dens_3d
#
# double volu_dens_3d(long int *h) {
# 	int i;
# 	unsigned long int iChi = 0, iVol = 0;
#
# 	for (i = 0; i < 256; i++) {
#         iVol += h[i];
# 	    if (i&1) iChi += h[i];
# 	}
#
# 	if(!iVol) return 0;
# 	else return (double)iChi/(double)iVol;
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def volu_dens_3d(h: np.ndarray[np.int64]) -> np.float64:
    iChi = 0
    iVol = 0
    for i in range(256):
        iVol += h[i]
        if i & 1:
            iChi += h[i]
    if iVol == 0:
        return np.float64(0.)
    else:
        return np.float64(iChi) / np.float64(iVol)


# /******************************************************************************/
# // {{{ surf_dens_3d
#
# double surf_dens_3d(long int *h, double res0, double res1, double res2) {
#     int i, l;
# 	unsigned long sv, le = 0;
# 	double wi[13], r[13], Sv, Lv, *Delta, *weight;
# 	int kl[13][2] = {{1,2},{1,4},{1,16},{1,8},{2,4},{1,32},{2,16},{1,64},{4,16},{1,128},{2,64},{4,32},{8,16}};
#
# 	Delta  = (double *)malloc(3*sizeof(double));
# 	weight = (double *)malloc(7*sizeof(double));
#
# 	for (i = 0; i < 7; i++) weight[i] = 0;
#
# 	r[0] = Delta[0] = res0;
# 	r[1] = Delta[1] = res1;
# 	r[2] = Delta[2] = res2;
# 	r[3] = r[4]  = sqrt(r[0]*r[0] + r[1]*r[1]);
# 	r[5] = r[6]  = sqrt(r[0]*r[0] + r[2]*r[2]);
# 	r[7] = r[8]  = sqrt(r[1]*r[1] + r[2]*r[2]);
# 	r[9] = r[10] = r[11] = r[12] = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
#
# 	weights(Delta, weight);
#  	wi[0]  = weight[0]; wi[1] = weight[1]; wi[2]  = weight[2]; wi[3]  = weight[3];
# 	wi[4]  = weight[3]; wi[5] = weight[5]; wi[6]  = weight[5]; wi[7]  = weight[4];
# 	wi[8]  = weight[4]; wi[9] = weight[6]; wi[10] = weight[6]; wi[11] = weight[6];
# 	wi[12] = weight[6];
#
#     for (l = 0; l < 256; l++) le += h[l];
#
#     Sv = Lv = 0;
#     for (i = 0; i < 13; i++) {
#         if (wi[i]) {
#             sv = 0;
#             for (l = 0; l < 256; l++) {
#                 sv += h[l] * (l == (l|kl[i][0])) * (0 == (l&kl[i][1]));
# 	            sv += h[l] * (l == (l|kl[i][1])) * (0 == (l&kl[i][0]));
#  	        }
# 	        Sv += wi[i]*(double)sv;
# 	        Lv += wi[i]*(double)le*r[i];
#  	    }
#  	}
#
# 	free(Delta);
# 	free(weight);
#
# 	if(!Lv) return 0;
# 	else return 0.25 * Sv/Lv;
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def surf_dens_3d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64, res2: np.float64) -> np.float64:
    kl = surf_dens_3d_kl
    Delta = np.empty(3, dtype=np.float64)
    weight = np.zeros(7, dtype=np.float64)
    r = np.empty(13, dtype=np.float64)
    wi = np.empty(13, dtype=np.float64)
    r[0] = Delta[0] = res0
    r[1] = Delta[1] = res1
    r[2] = Delta[2] = res2
    r[3] = r[4] = np.sqrt(r[0] * r[0] + r[1] * r[1])
    r[5] = r[6] = np.sqrt(r[0] * r[0] + r[2] * r[2])
    r[7] = r[8] = np.sqrt(r[1] * r[1] + r[2] * r[2])
    r[9] = r[10] = r[11] = r[12] = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    weights(Delta, weight)
    wi[0] = weight[0]
    wi[1] = weight[1]
    wi[2] = weight[2]
    wi[3] = weight[3]
    wi[4] = weight[3]
    wi[5] = weight[5]
    wi[6] = weight[5]
    wi[7] = weight[4]
    wi[8] = weight[4]
    wi[9] = weight[6]
    wi[10] = weight[6]
    wi[11] = weight[6]
    wi[12] = weight[6]
    le = np.sum(h)
    Sv = 0.
    Lv = 0.
    for i in range(13):
        if wi[i]:
            sv = 0
            for l in range(256):
                sv += h[l] * (l == (l | kl[i][0])) * (0 == (l & kl[i][1])) + h[l] * (l == (l | kl[i][1])) * (
                        0 == (l & kl[i][0]))
            Sv += wi[i] * sv
            Lv += wi[i] * le * r[i]
    if Lv == 0.:
        return np.float64(0.)
    else:
        return 0.25 * Sv / Lv


# /******************************************************************************/
# // {{{ curv_dens_3d
#
# double curv_dens_3d(long int *h, double res0, double res1, double res2) {
# /* Mean curvature in 3D is related to the 2D-Euler number on a 2D section plane.
#  * Within a 2x2x2 cube 13 different planes can be defined (see lang+99). The
#  * results for the different planes are weighted by the sin of the plane to the
#  * vertical axis */
#     int i, k, l;
# 	unsigned long mc, le = 0;
#   	int kr[9][4] = {{1, 2, 4,  8},{1, 2,16,32},{1, 4,16, 64},
#                     {1, 2,64,128},{4,16, 8,32},{1,32, 4,128},
#                     {2, 8,16, 64},{2, 4,32,64},{1,16, 8,128}};
# 	int kt[8][3] = {{1,64, 32},{2,16,128},{8,64,32},{4,16,128},
# 			        {2, 4,128},{8, 1, 64},{2, 4,16},{8, 1,32}};
# 	double wi[13], a[13], r[3], s, Mc, atr;
# 	double r01, r02, r12;
# 	double *Delta, *weight;
#
# 	Delta  = (double *)malloc(3*sizeof(double));
# 	weight = (double *)malloc(7*sizeof(double));
#
# 	for (i = 0; i < 7; i++) weight[i] = 0;
#
# 	r[0] = Delta[0] = res0;
# 	r[1] = Delta[1] = res1;
# 	r[2] = Delta[2] = res2;
#
# 	r01 = sqrt(r[0]*r[0] + r[1]*r[1]);
# 	r02 = sqrt(r[0]*r[0] + r[2]*r[2]);
# 	r12 = sqrt(r[1]*r[1] + r[2]*r[2]);
#
# 	s = (r01 + r02 + r12)/2;
#
# 	a[0] = r[0]*r[1];
#     a[1] = r[0]*r[2];
#     a[2] = r[1]*r[2];
# 	a[3] = a[4] = r[0]*r12;
# 	a[5] = a[6] = r[1]*r02;
# 	a[7] = a[8] = r[2]*r01;
# 	atr = sqrt(s*(s-r01)*(s-r02)*(s-r12));
# 	a[9] = a[10] = a[11] = a[12] = 2*atr;
#
# 	weights(Delta, weight);
#  	wi[0]  = weight[2];
#     wi[1]  = weight[1];
#     wi[2]  = weight[0];
#
#     wi[3]  = weight[4];
# 	wi[4]  = weight[4];
#     wi[5]  = weight[5];
#     wi[6]  = weight[5];
#     wi[7]  = weight[3];
# 	wi[8]  = weight[3];
#
#     wi[9]  = weight[6];
#     wi[10] = weight[6];
#     wi[11] = weight[6];
#     wi[12] = weight[6];
#
#     for (l = 0; l < 256; l++) le += h[l];
#
#     Mc = 0;
#     for (i = 0; i < 9; i++) {
#         mc = 0;
# 	    for (l = 0; l < 256; l++) {
# 		    for (k = 0; k < 4; k++) {
#                 mc += h[l] * (l==(l|kr[i][k])) * (0==(l&kr[i][(k+1)%4])) * (0==(l&kr[i][(k+2)%4])) * (0==(l&kr[i][(k+3)%4]));
# 			    mc -= h[l] * (l==(l|kr[i][k])) * (l==(l|kr[i][(k+1)%4])) * (l==(l|kr[i][(k+2)%4])) * (0==(l&kr[i][(k+3)%4]));
#             }
#         }
# 	    Mc += wi[i]/(4*a[i])*(double)mc;
#     }
#
#     for (i = 9; i < 13; i++) {
#         mc = 0;
# 	    for (l = 0; l < 256; l++) {
#             for (k = 0; k < 3; k++) {
#                 mc += h[l] * (l==(l|kt[i-9][k])) * (0==(l&kt[i-9][(k+1)%3])) * (0==(l&kt[i-9][(k+2)%3]));
#                 mc -= h[l] * (l==(l|kt[i-5][k])) * (l==(l|kt[i-5][(k+1)%3])) * (0==(l&kt[i-5][(k+2)%3]));
#             }
#         }
# 	    Mc += wi[i]/(3*a[i])*(double)mc;
#     }
#
# 	free(Delta);
# 	free(weight);
#
#     return (double)Mc/(double)(le) * 2.0/M_PI;
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def curv_dens_3d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64, res2: np.float64) -> np.float64:
    kr = curv_dens_3d_kr
    kt = curv_dens_3d_kt
    Delta = np.empty(3, dtype=np.float64)
    weight = np.zeros(7, dtype=np.float64)
    r = np.empty(13, dtype=np.float64)
    wi = np.empty(13, dtype=np.float64)
    Delta[0] = r[0] = res0
    Delta[1] = r[1] = res1
    Delta[2] = r[2] = res2

    r[3] = r[4] = np.sqrt(r[0] * r[0] + r[1] * r[1])
    r[5] = r[6] = np.sqrt(r[0] * r[0] + r[2] * r[2])
    r[7] = r[8] = np.sqrt(r[1] * r[1] + r[2] * r[2])
    r[9] = r[10] = r[11] = r[12] = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    weights(Delta, weight)
    wi[0] = weight[2]
    wi[1] = weight[1]
    wi[2] = weight[0]
    wi[3] = weight[4]
    wi[4] = weight[4]
    wi[5] = weight[5]
    wi[6] = weight[5]
    wi[7] = weight[3]
    wi[8] = weight[3]
    wi[9] = weight[6]
    wi[10] = weight[6]
    wi[11] = weight[6]
    wi[12] = weight[6]
    le = np.sum(h)
    Mc = 0.
    for i in range(9):
        mc = 0
        for l in range(256):
            for k in range(4):
                mc += h[l] * (l == (l | kr[i][k])) * (0 == (l & kr[i][(k + 1) % 4])) * (
                        0 == (l & kr[i][(k + 2) % 4])) * (0 == (l & kr[i][(k + 3) % 4]))
                mc -= h[l] * (l == (l | kr[i][k])) * (l == (l | kr[i][(k + 1) % 4])) * (
                        l == (l | kr[i][(k + 2) % 4])) * (0 == (l & kr[i][(k + 3) % 4]))
        Mc += wi[i] / (4 * r[i]) * mc
    for i in range(9, 13):
        mc = 0
        for l in range(256):
            for k in range(3):
                mc += h[l] * (l == (l | kt[i - 9][k])) * (0 == (l & kt[i - 9][(k + 1) % 3])) * (
                        0 == (l & kt[i - 9][(k + 2) % 3]))
                mc -= h[l] * (l == (l | kt[i - 5][k])) * (l == (l | kt[i - 5][(k + 1) % 3])) * (
                        0 == (l & kt[i - 5][(k + 2) % 3]))
        Mc += wi[i] / (3 * r[i]) * mc
    return Mc / le * 2.0 / np.pi


# /******************************************************************************/
# // {{{ eule_dens_3d
#
# /******************************************************************************/
#
# double eul6_dens_3d(long int *h, double res0, double res1, double res2) {
# // This function is shown on page 122 of ohser 2000
#     int i;
# 	long int iChi = 0, iVol = 0;
# 	int iu[256] = {
#         0, 1,  0,  0,  0,  0,  0, -1,  0, 1,  0,  0,  0,  0, 0,  0, //   0..  15
# 	    0, 0,  0, -1,  0, -1,  0, -2,  0, 0,  0, -1,  0, -1, 0, -1, //  16..  31
# 	    0, 1,  0,  0,  0,  0,  0, -1,  0, 1,  0,  0,  0,  0, 0,  0, //  32..  47
# 	    0, 0,  0,  0,  0, -1,  0, -1,  0, 0,  0,  0,  0, -1, 0,  0, //  48..  63
# 	    0, 1,  0,  0,  0,  0,  0, -1,  0, 1,  0,  0,  0,  0, 0,  0, //  64..  79
# 	    0, 0,  0, -1,  0,  0,  0, -1,  0, 0,  0, -1,  0,  0, 0,  0, //  80..  95
# 	    0, 1,  0,  0,  0,  0,  0,  0,  0, 1,  0,  0,  0,  0, 0,  0, //  96.. 111
# 	    0, 0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0, 0,  1, // 112.. 127
# 	    0, 1,  0,  0,  0,  0,  0, -1,  0, 1,  0,  0,  0,  0, 0,  0, // 128.. 143
# 	    0, 0,  0, -1,  0, -1,  0, -2,  0, 0,  0, -1,  0, -1, 0, -1, // 144.. 159
# 	    0, 1,  0,  0,  0,  0,  0, -1,  0, 1,  0,  0,  0,  0, 0,  0, // 160.. 175
# 	    0, 0,  0,  0,  0, -1,  0, -1,  0, 0,  0,  0,  0, -1, 0,  0, // 176.. 191
# 	    0, 1,  0,  0,  0,  0,  0, -1,  0, 1,  0,  0,  0,  0, 0,  0, // 192.. 207
# 	    0, 0,  0, -1,  0,  0,  0, -1,  0, 0,  0, -1,  0,  0, 0,  0, // 208.. 223
# 	    0, 1,  0,  0,  0,  0,  0,  0,  0, 1,  0,  0,  0,  0, 0,  0, // 224.. 223
# 	    0, 0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0, 0,  0, // 240.. 255
# 	};
#
#  	for (i = 0; i < 256; i++) {
#         iChi += iu[i]*h[i];
# 		iVol += h[i];
# 	}
#
# 	if(!iVol) return 0;
#  	else return 3.0/(4.0*M_PI)*(double)iChi/((double)iVol*res0*res1*res2);
# }
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eul6_dens_3d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64, res2: np.float64) -> np.float64:
    iu = eul6_dens_3d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    if iVol == 0:
        return np.float64(0.)
    else:
        return 3.0 / (4.0 * np.pi) * np.float64(iChi) / (np.float64(iVol) * res0 * res1 * res2)


# /******************************************************************************/
#
# double eu26_dens_3d(long int *h, double res0, double res1, double res2) {
# // This function is shown on page 123 of ohser 2000
# 	int i;
# 	long int iChi = 0, iVol = 0;
# 	int iu[256] = {
#          0,  3,  3,  0,  3,  0,  6,  -3,  3,  6,  0,  -3,  0,  -3,  -3,  0, //   0..  15
#          3,  0,  6, -3,  6, -3,  9,  -6,  6,  3,  3,  -6,  3,  -6,   0, -3, //  16..  31
#          3,  6,  0, -3,  6,  3,  3,  -6,  6,  9, -3,  -6,  3,   0,  -6, -3, //  32..  47
#          0, -3, -3,  0,  3, -6,  0,  -3,  3,  0, -6,  -3,  0,  -8,  -8,  0, //  48..  63
#          3,  6,  6,  3,  0, -3,  3,  -6,  6,  9,  3,   0, -3,  -6,  -6, -3, //  64..  79
#          0, -3,  3, -6, -3,  0,  0,  -3,  3,  0,  0,  -8, -6,  -3,  -8,  0, //  80..  95
#          6,  9,  3,  0,  3,  0,  0,  -8,  9, 12,  0,  -3,  0,  -3,  -8, -6, //  96.. 111
#         -3, -6, -6, -3, -6, -3, -8,   0,  0, -3, -8,  -6, -8,  -6, -12,  3, // 112.. 127
#          3,  6,  6,  3,  6,  3,  9,   0,  0,  3, -3,  -6, -3,  -6,  -6, -3, // 128.. 143
#          6,  3,  9,  0,  9,  0, 12,  -3,  3,  0,  0,  -8,  0,  -8,  -3, -6, // 144.. 159
#          0,  3, -3, -6,  3,  0,  0,  -8, -3,  0,  0,  -3, -6,  -8,  -3,  0, // 160.. 175
#         -3, -6, -6, -3,  0, -8, -3,  -6, -6, -8, -3,   0, -8, -12,  -6,  3, // 176.. 191
#          0,  3,  3,  0, -3, -6,  0,  -8, -3,  0, -6,  -8,  0,  -3,  -3,  0, // 192.. 207
#         -3, -6,  0, -8, -6, -3, -3,  -6, -6, -8, -8, -12, -3,   0,  -6,  3, // 208.. 223
#         -3,  0, -6, -8, -6, -8, -8, -12, -6, -3, -3,  -6, -3,  -6,   0,  3, // 224.. 223
#          0, -3, -3,  0, -3,  0, -6,   3, -3, -6,  0,   3,  0,   3,   3,  0, // 240.. 255
# 	};
#
#  	for (i = 0; i < 256; i++) {
#         iChi += iu[i]*h[i];
# 		iVol += h[i];
# 	}
#
# 	if(!iVol) return 0;
#  	else return 1.0/(32.0*M_PI)*(double)iChi/((double)iVol*res0*res1*res2);
# }
#
# // }}}

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eu26_dens_3d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64, res2: np.float64) -> np.float64:
    iu = eu26_dens_3d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)

    if iVol == 0:
        return np.float64(0.)
    else:
        return 1.0 / (32.0 * np.pi) * np.float64(iChi) / (np.float64(iVol) / res0 * res1 * res2)


# /******************************************************************************/
# // {{{ weights
#
# void weights(double *Delta,double *weight) {
# /* Parameters: the side lengths of the unit cell Delta[0..2], Returns: the
#  * weights weights[0..6] for the computation of the surface denstiy,
#  * Initial: N.O. Aragones Salazar , 18 Sep 1999 */
#     int i, j, k;
#     double delta_xy, delta_yz, delta_zx, delta;
#     double v[8][3][6];   /* vertex of the polyhedron */
#     double dir[8][3][6]; /* projection of the vertex over the unit sphere */
#     double  prod0, prod1, prod2;
#
#     delta_xy = sqrt(Delta[0]*Delta[0]+Delta[1]*Delta[1]);
#     delta_yz = sqrt(Delta[1]*Delta[1]+Delta[2]*Delta[2]);
#     delta_zx = sqrt(Delta[2]*Delta[2]+Delta[0]*Delta[0]);
#     delta = sqrt(Delta[0]*Delta[0]+Delta[1]*Delta[1]+Delta[2]*Delta[2]);
#
#     v[0][0][0] = 1;
#     v[0][1][0] = (delta_xy-Delta[0])/Delta[1];
#     v[0][2][0] = (delta-delta_xy)/Delta[2];
#     v[1][0][0] = 1;
#     v[1][1][0] = (delta-delta_zx)/Delta[1];
#     v[1][2][0] = (delta_zx-Delta[0])/Delta[2];
#     v[2][0][0] = v[1][0][0]; v[2][1][0] = -v[1][1][0]; v[2][2][0] =  v[1][2][0];
#     v[3][0][0] = v[0][0][0]; v[3][1][0] = -v[0][1][0]; v[3][2][0] =  v[0][2][0];
#     v[4][0][0] = v[0][0][0]; v[4][1][0] = -v[0][1][0]; v[4][2][0] = -v[0][2][0];
#     v[5][0][0] = v[1][0][0]; v[5][1][0] = -v[1][1][0]; v[5][2][0] = -v[1][2][0];
#     v[6][0][0] = v[1][0][0]; v[6][1][0] =  v[1][1][0]; v[6][2][0] = -v[1][2][0];
#     v[7][0][0] = v[0][0][0]; v[7][1][0] =  v[0][1][0]; v[7][2][0] = -v[0][2][0];
#
#     v[0][0][1] = (delta-delta_yz)/Delta[0];
#     v[0][1][1] = 1;
#     v[0][2][1] = (delta_yz-Delta[1])/Delta[2];
#     v[1][0][1] = (delta_xy-Delta[1])/Delta[0];
#     v[1][1][1] = 1;
#     v[1][2][1] = (delta-delta_xy)/Delta[2];
#     v[2][0][1] = v[1][0][1];
#     v[2][1][1] = v[1][1][1]; v[2][2][1] = -v[1][2][1]; v[3][0][1] =  v[0][0][1];
#     v[3][1][1] = v[0][1][1]; v[3][2][1] = -v[0][2][1]; v[4][0][1] = -v[0][0][1];
#     v[4][1][1] = v[0][1][1]; v[4][2][1] = -v[0][2][1]; v[5][0][1] = -v[1][0][1];
#     v[5][1][1] = v[1][1][1]; v[5][2][1] = -v[1][2][1]; v[6][0][1] = -v[1][0][1];
#     v[6][1][1] = v[1][1][1]; v[6][2][1] =  v[1][2][1]; v[7][0][1] = -v[0][0][1];
#     v[7][1][1] = v[0][1][1]; v[7][2][1] =  v[0][2][1];
#
#     v[0][0][2] = (delta_zx-Delta[2])/Delta[0];
#     v[0][1][2] = (delta-delta_zx)/Delta[1];
#     v[0][2][2] = 1;
#     v[1][0][2] = (delta-delta_yz)/Delta[0];
#     v[1][1][2] = (delta_yz-Delta[2])/Delta[1];
#     v[1][2][2] = 1;
#     v[2][0][2] = -v[1][0][2]; v[2][1][2] =  v[1][1][2]; v[2][2][2] = v[1][2][2];
#     v[3][0][2] = -v[0][0][2]; v[3][1][2] =  v[0][1][2]; v[3][2][2] = v[0][2][2];
#     v[4][0][2] = -v[0][0][2]; v[4][1][2] = -v[0][1][2]; v[4][2][2] = v[0][2][2];
#     v[5][0][2] = -v[1][0][2]; v[5][1][2] = -v[1][1][2]; v[5][2][2] = v[1][2][2];
#     v[6][0][2] =  v[1][0][2]; v[6][1][2] = -v[1][1][2]; v[6][2][2] = v[1][2][2];
#     v[7][0][2] =  v[0][0][2]; v[7][1][2] = -v[0][1][2]; v[7][2][2] = v[0][2][2];
#
#     for (k = 0; k <= 2; k++)
#         for (i = 0; i <= 7; i++)
#             for (j = 0; j <= 2; j++)
#                 dir[i][j][k] = v[i][j][k]/sqrt(pow(v[i][0][k],2)
#                              + pow(v[i][1][k],2)
#                              + pow(v[i][2][k],2));
#
#     for (k = 0; k <= 2; k++) {
#         for (i = 0; i <= 7; i++) {
#             prod0 = dir[i%8][0][k]*dir[(i+2)%8][0][k]
#                   + dir[i%8][1][k]*dir[(i+2)%8][1][k]
#                   + dir[i%8][2][k]*dir[(i+2)%8][2][k];
#             prod1 = dir[i%8][0][k]*dir[(i+1)%8][0][k]
#                   + dir[i%8][1][k]*dir[(i+1)%8][1][k]
#                   + dir[i%8][2][k]*dir[(i+1)%8][2][k];
#             prod2 = dir[(i+1)%8][0][k]*dir[(i+2)%8][0][k]
#                   + dir[(i+1)%8][1][k]*dir[(i+2)%8][1][k]
#                   + dir[(i+1)%8][2][k]*dir[(i+2)%8][2][k];
#             weight[k] = weight[k]
#                       + acos((prod0-prod1*prod2)/(sqrt(1.-prod1*prod1)*sqrt(1.-prod2*prod2)));
#         }
#         weight[k] = (weight[k]-6.*M_PI)/(4.*M_PI);
#     }
#
#     v[0][0][3] =  v[0][0][0]; v[0][1][3] = v[0][1][0]; v[0][2][3] =  v[0][2][0];
#     v[1][0][3] =  v[0][0][0]; v[1][1][3] = v[0][1][0]; v[1][2][3] = -v[0][2][0];
#     v[2][0][3] =  v[1][0][1]; v[2][1][3] = v[1][1][1]; v[2][2][3] = -v[1][2][1];
#     v[3][0][3] =  v[1][0][1]; v[3][1][3] = v[1][1][1]; v[3][2][3] =  v[1][2][1];
#     v[0][0][4] =  v[0][0][1]; v[0][1][4] = v[0][1][1]; v[0][2][4] =  v[0][2][1];
#     v[1][0][4] = -v[0][0][1]; v[1][1][4] = v[0][1][1]; v[1][2][4] =  v[0][2][1];
#     v[2][0][4] = -v[1][0][2]; v[2][1][4] = v[1][1][2]; v[2][2][4] =  v[1][2][2];
#     v[3][0][4] =  v[1][0][2]; v[3][1][4] = v[1][1][2]; v[3][2][4] =  v[1][2][2];
#
#     v[0][0][5] = v[0][0][2]; v[0][1][5] =  v[0][1][2]; v[0][2][5] = v[0][2][2];
#     v[1][0][5] = v[0][0][2]; v[1][1][5] = -v[0][1][2]; v[1][2][5] = v[0][2][2];
#     v[2][0][5] = v[1][0][0]; v[2][1][5] = -v[1][1][0]; v[2][2][5] = v[1][2][0];
#     v[3][0][5] = v[1][0][0]; v[3][1][5] =  v[1][1][0]; v[3][2][5] = v[1][2][0];
#
#     for (k = 3; k <= 5; k++)
#         for (i = 0; i <= 3; i++)
#             for (j = 0; j <= 2; j++)
#                 dir[i][j][k] = v[i][j][k]/sqrt(pow(v[i][0][k],2)
#                              + pow(v[i][1][k],2)
#                              + pow(v[i][2][k],2));
#
#     for (k = 3; k <= 5; k++) {
#         for (i = 0; i <= 3; i++) {
#             prod0 = dir[i%4][0][k]*dir[(i+2)%4][0][k]
#                   + dir[i%4][1][k]*dir[(i+2)%4][1][k]
#                   + dir[i%4][2][k]*dir[(i+2)%4][2][k];
#             prod1 = dir[i%4][0][k]*dir[(i+1)%4][0][k]
#                   + dir[i%4][1][k]*dir[(i+1)%4][1][k]
#                   + dir[i%4][2][k]*dir[(i+1)%4][2][k];
#             prod2 = dir[(i+1)%4][0][k]*dir[(i+2)%4][0][k]
#                   + dir[(i+1)%4][1][k]*dir[(i+2)%4][1][k]
#                   + dir[(i+1)%4][2][k]*dir[(i+2)%4][2][k];
#             weight[k] = weight[k]
#                       + acos((prod0-prod1*prod2)/(sqrt(1-pow(prod1,2))*sqrt(1-pow(prod2,2))));
#         }
#         weight[k] = (weight[k]-2*M_PI)/(4*M_PI);
#     }
#
#     weight[6] = (1-2*(weight[0]+weight[1]+weight[2])-4*(weight[3]+weight[4]+weight[5]))/8;
#
# }
#
# // }}}
@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def weights(Delta: np.ndarray[np.float64], weight: np.ndarray[np.float64]):
    delta_xy = np.sqrt(Delta[0] * Delta[0] + Delta[1] * Delta[1])
    delta_yz = np.sqrt(Delta[1] * Delta[1] + Delta[2] * Delta[2])
    delta_zx = np.sqrt(Delta[2] * Delta[2] + Delta[0] * Delta[0])
    delta = np.sqrt(Delta[0] * Delta[0] + Delta[1] * Delta[1] + Delta[2] * Delta[2])
    v = np.empty((8, 3, 6), dtype=np.float64)
    dir = np.empty((8, 3, 6), dtype=np.float64)
    v[0, 0, 0] = 1
    v[0, 1, 0] = (delta_xy - Delta[0]) / Delta[1]
    v[0, 2, 0] = (delta - delta_xy) / Delta[2]
    v[1, 0, 0] = 1
    v[1, 1, 0] = (delta - delta_zx) / Delta[1]
    v[1, 2, 0] = (delta_zx - Delta[0]) / Delta[2]
    v[2, 0, 0] = v[1, 0, 0]
    v[2, 1, 0] = -v[1, 1, 0]
    v[2, 2, 0] = v[1, 2, 0]
    v[3, 0, 0] = v[0, 0, 0]
    v[3, 1, 0] = -v[0, 1, 0]
    v[3, 2, 0] = v[0, 2, 0]
    v[4, 0, 0] = v[0, 0, 0]
    v[4, 1, 0] = -v[0, 1, 0]
    v[4, 2, 0] = -v[0, 2, 0]
    v[5, 0, 0] = v[1, 0, 0]
    v[5, 1, 0] = -v[1, 1, 0]
    v[5, 2, 0] = -v[1, 2, 0]
    v[6, 0, 0] = v[1, 0, 0]
    v[6, 1, 0] = v[1, 1, 0]
    v[6, 2, 0] = -v[1, 2, 0]
    v[7, 0, 0] = v[0, 0, 0]
    v[7, 1, 0] = v[0, 1, 0]
    v[7, 2, 0] = -v[0, 2, 0]

    v[0, 0, 1] = (delta - delta_yz) / Delta[0]
    v[0, 1, 1] = 1
    v[0, 2, 1] = (delta_yz - Delta[1]) / Delta[2]
    v[1, 0, 1] = (delta_xy - Delta[1]) / Delta[0]
    v[1, 1, 1] = 1
    v[1, 2, 1] = (delta - delta_xy) / Delta[2]
    v[2, 0, 1] = v[1, 0, 1]
    v[2, 1, 1] = v[1, 1, 1]
    v[2, 2, 1] = -v[1, 2, 1]
    v[3, 0, 1] = v[0, 0, 1]
    v[3, 1, 1] = v[0, 1, 1]
    v[3, 2, 1] = -v[0, 2, 1]
    v[4, 0, 1] = -v[0, 0, 1]
    v[4, 1, 1] = v[0, 1, 1]
    v[4, 2, 1] = -v[0, 2, 1]
    v[5, 0, 1] = -v[1, 0, 1]
    v[5, 1, 1] = v[1, 1, 1]
    v[5, 2, 1] = -v[1, 2, 1]
    v[6, 0, 1] = -v[1, 0, 1]
    v[6, 1, 1] = v[1, 1, 1]
    v[6, 2, 1] = v[1, 2, 1]
    v[7, 0, 1] = -v[0, 0, 1]
    v[7, 1, 1] = v[0, 1, 1]
    v[7, 2, 1] = v[0, 2, 1]

    v[0, 0, 2] = (delta_zx - Delta[2]) / Delta[0]
    v[0, 1, 2] = (delta - delta_zx) / Delta[1]
    v[0, 2, 2] = 1
    v[1, 0, 2] = (delta - delta_yz) / Delta[0]
    v[1, 1, 2] = (delta_yz - Delta[2]) / Delta[1]
    v[1, 2, 2] = 1
    v[2, 0, 2] = -v[1, 0, 2]
    v[2, 1, 2] = v[1, 1, 2]
    v[2, 2, 2] = v[1, 2, 2]
    v[3, 0, 2] = -v[0, 0, 2]
    v[3, 1, 2] = v[0, 1, 2]
    v[3, 2, 2] = v[0, 2, 2]
    v[4, 0, 2] = -v[0, 0, 2]
    v[4, 1, 2] = -v[0, 1, 2]
    v[4, 2, 2] = v[0, 2, 2]
    v[5, 0, 2] = -v[1, 0, 2]
    v[5, 1, 2] = -v[1, 1, 2]
    v[5, 2, 2] = v[1, 2, 2]
    v[6, 0, 2] = v[1, 0, 2]
    v[6, 1, 2] = -v[1, 1, 2]
    v[6, 2, 2] = v[1, 2, 2]
    v[7, 0, 2] = v[0, 0, 2]
    v[7, 1, 2] = -v[0, 1, 2]
    v[7, 2, 2] = v[0, 2, 2]
    for k in range(0, 3):
        for i in range(0, 8):
            norm = np.sqrt(v[i, 0, k] ** 2 + v[i, 1, k] ** 2 + v[i, 2, k] ** 2)
            for j in range(0, 3):
                dir[i, j, k] = v[i, j, k] / norm

    #     for (k = 0; k <= 2; k++) {
    #         for (i = 0; i <= 7; i++) {
    #             prod0 = dir[i%8][0][k]*dir[(i+2)%8][0][k]
    #                   + dir[i%8][1][k]*dir[(i+2)%8][1][k]
    #                   + dir[i%8][2][k]*dir[(i+2)%8][2][k];
    #             prod1 = dir[i%8][0][k]*dir[(i+1)%8][0][k]
    #                   + dir[i%8][1][k]*dir[(i+1)%8][1][k]
    #                   + dir[i%8][2][k]*dir[(i+1)%8][2][k];
    #             prod2 = dir[(i+1)%8][0][k]*dir[(i+2)%8][0][k]
    #                   + dir[(i+1)%8][1][k]*dir[(i+2)%8][1][k]
    #                   + dir[(i+1)%8][2][k]*dir[(i+2)%8][2][k];
    #             weight[k] = weight[k]
    #                       + acos((prod0-prod1*prod2)/(sqrt(1.-prod1*prod1)*sqrt(1.-prod2*prod2)));
    #         }
    #         weight[k] = (weight[k]-6.*M_PI)/(4.*M_PI);
    #     }

    for k in range(0, 3):
        for i in range(0, 8):
            prod0 = dir[i % 8, 0, k] * dir[(i + 2) % 8, 0, k] + dir[i % 8, 1, k] * dir[(i + 2) % 8, 1, k] + dir[
                i % 8, 2, k] * dir[(i + 2) % 8, 2, k]
            prod1 = dir[i % 8, 0, k] * dir[(i + 1) % 8, 0, k] + dir[i % 8, 1, k] * dir[(i + 1) % 8, 1, k] + dir[
                i % 8, 2, k] * dir[(i + 1) % 8, 2, k]
            prod2 = dir[(i + 1) % 8, 0, k] * dir[(i + 2) % 8, 0, k] + dir[(i + 1) % 8, 1, k] * dir[(i + 2) % 8, 1, k] + \
                    dir[(i + 1) % 8, 2, k] * dir[(i + 2) % 8, 2, k]
            weight[k] += np.arccos(
                (prod0 - prod1 * prod2) / (np.sqrt(1. - prod1 * prod1) * np.sqrt(1. - prod2 * prod2)))
        weight[k] = (weight[k] - 6. * np.pi) / (4. * np.pi)

    v[0, 0, 3] = v[0, 0, 0]
    v[0, 1, 3] = v[0, 1, 0]
    v[0, 2, 3] = v[0, 2, 0]
    v[1, 0, 3] = v[0, 0, 0]
    v[1, 1, 3] = v[0, 1, 0]
    v[1, 2, 3] = -v[0, 2, 0]
    v[2, 0, 3] = v[1, 0, 1]
    v[2, 1, 3] = v[1, 1, 1]
    v[2, 2, 3] = -v[1, 2, 1]
    v[3, 0, 3] = v[1, 0, 1]
    v[3, 1, 3] = v[1, 1, 1]
    v[3, 2, 3] = v[1, 2, 1]
    v[0, 0, 4] = v[0, 0, 1]
    v[0, 1, 4] = v[0, 1, 1]
    v[0, 2, 4] = v[0, 2, 1]
    v[1, 0, 4] = -v[0, 0, 1]
    v[1, 1, 4] = v[0, 1, 1]
    v[1, 2, 4] = v[0, 2, 1]
    v[2, 0, 4] = -v[1, 0, 2]
    v[2, 1, 4] = v[1, 1, 2]
    v[2, 2, 4] = v[1, 2, 2]
    v[3, 0, 4] = v[1, 0, 2]
    v[3, 1, 4] = v[1, 1, 2]
    v[3, 2, 4] = v[1, 2, 2]

    v[0, 0, 5] = v[0, 0, 2]
    v[0, 1, 5] = v[0, 1, 2]
    v[0, 2, 5] = v[0, 2, 2]
    v[1, 0, 5] = v[0, 0, 2]
    v[1, 1, 5] = -v[0, 1, 2]
    v[1, 2, 5] = v[0, 2, 2]
    v[2, 0, 5] = v[1, 0, 0]
    v[2, 1, 5] = -v[1, 1, 0]
    v[2, 2, 5] = v[1, 2, 0]
    v[3, 0, 5] = v[1, 0, 0]
    v[3, 1, 5] = v[1, 1, 0]
    v[3, 2, 5] = v[1, 2, 0]

    #     v[0][0][3] =  v[0][0][0]; v[0][1][3] = v[0][1][0]; v[0][2][3] =  v[0][2][0];
    #     v[1][0][3] =  v[0][0][0]; v[1][1][3] = v[0][1][0]; v[1][2][3] = -v[0][2][0];
    #     v[2][0][3] =  v[1][0][1]; v[2][1][3] = v[1][1][1]; v[2][2][3] = -v[1][2][1];
    #     v[3][0][3] =  v[1][0][1]; v[3][1][3] = v[1][1][1]; v[3][2][3] =  v[1][2][1];
    #     v[0][0][4] =  v[0][0][1]; v[0][1][4] = v[0][1][1]; v[0][2][4] =  v[0][2][1];
    #     v[1][0][4] = -v[0][0][1]; v[1][1][4] = v[0][1][1]; v[1][2][4] =  v[0][2][1];
    #     v[2][0][4] = -v[1][0][2]; v[2][1][4] = v[1][1][2]; v[2][2][4] =  v[1][2][2];
    #     v[3][0][4] =  v[1][0][2]; v[3][1][4] = v[1][1][2]; v[3][2][4] =  v[1][2][2];
    #
    #     v[0][0][5] = v[0][0][2]; v[0][1][5] =  v[0][1][2]; v[0][2][5] = v[0][2][2];
    #     v[1][0][5] = v[0][0][2]; v[1][1][5] = -v[0][1][2]; v[1][2][5] = v[0][2][2];
    #     v[2][0][5] = v[1][0][0]; v[2][1][5] = -v[1][1][0]; v[2][2][5] = v[1][2][0];
    #     v[3][0][5] = v[1][0][0]; v[3][1][5] =  v[1][1][0]; v[3][2][5] = v[1][2][0];

    #     for (k = 3; k <= 5; k++)
    #         for (i = 0; i <= 3; i++)
    #             for (j = 0; j <= 2; j++)
    #                 dir[i][j][k] = v[i][j][k]/sqrt(pow(v[i][0][k],2)
    #                              + pow(v[i][1][k],2)
    #                              + pow(v[i][2][k],2));
    #
    #     for (k = 3; k <= 5; k++) {
    #         for (i = 0; i <= 3; i++) {
    #             prod0 = dir[i%4][0][k]*dir[(i+2)%4][0][k]
    #                   + dir[i%4][1][k]*dir[(i+2)%4][1][k]
    #                   + dir[i%4][2][k]*dir[(i+2)%4][2][k];
    #             prod1 = dir[i%4][0][k]*dir[(i+1)%4][0][k]
    #                   + dir[i%4][1][k]*dir[(i+1)%4][1][k]
    #                   + dir[i%4][2][k]*dir[(i+1)%4][2][k];
    #             prod2 = dir[(i+1)%4][0][k]*dir[(i+2)%4][0][k]
    #                   + dir[(i+1)%4][1][k]*dir[(i+2)%4][1][k]
    #                   + dir[(i+1)%4][2][k]*dir[(i+2)%4][2][k];
    #             weight[k] = weight[k]
    #                       + acos((prod0-prod1*prod2)/(sqrt(1-pow(prod1,2))*sqrt(1-pow(prod2,2))));
    #         }
    #         weight[k] = (weight[k]-2*M_PI)/(4*M_PI);
    #     }
    #
    #     weight[6] = (1-2*(weight[0]+weight[1]+weight[2])-4*(weight[3]+weight[4]+weight[5]))/8;
    #
    # }

    for k in range(3, 6):
        for i in range(0, 4):
            norm = np.sqrt(v[i, 0, k] ** 2 + v[i, 1, k] ** 2 + v[i, 2, k] ** 2)
            for j in range(0, 3):
                dir[i, j, k] = v[i, j, k] / norm

    for k in range(3, 6):
        for i in range(0, 4):
            prod0 = dir[i % 4, 0, k] * dir[(i + 2) % 4, 0, k] + dir[i % 4, 1, k] * dir[(i + 2) % 4, 1, k] + dir[
                i % 4, 2, k] * dir[(i + 2) % 4, 2, k]
            prod1 = dir[i % 4, 0, k] * dir[(i + 1) % 4, 0, k] + dir[i % 4, 1, k] * dir[(i + 1) % 4, 1, k] + dir[
                i % 4, 2, k] * dir[(i + 1) % 4, 2, k]
            prod2 = dir[(i + 1) % 4, 0, k] * dir[(i + 2) % 4, 0, k] + dir[(i + 1) % 4, 1, k] * dir[
                (i + 2) % 4, 1, k] + dir[(i + 1) % 4, 2, k] * dir[(i + 2) % 4, 2, k]
            weight[k] += np.arccos(
                (prod0 - prod1 * prod2) / (np.sqrt(1. - prod1 * prod1) * np.sqrt(1. - prod2 * prod2)))
        weight[k] = (weight[k] - 2. * np.pi) / (4. * np.pi)

    weight[6] = (1 - 2 * (weight[0] + weight[1] + weight[2]) - 4 * (weight[3] + weight[4] + weight[5])) / 8
