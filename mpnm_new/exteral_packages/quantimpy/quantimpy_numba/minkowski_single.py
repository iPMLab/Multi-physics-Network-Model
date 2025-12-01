import os
os.environ["NUMBA_OPT"] = "max"
os.environ["NUMBA_SLP_VECTORIZE"] = "1"
os.environ["NUMBA_ENABLE_AVX"] = "1"
os.environ["NUMBA_FUNCTION_CACHE_SIZE"] = "1024"
import numba as nb
import numpy as np
# nb.config.reload_config()
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

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eul4_dens_2d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64) -> np.float64:
    iu = eul4_dens_2d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    return np.float64(iChi) / (np.float64(iVol) * res0 * res1) / np.pi

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eul8_dens_2d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64) -> np.float64:
    iu = eul8_dens_2d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    return np.float64(iChi) / (np.float64(iVol) * res0 * res1) / (12 * np.pi)

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

@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eul6_dens_3d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64, res2: np.float64) -> np.float64:
    iu = eul6_dens_3d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    if iVol == 0:
        return np.float64(0.)
    else:
        return 3.0 / (4.0 * np.pi) * np.float64(iChi) / (np.float64(iVol) * res0 * res1 * res2)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def eu26_dens_3d(h: np.ndarray[np.int64], res0: np.float64, res1: np.float64, res2: np.float64) -> np.float64:
    iu = eu26_dens_3d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)

    if iVol == 0:
        return np.float64(0.)
    else:
        return 1.0 / (32.0 * np.pi) * np.float64(iChi) / (np.float64(iVol) / res0 * res1 * res2)


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




#### CONSTANTS ####
eul8_dens_2d_iu = np.array((0, 3, 3, 0, 3, 0, 6, -3, 3, 6, 0, -3, 0, -3, -3, 0), dtype=np.int64)
eul4_dens_2d_iu = np.array((0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0), dtype=np.int64)
leng_dens_2d_kl = np.array(((1, 2),
                            (1, 4),
                            (1, 8),
                            (2, 4)), dtype=np.int64)
surf_dens_3d_kl = np.array(((1, 2),
                            (1, 4),
                            (1, 16),
                            (1, 8),
                            (2, 4),
                            (1, 32),
                            (2, 16),
                            (1, 64),
                            (4, 16),
                            (1, 128),
                            (2, 64),
                            (4, 32),
                            (8, 16)), dtype=np.int64)
curv_dens_3d_kr = np.array(((1, 2, 4, 8),
                            (1, 2, 16, 32),
                            (1, 4, 16, 64),
                            (1, 2, 64, 128),
                            (4, 16, 8, 32),
                            (1, 32, 4, 128),
                            (2, 8, 16, 64),
                            (2, 4, 32, 64),
                            (1, 16, 8, 128)), dtype=np.int64)

curv_dens_3d_kt = np.array(((1, 64, 32),
                            (2, 16, 128),
                            (8, 64, 32),
                            (4, 16, 128),
                            (2, 4, 128),
                            (8, 1, 64),
                            (2, 4, 16),
                            (8, 1, 32)), dtype=np.int64)
eul6_dens_3d_iu = np.array((0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, -1, 0, -1, 0, -2, 0, 0, 0, -1, 0, -1, 0, -1,
                            0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0,
                            0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                            0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, -1, 0, -1, 0, -2, 0, 0, 0, -1, 0, -1, 0, -1,
                            0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0,
                            0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,), dtype=np.int64)

eu26_dens_3d_iu = np.array((
    0, 3, 3, 0, 3, 0, 6, -3, 3, 6, 0, -3, 0, -3, -3, 0,
    3, 0, 6, -3, 6, -3, 9, -6, 6, 3, 3, -6, 3, -6, 0, -3,
    3, 6, 0, -3, 6, 3, 3, -6, 6, 9, -3, -6, 3, 0, -6, -3,
    0, -3, -3, 0, 3, -6, 0, -3, 3, 0, -6, -3, 0, -8, -8, 0,
    3, 6, 6, 3, 0, -3, 3, -6, 6, 9, 3, 0, -3, -6, -6, -3,
    0, -3, 3, -6, -3, 0, 0, -3, 3, 0, 0, -8, -6, -3, -8, 0,
    6, 9, 3, 0, 3, 0, 0, -8, 9, 12, 0, -3, 0, -3, -8, -6,
    -3, -6, -6, -3, -6, -3, -8, 0, 0, -3, -8, -6, -8, -6, -12, 3,
    3, 6, 6, 3, 6, 3, 9, 0, 0, 3, -3, -6, -3, -6, -6, -3,
    6, 3, 9, 0, 9, 0, 12, -3, 3, 0, 0, -8, 0, -8, -3, -6,
    0, 3, -3, -6, 3, 0, 0, -8, -3, 0, 0, -3, -6, -8, -3, 0,
    -3, -6, -6, -3, 0, -8, -3, -6, -6, -8, -3, 0, -8, -12, -6, 3,
    0, 3, 3, 0, -3, -6, 0, -8, -3, 0, -6, -8, 0, -3, -3, 0,
    -3, -6, 0, -8, -6, -3, -3, -6, -6, -8, -8, -12, -3, 0, -6, 3,
    -3, 0, -6, -8, -6, -8, -8, -12, -6, -3, -3, -6, -3, -6, 0, 3,
    0, -3, -3, 0, -3, 0, -6, 3, -3, -6, 0, 3, 0, 3, 3, 0), dtype=np.int64)
####CONSTANTS END####


def mk_functionals_2d(
        image:np.ndarray[bool],
        resolution = None,
        norm:bool=False,
        return_area:bool=True,
        return_length:bool=True,
        return_euler8:bool=True,
        ):
    
    '''
    r"""Compute the Minkowski functionals in 2D or 3D.
    This function computes the Minkowski functionals for the Numpy array `image`. Both
    2D and 3D arrays are supported. Optionally, the (anisotropic) resolution of the
    array can be provided using the Numpy array `res`. When a resolution array is
    provided it needs to be of the same dimension as the image array.

    Parameters
    ----------
    image : ndarray, bool
        Image can be either a 2D or 3D array of data type `bool`.
    res : ndarray, {int, float}, optional
        By default the resolution is assumed to be 1 <unit of length>/pixel in all directions.
        If a resolution is provided it needs to be of the same dimension as the
        image array.
    norm : bool, defaults to False
        When norm=True the functionals are normalized with the total area or
        volume of the image. Defaults to norm=False.

    Returns
    -------
    out : tuple, float
        In the case of a 2D image this function returns a tuple of the area,
        length, and the Euler characteristic. In the case of a 3D image this
        function returns a tuple of the volume, surface, curvature, and the
        Euler characteristic. The return data type is `float`.

    See Also
    --------
    ~quantimpy.minkowski.functions_open
    ~quantimpy.minkowski.functions_close

    Notes
    -----

    The definition of the Minkowski functionals follows the convention in the
    physics literature [3]_.

    Considering a 2D body, :math:`X`, with a smooth boundary, :math:`\delta X`,
    the following functionals are computed:

    .. math:: M_{0} (X) &= \int_{X} d s, \\
              M_{1} (X) &= \frac{1}{2 \pi} \int_{\delta X} d c, \text{ and } \\
              M_{2} (X) &= \frac{1}{2 \pi^{2}} \int_{\delta X} \left[\frac{1}{R} \right] d c,

    where :math:`d s` is a surface element and :math:`d c` is a circumference
    element. :math:`R` is the radius of the local curvature. This results in the
    following definitions for the surface area, :math:`S = M_{0} (X)`,
    circumference, :math:`C = 2 \pi M_{1} (X)`, and the 2D Euler characteristic,
    :math:`\chi (X) = \pi M_{2} (X)`.

    Considering a 3D body, :math:`X`, with a smooth boundary surface, :math:`\delta
    X`, the following functionals are computed:

    .. math:: M_{0} (X) &= V = \int_{X} d v, \\
              M_{1} (X) &= \frac{1}{8} \int_{\delta X} d s, \\
              M_{2} (X) &= \frac{1}{2 \pi^{2}} \int_{\delta X}  \frac{1}{2} \left[\frac{1}{R_{1}} + \frac{1}{R_{2}}\right] d s, \text{ and } \\
              M_{3} (X) &= \frac{3}{(4 \pi)^{2}} \int_{\delta X} \left[\frac{1}{R_{1} R_{2}}\right] d s,

    where :math:`d v` is a volume element and :math:`d s` is a surface element.
    :math:`R_{1}` and :math:`R_{2}` are the principal radii of curvature of
    surface element :math:`d s`. This results in the following definitions for
    the volume, :math:`V = M_{0} (X)`, surface area, :math:`S = 8 M_{1} (X)`,
    integral mean curvature, :math:`H = 2 \pi^{2} M_{2} (X)`, and the 3D Euler
    characteristic, :math:`\chi (X) = 4 \pi/3 M_{3} (X)`.
    '''
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

    minkowski0 = mk_functionals_3d(image0, res0,return_surface=True,return_curvature=False,return_euler26=False,return_volume=False)
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