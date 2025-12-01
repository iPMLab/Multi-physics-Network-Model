import numba as nb
import numpy as np


def mk_functionals(
    image,
    resolution=None,
    norm: bool = False,
    return_area: bool = True,
    return_length: bool = True,
    return_euler4: bool = False,
    return_euler8: bool = True,
    return_volume: bool = True,
    return_surface: bool = True,
    return_curvature: bool = True,
    return_euler6: bool = False,
    return_euler26: bool = True,
):
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
    """
    ndim = image.ndim
    dtype_image = image.dtype
    assert ndim in (2, 3), "image must be 2D or 3D"
    assert np.issubdtype(dtype_image, bool) or np.issubdtype(dtype_image, np.integer), (
        "image must be bool or integer"
    )
    if resolution is None:
        resolution = np.ones(ndim, dtype=np.float64)
    else:
        resolution = np.reshape(resolution, -1).astype(np.float64, copy=False)
        assert resolution.size == ndim or resolution.size == 1, (
            "resolution must be of size 1 or ndim"
        )
        if resolution.size == 1:
            resolution = np.repeat(resolution, ndim)
    image = np.ascontiguousarray(image)
    if ndim == 2:
        area, length, euler4, euler8 = nb_functionals_2d(
            image,
            res0=resolution[0],
            res1=resolution[1],
            norm=norm,
            return_area=return_area,
            return_length=return_length,
            return_euler4=return_euler4,
            return_euler8=return_euler8,
        )
    elif ndim == 3:
        volume, surface, curvature, euler6, euler26 = nb_functionals_3d(
            image,
            res0=resolution[0],
            res1=resolution[1],
            res2=resolution[2],
            norm=norm,
            return_volume=return_volume,
            return_surface=return_surface,
            return_curvature=return_curvature,
            return_euler6=return_euler6,
            return_euler26=return_euler26,
        )
    else:
        raise ValueError("image must be 2D or 3D")
    res = []
    if ndim == 2:
        if return_area:
            res.append(area)
        if return_length:
            res.append(length)
        if return_euler4:
            res.append(euler4)
        if return_euler8:
            res.append(euler8)
    if ndim == 3:
        # 3D
        if return_volume:
            res.append(volume)
        if return_surface:
            res.append(surface)
        if return_curvature:
            res.append(curvature)
        if return_euler6:
            res.append(euler6)
        if return_euler26:
            res.append(euler26)

    return tuple(res) if len(res) > 1 else res[0]


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_functionals_2d(
    image: np.ndarray,
    res0: np.float64,
    res1: np.float64,
    norm: bool,
    return_area: bool,
    return_length: bool,
    return_euler4: bool,
    return_euler8: bool,
):
    dim0, dim1 = image.shape
    size_density = (dim0 - 1) * (dim1 - 1)
    if norm:
        size_total = image.size * res0 * res1
    h = quant_2d(image)
    if return_area:
        area = size_density * area_dens_2d(h) * res0 * res1
        if norm:
            area /= size_total
    else:
        area = np.float64(0.0)
    if return_length:
        length = size_density * leng_dens_2d(h, res0, res1) * res0 * res1
        if norm:
            length /= size_total
    else:
        length = np.float64(0.0)
    if return_euler4:
        euler4 = size_density * eul4_dens_2d(h)
        if norm:
            euler4 /= size_total
    else:
        euler4 = np.float64(0.0)
    if return_euler8:
        euler8 = size_density * eul8_dens_2d(h)
        if norm:
            euler8 /= size_total
    else:
        euler8 = np.float64(0.0)
    return area, length, euler4, euler8


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_functionals_3d(
    image: np.ndarray,
    res0: np.float64,
    res1: np.float64,
    res2: np.float64,
    norm: bool,
    return_volume: bool,
    return_surface: bool,
    return_curvature: bool,
    return_euler6: bool,
    return_euler26: bool,
):
    dim0, dim1, dim2 = image.shape
    size_density = (dim0 - 1) * (dim1 - 1) * (dim2 - 1)
    if norm:
        size_total = image.size * res0 * res1 * res2
    h = quant_3d(image)
    if return_volume:
        volume = size_density * volu_dens_3d(h) * res0 * res1 * res2
        if norm:
            volume /= size_total
    else:
        volume = np.float64(0.0)
    if return_surface:
        surface = size_density * surf_dens_3d(h, res0, res1, res2) * res0 * res1 * res2
        if norm:
            surface /= size_total
    else:
        surface = np.float64(0.0)
    if return_curvature:
        curvature = (
            size_density * curv_dens_3d(h, res0, res1, res2) * res0 * res1 * res2
        )
        if norm:
            curvature /= size_total
    else:
        curvature = np.float64(0.0)
    if return_euler6:
        euler6 = size_density * eul6_dens_3d(h)
        if norm:
            euler6 /= size_total
    else:
        euler6 = np.float64(0.0)
    if return_euler26:
        euler26 = size_density * eu26_dens_3d(h)
        if norm:
            euler26 /= size_total
    else:
        euler26 = np.float64(0.0)

    return volume, surface, curvature, euler6, euler26


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def quant_2d(image: np.ndarray) -> np.ndarray:
    dim0, dim1 = image.shape
    h = np.zeros(16, dtype=np.int64)
    for y in range(dim0 - 1):
        pz0 = image[y]
        pz1 = image[y + 1]
        mask = (pz1[0] << 1) | pz0[0]
        for x in range(1, dim1):
            mask |= ((pz1[x] << 1) | pz0[x]) << 2
            h[mask] += 1
            mask >>= 2
    return h


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def quant_3d(image: np.ndarray) -> np.ndarray:
    dim0, dim1, dim2 = image.shape
    h = np.zeros(256, dtype=np.int64)
    for z in range(dim0 - 1):
        for y in range(dim1 - 1):
            pz0y0 = image[z, y]
            pz0y1 = image[z, y + 1]
            pz1y0 = image[z + 1, y]
            pz1y1 = image[z + 1, y + 1]

            # 初始化低四位掩码
            mask = pz0y0[0] | (pz1y0[0] << 1) | (pz0y1[0] << 2) | (pz1y1[0] << 3)
            for x in range(1, dim2):
                # 预加载当前z层的四个相邻点
                mask |= (
                    (pz0y0[x] << 4)
                    | (pz1y0[x] << 5)
                    | (pz0y1[x] << 6)
                    | (pz1y1[x] << 7)
                )
                h[mask] += 1
                mask >>= 4
    return h


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def area_dens_2d(h: np.ndarray) -> np.float64:
    iChi = np.sum(h[1::2])
    iVol = np.sum(h)
    if iVol == 0:
        return np.float64(0.0)
    else:
        return np.float64(iChi) / np.float64(iVol)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def leng_dens_2d(h: np.ndarray, res0: np.float64, res1: np.float64) -> np.float64:
    wi = np.empty(4, dtype=np.float64)
    r = np.empty(4, dtype=np.float64)
    kl = leng_dens_2d_kl

    r[0] = res0
    r[1] = res1
    r[2] = r[3] = np.sqrt(res0**2 + res1**2)

    wi[0] = np.arctan(res0 / res1) / np.pi
    wi[1] = np.arctan(res1 / res0) / np.pi
    wi[2] = wi[3] = (1 - wi[0] - wi[1]) / 2
    Ls = Ls_16
    iVol = np.sum(h)
    LI = np.sum(wi * r)
    II = np.float64(0.0)
    for i in range(4):
        kl_0, kl_1 = kl[i]
        II += wi[i] * np.sum(
            h
            * (
                ((Ls == Ls | kl_0) & (0 == Ls & kl_1))
                | ((Ls == Ls | kl_1) & (0 == Ls & kl_0))
            )
        )
    if LI == 0.0:
        return np.float64(0.0)
    else:
        return 0.25 * II / LI / iVol


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def eul4_dens_2d(h: np.ndarray) -> np.float64:
    iu = eul4_dens_2d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    return np.float64(iChi) / np.float64(iVol) / np.pi


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def eul8_dens_2d(h: np.ndarray) -> np.float64:
    iu = eul8_dens_2d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)
    return np.float64(iChi) / np.float64(iVol) / (12 * np.pi)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def volu_dens_3d(h: np.ndarray) -> np.float64:
    iChi = np.sum(h[1::2])
    iVol = np.sum(h)
    return np.float64(iChi) / np.float64(iVol) if iVol != 0 else np.float64(0.0)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def surf_dens_3d(
    h: np.ndarray, res0: np.float64, res1: np.float64, res2: np.float64
) -> np.float64:
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
    wi[3] = wi[4] = weight[3]
    wi[5] = wi[6] = weight[5]
    wi[7] = wi[8] = weight[4]
    wi[9] = wi[10] = wi[11] = wi[12] = weight[6]

    iVol = np.sum(h)
    Sv = np.float64(0.0)
    Ls = Ls_256
    Lv = np.sum(wi * r)
    for i in range(13):
        kl_0, kl_1 = kl[i]
        Sv += wi[i] * np.sum(
            h
            * (
                ((Ls == Ls | kl_0) & (0 == Ls & kl_1))
                | ((Ls == Ls | kl_1) & (0 == Ls & kl_0))
            )
        )

    return 0.25 * Sv / Lv / iVol if iVol != 0 else np.float64(0.0)


@nb.njit(parallel=False, cache=True, fastmath=False, nogil=True, error_model="numpy")
def curv_dens_3d(h: np.ndarray, res0: float, res1: float, res2: float) -> float:
    kr = curv_dens_3d_kr  # (9, 4)
    kt = curv_dens_3d_kt  # (8, 3)

    Delta = np.empty(3, dtype=np.float64)
    weight = np.zeros(7, dtype=np.float64)
    a = np.empty(13, dtype=np.float64)
    wi = np.empty(13, dtype=np.float64)

    Delta[0] = res0
    Delta[1] = res1
    Delta[2] = res2

    r01 = np.sqrt(res0 * res0 + res1 * res1)
    r02 = np.sqrt(res0 * res0 + res2 * res2)
    r12 = np.sqrt(res1 * res1 + res2 * res2)
    # --- Compute area factors a[0..12] ---
    a[0] = res0 * res1
    a[1] = res0 * res2
    a[2] = res1 * res2
    a[3] = a[4] = res2 * r01
    a[5] = a[6] = res1 * r02
    a[7] = a[8] = res0 * r12

    s = 0.5 * (r01 + r02 + r12)
    atr = np.sqrt(s * (s - r01) * (s - r02) * (s - r12))
    a[9] = a[10] = a[11] = a[12] = 2.0 * atr
    # --- Compute weights ---
    weights(Delta, weight)
    wi[0] = weight[0]
    wi[1] = weight[1]
    wi[2] = weight[2]

    wi[3] = wi[4] = weight[3]
    wi[5] = wi[6] = weight[5]
    wi[7] = wi[8] = weight[4]
    wi[9] = wi[10] = wi[11] = wi[12] = weight[6]

    iVol = np.sum(h)

    Ls = Ls_256
    Mc = np.float64(0)
    count = np.empty(256, dtype=np.int64)
    for i in range(9):
        count.fill(0)
        kr_i = kr[i]

        # k = 0
        main_mask = ((Ls | kr_i[0]) == Ls) & (Ls & kr_i[3] == 0)
        add_mask = main_mask & (Ls & kr_i[1] == 0) & (Ls & kr_i[2] == 0)
        sub_mask = main_mask & ((Ls | kr_i[1]) == Ls) & ((Ls | kr_i[2]) == Ls)
        count += add_mask
        count -= sub_mask

        # k = 1
        main_mask = ((Ls | kr_i[1]) == Ls) & (Ls & kr_i[0] == 0)
        add_mask = main_mask & (Ls & kr_i[2] == 0) & (Ls & kr_i[3] == 0)
        sub_mask = main_mask & ((Ls | kr_i[2]) == Ls) & ((Ls | kr_i[3]) == Ls)
        count += add_mask
        count -= sub_mask

        # k = 2
        main_mask = ((Ls | kr_i[2]) == Ls) & (Ls & kr_i[1] == 0)
        add_mask = main_mask & (Ls & kr_i[3] == 0) & (Ls & kr_i[0] == 0)
        sub_mask = main_mask & ((Ls | kr_i[3]) == Ls) & ((Ls | kr_i[0]) == Ls)
        count += add_mask
        count -= sub_mask

        # k = 3
        main_mask = ((Ls | kr_i[3]) == Ls) & (Ls & kr_i[2] == 0)
        add_mask = main_mask & (Ls & kr_i[0] == 0) & (Ls & kr_i[1] == 0)
        sub_mask = main_mask & ((Ls | kr_i[0]) == Ls) & ((Ls | kr_i[1]) == Ls)
        count += add_mask
        count -= sub_mask

        Mc += wi[i] / (4 * a[i]) * np.sum(h * count)

    for i in range(9, 13):
        count.fill(0)
        kt_i0 = kt[i - 9]  # shape: (3,)
        kt_i1 = kt[i - 5]

        # -----------------------
        # k = 0: indices (0,1,2)
        # -----------------------
        add_mask = (
            ((Ls | kt_i0[0]) == Ls) & ((Ls & kt_i0[1]) == 0) & ((Ls & kt_i0[2]) == 0)
        )
        sub_mask = (
            ((Ls | kt_i1[0]) == Ls) & ((Ls | kt_i1[1]) == Ls) & ((Ls & kt_i1[2]) == 0)
        )
        count += add_mask
        count -= sub_mask

        # -----------------------
        # k = 1: indices (1,2,0)
        # -----------------------
        add_mask = (
            ((Ls | kt_i0[1]) == Ls) & ((Ls & kt_i0[2]) == 0) & ((Ls & kt_i0[0]) == 0)
        )
        sub_mask = (
            ((Ls | kt_i1[1]) == Ls) & ((Ls | kt_i1[2]) == Ls) & ((Ls & kt_i1[0]) == 0)
        )
        count += add_mask
        count -= sub_mask

        # -----------------------
        # k = 2: indices (2,0,1)
        # -----------------------
        add_mask = (
            ((Ls | kt_i0[2]) == Ls) & ((Ls & kt_i0[0]) == 0) & ((Ls & kt_i0[1]) == 0)
        )
        sub_mask = (
            ((Ls | kt_i1[2]) == Ls) & ((Ls | kt_i1[0]) == Ls) & ((Ls & kt_i1[1]) == 0)
        )
        count += add_mask
        count -= sub_mask

        # 累加到全局结果
        Mc += wi[i] / (3 * a[i]) * np.sum(h * count)

    return 2.0 / np.pi * Mc / iVol if iVol != 0 else np.float64(0.0)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def eul6_dens_3d(h: np.ndarray) -> np.float64:
    iu = eul6_dens_3d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)

    return 3.0 / (4.0 * np.pi) * iChi / iVol if iVol != 0 else np.float64(0.0)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def eu26_dens_3d(h: np.ndarray) -> np.float64:
    iu = eu26_dens_3d_iu
    iChi = np.sum(iu * h)
    iVol = np.sum(h)

    return 1.0 / (32.0 * np.pi) * iChi / iVol if iVol != 0 else np.float64(0.0)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def weights(Delta: np.ndarray, weight: np.ndarray):
    delta_xy = np.sqrt(Delta[0] * Delta[0] + Delta[1] * Delta[1])
    delta_yz = np.sqrt(Delta[1] * Delta[1] + Delta[2] * Delta[2])
    delta_zx = np.sqrt(Delta[2] * Delta[2] + Delta[0] * Delta[0])
    delta = np.sqrt(Delta[0] * Delta[0] + Delta[1] * Delta[1] + Delta[2] * Delta[2])
    v = np.empty((8, 3, 6), dtype=np.float64)
    dirs = np.empty((8, 3, 6), dtype=np.float64)
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
    for k in range(0, 3):
        for i in range(0, 8):
            norm = np.sqrt(v[i, 0, k] ** 2 + v[i, 1, k] ** 2 + v[i, 2, k] ** 2)
            for j in range(0, 3):
                dirs[i, j, k] = v[i, j, k] / norm

    for k in range(0, 3):
        for i in range(0, 8):
            prod0 = (
                dirs[i % 8, 0, k] * dirs[(i + 2) % 8, 0, k]
                + dirs[i % 8, 1, k] * dirs[(i + 2) % 8, 1, k]
                + dirs[i % 8, 2, k] * dirs[(i + 2) % 8, 2, k]
            )
            prod1 = (
                dirs[i % 8, 0, k] * dirs[(i + 1) % 8, 0, k]
                + dirs[i % 8, 1, k] * dirs[(i + 1) % 8, 1, k]
                + dirs[i % 8, 2, k] * dirs[(i + 1) % 8, 2, k]
            )
            prod2 = (
                dirs[(i + 1) % 8, 0, k] * dirs[(i + 2) % 8, 0, k]
                + dirs[(i + 1) % 8, 1, k] * dirs[(i + 2) % 8, 1, k]
                + dirs[(i + 1) % 8, 2, k] * dirs[(i + 2) % 8, 2, k]
            )
            weight[k] += np.arccos(
                (prod0 - prod1 * prod2)
                / (np.sqrt(1.0 - prod1 * prod1) * np.sqrt(1.0 - prod2 * prod2))
            )
        weight[k] = (weight[k] - 6.0 * np.pi) / (4.0 * np.pi)

    for k in range(3, 6):
        for i in range(0, 4):
            norm = np.sqrt(v[i, 0, k] ** 2 + v[i, 1, k] ** 2 + v[i, 2, k] ** 2)
            for j in range(0, 3):
                dirs[i, j, k] = v[i, j, k] / norm

    for k in range(3, 6):
        for i in range(0, 4):
            prod0 = (
                dirs[i % 4, 0, k] * dirs[(i + 2) % 4, 0, k]
                + dirs[i % 4, 1, k] * dirs[(i + 2) % 4, 1, k]
                + dirs[i % 4, 2, k] * dirs[(i + 2) % 4, 2, k]
            )
            prod1 = (
                dirs[i % 4, 0, k] * dirs[(i + 1) % 4, 0, k]
                + dirs[i % 4, 1, k] * dirs[(i + 1) % 4, 1, k]
                + dirs[i % 4, 2, k] * dirs[(i + 1) % 4, 2, k]
            )
            prod2 = (
                dirs[(i + 1) % 4, 0, k] * dirs[(i + 2) % 4, 0, k]
                + dirs[(i + 1) % 4, 1, k] * dirs[(i + 2) % 4, 1, k]
                + dirs[(i + 1) % 4, 2, k] * dirs[(i + 2) % 4, 2, k]
            )
            weight[k] += np.arccos(
                (prod0 - prod1 * prod2)
                / (np.sqrt(1.0 - prod1 * prod1) * np.sqrt(1.0 - prod2 * prod2))
            )
        weight[k] = (weight[k] - 2.0 * np.pi) / (4.0 * np.pi)

    weight[6] = (
        1
        - 2 * (weight[0] + weight[1] + weight[2])
        - 4 * (weight[3] + weight[4] + weight[5])
    ) / 8


#### CONSTANTS ####
eul8_dens_2d_iu = np.array(
    (0, 3, 3, 0, 3, 0, 6, -3, 3, 6, 0, -3, 0, -3, -3, 0), dtype=np.int32
)
eul4_dens_2d_iu = np.array(
    (0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0), dtype=np.int32
)
leng_dens_2d_kl = np.array(((1, 2), (1, 4), (1, 8), (2, 4)), dtype=np.int32)
surf_dens_3d_kl = np.array(
    (
        (1, 2),
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
        (8, 16),
    ),
    dtype=np.int32,
)
curv_dens_3d_kr = np.array(
    (
        (1, 2, 4, 8),
        (1, 2, 16, 32),
        (1, 4, 16, 64),
        (1, 2, 64, 128),
        (4, 16, 8, 32),
        (1, 32, 4, 128),
        (2, 8, 16, 64),
        (2, 4, 32, 64),
        (1, 16, 8, 128),
    ),
    dtype=np.int32,
)

curv_dens_3d_kt = np.array(
    (
        (1, 64, 32),
        (2, 16, 128),
        (8, 64, 32),
        (4, 16, 128),
        (2, 4, 128),
        (8, 1, 64),
        (2, 4, 16),
        (8, 1, 32),
    ),
    dtype=np.int32,
)
eul6_dens_3d_iu = np.array(
    (
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        -1,
        0,
        -2,
        0,
        0,
        0,
        -1,
        0,
        -1,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        -1,
        0,
        -2,
        0,
        0,
        0,
        -1,
        0,
        -1,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        -1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ),
    dtype=np.int32,
)

eu26_dens_3d_iu = np.array(
    (
        0,
        3,
        3,
        0,
        3,
        0,
        6,
        -3,
        3,
        6,
        0,
        -3,
        0,
        -3,
        -3,
        0,
        3,
        0,
        6,
        -3,
        6,
        -3,
        9,
        -6,
        6,
        3,
        3,
        -6,
        3,
        -6,
        0,
        -3,
        3,
        6,
        0,
        -3,
        6,
        3,
        3,
        -6,
        6,
        9,
        -3,
        -6,
        3,
        0,
        -6,
        -3,
        0,
        -3,
        -3,
        0,
        3,
        -6,
        0,
        -3,
        3,
        0,
        -6,
        -3,
        0,
        -8,
        -8,
        0,
        3,
        6,
        6,
        3,
        0,
        -3,
        3,
        -6,
        6,
        9,
        3,
        0,
        -3,
        -6,
        -6,
        -3,
        0,
        -3,
        3,
        -6,
        -3,
        0,
        0,
        -3,
        3,
        0,
        0,
        -8,
        -6,
        -3,
        -8,
        0,
        6,
        9,
        3,
        0,
        3,
        0,
        0,
        -8,
        9,
        12,
        0,
        -3,
        0,
        -3,
        -8,
        -6,
        -3,
        -6,
        -6,
        -3,
        -6,
        -3,
        -8,
        0,
        0,
        -3,
        -8,
        -6,
        -8,
        -6,
        -12,
        3,
        3,
        6,
        6,
        3,
        6,
        3,
        9,
        0,
        0,
        3,
        -3,
        -6,
        -3,
        -6,
        -6,
        -3,
        6,
        3,
        9,
        0,
        9,
        0,
        12,
        -3,
        3,
        0,
        0,
        -8,
        0,
        -8,
        -3,
        -6,
        0,
        3,
        -3,
        -6,
        3,
        0,
        0,
        -8,
        -3,
        0,
        0,
        -3,
        -6,
        -8,
        -3,
        0,
        -3,
        -6,
        -6,
        -3,
        0,
        -8,
        -3,
        -6,
        -6,
        -8,
        -3,
        0,
        -8,
        -12,
        -6,
        3,
        0,
        3,
        3,
        0,
        -3,
        -6,
        0,
        -8,
        -3,
        0,
        -6,
        -8,
        0,
        -3,
        -3,
        0,
        -3,
        -6,
        0,
        -8,
        -6,
        -3,
        -3,
        -6,
        -6,
        -8,
        -8,
        -12,
        -3,
        0,
        -6,
        3,
        -3,
        0,
        -6,
        -8,
        -6,
        -8,
        -8,
        -12,
        -6,
        -3,
        -3,
        -6,
        -3,
        -6,
        0,
        3,
        0,
        -3,
        -3,
        0,
        -3,
        0,
        -6,
        3,
        -3,
        -6,
        0,
        3,
        0,
        3,
        3,
        0,
    ),
    dtype=np.int32,
)
Ls_16 = np.arange(16, dtype=np.int32)
Ls_256 = np.arange(256, dtype=np.int32)
####CONSTANTS END####
