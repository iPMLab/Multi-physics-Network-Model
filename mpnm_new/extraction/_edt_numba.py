import numba as nb
import numpy as np

INF_FLOAT32 = np.finfo(np.float32).max


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_squared_edt_1d(f, w, apply_envelope):
    n = f.size
    if n == 0:
        return
    w2 = w * w
    # Extract the 1D slice: f[0], f[stride], f[2*stride], ..., f[(n-1)*stride]
    ff = f.copy()
    # v: indices of parabolas in lower envelope
    v = np.empty(n, dtype=np.int32)
    # ranges: boundaries where each parabola dominates
    ranges = np.empty(n + 1, dtype=np.float32)
    k = 0
    v[0] = 0
    ranges[0] = -INF_FLOAT32
    ranges[1] = INF_FLOAT32

    # First pass: construct lower envelope
    for i in range(1, n):
        # Compute intersection between parabola i and v[k]
        while True:
            vk = v[k]
            diff = i - vk
            s = (ff[i] - ff[vk] + w2 * diff * (i + vk)) / (2.0 * w2 * diff)
            # Pop if intersection is before current boundary
            if k > 0 and s <= ranges[k]:
                k -= 1
            else:
                break

        k += 1
        v[k] = i
        ranges[k] = s
        ranges[k + 1] = INF_FLOAT32

    k = 0
    # Optional: clamp to distance to image border (prevents bleed beyond edges)
    if apply_envelope:
        for i in range(n):
            while ranges[k + 1] < i:
                k += 1
            # Evaluate parabola centered at v[k]
            vk = v[k]
            dist_sq = w2 * ((i - vk) ** 2)
            envelope = min(w * (i + 1), w * (n - i)) ** 2
            f[i] = min(envelope, f[i], dist_sq + ff[vk])
    # Second pass: evaluate the lower envelope at each point
    else:
        for i in range(n):
            while ranges[k + 1] < i:
                k += 1
            # Evaluate parabola centered at v[k]
            vk = v[k]
            dist_sq = w2 * ((i - vk) ** 2)
            f[i] = dist_sq + ff[vk]


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_edt(binary_img, wx=1.0, wy=1.0, wz=1.0, black_border=False):
    """
    Compute Euclidean Distance Transform for 2D or 3D binary image.

    Parameters
    ----------
    binary_img : (H, W) or (D, H, W) bool array
        True = foreground (object), False = background.
        Distance is computed to the nearest foreground pixel.
    wx, wy, wz : float
        Anisotropy weights for x, y, (and z) axes.
        For 2D, wz is ignored.
    black_border : bool
        If True, assume background outside image (standard EDT behavior).
        This is equivalent to padding with background.

    Returns
    -------
    dist : float array of same shape as input
        Euclidean distance transform.
    """
    assert binary_img.dtype == np.dtype("bool"), "Input must be boolean array"
    dist_sq = binary_img * INF_FLOAT32
    if binary_img.ndim == 2:
        sx, sy = binary_img.shape
        # X-pass (along axis 0: columns)
        for y in nb.prange(sy):
            nb_squared_edt_1d(dist_sq[:, y], wx, black_border)
        # Y-pass (along axis 1: rows)
        for x in nb.prange(sx):
            nb_squared_edt_1d(dist_sq[x, :], wy, black_border)
        dist_sq = np.sqrt(dist_sq)
        if dist_sq[0, 0] == INF_FLOAT32:
            dist_sq = np.full_like(dist_sq, np.inf)
        return dist_sq

    elif binary_img.ndim == 3:
        sz, sx, sy = binary_img.shape
        # X-pass: along axis 1 (columns in each y-z plane)
        for z in nb.prange(sz):
            for y in nb.prange(sy):
                nb_squared_edt_1d(dist_sq[z, :, y], wx, black_border)
        # Y-pass: along axis 2 (rows in each x-z plane)
        for z in nb.prange(sz):
            for x in nb.prange(sx):
                nb_squared_edt_1d(dist_sq[z, x, :], wy, black_border)
        # Z-pass: along axis 0 (depth)
        for x in nb.prange(sx):
            for y in nb.prange(sy):
                nb_squared_edt_1d(dist_sq[:, x, y], wz, black_border)
        dist_sq = np.sqrt(dist_sq)
        if dist_sq[0, 0, 0] == INF_FLOAT32:
            dist_sq = np.full_like(dist_sq, np.inf)
        return dist_sq

    else:
        raise ValueError("Only 2D and 3D binary images are supported.")
