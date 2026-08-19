import numba as nb
import numpy as np

# float32 最大值，用于 tofinite / toinfinite 哨兵替换
_FLOAT32_MAX = np.float32(np.finfo(np.float32).max)


# ──────────────────────────────────────────────
# Pass-1 用的 Rosenfeld-Pfaltz 1D 距离变换
# 对应 C++ squared_edt_1d_multi_seg
# ──────────────────────────────────────────────
@nb.njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _squared_edt_1d_multi_seg(segids, d, offset, n, stride, anisotropy, black_border):
    """
    segids : 展平后的 bool 数组（原始二值图）
    d      : 展平后的 float32 数组（输出，就地写入）
    offset : 在展平数组中的起始下标
    n      : 元素个数
    stride : 相邻元素在展平数组中的步长
    """
    if n == 0:
        return

    working_segid = segids[offset]

    # ---- 初始化第一个元素 ----
    if black_border:
        d[offset] = np.float32(anisotropy) if working_segid else np.float32(0.0)
    else:
        d[offset] = np.float32(0.0) if not working_segid else np.float32(np.inf)

    # ---- 前向扫描 ----
    for i in range(1, n):
        idx = offset + i * stride
        prev_idx = offset + (i - 1) * stride
        cur = segids[idx]
        if not cur:  # 背景 → 距离 0
            d[idx] = np.float32(0.0)
        elif cur == working_segid:  # 同一段 → 累加
            d[idx] = d[prev_idx] + np.float32(anisotropy)
        else:  # 新段开始
            d[idx] = np.float32(anisotropy)
            d[prev_idx] = (
                np.float32(anisotropy) if segids[prev_idx] else np.float32(0.0)
            )
            working_segid = cur

    # ---- 后向扫描 ----
    min_bound = 0
    if black_border:
        last_idx = offset + (n - 1) * stride
        d[last_idx] = np.float32(anisotropy) if segids[last_idx] else np.float32(0.0)
        min_bound = 1

    for i in range(n - 2, min_bound - 1, -1):
        idx = offset + i * stride
        next_idx = offset + (i + 1) * stride
        val = d[next_idx] + np.float32(anisotropy)
        if val < d[idx]:
            d[idx] = val

    # ---- 平方 ----
    for i in range(n):
        idx = offset + i * stride
        d[idx] *= d[idx]


# ──────────────────────────────────────────────
# Pass-2/3 用的 Felzenszwalb-Huttenlocher 抛物线法
# 对应 C++ squared_edt_1d_parabolic（合并了左右 envelope 两个重载）
# ──────────────────────────────────────────────
@nb.njit(cache=True, fastmath=True, nogil=True, error_model="numpy")
def _squared_edt_1d_parabolic(
    f, offset, n, stride, anisotropy, black_border_left, black_border_right
):
    if n == 0:
        return

    w2 = float(anisotropy) * float(anisotropy)  # 与 C++ 一致：double 精度

    v = np.empty(n, dtype=np.int64)
    ff = np.empty(n, dtype=np.float64)
    ranges = np.empty(n + 1, dtype=np.float64)

    v[0] = 0
    for i in range(n):
        ff[i] = f[offset + i * stride]  # float32 → float64

    ranges[0] = -np.inf
    ranges[1] = np.inf

    # ---- 第一遍：构建下包络 ----
    k = 0
    for i in range(1, n):
        diff = i - v[k]
        factor1 = diff * w2
        factor2 = i + v[k]
        s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1)

        while k > 0 and s <= ranges[k]:
            k -= 1
            diff = i - v[k]
            factor1 = diff * w2
            factor2 = i + v[k]
            s = (ff[i] - ff[v[k]] + factor1 * factor2) / (2.0 * factor1)

        k += 1
        v[k] = i
        ranges[k] = s
        ranges[k + 1] = np.inf

    # ---- 第二遍：求值 + 可选 envelope ----
    k = 0
    for i in range(n):
        while ranges[k + 1] < i:
            k += 1

        vk = v[k]
        diff = i - vk
        val = w2 * diff * diff + ff[vk]

        if black_border_left and black_border_right:
            env_l = w2 * (i + 1) * (i + 1)
            env_r = w2 * (n - i) * (n - i)
            envelope = env_l if env_l < env_r else env_r
            if envelope < val:
                val = envelope
        elif black_border_left:
            envelope = w2 * (i + 1) * (i + 1)
            if envelope < val:
                val = envelope
        elif black_border_right:
            envelope = w2 * (n - i) * (n - i)
            if envelope < val:
                val = envelope

        f[offset + i * stride] = np.float32(val)  # float64 → float32


# ──────────────────────────────────────────────
# 主入口：2D / 3D EDT
# 对应 C++ edt::_binary_edt3dsq
# ──────────────────────────────────────────────
@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_edt(binary_img, wx=1.0, wy=1.0, wz=1.0, black_border=False):
    """
    binary_img : (H, W) 或 (D, H, W) bool 数组
                 True = 前景, False = 背景
    返回与输入同形的 float32 距离图
    """
    ndim = binary_img.ndim

    # ==================== 2D ====================
    if ndim == 2:
        sy, sx = binary_img.shape
        voxels = sy * sx
        workspace = np.empty(voxels, dtype=np.float32)
        img_flat = binary_img.ravel()
        wx32 = np.float32(wx)
        wy32 = np.float32(wy)

        # Pass 1 — X（Rosenfeld-Pfaltz）
        for y in nb.prange(sy):
            _squared_edt_1d_multi_seg(
                img_flat, workspace, sx * y, sx, 1, wx32, black_border
            )

        # tofinite
        if not black_border:
            for i in nb.prange(voxels):
                if np.isinf(workspace[i]):
                    workspace[i] = _FLOAT32_MAX

        # Pass 2 — Y（抛物线，跳过前导零）
        for x in nb.prange(sx):
            y = 0
            while y < sy and workspace[x + sx * y] == 0.0:
                y += 1
            if y < sy:
                _squared_edt_1d_parabolic(
                    workspace,
                    x + sx * y,
                    sy - y,
                    sx,
                    wy32,
                    black_border or (y > 0),
                    black_border,
                )

        # toinfinite
        if not black_border:
            for i in nb.prange(voxels):
                if workspace[i] >= _FLOAT32_MAX:
                    workspace[i] = np.float32(np.inf)

        return np.sqrt(workspace).reshape(sy, sx)

    # ==================== 3D ====================
    elif ndim == 3:
        sz, sy, sx = binary_img.shape
        sxy = sx * sy
        voxels = sz * sxy
        workspace = np.empty(voxels, dtype=np.float32)
        img_flat = binary_img.ravel()
        wx32 = np.float32(wx)
        wy32 = np.float32(wy)
        wz32 = np.float32(wz)

        # Pass 1 — X（Rosenfeld-Pfaltz）
        total_x = sz * sy
        for i in nb.prange(total_x):
            z = i // sy
            y = i % sy
            _squared_edt_1d_multi_seg(
                img_flat, workspace, sx * y + sxy * z, sx, 1, wx32, black_border
            )

        # tofinite
        if not black_border:
            for i in nb.prange(voxels):
                if np.isinf(workspace[i]):
                    workspace[i] = _FLOAT32_MAX

        # Pass 2 — Y（抛物线，跳过前导零）
        total_y = sz * sx
        for i in nb.prange(total_y):
            z = i // sx
            x = i % sx
            base = x + sxy * z
            y = 0
            while y < sy and workspace[base + sx * y] == 0.0:
                y += 1
            if y < sy:
                _squared_edt_1d_parabolic(
                    workspace,
                    base + sx * y,
                    sy - y,
                    sx,
                    wy32,
                    black_border or (y > 0),
                    black_border,
                )

        # Pass 3 — Z（抛物线，跳过前导零）
        total_z = sy * sx
        for i in nb.prange(total_z):
            y = i // sx
            x = i % sx
            base = x + sx * y
            z = 0
            while z < sz and workspace[base + sxy * z] == 0.0:
                z += 1
            if z < sz:
                _squared_edt_1d_parabolic(
                    workspace,
                    base + sxy * z,
                    sz - z,
                    sxy,
                    wz32,
                    black_border or (z > 0),
                    black_border,
                )

        # toinfinite
        if not black_border:
            for i in nb.prange(voxels):
                if workspace[i] >= _FLOAT32_MAX:
                    workspace[i] = np.float32(np.inf)

        return np.sqrt(workspace).reshape(sz, sy, sx)

    else:
        raise ValueError("Only 2D and 3D binary images are supported.")


# ──────────────────────────────────────────────
# Medial-surface clipping
# 对应 C++ medialSurface::calc_distmaps 中的 clip 部分
# ──────────────────────────────────────────────
@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_classic_edt(binary_img, _clipROutyz=0.5, _clipROutx=0.5):
    dt = nb_edt(binary_img)
    nz, ny, nx = dt.shape
    for z in nb.prange(nz):
        for y in nb.prange(ny):
            for x in nb.prange(nx):
                if binary_img[z, y, x]:
                    limit = dt[z, y, x] - 0.5

                    iSqr = min(y + 2, ny - y + 1)
                    if iSqr < limit:
                        limit = max(
                            (1.0 - _clipROutyz) * limit + _clipROutyz * iSqr, 0.01
                        )

                    iSqr = min(z + 2, nz - z + 1)
                    if iSqr < limit:
                        limit = max(
                            (1.0 - _clipROutyz) * limit + _clipROutyz * iSqr, 0.01
                        )

                    iSqr = min(x + 2, nx - x + 1)
                    if iSqr < limit:
                        limit = max(
                            (1.0 - _clipROutx) * limit + _clipROutx * iSqr, 0.1
                        )  # 注意这里是 0.1

                    dt[z, y, x] = limit

    return dt
