import numpy as np
import numba as nb

# from numpy import linalg as LA
from numba.core import types
from numba.typed import Dict

_0p5 = np.float32(0.5)


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def nb_parallel_sum(arr):
    arr = arr.reshape(-1)
    _sum = 0
    for i in nb.prange(arr.size):
        _sum += arr[i]
    return _sum


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def nb_where(arr, nVxls):
    indices = np.empty(nVxls, dtype=np.int64)
    _, ny, nx = arr.shape
    arr = arr.reshape(-1)
    nzyx = arr.size
    index = 0
    for i in range(nzyx):
        if arr[i]:
            indices[index] = i
            index += 1

    xs = indices % nx
    indices = indices // nx
    zs = indices // ny
    ys = indices % ny
    zsysxs = np.empty((zs.size, 3), dtype=np.int64)
    for ipar in nb.prange(zs.size):
        zsysxs_i = zsysxs[ipar]
        zsysxs_i[0] = zs[ipar]
        zsysxs_i[1] = ys[ipar]
        zsysxs_i[2] = xs[ipar]
    return zsysxs


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def smooth_radius(image, dt, zsysxs_v):
    nz, ny, nx = dt.shape
    print("smoothing R")
    delR = np.empty(image.shape, dtype=np.float32)
    for ipar in nb.prange(zsysxs_v.shape[0]):
        zo, yo, xo = zsysxs_v[ipar]
        sum_r = 0.0
        counter = 0
        for zi in range(-1, 2):
            for yi in range(-1, 2):
                for xi in range(-1, 2):
                    z = zo + zi
                    y = yo + yi
                    x = xo + xi
                    if (
                        z < 0
                        or z >= nz
                        or y < 0
                        or y >= ny
                        or x < 0
                        or x >= nx
                        or ~image[z, y, x]
                    ):
                        continue
                    sum_r += dt[z, y, x]
                    counter += 1
        delR[zo, yo, xo] = 4.0 * sum_r / (3 * counter + 27) - dt[zo, yo, xo]

    for ipar in nb.prange(zsysxs_v.shape[0]):
        zo, yo, xo = zsysxs_v[ipar]
        sum_del_r = 0.0
        counter = 0
        for zi in range(-1, 2):
            for yi in range(-1, 2):
                for xi in range(-1, 2):
                    z = zo + zi
                    y = yo + yi
                    x = xo + xi
                    if (
                        z < 0
                        or z >= nz
                        or y < 0
                        or y >= ny
                        or x < 0
                        or x >= nx
                        or ~image[z, y, x]
                    ):
                        continue
                    sum_del_r += delR[z, y, x]
                    counter += 1

        dt[zo, yo, xo] += min(
            max(
                0.02 * (delR[zo, yo, xo] - 0.99 * 2.0 * sum_del_r / (counter + 27)),
                -0.005,
            ),
            0.01,
        )
    # 计算最大半径
    print("maxrrr", np.max(dt))
    return dt


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def get_masterball(ball_boss, ball_index):
    while ball_boss[ball_index] != ball_index:
        ball_index = ball_boss[ball_index]
    return ball_index


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def get_ball_level(ball_boss, ball_index):
    level = 1
    current = ball_index
    while ball_boss[current] != current:
        current = ball_boss[current]
        level += 1
    return level


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def inParents(ball_boss, ball_index_i, ball_index_j):
    """Check if ball_index_j is a parent of ball_index_i in the boss tree"""
    current = ball_boss[ball_index_i]
    while True:
        if current == ball_index_j:
            return True
        if current == ball_boss[current]:
            return False
        current = ball_boss[current]


# def paradox_pre_removeincludedballI(image, dt, isball, _minRp): Avoid too long name, due to numba cache limit
@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def paradox_pre_rmincludedballI(image, dt, isball, _minRp):
    nz, ny, nx = image.shape
    nz_half = (nz + 1) // 2
    ny_half = (ny + 1) // 2
    nx_half = (nx + 1) // 2
    for _z in nb.prange(0, nz_half):
        zo = _z * 2
        for _y in nb.prange(0, ny_half):
            yo = _y * 2
            for _x in nb.prange(0, nx_half):
                xo = _x * 2
                max_r = -1
                max_z = -1
                max_y = -1
                max_x = -1
                for zi in range(2):
                    for yi in range(2):
                        for xi in range(2):
                            z = zo + zi
                            y = yo + yi
                            x = xo + xi
                            if (
                                z < 0
                                or z >= nz
                                or y < 0
                                or y >= ny
                                or x < 0
                                or x >= nx
                                or ~image[z, y, x]
                            ):
                                continue
                            r = dt[z, y, x]
                            if r <= max_r or r <= _minRp:
                                continue
                            max_r = r
                            max_z = z
                            max_y = y
                            max_x = x
                if max_z >= 0:
                    isball[max_z, max_y, max_x] = True


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def paradox_removeincludedballI(
    ball_indices, ball_R, image, dt, isball, _RCorsnf, _RCorsn, _MSNoise
):
    removed_ball = 0
    for i in range(ball_indices.shape[0]):
        zo, yo, xo = ball_indices[i]
        if ~isball[zo, yo, xo]:
            continue
        nz, ny, nx = image.shape
        ri = ball_R[i]
        mbmbDist = _RCorsnf * ri + _RCorsn
        r_sphere = ri + 0.55
        r_sphere_2 = r_sphere**2

        # voxel center = index + 0.5
        # |(z+0.5) - cz| <= radius
        # → z >= cz - radius - 0.5
        # → z <= cz + radius - 0.5
        z_start = max(int(np.ceil(zo - _0p5 - r_sphere)), 0)
        z_end = min(int(np.floor(zo - _0p5 + r_sphere)) + 1, nz)

        for z in range(z_start, z_end):
            dz = z + _0p5 - zo
            ry2 = r_sphere_2 - dz * dz
            if ry2 <= 0.0:
                continue
            ry = np.sqrt(ry2)

            y_start = max(int(np.ceil(yo - _0p5 - ry)), 0)
            y_end = min(int(np.floor(yo - _0p5 + ry)) + 1, ny)

            for y in range(y_start, y_end):
                dy = y + _0p5 - yo
                rx2 = ry2 - dy * dy
                if rx2 <= 0.0:
                    continue
                rx = np.sqrt(rx2)

                x_start = max(int(np.ceil(xo - _0p5 - rx)), 0)
                x_end = min(int(np.floor(xo - _0p5 + rx)) + 1, nx)

                for x in range(x_start, x_end):
                    dx = x + _0p5 - xo
                    if ~isball[z, y, x] or (z == zo and y == yo and x == xo):
                        continue
                    rj = dt[z, y, x]
                    if rj > ri:
                        continue
                    D = np.sqrt(dz**2 + dy**2 + dx**2)
                    if D >= mbmbDist and D + rj >= r_sphere + _MSNoise:
                        continue
                    isball[z, y, x] = False
                    removed_ball += 1

    print(f"removed ball {removed_ball}")


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True, error_model="numpy")
def moveUphill(ball_indices, ball_findices, ball_R, image, dt):
    nz, ny, nx = image.shape
    for i in nb.prange(ball_indices.shape[0]):
        disp = np.array([0, 0, 0], dtype=np.float32)
        iz, iy, ix = ball_indices[i]
        vi_r = dt[iz, iy, ix]
        vjm_z = iz - 1
        vjm_y = iy
        vjm_x = ix
        vjp_z = iz + 1
        vjp_y = iy
        vjp_x = ix
        if 0 <= vjm_z and vjp_z < nz:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[0] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        vjm_z = iz
        vjm_y = iy - 1
        vjm_x = ix
        vjp_z = iz
        vjp_y = iy + 1
        vjp_x = ix
        if 0 <= vjm_y and vjp_y < ny:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[1] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        vjm_z = iz
        vjm_y = iy
        vjm_x = ix - 1
        vjp_z = iz
        vjp_y = iy
        vjp_x = ix + 1
        if 0 <= vjm_x and vjp_x < nx:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                if abs(gp - gm) > 0.01:
                    disp[2] = max(-0.49, min(0.49, -0.5 * (gp + gm) / (gp - gm)))

        ball_findices[i] = ball_indices[i] + disp + _0p5
        ball_R[i] = vi_r + 0.95 * np.sqrt(disp[0] ** 2 + disp[1] ** 2 + disp[2] ** 2)


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def moveUphillp1(ball_indices, ball_findices, ball_R, image, dt, isball):
    nz, ny, nx = image.shape
    for i in range(ball_indices.shape[0]):
        disp_z = 0.0
        disp_y = 0.0
        disp_x = 0.0
        grad_z = 0.0
        grad_y = 0.0
        grad_x = 0.0
        iz, iy, ix = ball_indices[i]
        vi_r = dt[iz, iy, ix]
        vjm_z = iz - 1
        vjm_y = iy
        vjm_x = ix
        vjp_z = iz + 1
        vjp_y = iy
        vjp_x = ix
        if 0 <= vjm_z and vjp_z < nz:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad_z = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp_z = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        vjm_z = iz
        vjm_y = iy - 1
        vjm_x = ix
        vjp_z = iz
        vjp_y = iy + 1
        vjp_x = ix
        if 0 <= vjm_y and vjp_y < ny:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad_y = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp_y = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        vjm_z = iz
        vjm_y = iy
        vjm_x = ix - 1
        vjp_z = iz
        vjp_y = iy
        vjp_x = ix + 1
        if 0 <= vjm_x and vjp_x < nx:
            if image[vjm_z, vjm_y, vjm_x] and image[vjp_z, vjp_y, vjp_x]:
                vjm_r = dt[vjm_z, vjm_y, vjm_x]
                vjp_r = dt[vjp_z, vjp_y, vjp_x]
                gp = vjp_r - vi_r
                gm = vi_r - vjm_r
                grad_x = 0.5 * (gp + gm)
                if abs(gp - gm) > 0.01:
                    disp_x = max(-0.59, min(0.59, -0.5 * (gp + gm) / (gp - gm)))
        disp_z += 1.4 * grad_z
        disp_y += 1.4 * grad_y
        disp_x += 1.4 * grad_x
        disp_norm = 0.55 * np.sqrt(disp_z**2 + disp_y**2 + disp_x**2) + 0.05
        disp_z /= disp_norm
        disp_y /= disp_norm
        disp_x /= disp_norm
        vxlj_z = np.int32(iz + disp_z)
        vxlj_y = np.int32(iy + disp_y)
        vxlj_x = np.int32(ix + disp_x)
        if (
            vxlj_z < 0
            or vxlj_z >= nz
            or vxlj_y < 0
            or vxlj_y >= ny
            or vxlj_x < 0
            or vxlj_x >= nx
            or (vxlj_z == iz and vxlj_y == iy and vxlj_x == ix)
            or isball[vxlj_z, vxlj_y, vxlj_x]
            or dt[vxlj_z, vxlj_y, vxlj_x] <= ball_R[i]
        ):
            continue

        isball[iz, iy, ix] = False
        isball[vxlj_z, vxlj_y, vxlj_x] = True
        ball_indices[i, 0] = vxlj_z
        ball_indices[i, 1] = vxlj_y
        ball_indices[i, 2] = vxlj_x
        ball_findices[i, 0] = vxlj_z + _0p5
        ball_findices[i, 1] = vxlj_y + _0p5
        ball_findices[i, 2] = vxlj_x + _0p5
        ball_R[i] = dt[vxlj_z, vxlj_y, vxlj_x]


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def makeFriend(ball_R, ball_boss, ball_indices, ball_findices, vi, vj):
    if ball_R[vj] > ball_R[vi]:
        vi, vj = vj, vi


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
    inline="always",
    forceinline=True,
)
def competeForParent(
    bi,
    bj,
    ball_findices,
    ball_R,
    ball_boss,
    image,
    dt,
    _MSNoise,
    _midRf,
    _vmvRadRelNf,
    _lenNf,
):

    def ratio_boss(boss_R, self_R, d, n):
        return (boss_R - self_R + 2.0 * n) / (d + 0.25)

    def ratio_peer(other_R, self_R, d, n):
        return (other_R - self_R + 2.0 * n + 0.01) / (d + 0.2)

    def ratio_near(boss_R, self_R, d, n, off):
        return (boss_R - self_R + 2.0 * n) / (d + off)

    nz, ny, nx = image.shape
    noise = _MSNoise
    ri = ball_R[bi]
    rj = ball_R[bj]
    riSq = ri**2
    rjSq = rj**2
    fzi, fyi, fxi = ball_findices[bi]
    fzj, fyj, fxj = ball_findices[bj]
    dSq = (fzi - fzj) ** 2 + (fyi - fyj) ** 2 + (fxi - fxj) ** 2
    dSqrt = np.sqrt(dSq)
    wsinv = 1.0 / (riSq + rjSq)
    middlevxlz = np.int32((fzi * rjSq + fzj * riSq) * wsinv)
    middlevxly = np.int32((fyi * rjSq + fyj * riSq) * wsinv)
    middlevxlx = np.int32((fxi * rjSq + fxj * riSq) * wsinv)
    if (
        middlevxlz < 0
        or middlevxlz >= nz
        or middlevxly < 0
        or middlevxly >= ny
        or middlevxlx < 0
        or middlevxlx >= nx
        or ~image[middlevxlz, middlevxly, middlevxlx]
    ):
        return
    middlevxl_R = dt[middlevxlz, middlevxly, middlevxlx]
    minR = min(ri, rj)
    if middlevxl_R < minR * _midRf - 0.5 or 1.01 * dSqrt > ri + rj + 1.0 + noise:
        return

    if ball_boss[bj] == bj and get_masterball(ball_boss, bi) != bj:
        if ri >= rj:
            ball_boss[bj] = bi
        elif ball_R[ball_boss[bi]] <= rj:
            ball_boss[bi] = bj
        elif ri >= rj - noise and ri * _vmvRadRelNf + 1.0 * noise >= rj:
            ball_boss[bj] = bi
    elif ball_boss[bi] == bi and get_masterball(ball_boss, bj) != bi:
        if rj >= ri:
            ball_boss[bi] = bj
        elif ball_R[ball_boss[bj]] <= ri:
            ball_boss[bj] = bi
        elif rj >= ri - noise and rj * _vmvRadRelNf + 1.0 * noise >= ri:
            ball_boss[bi] = bj

    mbi = get_masterball(ball_boss, bi)
    mbj = get_masterball(ball_boss, bj)

    if mbi == bj or mbj == bi:
        return

    if mbi == mbj:
        leveli = get_ball_level(ball_boss, bi)
        levelj = get_ball_level(ball_boss, bj)
        level_diff = leveli - levelj
        bbi = ball_boss[bi]
        bbj = ball_boss[bj]
        bbi_R = ball_R[bbi]
        bbj_R = ball_R[bbj]
        dist_bbi_bi = np.sqrt(np.sum((ball_findices[bbi] - ball_findices[bi]) ** 2))
        dist_bbj_bj = np.sqrt(np.sum((ball_findices[bbj] - ball_findices[bj]) ** 2))
        dist_bibj = dSqrt

        if level_diff < -1 and ratio_boss(bbj_R, rj, dist_bbj_bj, noise) < ratio_peer(
            ri, rj, dist_bibj, noise
        ):
            ball_boss[bj] = bi
        elif level_diff > 1 and ratio_boss(bbi_R, ri, dist_bbi_bi, noise) < ratio_peer(
            rj, ri, dist_bibj, noise
        ):
            ball_boss[bi] = bj
        elif (
            level_diff > 0
            and ratio_near(bbi_R, ri, dist_bbi_bi, noise, 1.2)
            < ratio_near(rj, ri, dist_bibj, noise, 1.3)
            and not inParents(ball_boss, bj, bi)
        ):
            ball_boss[bi] = bj
        elif (
            level_diff < 0
            and ratio_near(bbj_R, rj, dist_bbj_bj, noise, 1.2)
            < ratio_near(ri, rj, dist_bibj, noise, 1.3)
            and not inParents(ball_boss, bi, bj)
        ):
            ball_boss[bj] = bi
    else:  # mvi != mvj:
        mbi_R = ball_R[mbi]
        mbj_R = ball_R[mbj]
        avg_R = 0.5 * (mbi_R + mbj_R)
        if (
            np.sum((ball_findices[mbi] - ball_findices[mbj]) ** 2)
            < _lenNf * (avg_R + 2.0 * noise) ** 2
        ):
            if mbi_R < mbj_R:
                bi, bj, mbi, mbj = bj, bi, mbj, mbi

            mbj_R = ball_R[mbj]
            if (
                mbj_R < _vmvRadRelNf * ball_R[bj] + noise
                and mbj_R < _vmvRadRelNf * ball_R[bi] + noise
                and mbj_R < _vmvRadRelNf * ball_R[ball_boss[bi]] + noise
            ):
                while (
                    ball_boss[bj] != bj
                    and mbj_R < _vmvRadRelNf * ball_R[ball_boss[bj]] + noise
                ):
                    temp = ball_boss[bj]
                    ball_boss[bj] = bi
                    bi = bj
                    bj = temp
                if ball_boss[bj] == bj and get_masterball(ball_boss, bi) != bj:
                    ball_boss[bj] = bi
        if bi != ball_boss[bj]:
            mbi = get_masterball(ball_boss, bi)
            mbj = get_masterball(ball_boss, bj)
            leveli = get_ball_level(ball_boss, bi)
            levelj = get_ball_level(ball_boss, bj)
            dist_avg = (
                np.sqrt(np.sum((ball_findices[mbj] - ball_findices[mbi]) ** 2))
                + 0.5 * noise
            )
            babb = np.array([bi, bj], dtype=np.int32)
            lalb = np.array([leveli, levelj], dtype=np.int32)

            def balance_step(babb, ma, mb, lalb):
                ba, bb = babb
                levela, levelb = lalb
                if levela < levelb:
                    return False
                dist_maba = np.sqrt(
                    np.sum((ball_findices[ma] - ball_findices[ba]) ** 2)
                )
                dist_mbba = np.sqrt(
                    np.sum((ball_findices[mb] - ball_findices[ba]) ** 2)
                )
                if (ball_R[ball_boss[ba]] - ball_R[ba] + 0.55 * noise) / (
                    dist_maba + dist_avg
                ) >= (ball_R[bb] - ball_R[ba] + 0.5 * noise) / (dist_mbba + dist_avg):
                    return False
                temp = ball_boss[ba]
                ball_boss[ba] = bb
                babb[1] = ba
                babb[0] = temp
                levela -= 1
                levelb += 1
                return True

            while balance_step(
                babb,
                mbi,
                mbj,
                lalb,
            ):
                pass

            babb[0], babb[1] = babb[1], babb[0]
            lalb[0], lalb[1] = lalb[1], lalb[0]

            while balance_step(babb, mbj, mbi, lalb):
                pass


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    error_model="numpy",
)
def float_range(start, stop, step):
    x = start
    n = 0
    while x < stop:
        yield x
        n += 1
        x = start + n * step


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
def findBoss(
    ball_indices,
    ball_findices,
    ball_R,
    ball_master,
    image,
    dt,
    isball,
    _MSNoise,
    _midRf,
    _vmvRadRelNf,
    _lenNf,
):
    nz, ny, nx = image.shape
    nBalls = ball_indices.shape[0]
    whichball = Dict.empty(
        key_type=types.int64,  # 或 types.int32，取决于你的索引范围
        value_type=types.int64,
        n_keys=nBalls,
    )
    keys = (
        ball_indices[:, 0] * ny * nx + ball_indices[:, 1] * nx + ball_indices[:, 2]
    ).astype(np.int64)
    for i in range(nBalls):
        whichball[keys[i]] = np.int64(i)

    for i in range(nBalls):
        fzo, fyo, fxo = ball_findices[i]
        izo, iyo, ixo = ball_indices[i]

        r_sphere = ball_R[i] * 0.6 + 2.0 * _MSNoise + 2.0
        r_sphere_2 = r_sphere**2

        z_start = max(int(np.ceil(fzo - _0p5 - r_sphere)), 0)
        z_end = min(int(np.floor(fzo - _0p5 + r_sphere)) + 1, nz)

        for z in range(z_start, z_end):
            dz = z + _0p5 - fzo
            ry2 = r_sphere_2 - dz * dz
            if ry2 <= 0.0:
                continue
            ry = np.sqrt(ry2)

            y_start = max(int(np.ceil(fyo - _0p5 - ry)), 0)
            y_end = min(int(np.floor(fyo - _0p5 + ry)) + 1, ny)

            for y in range(y_start, y_end):
                dy = y + _0p5 - fyo
                rx2 = ry2 - dy * dy
                if rx2 <= 0.0:
                    continue
                rx = np.sqrt(rx2)

                x_start = max(int(np.ceil(fxo - _0p5 - rx)), 0)
                x_end = min(int(np.floor(fxo - _0p5 + rx)) + 1, nx)

                for x in range(x_start, x_end):
                    if ~isball[z, y, x] or (z == izo and y == iyo and x == ixo):
                        continue
                    vj = whichball[np.int64((z * ny + y) * nx + x)]
                    competeForParent(
                        np.int64(i),
                        vj,
                        ball_findices,
                        ball_R,
                        ball_master,
                        image,
                        dt,
                        _MSNoise,
                        _midRf,
                        _vmvRadRelNf,
                        _lenNf,
                    )
