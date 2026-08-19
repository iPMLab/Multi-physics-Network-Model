import numba as nb
import numpy as np
from ._extraction_functions_numba import get_masterball, _0p5


@nb.njit(
    parallel=False,
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
    forceinline=True,
)
def get_max_count_nei(neis, n_neIs):
    max_count = 0
    max_val = -257

    for i in range(n_neIs):
        count = 0
        val = neis[i]

        for j in range(i, n_neIs):
            count += neis[j] == val

        if count > max_count or (count == max_count and val < max_val):
            max_count = count
            max_val = val

    return max_val, max_count


@nb.njit(parallel=False, cache=True, fastmath=True, nogil=True)
def CreateVElem(img_bool, dt, isball, ball_indices, ball_findices, ball_R, ball_boss):
    raw_value = -1
    nz, ny, nx = img_bool.shape
    VElems = np.full((nz + 2, ny + 2, nx + 2), raw_value, dtype=np.int32)
    poreIs = np.where(ball_boss == np.arange(ball_boss.shape[0]))[0]
    for ind in nb.prange(poreIs.size):
        pore_ind = poreIs[ind]
        zm, ym, xm = ball_indices[pore_ind] + 1
        VElems[zm, ym, xm] = ind

    for ball_index in range(ball_findices.shape[0]):
        masterball = get_masterball(ball_boss, ball_index)
        zm, ym, xm = ball_indices[masterball] + 1
        Vm = VElems[zm, ym, xm]
        fzo, fyo, fxo = ball_findices[ball_index]
        r = ball_R[ball_index]

        r_sphere = max(r * 0.5 - 1.0, 1.001)
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
                    if ~img_bool[z, y, x]:
                        continue
                    zVE = z + 1
                    yVE = y + 1
                    xVE = x + 1
                    Vi = VElems[zVE, yVE, xVE]
                    if Vi == raw_value:
                        VElems[zVE, yVE, xVE] = Vm
                        continue
                    if Vi == Vm:
                        continue

                    if isball[z, y, x] or ball_R[Vi] >= r:
                        continue
                    fz_old = z + _0p5 - ball_findices[Vi, 0]
                    fy_old = y + _0p5 - ball_findices[Vi, 1]
                    fx_old = x + _0p5 - ball_findices[Vi, 2]
                    fz_new = z + _0p5 - ball_findices[masterball, 0]
                    fy_new = y + _0p5 - ball_findices[masterball, 1]
                    fx_new = x + _0p5 - ball_findices[masterball, 2]
                    if (
                        fz_new**2 + fy_new**2 + fx_new**2
                        >= fz_old**2 + fy_old**2 + fx_old**2
                    ):
                        continue
                    VElems[zVE, yVE, xVE] = Vm

    return VElems, poreIs


@nb.njit(
    parallel=True,
    cache=True,
    fastmath=True,
    nogil=True,
    inline="always",
    forceinline=True,
)
def assign_array(src, dst):
    src_flat = src.reshape(-1)
    dst_flat = dst.reshape(-1)
    for i in nb.prange(dst.size):
        dst_flat[i] = src_flat[i]


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"ngrowPors changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_X2(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"  ngrowX3:{n_changes},")

    n_changes = 0
    assign_array(VElems, voxls)
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"{n_changes},")

    n_changes = 0
    assign_array(VElems, voxls)
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        if bgn <= voxls[z, y, x - 1]:
            VElems[z, y, x] = voxls[z, y, x - 1]
            n_changes += 1
        elif bgn <= voxls[z, y, x + 1]:
            VElems[z, y, x] = voxls[z, y, x + 1]
            n_changes += 1
        elif bgn <= voxls[z, y - 1, x]:
            VElems[z, y, x] = voxls[z, y - 1, x]
            n_changes += 1
        elif bgn <= voxls[z, y + 1, x]:
            VElems[z, y, x] = voxls[z, y + 1, x]
            n_changes += 1
        elif bgn <= voxls[z - 1, y, x]:
            VElems[z, y, x] = voxls[z - 1, y, x]
            n_changes += 1
        elif bgn <= voxls[z + 1, y, x]:
            VElems[z, y, x] = voxls[z + 1, y, x]
            n_changes += 1

    print(f"  ngrowX2:{n_changes}  ")

    return VElems, n_changes


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        R = dt_p1[z, y, x]
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        R_current = dt_p1[z, y, x - 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z, y, x + 1]
        R_current = dt_p1[z, y, x + 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z, y - 1, x]
        R_current = dt_p1[z, y - 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z, y + 1, x]
        R_current = dt_p1[z, y + 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z - 1, y, x]
        R_current = dt_p1[z - 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1
        V_current = voxls[z + 1, y, x]
        R_current = dt_p1[z + 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            if R_current > R:
                neIs[n_neIs] = V_current
                n_neIs += 1

        if nDifferentID >= 3:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 3:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedStrict changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        R = dt_p1[z, y, x]
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        R_current = dt_p1[z, y, x - 1]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        R_current = dt_p1[z, y, x + 1]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        R_current = dt_p1[z, y - 1, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        R_current = dt_p1[z, y + 1, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        R_current = dt_p1[z - 1, y, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        R_current = dt_p1[z + 1, y, x]
        if bgn <= V_current and R_current > R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        if nDifferentID >= 2:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 2:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedian changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        R = dt_p1[z, y, x]
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        R_current = dt_p1[z, y, x - 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        R_current = dt_p1[z, y, x + 1]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        R_current = dt_p1[z, y - 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        R_current = dt_p1[z, y + 1, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        R_current = dt_p1[z - 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        R_current = dt_p1[z + 1, y, x]
        if bgn <= V_current and R_current >= R:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1

        if nDifferentID >= 2:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 2:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedEqs changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        if VElems[z, y, x] != raw_value:
            continue
        nDifferentID = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        if bgn <= V_current:
            nDifferentID += 1
            neIs[n_neIs] = V_current
            n_neIs += 1

        if nDifferentID >= 2:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count >= 2:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"ngMedLoose changes: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def median_elem(zsysxs_v, VElems, voxls, bgn):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0

    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        pID = voxls[z, y, x]
        if pID < bgn:
            continue
        n_same = 0
        n_diff = 0
        neIs = np.empty(6, dtype=np.int32)
        n_neIs = 0
        V_current = voxls[z, y, x - 1]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y, x + 1]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y - 1, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z, y + 1, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z - 1, y, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        V_current = voxls[z + 1, y, x]
        if V_current == pID:
            n_same += 1
        elif bgn <= V_current:
            n_diff += 1
            neIs[n_neIs] = V_current
            n_neIs += 1
        if n_diff > n_same:
            max_count_nei, max_count = get_max_count_nei(neIs, n_neIs)
            if max_count > n_same:
                VElems[z, y, x] = max_count_nei
                n_changes += 1

    print(f"nMedian: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def retreat_pores_median(zsysxs_v, VElems, voxls, bgn, raw_value):
    nVxls = zsysxs_v.shape[0]
    assign_array(VElems, voxls)
    n_changes = 0
    for ipar in nb.prange(nVxls):
        z, y, x = zsysxs_v[ipar]
        pID = VElems[z, y, x]
        if pID < bgn:
            continue
        nSameID = 0
        nDiffereID = 0
        V_current = voxls[z, y, x - 1]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z, y, x + 1]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z, y - 1, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z, y + 1, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z - 1, y, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1
        V_current = voxls[z + 1, y, x]
        if V_current == pID:
            nSameID += 1
        elif bgn <= V_current:
            nDiffereID += 1

        if nDiffereID > 0 and nSameID > 0:
            VElems[z, y, x] = raw_value
            n_changes += 1

    print(f"nRetreat: {n_changes}")
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def refine_with_master_ball(VElems, ball_boss, ball_indices):
    nBalls = ball_indices.shape[0]
    for ipar in nb.prange(nBalls):
        ball_master = np.int64(ipar)
        while ball_boss[ball_master] != ball_master:
            ball_master = ball_boss[ball_master]
        zi, yi, xi = ball_indices[ipar]
        zm, ym, xm = ball_indices[ball_master]
        VElems[zi + 1, yi + 1, xi + 1] = VElems[zm + 1, ym + 1, xm + 1]
    return VElems


@nb.njit(parallel=True, cache=True, fastmath=True, nogil=True)
def refine_with_boss_ball(VElems, ball_indices, poreIs):
    for ipar in nb.prange(poreIs.size):
        ball_idx = poreIs[ipar]
        zi, yi, xi = ball_indices[ball_idx]
        VElems[zi + 1, yi + 1, xi + 1] = np.int32(ipar)
    return VElems
