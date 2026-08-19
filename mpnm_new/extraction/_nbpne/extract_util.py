import numpy as np

from ._edt_numba import nb_classic_edt, nb_edt as edt


from ._extraction_functions_numba import (
    _0p5,
    nb_parallel_sum,
    nb_where,
    smooth_radius,
    paradox_pre_rmincludedballI,
    paradox_removeincludedballI,
    moveUphill,
    moveUphillp1,
    findBoss,
)

from ._blocknet_numba import (
    grow_pores,
    grow_pores_med_eqs,
    grow_pores_med_eqs_loose,
    grow_pores_med_strict,
    grow_pores_median,
    grow_pores_X2,
    CreateVElem,
    median_elem,
    retreat_pores_median,
    refine_with_master_ball,
    refine_with_boss_ball,
)
import gc


class Defaults:
    def __init__(self, avgR):
        self.avgR = avgR  # 存储 avgR 以便后续计算

    @property
    def minRp(self):
        return min(1.25, self.avgR * 0.25) + 0.5

    @property
    def clipROutx(self):
        return 0.05

    @property
    def clipROutyz(self):
        return 0.98

    @property
    def midRf(self):
        return 0.7

    @property
    def MSNoise(self):
        return 1.0 * abs(self.minRp) + 1.0

    @property
    def lenNf(self):
        return 0.6

    @property
    def vmvRadRelNf(self):
        return 1.1

    @property
    def nRSmoothing(self):
        return 3

    @property
    def RCorsnf(self):
        return 0.15

    @property
    def RCorsn(self):
        return abs(self.minRp)


class Balls_cls:
    def __init__(self, isball, dt, sort=True) -> None:
        self.initialize(isball, dt, sort=sort)

    def initialize(self, isball, dt, sort=True) -> None:
        self.indices = nb_where(isball, nb_parallel_sum(isball))
        self.findices = self.indices + _0p5
        self.R = dt[self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]]
        if sort:
            self.sort()

    def update(self, isball=None, sort=True):
        if isball is not None:
            keep = isball[self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]]
            self.indices = self.indices[keep]
            self.findices = self.findices[keep]
            self.R = self.R[keep]
        if sort:
            self.sort()

    def sort(self):
        sorted_indices = np.argsort(-self.R, kind="stable")
        self.indices = self.indices[sorted_indices]
        self.findices = self.findices[sorted_indices]
        self.R = self.R[sorted_indices]
        return self


def extract(img_bool):
    nVxls = nb_parallel_sum(img_bool)
    print("num_voxels:", nVxls)

    zsysxs_v = nb_where(img_bool, nVxls)
    # _clipROutyz=0.98, _clipROutx=0.05

    # print(nb_where.inspect_types())
    dt = nb_classic_edt(img_bool, _clipROutyz=0.98, _clipROutx=0.05)
    # dt = edt(img_bool, black_border=True)
    # plt.imshow(dt[3])
    # plt.show()
    # dt = gaussian_filter(dt, 0.3)
    avgR = np.mean(dt, where=img_bool)
    print("avgR:", avgR)
    defaults = Defaults(avgR)
    for _ in range(defaults.nRSmoothing):
        smooth_radius(img_bool, dt, zsysxs_v)

    isball = np.zeros_like(img_bool, dtype=bool)
    paradox_pre_rmincludedballI(img_bool, dt, isball, defaults.minRp)
    print(f"remained balls: {nb_parallel_sum(isball)}")
    Balls = Balls_cls(isball, dt, sort=True)
    paradox_removeincludedballI(
        Balls.indices,
        Balls.R,
        img_bool,
        dt,
        isball,
        defaults.RCorsnf,
        defaults.RCorsn,
        defaults.MSNoise,
    )

    Balls.update(isball, sort=False)
    print("num_balls:", Balls.indices.shape[0])

    moveUphill(Balls.indices, Balls.findices, Balls.R, img_bool, dt)
    Balls.sort()
    moveUphillp1(Balls.indices, Balls.findices, Balls.R, img_bool, dt, isball)
    moveUphill(Balls.indices, Balls.findices, Balls.R, img_bool, dt)
    Balls.update(isball, sort=True)

    Balls.boss = np.arange(Balls.indices.shape[0], dtype=np.int64)
    findBoss(
        Balls.indices,
        Balls.findices,
        Balls.R,
        Balls.boss,
        img_bool,
        dt,
        isball,
        defaults.MSNoise,
        defaults.midRf,
        defaults.vmvRadRelNf,
        defaults.lenNf,
    )

    VElems, poreIs = CreateVElem(
        img_bool, dt, isball, Balls.indices, Balls.findices, Balls.R, Balls.boss
    )
    del img_bool
    dt_p1 = np.pad(dt, 1, mode="constant", constant_values=-1)
    del dt, isball
    gc.collect()
    voxls = np.empty_like(VElems)

    zsysxs_v += 1
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)

    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)

    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_strict(zsysxs_v, dt_p1, VElems, voxls, 0, -1)

    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)

    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)

    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)

    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)

    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)

    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems, _ = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)
    VElems, _ = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)
    VElems, _ = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)
    VElems, _ = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)

    VElems = median_elem(zsysxs_v, VElems, voxls, 0)
    VElems = median_elem(zsysxs_v, VElems, voxls, 0)
    VElems = median_elem(zsysxs_v, VElems, voxls, 0)
    VElems = median_elem(zsysxs_v, VElems, voxls, 0)
    VElems = median_elem(zsysxs_v, VElems, voxls, 0)

    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    while True:
        VElems, num_changes = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)
        if num_changes == 0:
            break
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)

    VElems = retreat_pores_median(zsysxs_v, VElems, voxls, 0, -1)

    VElems = refine_with_master_ball(VElems, Balls.boss, Balls.indices)

    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_median(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems, _ = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)
    VElems = grow_pores_med_eqs(zsysxs_v, dt_p1, VElems, voxls, 0, -1)
    VElems = grow_pores(zsysxs_v, VElems, voxls, 0, -1)
    VElems, _ = grow_pores_X2(zsysxs_v, VElems, voxls, 0, -1)

    VElems = median_elem(zsysxs_v, VElems, voxls, 0)
    VElems = refine_with_master_ball(VElems, Balls.boss, Balls.indices)
    VElems = grow_pores_med_eqs_loose(zsysxs_v, VElems, voxls, 0, -1)
    VElems = refine_with_boss_ball(VElems, Balls.indices, poreIs)
    gc.collect()
    return VElems
