import numpy as np
from pathlib import Path
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from skimage.measure import marching_cubes
from skimage.segmentation import watershed
import scipy.ndimage as spim
import scipy.spatial as sptl
from joblib import Parallel, delayed
from collections import OrderedDict
from functools import partial
from ._extraction_numba import (
    nb_buildq,
    nb_functionals_2d,
    nb_functionals_3d,
    nb_max_filter_non_padding,
    nb_binary_dilation,
    nb_binary_erosion,
    nb_compute_surface_area_voxel,
    nb_get_objects_volume,
    nb_edt,
)
from sparse_dot_mkl import dot_product_mkl
from ..network import load_Statoil, read_pypne
from ..network._classical_pn_keys import (
    Pore1_names,
    Pore2_names,
    Throat1_names,
    Throat2_names,
)
from ..util import (
    unique_uint_nonzero,
    remap,
    unique_rows,
    find_throat_conns_map,
)

dirs = ("x-", "x+", "y-", "y+", "z-", "z+")


def edt(image, black_border: bool = False, wx=1, wy=1, wz=1):
    return nb_edt(image, black_border=black_border, wx=wx, wy=wy, wz=wz)


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


def check_pnextract_result(img, path, name=None, prefix=None):
    """
    Read the network from the given path remove pores not in image
    Parameters
    ----------
    path : type_str
        The path of the network files.
    name : type_str, optional
        The name of the network, by default None.
    prefix : type_str, optional
        The prefix of the network files, by default None.
    Returns
    -------
    img : np.ndarray
        The image with remaped labels.
    """
    name = name if name is not None else prefix
    Path_mpnm = Path(path)
    Path_node1 = Path_mpnm / f"{name}_node1.dat"
    Path_node2 = Path_mpnm / f"{name}_node2.dat"
    Path_link1 = Path_mpnm / f"{name}_link1.dat"
    Path_link2 = Path_mpnm / f"{name}_link2.dat"

    Pores, Throats = load_Statoil(path, name, prefix)
    Pores1 = Pores.loc[:, Pore1_names]

    labels_img = unique_uint_nonzero(img).astype(np.int32, copy=False)
    labels_remap = np.arange(1, labels_img.max() + 1, dtype=np.int32)

    if np.array_equal(labels_img, labels_remap):
        print("PNEXTRACT: Image does not need to be modified")
        return img
    else:
        print("PNEXTRACT: Image needs to be modified")
        img = remap(img, labels_img, labels_remap)
        pores_not_in_img = np.setdiff1d(
            Pores.loc[:, "index"], labels_img, assume_unique=True
        )
        print(pores_not_in_img)
        Pores = Pores[Pores.loc[:, "index"].isin(labels_img)].reset_index(drop=True)
        Pores.loc[:, "index"] = np.arange(1, len(Pores) + 1)
        Throats = Throats[
            ~(
                Throats.loc[:, "pore_1_index"].isin(pores_not_in_img)
                | Throats.loc[:, "pore_2_index"].isin(pores_not_in_img)
            )
        ].reset_index(drop=True)
        Throats.loc[:, ["pore_1_index", "pore_2_index"]] = remap(
            Throats.loc[:, ["pore_1_index", "pore_2_index"]].to_numpy(),
            labels_img,
            labels_remap,
        )
        # Throats.loc[:, 'index'] = Throats.index + 1
        Throats.loc[:, "index"] = np.arange(1, len(Throats) + 1)

        Pores1 = pd.concat(
            (Pores[Pore1_names].iloc[[-1]], Pores.loc[:, Pore1_names]),
            axis=0,
            ignore_index=True,
        )
        # add a temporary column to avoid openpnm column error
        Pores1["temp"] = 0
        Pores1.to_csv(
            Path_node1, header=False, index=False, sep=" ", float_format="%1.6E"
        )

        Pores2 = Pores.loc[:, Pore2_names]
        Pores2.to_csv(
            Path_node2, header=False, index=False, sep=" ", float_format="%1.6E"
        )
        Throats1 = pd.concat(
            (Throats[Throat1_names].iloc[[-1]], Throats.loc[:, Throat1_names]),
            axis=0,
            ignore_index=True,
        )
        Throats1.to_csv(
            Path_link1, header=False, index=False, sep=" ", float_format="%1.6E"
        )
        Throats2 = Throats.loc[:, Throat2_names]
        Throats2.to_csv(
            Path_link2, header=False, index=False, sep=" ", float_format="%1.6E"
        )
    return img


def buildq(variable_indices):
    """
    Builds the filterq matrix for the given variables.
    """
    num_variables = variable_indices.max() + 1
    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices,
        ((1, 1),) * variable_indices.ndim,
        mode="constant",
        constant_values=-1,
    )
    rows_cols_data = nb_buildq(variable_indices)
    filterq = sparse.coo_array(
        (
            rows_cols_data[2].astype(np.float64, copy=False),
            (rows_cols_data[0], rows_cols_data[1]),
        ),
        shape=(3 * num_variables, num_variables),
    ).tocsr()
    filterq = dot_product_mkl(filterq.T, filterq)
    return filterq


def _jacobi(
    filterq,
    x0: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    max_iters: int = 10,
    rel_tol: float = 1e-6,
    weight: float = 0.5,
):
    """Jacobi method with constraints."""

    jacobi_r = filterq.tolil()
    shp = jacobi_r.shape
    jacobi_d = 1.0 / filterq.diagonal()
    jacobi_r.setdiag((0,) * shp[0])
    jacobi_r = jacobi_r.tocsr()

    x = x0

    # We check the stopping criterion each 10 iterations
    check_each = 10
    cum_rel_tol = 1 - (1 - rel_tol) ** check_each

    energy_now = np.dot(x, dot_product_mkl(filterq, x)) / 2
    # logging.debug("Energy at iter %d: %.6g", 0, energy_now)
    for i in range(max_iters):
        x_1 = -jacobi_d * dot_product_mkl(jacobi_r, x)
        x = weight * x_1 + (1 - weight) * x

        # Constraints.
        x = np.maximum(x, lower_bound)
        x = np.minimum(x, upper_bound)

        # Stopping criterion
        if (i + 1) % check_each == 0:
            # Update energy
            energy_before = energy_now
            energy_now = np.dot(x, dot_product_mkl(filterq, x)) / 2

            # logging.debug("Energy at iter %d: %.6g", i + 1, energy_now)

            # Check stopping criterion
            cum_rel_improvement = (energy_before - energy_now) / energy_before
            if cum_rel_improvement < cum_rel_tol:
                break

    return x


def constrained_smooth(img, band_radius: int = 4, max_iters: int = 500, rel_tol=1e-6):
    """
    Implementation of the smoothing method from

    "Surface Extraction from Binary Volumes with Higher-Order Smoothness"
    Victor Lempitsky, CVPR10
    """

    # # Compute the distance map, the border and the band.
    binary_array = np.where(img > 0, True, False)

    # Compute the band and the border.
    dist_func = partial(edt)
    distance = np.where(
        binary_array, dist_func(binary_array) - 0.5, -dist_func(~binary_array) + 0.5
    )
    # border = np.abs(distance) < 1
    band = np.abs(distance) <= band_radius

    num_variables = np.count_nonzero(band)

    variable_indices = np.full_like(band, -1, dtype=np.int32)
    variable_indices[band] = np.arange(num_variables)
    # Compute filterq.
    filterq = buildq(variable_indices)
    # Initialize the variables.
    res = np.asarray(distance, dtype=np.double)
    x = res[band]
    upper_bound = np.where(x < 0, x, np.inf)
    lower_bound = np.where(x > 0, x, -np.inf)

    upper_bound[np.abs(upper_bound) < 1] = 0
    lower_bound[np.abs(lower_bound) < 1] = 0
    # Solve.
    x = _jacobi(
        filterq=filterq,
        x0=x,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_iters=max_iters,
        rel_tol=rel_tol,
    )

    res[band] = x
    return res


def calculate_surface_area_mk(image):
    return (
        mk_functionals(
            image,
            return_curvature=False,
            return_surface=True,
            return_volume=False,
            return_euler26=False,
        )
        * 8
    )


def calculate_surface_area_mc(image, level=0.5):
    verts, faces, normals, values = marching_cubes(image, level=level, step_size=1)
    # Fancy indexing to define two vector arrays from triangle vertices
    actual_verts = verts[faces]
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    # del actual_verts
    # Area of triangle in 3D = 1/2 * Euclidean norm of cross product
    return ((np.cross(a, b) ** 2).sum(axis=1) ** 0.5).sum() / 2.0


def calculate_surface_area_voxel(image):
    area = nb_compute_surface_area_voxel(image)
    return area


def calculate_surface_area_constrained_smooth(image):
    import mcubes

    # image = constrained_smooth(image)
    image_ = mcubes.smooth(image, method="constrained", max_iters=int(1e20))
    area = calculate_surface_area_mc(image_, level=0)
    return area


def calculate_boundary_area(mix_image, Path_dict):
    """
    use pnextract order of array
    """
    Csvs_dict = Path_dict["Csvs"]
    labels = np.arange(1, mix_image.max() + 1, dtype=np.int32)
    boundaries_areas = np.zeros((len(labels), 7), dtype=np.int32)
    boundaries_areas[:, 0] = labels
    slice_left = (slice(None), slice(None), slice(0))
    slice_right = (slice(None), slice(None), slice(-1))
    slice_front = (slice(None), slice(0), slice(None))
    slice_back = (slice(None), slice(-1), slice(None))
    slice_bottom = (slice(0), slice(None), slice(None))
    slice_top = (slice(-1), slice(None), slice(None))
    slices = (slice_left, slice_right, slice_front, slice_back, slice_bottom, slice_top)
    for i, slice_ in enumerate(slices):
        labels_i, counts_i = unique_uint_nonzero(mix_image[slice_], return_counts=True)
        boundaries_areas[:, i + 1][np.isin(labels, labels_i)] = counts_i
    boundaries_areas = pd.DataFrame(
        boundaries_areas, columns=["label", "x-", "x+", "y-", "y+", "z-", "z+"]
    )
    boundaries_areas.to_csv(Csvs_dict["boundaries_areas"])

    return boundaries_areas


def calculate_boundary_area2(mix_image):
    """
    use pnextract order of array
    """
    # Csvs_dict = Path_dict["Csvs"]
    labels = np.arange(1, mix_image.max() + 1, dtype=np.int32)
    boundaries_areas = np.zeros((len(labels), 7), dtype=np.int32)
    boundaries_areas[:, 0] = labels
    # shapes = np.array(((mix_image.shape[0],mix_image.shape[1]),
    #                    (mix_image.shape[0],mix_image.shape[1]),
    #                    (mix_image.shape[0],mix_image.shape[2]),
    #                    (mix_image.shape[0],mix_image.shape[2]),
    #                    (mix_image.shape[1],mix_image.shape[2]),
    #                    (mix_image.shape[1],mix_image.shape[2]),
    #                    ))
    # scales = ((shapes[:,0]-1)*(shapes[:,1]-1))/(shapes[:,0]*shapes[:,1])
    slices = [
        (slice(None), slice(None), 0),  # 左边界（X=0）
        (slice(None), slice(None), -1),  # 右边界（X=-1）
        (slice(None), 0, slice(None)),  # 前边界（Y=0）
        (slice(None), -1, slice(None)),  # 后边界（Y=-1）
        (0, slice(None), slice(None)),  # 底边界（Z=0）
        (-1, slice(None), slice(None)),  # 顶边界（Z=-1）
    ]
    for i, slice_ in enumerate(slices):
        labels_i, counts_i = unique_uint_nonzero(mix_image[slice_], return_counts=True)
        boundaries_areas[:, i + 1][np.isin(labels, labels_i, kind="table")] = (
            counts_i  # *scales[i]
        )
    boundaries_areas_dict = OrderedDict()
    boundaries_areas_dict["pore.surface_area_x-"] = boundaries_areas[:, 1]
    boundaries_areas_dict["pore.surface_area_x+"] = boundaries_areas[:, 2]
    boundaries_areas_dict["pore.surface_area_y-"] = boundaries_areas[:, 3]
    boundaries_areas_dict["pore.surface_area_y+"] = boundaries_areas[:, 4]
    boundaries_areas_dict["pore.surface_area_z-"] = boundaries_areas[:, 5]
    boundaries_areas_dict["pore.surface_area_z+"] = boundaries_areas[:, 6]

    return boundaries_areas_dict


def maximum_filter(image, footprint, mode="reflect"):
    if mode == "reflect":
        mode = "symmetric"
    image_padded = np.pad(
        image,
        tuple(
            (footprint.shape[i] // 2, footprint.shape[i] // 2)
            for i in range(image.ndim)
        ),
        mode=mode,
    )
    return nb_max_filter_non_padding(image_padded, footprint)


def binary_dilation(image, structure=None, iterations=1):
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        structure = np.asarray(structure, dtype=bool)
    image = np.asarray(image, dtype=bool)
    return nb_binary_dilation(image=image, structure=structure, iterations=iterations)


def binary_erosion(image, structure=None, iterations=1):
    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        structure = np.asarray(structure, dtype=bool)
    image = np.asarray(image, dtype=bool)
    return nb_binary_erosion(image=image, structure=structure, iterations=iterations)


def ps_round(r, ndim, smooth=True):
    r"""
    Creates round structuring element with the given radius and dimensionality

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    strel : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_round.html>`_
    to view online example.

    """
    rad = int(np.ceil(r))
    other = np.ones((2 * rad + 1,) * ndim, dtype=bool)
    other[(rad,) * ndim] = False
    if smooth:
        ball = edt(other) < r
    else:
        ball = edt(other) <= r
    return ball


def extend_slice(slices, shape, pad=1):
    r"""
    Adjust slice indices to include additional voxles around the slice.

    This function does bounds checking to ensure the indices don't extend
    outside the image.

    Parameters
    ----------
    slices : list of slice objects
         A list (or tuple) of N slice objects, where N is the number of
         dimensions in the image.
    shape : array_like
        The shape of the image into which the slice objects apply.  This is
        used to check the bounds to prevent indexing beyond the image.
    pad : int or list of ints
        The number of voxels to expand in each direction.

    Returns
    -------
    slices : list of slice objects
        A list slice of objects with the start and stop attributes respectively
        incremented and decremented by 1, without extending beyond the image
        boundaries.

    Examples
    --------
    >>> from scipy.ndimage import label, find_objects
    >>> from porespy.tools import extend_slice
    >>> im = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]])
    >>> labels = label(im)[0]
    >>> s = find_objects(labels)

    Using the slices returned by ``find_objects``, set the first label to 3

    >>> labels[s[0]] = 3
    >>> print(labels)
    [[3 0 0]
     [3 0 0]
     [0 0 2]]

    Next extend the slice, and use it to set the values to 4

    >>> s_ext = extend_slice(s[0], shape=im.shape, pad=1)
    >>> labels[s_ext] = 4
    >>> print(labels)
    [[4 4 0]
     [4 4 0]
     [4 4 2]]

    As can be seen by the location of the 4s, the slice was extended by 1, and
    also handled the extension beyond the boundary correctly.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/extend_slice.html>`_
    to view online example.

    """
    shape = np.asarray(shape)
    pad = np.asarray(pad, dtype=np.int64) * (shape > 0)
    res = tuple(
        slice(max(s.start - pad[i], 0), min(s.stop + pad[i], shape[i]), None)
        for i, s in enumerate(slices)
    )
    return res


def find_peaks(dt, r_max=4, strel=None, sigma=None):
    r"""
    Finds local maxima in the distance transform

    Parameters
    ----------
    dt : ndarray
        The distance transform of the pore space.  This may be calculated
        and filtered using any means desired.
    r_max : scalar
        The radius of the spherical element used in the maximum filter.
        This controls the localness of any maxima. The default is 4 voxels.
    strel : ndarray
        Instead of supplying ``r_max``, this argument allows a custom
        structuring element allowing control over both size and shape.
    sigma : float or list of floats
        If given, then a gaussian filter is applied to the distance transform
        using this value for the kernel
        (i.e. ``scipy.ndimage.gaussian_filter(dt, sigma)``)
    divs : int or array_like
        The number of times to divide the image for parallel processing.
        If ``1`` then parallel processing does not occur.  ``2`` is
        equivalent to ``[2, 2, 2]`` for a 3D image. The number of cores
        used is specified in ``porespy.settings.ncores`` and defaults to
        all cores.

    Returns
    -------
    image : ndarray
        An array of booleans with ``True`` values at the location of any
        local maxima.

    Notes
    -----
    It is also possible ot the ``peak_local_max`` function from the
    ``skimage.feature`` module as follows:

    ``peaks = peak_local_max(image=dt, min_distance=r, exclude_border=0,
    indices=False)``

    The *skimage* function automatically uses a square structuring element
    which is significantly faster than using a circular or spherical
    element.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_peaks.html>`_
    to view online example.

    """
    im = dt > 0
    if strel is None:
        strel = ps_round(r=r_max, ndim=im.ndim)
    if sigma is not None:
        dt = spim.gaussian_filter(dt, sigma=sigma)
    mx = maximum_filter(dt + 2.0 * (~im), footprint=strel, mode="constant")
    peaks = (dt == mx) * im
    return peaks


def trim_saddle_points(peaks, dt, maxiter=20):
    r"""
    Removes peaks that were mistakenly identified because they lied on a
    saddle or ridge in the distance transform that was not actually a true
    local peak.

    Parameters
    ----------
    peaks : ndarray
        A boolean image containing ``True`` values to mark peaks in the
        distance transform (``dt``)
    dt : ndarray
        The distance transform of the pore space for which the peaks
        are sought.
    maxiter : int
        The number of iteration to use when finding saddle points.
        The default value is 20.

    Returns
    -------
    image : ndarray
        An image with fewer peaks than the input image

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmentation".  Physical Review E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_saddle_points.html>`_
    to view online example.

    """
    new_peaks = np.zeros_like(peaks, dtype=bool)

    cube = np.ones((3,) * dt.ndim, dtype=bool)

    labels, N = spim.label(peaks > 0)
    slices = spim.find_objects(labels)
    for i, s in tqdm(enumerate(slices)):
        sx = extend_slice(s, shape=peaks.shape, pad=maxiter)
        peaks_i = labels[sx] == i + 1
        dt_i = dt[sx]
        im_i = dt_i > 0
        for it in range(maxiter):
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=cube)
            peaks_max = peaks_dil * np.max(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) & im_i
            if np.all(peaks_extended == peaks_i):
                new_peaks[sx] |= peaks_i
                break  # Found a true peak
            elif ~np.all(peaks_extended & peaks_i):
                break  # Found a saddle point
            peaks_i = peaks_extended

        # if iters >= maxiter:
        #     Warning(
        #         "Maximum number of iterations reached, consider "
        #         + "running again with a larger value of max_iters"
        #     )
    return new_peaks & peaks


def trim_nearby_peaks(peaks, dt, f=1):
    r"""
    Removes peaks that are nearer to another peak than to solid

    Parameters
    ----------
    peaks : ndarray
        A image containing nonzeros values indicating peaks in the distance
        transform (``dt``).  If ``peaks`` is boolean, a boolean is returned;
        if ``peaks`` have already been labelled, then the original labels
        are returned, missing the trimmed peaks.
    dt : ndarray
        The distance transform of the pore space
    f : scalar
        Controls how close peaks must be before they are considered near
        to each other. Sets of peaks are tagged as too near if
        ``d_neighbor < f * d_solid``.

    Returns
    -------
    image : ndarray
        An array the same size and type as ``peaks`` containing a subset of
        the peaks in the original image.

    Notes
    -----
    Each pair of peaks is considered simultaneously, so for a triplet of nearby
    peaks, each pair is considered.  This ensures that only the single peak
    that is furthest from the solid is kept.  No iteration is required.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmenation". Physical Review
    E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_nearby_peaks.html>`_
    to view online example.

    """
    cube = np.ones((3,) * dt.ndim, dtype=bool)

    labels, N = spim.label(peaks > 0, structure=cube)
    crds = spim.measurements.center_of_mass(
        peaks > 0, labels=labels, index=np.arange(1, N + 1)
    )
    crds = np.vstack(crds).astype(int)  # Convert to numpy array of ints
    L = dt[tuple(crds.T)]  # Get distance to solid for each peak
    # Add tiny amount to joggle points to avoid equal distances to solid
    # arange was added instead of random values so the results are repeatable
    L = L + np.arange(len(L)) * 1e-6

    tree = sptl.cKDTree(data=crds)
    # Find list of nearest peak to each peak
    temp = tree.query(x=crds, k=2, workers=-1)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory
    hits = (dist_to_neighbor <= f * L).nonzero()[0]
    # Drop peak that is closer to the solid than it's neighbor

    drop_peaks = np.where(
        L[hits] < L[nearest_neighbor[hits]], hits, nearest_neighbor[hits]
    )
    drop_peaks = np.unique(drop_peaks)
    new_peaks = ~np.isin(labels, drop_peaks + 1, kind="table") * peaks
    return new_peaks


def snow_image_segmentation(
    image: np.ndarray,
    target_value: int = 1,
    sigma: float = 0.4,
    r_max: float = None,
    unsegmented_value: int = -1,
):
    """ """
    image = image == target_value
    image_inversed = ~image
    dt_original = edt(image)
    dt_r_mean = np.mean(dt_original[image])
    dt_soomthed = spim.gaussian_filter(input=dt_original, sigma=sigma)
    dt_soomthed[image_inversed] = 0.0
    if r_max is None:
        r_max = np.round(dt_r_mean + 0.5)
    peaks = find_peaks(dt=dt_soomthed, r_max=r_max)
    peaks = trim_saddle_points(peaks=peaks, dt=dt_soomthed)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt_original)
    peaks = spim.label(peaks)[0]
    image_VElems = watershed(
        -dt_soomthed, markers=peaks, mask=image
    )  # Watershed algarithm

    labels = unique_uint_nonzero(image_VElems)
    if np.array_equal(labels, np.arange(1, np.max(labels) + 1, dtype=np.int32)):
        pass
        # print("SNOW EXTRACTION: all labels are continuous, no need to relabel")
    else:
        print("SNOW EXTRACTION: Labels are not continuous, relabelling the labels")
        image_VElems = remap(labels, np.arange(1, labels.size + 1, dtype=np.int32))
    image_VElems[image & (image_VElems == 0)] = unsegmented_value
    image_VElems = image_VElems.astype(np.int32, copy=False)
    return image_VElems


def pnextract(
    image,
    resolution=1.0,
    target_value=0,
    n_workers=1,
    verbose=False,
    cut_image=False,
    check_result=False,
    config_dict=None,
):
    import pypne

    image = (image != target_value).astype(np.uint8, copy=False)
    image_VElems, pn_pne = pypne.pnextract(
        image,
        resolution,
        config_settings=config_dict,
        n_workers=n_workers,
        verbose=verbose,
    )
    image_VElems -= 1  # start from 0
    if cut_image:
        image_VElems = image_VElems[1:-1, 1:-1, 1:-1]
    if check_result:
        image_VElems, pn_pne = check_pypne_result(image_VElems, pn_pne)
    return image_VElems, pn_pne


def check_pypne_result(image_VElems, pn_pne):
    labels_pn = pn_pne["pore._id"]
    labels_image = unique_uint_nonzero(image_VElems)
    labels_remap = np.arange(1, len(labels_image) + 1, dtype=np.int32)

    labels_missing = np.setdiff1d(labels_pn, labels_image, assume_unique=True)
    if labels_missing.size > 0:
        pores_missing = labels_missing - 1
        throats_missing = np.where(
            np.isin(pn_pne["throat.pore_1_index"] - 1, pores_missing)
            | np.isin(pn_pne["throat.pore_2_index"] - 1, pores_missing)
        )[0]
        print(f"Missing labels: {labels_missing}")
        # print(f"Missing pores: {pores_missing}")
        print(f"Missing throats: {throats_missing}")
        for key in pn_pne.keys():
            if key.startswith("pore."):
                pn_pne[key] = np.delete(pn_pne[key], pores_missing, axis=0)
            if key.startswith("throat."):
                pn_pne[key] = np.delete(pn_pne[key], throats_missing, axis=0)
        pn_pne["pore._id"] = labels_remap
        pn_pne["throat.pore_1_index"] = remap(
            pn_pne["throat.pore_1_index"], labels_image, labels_remap
        )
        pn_pne["throat.pore_2_index"] = remap(
            pn_pne["throat.pore_2_index"], labels_image, labels_remap
        )
        image_VElems = remap(image_VElems, labels_image, labels_remap).astype(
            np.int32, copy=False
        )

    return image_VElems, pn_pne


def get_objects_volume(labeled_image):
    """
    labeled_image: 3D labeled image, should be continuous and start from 1, 0 is background
    """
    labels = unique_uint_nonzero(labeled_image)
    label_min = labels.min()
    label_max = labels.max()
    assert (
        np.array_equal(labels, np.arange(label_min, label_max + 1, dtype=np.int32))
        and label_min == 1
    ), "Labels are not continuous or not start from 1"

    z_min, z_max, y_min, y_max, x_min, x_max, volume = nb_get_objects_volume(
        labeled_image, label_max=label_max
    )
    phase_props_table = np.column_stack(
        (
            z_min,
            y_min,
            x_min,
            z_max,
            y_max,
            x_max,
            volume,
        )
    ).astype(np.int32, copy=False)
    return phase_props_table


# calculate_surface_area_extraction = calculate_surface_area_voxel
calculate_surface_area_extraction = calculate_surface_area_mc
# calculate_surface_area_extraction = calculate_surface_area_mk


def calculate_pore_properties_from_image(
    label, labeled_image, phase_props_table, offset=3, pad=2
):
    """
    label,pore.z,pore.y,pore.x,pore.radius,pore.surface_area,pore.volume
    """
    result = np.empty(7, dtype=np.float32)

    zyx_min_max = get_label_box(label, phase_props_table, offset=offset)
    zyx_min = zyx_min_max[0]
    zyx_max = zyx_min_max[1]
    region = labeled_image[
        zyx_min[0] : zyx_max[0], zyx_min[1] : zyx_max[1], zyx_min[2] : zyx_max[2]
    ]

    region_bool_i = region == label
    dt_i = edt(region_bool_i, black_border=False)
    center_index = np.unravel_index(dt_i.argmax(), dt_i.shape)
    radius_i = dt_i[center_index[0], center_index[1], center_index[2]]
    center_index += zyx_min
    region_bool_i = np.pad(
        region_bool_i,
        ((pad, pad), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=False,
    )

    # Volume i
    volume_i = phase_props_table[label, 6]
    # Surface area i
    area_i = calculate_surface_area_extraction(region_bool_i)
    result[0] = label
    result[1:4] = center_index[::-1] + 0.5
    result[4] = radius_i
    result[5] = area_i
    result[6] = volume_i
    return result


##### calculate_throat_properties_from_image #####
dilate_structure = np.array(
    (
        ((False, False, False), (False, True, False), (False, False, False)),
        ((False, True, False), (True, True, True), (False, True, False)),
        ((False, False, False), (False, True, False), (False, False, False)),
    )
)


def calculate_throat_properties_from_image(
    label, labeled_image, phase_props_table, offset=3, pad=2
):
    zyx_min_max = get_label_box(label, phase_props_table, offset=offset)
    zyx_min = zyx_min_max[0]
    zyx_max = zyx_min_max[1]
    region = labeled_image[
        zyx_min[0] : zyx_max[0], zyx_min[1] : zyx_max[1], zyx_min[2] : zyx_max[2]
    ]
    region = np.pad(
        region, ((pad, pad), (pad, pad), (pad, pad)), mode="constant", constant_values=0
    )
    region_bool_i = region == label
    area_i = calculate_surface_area_extraction(region_bool_i)
    # structure = np.ones((3,) * region_bool_i.ndim, dtype=bool)
    region_dilated_bool_i = spim.binary_dilation(
        region_bool_i, structure=dilate_structure
    )
    neighbor_labels = np.setdiff1d(
        unique_uint_nonzero(region[region_dilated_bool_i]), label, assume_unique=True
    )
    num_neighbors = neighbor_labels.size
    if num_neighbors == 0:
        return np.array((label, -1, -1), dtype=np.int32)

    table = np.empty((num_neighbors, 3), dtype=np.float32)
    for i, neighbor_label in enumerate(neighbor_labels):
        area_nei = calculate_surface_area_extraction(region == neighbor_label)
        area_total = calculate_surface_area_extraction(
            np.isin(region, (label, neighbor_label), kind="table")
        )
        table[i, 0] = label
        table[i, 1] = neighbor_label
        table[i, 2] = (area_i + area_nei - area_total) / 2
    table = table[table[:, 2] > 0]
    if table.size == 0:
        return np.array((label, -1, -1), dtype=np.int32)
    else:
        return table


###############


def extract_from_image(
    labeled_image,
    resolution,
    phase_props_table=None,
    n_workers=1,
    backend="loky",
    seps=None,
):
    """
    labeled_image: 3D labeled image, should be continuous and start from 1, 0 is background
    seps : list of ints,start of each phase, if None, only one phase is assumed
    eg : 1-20 is phase1, 21-40 is phase2, 41-60 is phase3
    seps = [1,21,41,61]
    """

    labels = unique_uint_nonzero(labeled_image)
    label_max = labels.max() + 1
    if np.all(labels != np.arange(1, label_max)):
        raise ValueError("Labels are not continuous or not start from 1")
    if seps is None:
        seps = [1, label_max]
    if phase_props_table is None:
        phase_props_table = get_objects_volume(labeled_image)
    pore_props = np.vstack(
        Parallel(n_jobs=n_workers, backend=backend)(
            delayed(calculate_pore_properties_from_image)(
                label, labeled_image, phase_props_table
            )
            for label in tqdm(labels)
        )
    )
    throat_props = np.vstack(
        Parallel(n_jobs=n_workers, backend=backend)(
            delayed(calculate_throat_properties_from_image)(
                label, labeled_image, phase_props_table
            )
            for label in tqdm(labels)
        )
    )
    throat_props = throat_props[throat_props[:, 2] > 0]
    throat_props[:, 0:2] = np.sort(throat_props[:, 0:2], axis=1)
    connections_unique, connections_indices, connections_count = unique_rows(
        np.ascontiguousarray(throat_props[:, 0:2], dtype=np.int32),
        keepdims=False,
        return_inverse=True,
        return_counts=True,
    )
    contact_area_average = (
        np.bincount(connections_indices, weights=throat_props[:, 2]) / connections_count
    )
    throat_props = np.column_stack((connections_unique, contact_area_average))
    throat_props = throat_props[np.lexsort((throat_props[:, 1], throat_props[:, 0]))]

    pore_props_dict = {
        "pore._id": (pore_props[:, 0] - 1).astype(np.int32, copy=False),
        "pore.coords": pore_props[:, 1:4],
        "pore.radius": pore_props[:, 4],
        "pore.area": pore_props[:, 5],
        "pore.volume": pore_props[:, 6],
    }
    throat_props_dict = {
        "throat.conns": (throat_props[:, 0:2] - 1).astype(np.int32, copy=False),
        "throat.area": throat_props[:, 2],
    }
    boundaries_areas_dict = calculate_boundary_area2(labeled_image)

    multi_net = {}
    multi_net["pore._id"] = pore_props_dict["pore._id"]
    num_pore_multi_net = multi_net["pore._id"].size
    multi_net["pore.label"] = multi_net["pore._id"].copy()
    multi_net["pore.all"] = np.ones(num_pore_multi_net, dtype=bool)
    multi_net["pore.shape_factor"] = np.ones(num_pore_multi_net, dtype=np.float32)
    multi_net["pore.real_shape_factor"] = multi_net["pore.shape_factor"].copy()

    multi_net["pore.coords"] = pore_props_dict["pore.coords"] * resolution
    multi_net["pore.radius"] = pore_props_dict["pore.radius"] * resolution
    multi_net["pore.area"] = pore_props_dict["pore.area"] * resolution**2
    multi_net["pore.volume"] = pore_props_dict["pore.volume"] * resolution**3
    multi_net["pore.void"] = ~multi_net["pore.all"]
    multi_net["pore.solid"] = ~multi_net["pore.all"]

    # add boundary areas
    multi_net["pore.surface_all"] = np.zeros(num_pore_multi_net, dtype=bool)
    for i, (k, v) in enumerate(boundaries_areas_dict.items()):
        multi_net[f"pore.surface_{dirs[i]}"] = v.astype(bool)
        multi_net[f"pore.surface_area_{dirs[i]}"] = v * resolution**2
        multi_net["pore.surface_all"] = (
            multi_net["pore.surface_all"] | multi_net[f"pore.surface_{dirs[i]}"]
        )

    multi_net["throat.conns"] = throat_props_dict["throat.conns"]
    multi_net["throat._id"] = np.arange(multi_net["throat.conns"].shape[0])
    multi_net["throat.label"] = multi_net["throat._id"].copy()
    num_throat_multi_net = multi_net["throat.conns"].shape[0]
    multi_net["throat.all"] = np.ones(num_throat_multi_net, dtype=bool)
    multi_net["throat.area"] = throat_props_dict["throat.area"] * resolution**2
    multi_net["throat.solid"] = ~multi_net["throat.all"]
    multi_net["throat.void"] = ~multi_net["throat.all"]
    multi_net["throat.connect"] = ~multi_net["throat.all"]
    multi_net["throat.length"] = np.linalg.norm(
        multi_net["pore.coords"][multi_net["throat.conns"][:, 0]]
        - multi_net["pore.coords"][multi_net["throat.conns"][:, 1]],
        axis=1,
    )
    multi_net["throat.total_length"] = multi_net["throat.length"].copy()
    multi_net["throat.radius"] = np.sqrt(multi_net["throat.area"] / np.pi)
    multi_net["throat.shape_factor"] = np.ones(num_throat_multi_net, dtype=np.float32)
    multi_net["throat.real_shape_factor"] = multi_net["throat.shape_factor"].copy()
    nets = []
    nets.append(multi_net)
    if len(seps) == 2:
        return nets

    if min(seps) < 1 or max(seps) > label_max:
        raise ValueError("seps should be within [1, MAX_LABEL+1]")

    for i in range(len(seps) - 1):
        # multi_net = multi_net.copy()
        sep_start = seps[i]
        sep_end = seps[i + 1]
        phase_net = {}
        sep_start_net = sep_start - 1
        sep_end_net = sep_end - 1
        pore__id_o = np.arange(sep_start_net, sep_end_net)
        num_pore = sep_end_net - sep_start_net
        phase_net["pore._id"] = np.arange(num_pore)
        num_pore = phase_net["pore._id"].size
        phase_net["pore.label"] = phase_net["pore._id"].copy()
        phase_net["pore.all"] = multi_net["pore.all"][sep_start_net:sep_end_net]
        phase_net["pore.shape_factor"] = multi_net["pore.shape_factor"][
            sep_start_net:sep_end_net
        ]
        phase_net["pore.real_shape_factor"] = multi_net["pore.real_shape_factor"][
            sep_start_net:sep_end_net
        ]
        phase_net["pore.coords"] = multi_net["pore.coords"][sep_start_net:sep_end_net]
        phase_net["pore.radius"] = multi_net["pore.radius"][sep_start_net:sep_end_net]
        phase_net["pore.area"] = multi_net["pore.area"][sep_start_net:sep_end_net]
        phase_net["pore.volume"] = multi_net["pore.volume"][sep_start_net:sep_end_net]
        phase_net["pore.void"] = ~phase_net["pore.all"]
        phase_net["pore.solid"] = ~phase_net["pore.all"]
        get_surface_from_mpn(
            mpn_0=phase_net,
            mpn_1=multi_net,
            dirs=dirs,
            start=sep_start_net,
            end=sep_end_net,
        )
        phase_throat_bool = np.all(
            np.isin(multi_net["throat.conns"], pore__id_o, kind="table"),
            axis=1,
        )
        phase_net["throat.conns"] = multi_net["throat.conns"][phase_throat_bool]
        phase_net["throat.conns"] = (phase_net["throat.conns"] - sep_start_net).astype(
            np.int32, copy=False
        )

        num_throat = phase_net["throat.conns"].shape[0]
        phase_net["throat.all"] = multi_net["throat.all"][phase_throat_bool]
        phase_net["throat._id"] = np.arange(num_throat)
        phase_net["throat.label"] = phase_net["throat._id"].copy()
        phase_net["throat.area"] = multi_net["throat.area"][phase_throat_bool]
        phase_net["throat.solid"] = ~phase_net["throat.all"]
        phase_net["throat.void"] = ~phase_net["throat.all"]
        phase_net["throat.connect"] = ~phase_net["throat.all"]
        phase_net["throat.length"] = multi_net["throat.length"][phase_throat_bool]
        phase_net["throat.total_length"] = multi_net["throat.total_length"][
            phase_throat_bool
        ]
        phase_net["throat.radius"] = multi_net["throat.radius"][phase_throat_bool]
        phase_net["throat.shape_factor"] = multi_net["throat.shape_factor"][
            phase_throat_bool
        ]
        phase_net["throat.real_shape_factor"] = multi_net["throat.real_shape_factor"][
            phase_throat_bool
        ]
        nets.append(phase_net)

    return nets


def get_label_box(i, phase_props_table, offset=1):
    """
    phase_props_table should be contiuous and start from 1
    """

    index = np.empty((2, 3), dtype=np.int32)
    index[0] = np.maximum(phase_props_table[i, 0:3] - offset, 0)
    index[1] = phase_props_table[i, 3:6] + offset
    return index


def multi_phase_segmentation(image, config_list, n_workers=1, backend="loky"):
    pnextract_partial = partial(
        pnextract,
        image=image,
        cut_image=True,
        check_result=True,
    )
    snow_image_segmentation_partial = partial(
        snow_image_segmentation,
        image=image,
        unsegmented_value=-1,
    )
    segmentation_funcs = []
    segmentation_args = []
    num_config = len(config_list)
    for i, config in enumerate(config_list):
        if "method" not in config:
            raise ValueError(
                "method is not provided in the config_dict, valid values are pne and snow"
            )
        elif config["method"] == "pne":
            segmentation_funcs.append(pnextract_partial)
            segmentation_args.append(
                {
                    "target_value": config["target_value"],
                    "resolution": config["resolution"],
                    "config_dict": config.get("config_pne"),
                    "n_workers": config.get("n_workers", 1),
                }
            )
        elif config["method"] == "snow":
            segmentation_funcs.append(snow_image_segmentation_partial)
            segmentation_args.append(
                {
                    "target_value": config["target_value"],
                }
            )
    res_extracted = Parallel(n_jobs=n_workers, backend=backend)(
        delayed(func)(**args)
        for func, args in zip(segmentation_funcs, segmentation_args)
    )
    pns = [
        None,
    ] * num_config
    images_labeled = [
        None,
    ] * num_config
    seps = [
        None,
    ] * num_config
    for i, res in enumerate(res_extracted):
        if config_list[i]["method"] == "pne":
            pns[i] = res[1]
            images_labeled[i] = res[0]
        elif config_list[i]["method"] == "snow":
            images_labeled[i] = res

    # if fill_unlabeled:
    #     for i, labeled_image in enumerate(images_labeled):
    #         label_max = np.max(labeled_image)
    #         target_value = config_list[i]["target_value"]
    #         image_seg = labeled_image > 0
    #         image_unseg = (~image_seg) & (image == target_value)
    #         unseg_labeled = label(image_unseg, connectivity=2, background=False)
    #         phase_props_table = get_phase_props_table(unseg_labeled)
    #         pores_added = 0
    #         for j, label_unseg in enumerate(phase_props_table[:, 0]):
    #             zyx_min_max = get_label_box(label_unseg, phase_props_table)
    #             zyx_min = zyx_min_max[0]
    #             zyx_max = zyx_min_max[1]

    #             region_seg = labeled_image[
    #                 zyx_min[0] : zyx_max[0],
    #                 zyx_min[1] : zyx_max[1],
    #                 zyx_min[2] : zyx_max[2],
    #             ]
    #             region_unseg_bool = (
    #                 unseg_labeled[
    #                     zyx_min[0] : zyx_max[0],
    #                     zyx_min[1] : zyx_max[1],
    #                     zyx_min[2] : zyx_max[2],
    #                 ]
    #                 == label_unseg
    #             )

    #             # structure = np.array(
    #             #     [
    #             #         [
    #             #             [False, False, False],
    #             #             [False, True, False],
    #             #             [False, False, False],
    #             #         ],
    #             #         [
    #             #             [False, True, False],
    #             #             [True, True, True],
    #             #             [False, True, False],
    #             #         ],
    #             #         [
    #             #             [False, False, False],
    #             #             [False, True, False],
    #             #             [False, False, False],
    #             #         ],
    #             #     ]
    #             # )
    #             structure = np.ones((3, 3, 3), dtype=bool)
    #             region_unseg_dilated_bool = spim.binary_dilation(
    #                 region_unseg_bool, structure=structure
    #             )
    #             neighbor_labels = region_seg[region_unseg_dilated_bool]

    #             z, y, x = np.where(region_unseg_bool)
    #             z += zyx_min[0]
    #             y += zyx_min[1]
    #             x += zyx_min[2]
    #             neighbor_labels = neighbor_labels[neighbor_labels > 0]
    #             if neighbor_labels.size > 0:
    #                 neighbor_most = np.argmax(np.bincount(neighbor_labels))

    #                 labeled_image[z, y, x] = neighbor_most

    #             else:
    #                 pass
    #         #         labeled_image[z, y, x] = label_max + 1 + pores_added
    #         #         pores_added += 1
    #         # print(f"Added {pores_added} pores to target value {target_value}")

    current_phase_max = 0
    image_mixed = np.zeros(image.shape, dtype=np.int32)
    for i, labeled_image in enumerate(images_labeled):
        image_mixed = (
            np.where(labeled_image <= 0, 0, labeled_image + current_phase_max)
            + image_mixed
        )
        current_phase_max = np.max(image_mixed).astype(np.int32, copy=False)
        seps[i] = current_phase_max + 1
    seps = [1, *seps]
    image_mixed = image_mixed.astype(np.int32, copy=False)
    return image_mixed, pns, seps


def dualn_phase_extraction(
    image,
    resolution,
    config_list,
    n_workers_segmentation=1,
    n_workers_extraction=1,
    backend="loky",
):
    image_mix, nets, seps = multi_phase_extraction(
        image=image,
        resolution=resolution,
        config_list=config_list,
        # fill_unlabeled=fill_unlabeled,
        n_workers_segmentation=n_workers_segmentation,
        n_workers_extraction=n_workers_extraction,
        backend=backend,
    )
    """
    first config is void, second config is solid
    See multi_phase_extraction
    """
    dualn = nets[0]
    pn = nets[1]
    sn = nets[2]
    num_pn_pore = pn["pore._id"].size

    ## Void Solid Properties
    pn["pore.void"] = pn["pore.all"].copy()
    pn["pore.solid"] = ~pn["pore.all"]
    pn["throat.void"] = pn["throat.all"].copy()
    pn["throat.solid"] = ~pn["throat.all"]

    sn["pore.solid"] = sn["pore.all"].copy()
    sn["pore.void"] = ~sn["pore.all"]
    sn["throat.solid"] = sn["throat.all"].copy()
    sn["throat.void"] = ~sn["throat.all"]

    ## Dual Properties
    dualn["pore.void"][:num_pn_pore] = True
    dualn["pore.solid"][num_pn_pore:] = True
    dualn["throat.void"] = np.all(dualn["throat.conns"] < num_pn_pore, axis=1)
    dualn["throat.solid"] = np.all(dualn["throat.conns"] >= num_pn_pore, axis=1)
    dualn["throat.connect"] = ~(dualn["throat.void"] | dualn["throat.solid"])
    return image_mix, [dualn, pn, sn], seps


def get_surface_from_mpn(mpn_0, mpn_1, dirs, start, end):
    mpn_0["pore.surface_all"] = mpn_1["pore.surface_all"][start:end]
    for i in range(len(dirs)):
        mpn_0[f"pore.surface_{dirs[i]}"] = mpn_1[f"pore.surface_{dirs[i]}"][start:end]
        mpn_0[f"pore.surface_area_{dirs[i]}"] = mpn_1[f"pore.surface_area_{dirs[i]}"][
            start:end
        ]
    return mpn_0


def multi_phase_extraction(
    image,
    resolution,
    config_list,
    n_workers_segmentation=1,
    n_workers_extraction=1,
    backend="loky",
):
    """
    config_list : a list of configs
    in each config, there are snow and pne keys
    if method is pne
    config = {
        method: "pne",
        target_value: value to extract,
        config_pne: see pypne's documentation for details (default None)
        pn_mode : Optional[str] "image","origin","origin_image" (default "image")
            "image" : pores and throats are extracted from image, without using pnextracted classic pore network
            "origin" : using classic pore network
            "origin_image" : using classic pore network throat.conns, adjusting pores' properties and throats' properties according to image
    }
    if method is snow
    config = {
        method: "snow",
        target_value: value to extract,
        sigma: float (default 0.4),
        r_max: float (default None),
        unsegmented_value: int (default -1)
    }
    """
    pne_config_keys_required = {"target_value"}
    snow_config_keys_required = {"target_value"}
    # check config_list
    for i, config in enumerate(config_list):
        if "method" not in config:
            raise ValueError(
                "method is not provided in the config_dict, valid values are pne and snow"
            )
        elif config["method"] == "pne":
            keys_lack = pne_config_keys_required - set(config.keys())
            if config.get("resolution", None) is not None:
                print(
                    "Warning:resolution of pne is not used in multi_phase_extraction, using resolution of function argument"
                )
            config["resolution"] = resolution

            if keys_lack:
                raise ValueError(f"config_dict[{i}] lacks keys {keys_lack}")
        elif config["method"] == "snow":
            keys_lack = snow_config_keys_required - set(config.keys())
            if keys_lack:
                raise ValueError(f"config_dict[{i}] lacks keys {keys_lack}")

    image_mixed, pns_pne, seps = multi_phase_segmentation(
        image=image,
        config_list=config_list,
        n_workers=n_workers_segmentation,
        backend=backend,
    )
    import time

    t0 = time.time()
    nets = extract_from_image(
        labeled_image=image_mixed,
        resolution=resolution,
        phase_props_table=None,
        seps=seps,
        n_workers=n_workers_extraction,
        backend=backend,
    )
    net_mixed = nets[0]
    for i, config in enumerate(config_list):
        if config["method"] == "pne":
            pn_mode = config.get("pn_mode", "image")
            if pn_mode == "image":
                pass
            else:
                # origin pn
                net_pne = read_pypne(pns_pne[i])
                pore__id = net_pne["pore._id"]
                pore_start = seps[i] - 1
                pore__id += pore_start
                sep_start_net, sep_end_net = np.min(pore__id), np.max(pore__id) + 1
                get_surface_from_mpn(
                    mpn_0=net_pne,
                    mpn_1=net_mixed,
                    dirs=dirs,
                    start=sep_start_net,
                    end=sep_end_net,
                )
                throat_bool = ~np.all(
                    np.isin(net_mixed["throat.conns"], pore__id, kind="table"),
                    axis=1,
                )
                for key in net_mixed.keys():
                    net_mixed[key] = net_mixed[key][throat_bool]

                if pn_mode == "origin":
                    for key in net_mixed.keys():
                        if key.startswith("pore."):
                            net_mixed[key][pore__id] = net_pne[key]
                        if key.startswith("throat."):
                            net_mixed[key] = np.concatenate(
                                (net_mixed[key], net_pne[key]), axis=0
                            )

                elif pn_mode == "origin_image":
                    throat_map = find_throat_conns_map(
                        net_pne["throat.conns"], net_mixed["throat.conns"]
                    )
                    throat_0 = throat_map[:, 0]
                    throat_1 = throat_map[:, 1]
                    for key in net_pne.keys():
                        if key.startswith("pore."):
                            net_pne[key] = net_mixed[key][pore__id]
                        if key.startswith("throat."):
                            net_pne[key][throat_0] = net_mixed[key][throat_1]
                    for key in net_mixed.keys():
                        if key.startswith("pore."):
                            net_mixed[key][pore__id] = net_pne[key]
                        if key.startswith("throat."):
                            net_mixed[key] = np.concatenate(
                                (net_mixed[key], net_pne[key]), axis=0
                            )
                net_mixed["throat.conns"] = np.sort(net_mixed["throat.conns"], axis=1)
                net_pne["throat.conns"] = np.sort(net_pne["throat.conns"], axis=1)
                if len(seps) == 1:
                    nets[0] = net_pne
                else:
                    nets[0] = net_mixed
                    nets[i + 1] = net_pne
    return image_mixed, nets, seps
