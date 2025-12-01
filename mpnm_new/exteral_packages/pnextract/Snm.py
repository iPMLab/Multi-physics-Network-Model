#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:09:04 2021

@author: mingliangqu
"""

import numpy as np
from skimage.segmentation import watershed
import time
from edt import edt
import scipy.ndimage as spim
from tqdm import tqdm
import scipy.spatial as sptl


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
    mx = spim.maximum_filter(dt + 2.0 * (~im), footprint=strel)
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
            elif np.all(peaks_extended & peaks_i) == False:
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
    hits = np.where(dist_to_neighbor <= f * L)[0]
    # Drop peak that is closer to the solid than it's neighbor

    drop_peaks = np.where(
        L[hits] < L[nearest_neighbor[hits]], hits, nearest_neighbor[hits]
    )
    drop_peaks = np.unique(drop_peaks)
    new_peaks = ~np.isin(labels, drop_peaks + 1, kind="table") * peaks
    return new_peaks


def solid_network_extraction(
    Path_dict,
    solid_value=1,
    resolution=None,
    size=None,
    sigma=None,
    r_max=None,
    n_workers_dt=10,
):
    print("\nsolid extraction starting\n")
    t0 = time.time()

    Images_dict = Path_dict["Images"]
    image = np.fromfile(Path_dict["Path_input"], dtype=np.uint8).reshape(
        size[2], size[1], size[0]
    )
    image = np.where(image == solid_value, True, False)
    image_inverse = ~image
    if Images_dict.get("solid") is not None:
        image.astype(np.uint8).tofile(Images_dict["solid"])

    dt_o = edt(image, parallel=-1)  # distance_map
    dt_r_ave = np.mean(dt_o[image])
    if sigma is None:
        sigma = 0.4
    dt_s = spim.gaussian_filter(input=dt_o, sigma=sigma)
    dt_s[image_inverse] = 0.0
    if r_max is None:
        r_max = np.round(dt_r_ave + 0.5)
    peaks = find_peaks(dt=dt_s, r_max=r_max)
    peaks = trim_saddle_points(peaks=peaks, dt=dt_s)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt_o)
    peaks, N = spim.label(peaks)
    array_solid_VElems = watershed(
        -dt_s, markers=peaks, mask=image
    )  # Watershed algarithm
    """
    Solid pixels which are too small is -1
    array_solid_solid only include segmented solid pixels (equals to 1), excluding small solid pixels that are ignored (equals to 0)
    """
    array_solid_VElems = np.where(
        (image == 1) & (array_solid_VElems == 0), -1, array_solid_VElems
    )
    if Images_dict.get("solid_VElems") is not None:
        array_solid_VElems.astype(np.int32, copy=False).tofile(
            Images_dict["solid_VElems"]
        )

    # for i in range(len(values)):
    #     print(values[i], counts[i])
    if Images_dict.get("solid_solid") is not None:
        array_solid_solid = np.where(array_solid_VElems < 1, 0, array_solid_VElems)
        array_solid_solid.astype(np.int32, copy=False).tofile(
            Images_dict["solid_solid"]
        )
    tend = time.time()
    print(
        "\n========================================\n\
finish solid network extraction\ntime cost：%.6fs\n\
========================================"
        % (tend - t0)
    )
    return array_solid_VElems.astype(np.int32, copy=False)


if __name__ == "__main__":
    resolution = 0.025 / 300
    size = [300, 300, 300]
    path = "/2/Documents/yjp/image_300_final/"
    file_name = "image_300_300_300"
    path_solid_network = path + "/solid_network/"
    path_images = path + "/images/"
    resolution = 0.025 / 300
    n_workers_dt = 10
    solid_network_extraction(
        path, path_solid_network, path_images, file_name, resolution, size, n_workers_dt
    )
