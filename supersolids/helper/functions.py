#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Functions for Potential and initial wave function :math:`\psi_0`

"""

import functools
import itertools
import sys

import numpy as np
from scipy.integrate import quad
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.special import jv
from scipy import stats, ndimage
from typing import Tuple, Callable, Optional, List

from supersolids.helper import constants, get_version
cp, cuda_used = get_version.check_cupy_used(np)

from supersolids.helper.run_time import run_time
from supersolids.helper.Box import Box
from supersolids.helper.Resolution import Resolution


def check_ResBox(Res: Resolution, MyBox: Box):
    assert isinstance(Res, Resolution), f"Res: {type(Res)} is not type {type(Resolution)}"
    assert isinstance(MyBox, Box), f"box: {type(MyBox)} is not type {type(Box)}"
    assert MyBox.dim == Res.dim, (f"Dimension of Box ({MyBox.dim}) and "
                                  f"Res ({Res.dim}) needs to be equal.")

    if Res.dim > 3:
        sys.exit("Spatial dimension over 3. This is not implemented.")

    return Res, MyBox


def get_grid_helper(Res: Resolution, MyBox: Box, index: int):
    try:
        x0, x1 = MyBox.get_bounds_by_index(index)
        res = Res.get_bounds_by_index(index)
        box_len = x1 - x0
        x: cp.ndarray = cp.linspace(x0, x1, res, endpoint=False)
        dx: float = box_len / float(res - 1)
        dkx: float = 2.0 * np.pi / box_len
        kx: cp.ndarray = cp.fft.fftfreq(res, d=1.0 / (dkx * res))

    except KeyError:
        sys.exit(
            f"Keys x0, x1 of box needed, "
            f"but it has the keys: {MyBox.keys()}, "
            f"Key x of res needed, "
            f"but it has the keys: {Res.keys()}")

    return x, dx, kx, dkx


def get_grid(Res: Resolution, MyBox: Box):
    x0, x1 = MyBox.get_bounds_by_index(0)
    res_x = Res.get_bounds_by_index(0)
    y0, y1 = MyBox.get_bounds_by_index(1)
    res_y = Res.get_bounds_by_index(1)
    z0, z1 = MyBox.get_bounds_by_index(2)
    res_z = Res.get_bounds_by_index(2)

    try:
        x_mesh, y_mesh, z_mesh = cp.mgrid[x0: x1: complex(0, res_x),
                                          y0: y1: complex(0, res_y),
                                          z0: z1: complex(0, res_z)
                                          ]
    except KeyError:
        sys.exit(
            f"Keys x0, x1, y0, y1, z0, z1 of box needed, "
            f"but it has the keys: {MyBox.keys()}, "
            f"Keys x, y, z of res needed, "
            f"but it has the keys: {Res.keys()}")

    return x_mesh, y_mesh, z_mesh


def BoxResAssert(Res, Box):
    assert len(Res) <= 3, "Dimension of Res needs to be smaller than 3."
    assert len(Box) <= 6, ("Dimension of Box needs to be smaller than 6, "
                           "as the maximum dimension of the problem is 3.")
    assert len(Box) == 2 * len(Res), (
        f"Dimension of Box is {len(Box)}, but needs "
        f"to be 2 times higher than of Res, "
        f"which currently is {len(Res)}.")


def aResAssert(Res, a):
    assert len(a) == len(Res), (
        f"Dimension of Amplitudes is {len(a)}, but needs "
        f"to be the same as dimension of Res, "
        f"which currently is {len(Res)}.")


def lambda_parsed(s):
    return eval(s, globals())


def identity(*args):
    return args


def tensor_grid_mult(tensor, tensor_vec):
    tensor_vec_result = cp.einsum("ij..., j...->i...", tensor, tensor_vec)

    return tensor_vec_result


def array_to_tensor_grid(arr: cp.ndarray, res_x: int, res_y: int, res_z: int):
    number_of_mixtures: int = arr.shape[0]

    arr_grid: List[cp.ndarray] = []
    for elem in cp.nditer(arr):
        arr_grid.append(cp.full((res_x, res_y, res_z), elem))
    tensor_grid_1d = cp.array(arr_grid)
    tensor_grid_2d = tensor_grid_1d.reshape(
        (number_of_mixtures, number_of_mixtures, res_x, res_y, res_z))

    return tensor_grid_2d


def arr_tensor_mult(self, arr, tensor_vec):
    tensor = array_to_tensor_grid(arr)
    tensor_result = tensor_grid_mult(tensor, tensor_vec)

    return tensor_result


def fft_plot(t, property_all):
    T = t[-1]
    N = len(t)
    sample_rate = N / T
    freq = cp.fft.rfftfreq(len(t), 1.0 / sample_rate)
    property_fft = cp.abs(cp.fft.rfft(property_all))

    return freq, property_fft


def get_droplet_edges(prob_droplets, peaks_index_3d, cut_axis):
    if cut_axis == 0:
        a = prob_droplets[:, peaks_index_3d[1], peaks_index_3d[2]]
    elif cut_axis == 1:
        a = prob_droplets[peaks_index_3d[0], :, peaks_index_3d[2]]
    elif cut_axis == 2:
        a = prob_droplets[peaks_index_3d[0], peaks_index_3d[1], :]
    else:
        sys.exit("Not implemented. Choose distance_axis 0, 1, 2.")

    zeros = cp.ndarray.flatten(cp.argwhere(a == 0))
    zeros_left = zeros[zeros < peaks_index_3d[cut_axis]]
    zeros_right = zeros[zeros > peaks_index_3d[cut_axis]]
    edge_left = max(zeros_left)
    edge_right = min(zeros_right)

    return edge_left, edge_right


def extract_droplet(prob_droplets, peaks_index_3d):
    edges = []
    for cut_axis in [0, 1, 2]:
        edges.append(get_droplet_edges(prob_droplets, peaks_index_3d, cut_axis))

    single_droplet = prob_droplets[slice(*edges[0]), slice(*edges[1]), slice(*edges[2])]

    return single_droplet, edges


def peaks_sort(peaks_indices, peaks_height, number_of_peaks):
    # sort peaks by height
    zipped_sorted_by_height = zip(*sorted(zip(peaks_indices, peaks_height), key=lambda t: t[1]))
    a, b = map(cp.array, zipped_sorted_by_height)

    # get the highest peaks (the n biggest, where n is number_of_peaks)
    peaks_sorted_indices = a[-number_of_peaks:]
    peaks_sorted_height = b[-number_of_peaks:]

    return peaks_sorted_indices, peaks_sorted_height


def peaks_sort_along(peaks_indices, peaks_height, number_of_peaks, axis):
    _, peaks_sorted_height = peaks_sort(peaks_indices, peaks_height, number_of_peaks)
    if axis in [0, 1, 2]:
        # get the highest peaks in a sorted fashion (the n biggest, where n is number_of_peaks)
        sorting_indices = cp.argsort(peaks_height)[-number_of_peaks:]
        peaks_sorted_indices = peaks_indices[sorting_indices]
    else:
        sys.exit(f"No such axis. Choose 0, 1 or 2 for axis x, y or z.")

    return peaks_sorted_indices, peaks_sorted_height


def get_peaks(prob):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(3, 3)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(prob, footprint=neighborhood) == prob
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (prob == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    peaks_mask = local_max ^ eroded_background

    peaks_height = prob[peaks_mask]
    peaks_indices = cp.argwhere(peaks_mask)

    return peaks_indices, peaks_height

def binary_structures():
    structure_star = ndimage.generate_binary_structure(3, 1)
    structure_vertical = ndimage.generate_binary_structure(3, 1)
    structure_horizontal = ndimage.generate_binary_structure(3, 1)
    structure_vertical[1, 0, :] = False
    structure_vertical[1, 2, :] = False
    structure_horizontal[1, :, 0] = False
    structure_horizontal[1, :, 2] = False

    return structure_star, structure_vertical, structure_horizontal

def fill_holes(region, structure_vertical, structure_horizontal):
    # fill holes in regions
    region_1 = ndimage.binary_dilation(region, structure=structure_vertical)
    region_2 = ndimage.binary_dilation(region_1, structure=structure_horizontal)

    return region_2


def get_meshgrid(x, y):
    x_mesh, y_mesh = cp.meshgrid(x, y)
    pos = cp.empty(x_mesh.shape + (2,))
    pos[:, :, 0] = x_mesh
    pos[:, :, 1] = y_mesh

    return x_mesh, y_mesh, pos


def get_meshgrid_3d(x, y, z):
    # WARNING: cp.meshgrid and mgrid have different structure,
    # resulting in fact x and y NEED to be swapped here (it is NOT a typo)
    x_mesh, y_mesh, z_mesh = cp.meshgrid(y, x, z)
    pos = cp.empty(x_mesh.shape + (3,))
    pos[:, :, :, 0] = x_mesh
    pos[:, :, :, 1] = y_mesh
    pos[:, :, :, 2] = z_mesh

    return x_mesh, y_mesh, z_mesh, pos


def check_provided_lists(number_of_mixtures: int,
                         a_s_list: cp.ndarray,
                         a_dd_list: cp.ndarray,
                         ):
    combinations = list(
        itertools.combinations_with_replacement(
            range(1, number_of_mixtures + 1),
            number_of_mixtures
        )
    )
    print(f"a_s and a_dd need to be provided as a list with the given order of combinations: "
          f"{combinations}.")

    if len(a_s_list) != len(combinations):
        sys.exit(f"a_s: {a_s_list} does not have the same length as combinations.")

    if len(a_dd_list) != len(combinations):
        sys.exit(f"a_dd: {a_dd_list} does not have the same length as combinations.")


def get_mu_combinations(dipol_list: list):
    number_of_mixtures = len(dipol_list)
    mu_combinations = list(
        itertools.combinations_with_replacement(
            dipol_list,
            number_of_mixtures
        )
    )

    mu_prod_combinations = cp.fromiter(map(cp.prod, mu_combinations), dtype=float)

    return mu_prod_combinations


def get_parameters_mixture(l_0: float,
                           number_of_mixtures: int,
                           a_dd_list: list,
                           a_s_list: list,
                           ) -> Tuple[cp.ndarray, cp.ndarray]:
    a_s_array = combinations2array(number_of_mixtures, a_s_list)
    a_dd_array = combinations2array(number_of_mixtures, a_dd_list)

    a_s_array_dimless = dimensionless(a_s_array, l_0)
    a_dd_array_dimless = dimensionless(a_dd_array, l_0)

    return a_s_array_dimless, a_dd_array_dimless


def dimensionless(arr, l_0):
    return arr / l_0


def combinations2array(number_of_mixtures: int,
                       combinations_list: list,
                       ) -> cp.ndarray:
    triu_indeces = cp.triu_indices(number_of_mixtures)
    triu = cp.zeros(shape=(number_of_mixtures, number_of_mixtures))
    triu[triu_indeces] = combinations_list
    arr = symmetric_mat(triu)

    return arr


def symmetric_mat(arr: cp.ndarray, axis=None) -> cp.ndarray:
    if axis:
        z_len = cp.shape(arr)[axis]
        result = arr[:, :, 0] + arr[:, :, 0].T - cp.diag(arr[:, :, 0].diagonal())
        for i in range(1, z_len):
            result = cp.stack((result,
                               arr[:, :, i] + arr[:, :, i].T - cp.diag(arr[:, :, i].diagonal())
                               ), axis=axis)

    else:
        result = arr + arr.T - cp.diag(arr.diagonal())

    return result


def w_dimensionsless(dimensionless_factor: float,
                     w_x: float = 2.0 * np.pi * 30.0,
                     w_y: float = 2.0 * np.pi * 30.0,
                     w_z: float = 2.0 * np.pi * 30.0,
                     ) -> (float, float, float):
    w_x_dimensionless = w_x * dimensionless_factor
    w_y_dimensionless = w_y * dimensionless_factor
    w_z_dimensionless = w_z * dimensionless_factor

    return w_x_dimensionless, w_y_dimensionless, w_z_dimensionless


def get_parameters(N: int = 10 ** 4,
                   m: float = 164 * constants.u_in_kg,
                   a_s: float = 90.0 * constants.a_0,
                   a_dd: float = 130.0 * constants.a_0,
                   w_x: float = 2.0 * np.pi * 30.0):
    a_s_l_ho_ratio, e_dd = g_qf_helper(m=m, a_s=a_s, a_dd=a_dd, w_x=w_x)
    g_qf = get_g_qf(N, a_s_l_ho_ratio, e_dd)
    g = get_g(N, a_s_l_ho_ratio)

    return g, g_qf, e_dd, a_s_l_ho_ratio


def get_g(N: int, a_s_l_ho_ratio: float):
    g = 4.0 * np.pi * a_s_l_ho_ratio * N

    return g


def g_qf_helper(m: float = 164 * constants.u_in_kg,
                a_s: float = 90.0 * constants.a_0,
                a_dd: float = 130.0 * constants.a_0,
                w_x: float = 2.0 * np.pi * 30.0):
    l_ho = get_l_ho(m, w_x)
    if a_s == 0.0 and a_dd == 0.0:
        e_dd = 0.0
    else:
        e_dd = a_dd / a_s
    a_s_l_ho_ratio = a_s / l_ho

    return a_s_l_ho_ratio, e_dd


def new_int(epsilon_dd: float):
    func = lambda u: cp.real((1 + epsilon_dd * (3 * u ** 2.0 - 1.0)) ** 2.5)
    try:
        integral = quad(func, 0.0, 1.0)[0]
    except:
        print(f"\nWARNING: epsilon_dd: {epsilon_dd} is over 1.0, leading to complex values in new_int.\n")

    return integral


def get_g_qf(N: int, a_s_l_ho_ratio: float, epsilon_dd: float):
    g_qf = (32.0 / (3.0 * cp.sqrt(np.pi))
            * 4.0 * np.pi * a_s_l_ho_ratio ** (5.0 / 2.0)
            * N ** (3.0 / 2.0)
            * new_int(epsilon_dd))

    return g_qf


def get_l_ho(m: float = 164.0 * constants.u_in_kg,
             w_x: float = 2.0 * np.pi * 30.0):
    l_ho = cp.sqrt(constants.hbar / (m * w_x))
    return l_ho


def get_alphas(w_x: float = 2.0 * np.pi * 30.0,
               w_y: float = 2.0 * np.pi * 30.0,
               w_z: float = 2.0 * np.pi * 30.0):
    alpha_y = w_y / w_x
    alpha_z = w_z / w_x

    return alpha_y, alpha_z


def psi_gauss_2d_pdf(pos, mu=cp.array(
    [0.0, 0.0]), var=cp.array([[1.0, 0.0], [0.0, 1.0]])):
    """
    Gives values according to gaus dirstribution (2D)
    with meshgrid of x,y as input

    :param pos: stacked meshgrid of an x (1D) and y (1D)
    :param mu: Mean of gauss
    :param var: Variance of gauss

    :param z_mesh: values according to gaus dirstribution (2D)
        with meshgrid of x,y as input

    """
    cov = cp.diag(var ** 2)
    rv = stats.multivariate_normal(mean=mu, cov=cov)
    z_mesh = rv.pdf(pos)

    return z_mesh


def psi_gauss_2d(x, y,
                 a_x: float = 1.0, a_y: float = 1.0,
                 x_0: float = 0.0, y_0: float = 0.0,
                 k_0: float = 0.0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0, y_0

    :param x: mathematical variable

    :param y: mathematical variable

    :param a_x: Stretching factor in x direction (cp.sqrt(2) * std_deviation)

    :param a_y: Stretching factor in y direction (cp.sqrt(2) * std_deviation)

    :param x_0: Mean spatial x of pulse

    :param y_0: Mean spatial y of pulse

    :param k_0: Group velocity of pulse

    """

    return (
            (a_x * a_y * np.pi) ** -0.5
            * cp.exp(-0.5 * (
            ((x - x_0) / a_x) ** 2.0
            + ((y - y_0) / a_y) ** 2.0
    )
                     + 1j * x * k_0)
    )


def psi_gauss_3d(x, y, z,
                 a_x: float = 1.0, a_y: float = 1.0, a_z: float = 1.0,
                 x_0: float = 0.0, y_0: float = 0.0, z_0: float = 0.0,
                 k_0: float = 0.0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0

    :param x: mathematical variable

    :param y: mathematical variable

    :param z: mathematical variable

    :param a_x: Stretching factor in x direction (cp.sqrt(2) * std_deviation)

    :param a_y: Stretching factor in y direction (cp.sqrt(2) * std_deviation)

    :param a_z: Stretching factor in z direction (cp.sqrt(2) * std_deviation)

    :param x_0: Mean spatial x of pulse

    :param y_0: Mean spatial y of pulse

    :param z_0: Mean spatial z of pulse

    :param k_0: Group velocity of pulse

    """

    return ((a_x * a_y * a_z * np.pi ** (3.0 / 2.0)) ** -0.5
            * cp.exp(-0.5 * (
                    ((x - x_0) / a_x) ** 2.0
                    + ((y - y_0) / a_y) ** 2.0
                    + ((z - z_0) / a_z) ** 2.0)
                     + 1j * x * k_0))


def psi_gauss_1d(x, a: float = 1.0, x_0: float = 0.0, k_0: float = 0.0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0

    :param x: mathematical variable

    :param a: Amplitude of pulse

    :param x_0: Mean spatial x of pulse

    :param k_0: Group velocity of pulse

    """

    return ((a * cp.sqrt(np.pi)) ** (-0.5)
            * cp.exp(-0.5 * ((x - x_0) * 1. / a) ** 2 + 1j * x * k_0))


def psi_pdf(x, loc: float = 0.0, scale: float = 1.0):
    """
    Mathematical function of gauss pulse

    :param x: mathematical variable

    :param loc: Localization of pulse centre

    :param scale: Scale of pulse

    """
    return stats.norm.pdf(x, loc=loc, scale=scale)


def psi_rect(x, x_min: float = -0.5, x_max: float = 0.5, a: float = 1.0):
    """
    Mathematical function of rectangular pulse
    between x_min and x_max with amplitude a

    :param x: mathematical variable

    :param x_min: Minimum x value of pulse (spatial)

    :param x_max: Maximum x value of pulse (spatial)

    :param a: Amplitude of pulse

    """

    pulse = cp.select([x < x_min, x < x_max, x_max < x], [0, a, 0])
    assert pulse.any(), ("Pulse is completely 0. Resolution is too small. "
                         "Resolution needs to be set, "
                         "as fft is used onto the pulse.")

    return pulse


def psi_gauss_solution(x):
    """
     Mathematical function of solution of non-linear Schroedinger for g=0

     :param x: mathematical variable

    """

    return cp.exp(-x ** 2) / cp.sqrt(np.pi)


def thomas_fermi_1d(x, g: float = 0.0):
    """
    Mathematical function of Thomas-Fermi distribution with coupling constant g

    :param x: mathematical variable

    :param g: coupling constant

    """

    if g != 0:
        # mu is the chemical potential
        mu = mu_1d(g)

        # this needs to be >> 1, e.g 5.3
        # print(cp.sqrt(2 * mu))

        return mu * (1 - ((x ** 2) / (2 * mu))) / g

    else:
        print(f"Thomas-Fermi not possible for g=0.0. But you specified g={g}. Returning None.")
        return None


def thomas_fermi_2d(x, y, g: float = 0.0):
    """
    Mathematical function of Thomas-Fermi distribution with coupling constant g

    :param x: mathematical variable

    :param y: mathematical variable

    :param g: coupling constant

    """

    if g != 0:
        # mu is the chemical potential
        mu = mu_2d(g)

        # this needs to be >> 1, e.g 5.3
        # print(cp.sqrt(2 * mu))

        return mu * (1 - ((x ** 2 + y ** 2) / (2 * mu))) / g

    else:
        print(f"Thomas-Fermi not possible for g=0.0. But you specified g={g}. Returning None.")
        return None


def thomas_fermi_2d_pos(pos, g: float = 0.0):
    x = pos[:, :, 0]
    y = pos[:, :, 1]

    return thomas_fermi_2d(x, y, g=g)


def thomas_fermi_3d(x, y, z, g: float = 0.0):
    """
    Mathematical function of Thomas-Fermi distribution with coupling constant g

    :param x: mathematical variable

    :param y: mathematical variable

    :param z: mathematical variable

    :param g: coupling constant

    """

    if g != 0:
        # mu is the chemical potential
        mu = mu_3d(g)

        # this needs to be >> 1, e.g 5.3
        # print(cp.sqrt(2 * mu))

        return mu * (1 - ((x ** 2 + y ** 2 + z ** 2) / (2 * mu))) / g

    else:
        print(f"Thomas-Fermi not possible for g=0.0. But you specified g={g}. Returning None.")
        return None


def mu_1d(g: float = 0.0):
    # mu is the chemical potential
    mu = ((3.0 * g) / (4.0 * cp.sqrt(2.0))) ** (2.0 / 3.0)

    return mu


def mu_2d(g: float = 0.0):
    # mu is the chemical potential
    mu = cp.sqrt(g / np.pi)

    return mu


def mu_3d(g: float = 0.0):
    # mu is the chemical potential
    mu = ((15 * g) / (16 * cp.sqrt(2) * cp.pi)) ** (2 / 5)

    return mu


def v_harmonic_1d(x):
    return 0.5 * x ** 2


def v_harmonic_2d(pos, alpha_y: float = 1.0):
    x = pos[:, :, 0]
    y = pos[:, :, 1]

    return v_2d(x, y, alpha_y=1.0)


def v_2d(x, y, alpha_y=1.0):
    return 0.5 * (x ** 2 + y ** 2)


def v_harmonic_3d(x, y, z, alpha_y: float = 1.0, alpha_z: float = 1.0, lH0: float = 1.0):
    return 0.5 * (x ** 2 + (alpha_y * y) ** 2 + (alpha_z * z) ** 2) / (lH0 ** 4.0)


def get_r_cut(k_mesh: cp.ndarray, r_cut: float = 1.0):
    kr_singular = k_mesh * r_cut

    # remove known singularity at [0, 0, 0], for calculation
    if kr_singular[0, 0, 0] == 0.0:
        kr_singular[0, 0, 0] = 1.0
    else:
        print(f"WARNING: kr_singular[0, 0, 0] = {kr_singular[0, 0, 0]}, but expected 0.")

    # FFT of a symmetric box-function
    r_cut_mesh = (1.0
                  + (3.0 / kr_singular ** 2.0) * cp.cos(kr_singular)
                  - (3.0 / kr_singular ** 3.0) * cp.sin(kr_singular))

    # set known value at [0, 0, 0]
    if r_cut_mesh[0, 0, 0]:
        r_cut_mesh[0, 0, 0] = 0.0

    return r_cut_mesh


def dipol_dipol(u):
    dipol = (4.0 * np.pi / 3.0) * (3.0 * u ** 2.0 - 1.0)

    return dipol


def dipol_dipol_interaction(kx_mesh: cp.ndarray,
                            ky_mesh: cp.ndarray,
                            kz_mesh: cp.ndarray,
                            r_cut: float = 1.0,
                            use_cut_off: bool = False):
    k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0 + kz_mesh ** 2.0
    # for [0, 0, 0] there is a singularity and factor/k_squared is 0/0, so we
    # arbitrary set the divisor to 1.0
    k_mesh: cp.ndarray = cp.sqrt(k_squared)
    k_mesh_singular_free = cp.where(k_mesh == 0.0, 1.0, k_mesh)
    k_mesh_singular_index = cp.where(k_mesh == 0.0)

    if use_cut_off:
        r_cut_mesh: cp.ndarray = get_r_cut(k_mesh_singular_free, r_cut=r_cut)
    else:
        r_cut_mesh: float = 1.0

    V_k_val = r_cut_mesh * dipol_dipol(kz_mesh / k_mesh_singular_free)
    # set the the arbitrary value assigned to the singularity to 0
    V_k_val[k_mesh_singular_index] = 0.0

    return V_k_val


def get_V_k_val_ddi(kx_mesh, ky_mesh, kz_mesh,
                    rho_lin: cp.ndarray,
                    z_lin: cp.ndarray,
                    ):
    """
    Explicit calculation of the Fourier transform with the cylindrical cut-off

    """
    z_cut = z_lin[-1]

    k_rho2_mesh = kx_mesh ** 2.0 + ky_mesh ** 2.0
    k_rho_mesh = cp.sqrt(k_rho2_mesh)
    k_r2_mesh = k_rho2_mesh + kz_mesh ** 2.0

    # remove artifical singularity
    k_r2_singular_free = cp.where(k_r2_mesh == 0.0, 1.0, k_r2_mesh)
    cos2a = kz_mesh ** 2.0 / k_r2_singular_free

    sin2a = 1 - cos2a
    sinacosa = cp.sqrt(sin2a * cos2a)
    term1 = cos2a - 1.0 / 3.0
    term2 = cp.exp(-z_cut * k_rho_mesh) * (sin2a * cp.cos(z_cut * kz_mesh)
                                           - sinacosa * cp.sin(z_cut * kz_mesh)
                                           )
    term3_slow = get_rho_integral_slow(k_rho_mesh, kz_mesh, rho_lin, z_lin)
    term3 = get_rho_integral(k_rho_mesh, kz_mesh, rho_lin, z_lin, compare=term3_slow)

    return 4.0 * np.pi * (term1 + term2 + term3)


def get_rho_integral_slow(k_rho_mesh: cp.ndarray,
                          kz_mesh: cp.ndarray,
                          rho_lin: cp.ndarray,
                          z_lin: cp.ndarray,
                          ):
    drho = rho_lin[1] - rho_lin[0]
    dz = z_lin[1] - z_lin[0]

    with run_time(name="get_rho_integral_slow"):
        it = cp.nditer([k_rho_mesh, kz_mesh], flags=['external_loop'])
        shape = cp.shape(kz_mesh)
        out = []
        # it = cp.nditer([k_rho_mesh, kz_mesh, None], flags=['external_loop'])
        # out = cp.zeros(shape=shape)
        with it:
            iter = cp.ndindex(shape)
            for k_rho1, kz1 in it:
                for k_rho, kz in zip(k_rho1, kz1):
                    index = next(iter)
                    integrand_rho_z = bessel_func(rho_lin[:, cp.newaxis], z_lin, k_rho, kz)
                    out.append(cp.sum(cp.sum(integrand_rho_z)) * drho * dz)
                    # out[index] = cp.sum(cp.sum(integrand_rho_z)) * drho * dz

        out = cp.array(out).reshape(shape)

    return out


def triu_list2array(triu_list, triu_ind, shape):
    triu = cp.zeros(shape=shape)
    triu[triu_ind] = triu_list

    return triu


def get_rho_integral(k_rho_mesh: cp.ndarray,
                     kz_mesh: cp.ndarray,
                     rho_lin: cp.ndarray,
                     z_lin: cp.ndarray,
                     compare,
                     ):
    drho = rho_lin[1] - rho_lin[0]
    dz = z_lin[1] - z_lin[0]
    x_size, y_size = k_rho_mesh.shape[0], k_rho_mesh.shape[1]

    with run_time(name="get_rho_integral"):
        x_len = int((x_size / 2.0) + 1.0)
        y_len = int((y_size / 2.0) + 1.0)
        k_rho_mesh_halved = k_rho_mesh[0:x_len, 0:y_len, :]
        kz_mesh_halved = kz_mesh[0:x_len, 0:y_len, :]
        z_len = cp.shape(kz_mesh_halved)[2]

        triu_ind = cp.triu_indices(n=x_len, m=y_len, k=-1)
        g = [k_rho_mesh_halved[:, :, i][triu_ind] for i in range(0, z_len)]
        h = [kz_mesh_halved[:, :, i][triu_ind] for i in range(0, z_len)]
        out = []
        for k_rho_rank, kz_rank in zip(g, h):
            inner = []
            for k_rho, kz in zip(k_rho_rank, kz_rank):
                integrand_rho_z = bessel_func(rho_lin[:, cp.newaxis], z_lin, k_rho, kz)
                inner.append(cp.sum(cp.sum(integrand_rho_z)) * drho * dz)
            out.append(inner)

        trius = [triu_list2array(inner, triu_ind, (x_len, y_len)) for inner in out]
        out = cp.stack(trius, axis=2)

    out_n_n = out[1:int(x_len - 1), :, :]
    a1 = cp.apply_over_axes(symmetric_mat, out_n_n, axes=2)
    z_len = cp.shape(a1)[2]
    result = []
    for i in range(0, z_len):
        b = cp.rot90(a1[:, :, i], 1, axes=(0, 1)).T
        c = cp.rot90(b, 2, axes=(0, 1)).T
        d = cp.rot90(c, 3, axes=(0, 1)).T
        e = cp.vstack((a1[:, :, i], b))
        f = cp.vstack((d, c))
        g = cp.hstack((e, f))
        first = cp.hstack((out[0, :, i], out[0, :, i][::-1]))
        h = cp.insert(g, 0, first, axis=0)
        middle = cp.hstack((out[int(x_len - 1), :, i], out[int(x_len - 1), :, i][::-1]))
        j = cp.insert(h, x_len, middle, axis=0)
        j_size_y = cp.size(j)[1]
        y_too_much = y_size - j_size_y
        for counter in range(0, y_too_much):
            k = cp.delete(j, int(y_size + 1), axis=1)
        # m = cp.hstack((out[i, :, :], k))
        result.append(k)

    result = cp.reshape(result, cp.shape(k_rho_mesh))

    return result


def get_rho_integral_quad(kx_mesh: float,
                          ky_mesh: float,
                          kz_mesh: float,
                          z_mesh: float,
                          rho_cut: float = 1.0,
                          z_cut: float = 1.0):
    shape = kx_mesh.shape
    k_rho_mesh = cp.sqrt(kx_mesh ** 2.0 + ky_mesh ** 2.0)

    result = []
    with run_time(name="get_rho_integral_quad"):
        it = cp.nditer([z_mesh, kz_mesh, k_rho_mesh], flags=['multi_index'])
        for z, kz, k_rho in it:
            # print(f"triple: {z}, {kz}, {k_rho}")
            print(it.multi_index)
            result.append(quad(bessel_func(z, k_rho, kz), rho_cut, np.inf)[0])
        result = cp.array(result).reshape(kz_mesh.shape)

    return result


def get_V_k_val_ddi_fft_where(x_mesh: float,
                              y_mesh: float,
                              z_mesh: float,
                              rho_cut: float = 1.0,
                              z_cut: float = 1.0):
    with run_time(name="fft V_ddi"):
        rho2_mesh = x_mesh ** 2.0 + y_mesh ** 2.0
        rho_mesh = cp.sqrt(rho2_mesh)
        r_mesh = cp.sqrt(rho2_mesh + z_mesh ** 2.0)
        cos_theta = z_mesh / r_mesh

        ddi = (1.0 - cos_theta ** 2.0) / (r_mesh ** 3.0)
        zeros = cp.zeros(shape=ddi.shape)
        cond_cylinder = cp.logical_and(cp.abs(z_mesh) < z_cut, rho_mesh < rho_cut)
        ddi_cut = cp.where(cond_cylinder, ddi, zeros)

        V_k_val = cp.fft.fftn(ddi_cut)

        V_k_val_real = V_k_val.real

    return V_k_val_real


def get_V_k_val_ddi_fft(x_mesh: float,
                        y_mesh: float,
                        z_mesh: float,
                        x_cut: List[float],
                        y_cut: List[float],
                        z_cut: List[float]):
    with run_time(name="fft V_ddi xyz cut"):
        r_mesh = cp.sqrt(x_mesh ** 2.0 + y_mesh ** 2.0 + z_mesh ** 2.0)

        r_mesh_singular_free = cp.where(r_mesh == 0.0, 1.0, r_mesh)
        cos_theta = z_mesh / r_mesh_singular_free

        ddi = (1.0 - 3.0 * cos_theta ** 2.0) / (r_mesh ** 3.0)
        zeros = cp.zeros(shape=ddi.shape)
        cond_x = (x_cut[0] < x_mesh) & (x_mesh < x_cut[1])
        cond_y = (y_cut[0] < y_mesh) & (y_mesh < y_cut[1])
        cond_z = (z_cut[0] < z_mesh) & (z_mesh < z_cut[1])
        cond_xyz_cut = cond_x & cond_y & cond_z
        ddi_cut = cp.where(cond_xyz_cut, ddi, zeros)

        V_k_val = cp.fft.fftn(ddi_cut)

        V_k_val_real = V_k_val.real

    return V_k_val_real


def bessel_func(rho, z, k_rho, kz):
    bessel = (- cp.cos(kz * z) * rho
              * (rho ** 2.0 - 2.0 * z ** 2.0) / ((rho ** 2.0 + z ** 2.0) ** 2.5)
              * jv(0, rho * k_rho)
              )

    return bessel


def f_kappa(kappa: cp.ndarray, epsilon: float = 10 ** -10) -> float:
    k2_1 = (kappa ** 2.0 - 1.0 + epsilon)
    result = ((2.0 * kappa ** 2.0 + 1.0) - (3.0 * kappa ** 2.0) * atan_special(
        k2_1)) / k2_1

    return result


@cp.vectorize
def atan_special(x):
    if x > 0:
        result = cp.arctan(cp.sqrt(x)) / cp.sqrt(x)
    elif x == 0:
        result = 0.0
    else:
        result = cp.arctanh(cp.sqrt(-x)) / cp.sqrt(-x)

    return result


def func_125(kappa: float, alpha_z: float, e_dd: float, epsilon: float = 10 ** -10):
    k2_1 = (kappa ** 2.0 - 1.0 + epsilon)
    a = 3.0 * kappa * e_dd * ((alpha_z ** 2.0 / 2.0 + 1.0) * (f_kappa(kappa) / k2_1) - 1.0)
    b = (e_dd - 1.0) * (kappa ** 2.0 - alpha_z ** 2.0)
    return a + b


def func_124(kappa: float, e_dd: float, N: float, a_s_l_ho_ratio: float):
    factor = (15.0 * N * kappa * a_s_l_ho_ratio)
    b = 1.5 * ((kappa ** 2.0 * f_kappa(kappa)) / (kappa ** 2.0 - 1.0)) - 1.0
    c = (1.0 + e_dd * b)
    R_r = (factor * (1.0 + e_dd * b)) ** (1.0 / 5.0)
    return R_r


def get_R_rz(kappa: float, e_dd: float, N: int, a_s_l_ho_ratio: float):
    R_r = func_124(kappa=kappa, e_dd=e_dd, N=N, a_s_l_ho_ratio=a_s_l_ho_ratio)
    R_z = R_r / kappa

    return R_r, R_z


def get_kappa(alpha_z: float, e_dd: float,
              x_min: float = 3.0, x_max: float = 5.0, res: int = 1000):
    kappa_array: cp.ndarray = cp.linspace(x_min, x_max, res,
                                          endpoint=False)
    y = func_125(kappa_array, alpha_z, e_dd)
    if y[-1] > 0:
        kappa_root = min(kappa_array[y >= 0.0])
    else:
        kappa_root = min(kappa_array[y <= 0.0])

    return kappa_root


def density_in_trap(x: float, y: float, z: float,
                    R_r: float, R_z: float, g: float = 0.0):
    r = cp.sqrt(x ** 2.0 + y ** 2.0)
    n_0 = 15.0 / (8.0 * cp.pi * R_z * R_r ** 2.0)
    a = (r / R_r) ** 2.0 + (z / R_z) ** 2.0

    n_r = cp.where(a > 1, 0.0, n_0 * (1.0 - a))

    return n_r


def density_in_trap_r(r: float, z: float, R_r: float, R_z: float,
                      g: float = 0.0):
    n_0 = 15.0 / (8.0 * np.pi * R_r ** 2.0 * R_z)
    return n_0 * (1.0 - (r ** 2.0 / R_r ** 2.0) - (z ** 2.0 / R_z ** 2.0))


def camera_func_r(frame: int,
                  r_0: float = 10.0,
                  phi_0: float = 45.0,
                  z_0: float = 20.0,
                  r_per_frame: float = 10.0) -> float:
    r = r_0 + r_per_frame * frame
    return r


def camera_func_phi(frame: int,
                    r_0: float = 10.0,
                    phi_0: float = 45.0,
                    z_0: float = 20.0,
                    phi_per_frame: float = 10.0) -> float:
    phi = phi_0 + (2.0 * np.pi / 360.0) * phi_per_frame * frame
    return phi


def camera_func_z(frame: int,
                  r_0: float = 10.0,
                  phi_0: float = 45.0,
                  z_0: float = 20.0,
                  z_per_frame: float = 10.0) -> float:
    z = z_0 + z_per_frame * frame
    return z


def camera_3d_trajectory(frame: int,
                         r_func: Callable = None,
                         phi_func: Callable = None,
                         z_func: Callable = None,
                         r_0: float = 10.0,
                         phi_0: float = 45.0,
                         z_0: float = 20.0) -> Tuple[float, float, float]:
    """
    Computes r, phi, z as the components of the camera position
    in the animation for the given frame.
    Depending on, if a callable function is given for the components,
    it is applied to the parameters
    or the start values are used.

    :param frame: Index of the frame in the animation

    :param r_func: r component of the movement of the camera.

    :param phi_func: phi component of the movement of the camera.

    :param z_func: z component of the movement of the camera.

    :param r_0: r component of the starting point of the camera movement.

    :param phi_0: phi component of the starting point of the camera movement.

    :param z_0: z component of the starting point of the camera movement.

    :return: r, phi, z as the components of the camera position
        in the animation for the given frame.

    """
    if r_func is None:
        r = r_0
    else:
        r = r_func(frame)
    if phi_func is None:
        phi = phi_0
    else:
        phi = phi_func(frame)
    if z_func is None:
        z = z_0
    else:
        z = z_func(frame)

    return r, phi, z


def noise_mesh(min: float = 0.8,
               max: float = 1.2,
               shape: Tuple[int, int, int] = (64, 64, 64)) -> cp.ndarray:
    noise = min + (max - min) * cp.random.rand(*shape)

    return noise


def dt_adaptive(t, dt) -> float:
    # TODO: adaptiv dt
    if t > 2.4:
        dt_adapted = 5 * 10 ** -4
    elif t > 1.5:
        dt_adapted = 1 * 10 ** -3
    elif t > 1.2:
        dt_adapted = 2 * 10 ** -3
    else:
        dt_adapted = dt

    return dt_adapted


# Script runs, if script is run as main script (called by python *.py)
if __name__ == '__main__':
    # due to fft of the points the res needs to be 2 **
    # resolution_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    # constants needed for the Schroedinger equation
    max_timesteps = 10
    dt = 0.05

    # functions needed for the Schroedinger equation (e.g. potential: V,
    # initial wave function: psi_0)
    V_1d = v_harmonic_1d
    V_2d = v_harmonic_2d
    V_3d = v_harmonic_3d

    # functools.partial sets all arguments except x,
    # as multiple arguments for Schroedinger aren't implement yet
    psi_0_1d = functools.partial(psi_gauss_1d, a=1, x_0=0, k_0=0)
    psi_0_2d = functools.partial(psi_gauss_2d_pdf, mu=cp.array(
        [0.0, 0.0]), var=cp.array([1.0, 1.0]))
    psi_0_3d = functools.partial(psi_gauss_3d, a=1, x_0=0, y_0=0, z_0=0, k_0=0)

    # testing for 2d plot
    L = 10
    x = cp.linspace(-L, L, resolution, endpoint=False)
    y = cp.linspace(-L, L, resolution, endpoint=False)
    x_mesh, y_mesh, pos = get_meshgrid(x, y)
