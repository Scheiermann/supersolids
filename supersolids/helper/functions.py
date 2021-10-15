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
from scipy.special import jv
from scipy import stats
from typing import Tuple, Callable, Optional, List

from supersolids.helper import constants
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
        x: np.ndarray = np.linspace(x0, x1, res, endpoint=False)
        dx: float = box_len / float(res - 1)
        dkx: float = 2.0 * np.pi / box_len
        kx: np.ndarray = np.fft.fftfreq(res, d=1.0 / (dkx * res))

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
        x_mesh, y_mesh, z_mesh = np.mgrid[x0: x1: complex(0, res_x),
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


def fft_plot(t, property_all):
    T = t[-1]
    N = len(t)
    sample_rate = N / T
    freq = np.fft.rfftfreq(len(t), 1.0 / sample_rate)
    property_fft = np.abs(np.fft.rfft(property_all))

    return freq, property_fft


def get_meshgrid(x, y):
    x_mesh, y_mesh = np.meshgrid(x, y)
    pos = np.empty(x_mesh.shape + (2,))
    pos[:, :, 0] = x_mesh
    pos[:, :, 1] = y_mesh

    return x_mesh, y_mesh, pos


def get_meshgrid_3d(x, y, z):
    # WARNING: np.meshgrid and mgrid have different structure,
    # resulting in fact x and y NEED to be swapped here (it is NOT a typo)
    x_mesh, y_mesh, z_mesh = np.meshgrid(y, x, z)
    pos = np.empty(x_mesh.shape + (3,))
    pos[:, :, :, 0] = x_mesh
    pos[:, :, :, 1] = y_mesh
    pos[:, :, :, 2] = z_mesh

    return x_mesh, y_mesh, z_mesh, pos


def check_provided_lists(number_of_mixtures: int,
                         a_s_array: np.ndarray,
                         a_dd_array: np.ndarray,
                         ):
    combinations = list(
                        itertools.combinations_with_replacement(
                            range(1, number_of_mixtures + 1),
                            number_of_mixtures
                            )
                        )
    print(f"a_s and a_dd need to be provided as a list with the given order of combinations: "
          f"{combinations}.")

    if len(a_s_array) != len(combinations):
        sys.exit(f"a_s: {a_s_array} does not have the same length as combinations.")

    if len(a_dd_array) != len(combinations):
        sys.exit(f"a_dd: {a_dd_array} does not have the same length as combinations.")


def generate_a_dd(mu_list: list,
                  a_dd_factor: float,
                  ):
    number_of_mixtures = len(mu_list)
    mu_combinations = list(
        itertools.combinations_with_replacement(
            mu_list,
            number_of_mixtures
        )
    )

    mu_prod_combinations = np.fromiter(map(np.prod, mu_combinations), dtype=float)
    a_dd_list = mu_prod_combinations * a_dd_factor

    return a_dd_list


def get_parameters_mixture(l_0: float,
                           mu_list: list,
                           a_s_list: list,
                           a_dd_factor: float,
                           ) -> (np.ndarray, np.ndarray):
    number_of_mixtures = len(mu_list)
    a_dd_list = generate_a_dd(mu_list, a_dd_factor)
    check_provided_lists(number_of_mixtures, a_s_list, a_dd_list)

    a_s_array = dimensionless(combinations2array(number_of_mixtures, a_s_list), l_0)
    a_dd_array = dimensionless(combinations2array(number_of_mixtures, a_dd_list), l_0)

    a_s_array_dimless = dimensionless(a_s_array, l_0)
    a_dd_array_dimless = dimensionless(a_dd_array, l_0)

    return a_s_array_dimless, a_dd_array_dimless


def dimensionless(arr, l_0):
    return arr / l_0


def combinations2array(number_of_mixtures: int,
                       combinations_list: list,
                       ) -> np.ndarray:
    triu_indeces = np.triu_indices(number_of_mixtures)
    triu = np.zeros(shape=(number_of_mixtures, number_of_mixtures))
    triu[triu_indeces] = combinations_list
    arr = symmetric_mat(triu)

    return arr


def symmetric_mat(arr: np.ndarray) -> np.ndarray:
    return arr + arr.T - np.diag(arr.diagonal())


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
    func = lambda u: (1 + epsilon_dd * (3 * u ** 2.0 - 1.0)) ** 2.5
    integral = quad(func, 0.0, 1.0)[0]

    return integral


def get_g_qf(N: int, a_s_l_ho_ratio: float, epsilon_dd: float):
    g_qf = (32.0 / (3.0 * np.sqrt(np.pi))
            * 4.0 * np.pi * a_s_l_ho_ratio ** (5.0 / 2.0)
            * N ** (3.0 / 2.0)
            * new_int(epsilon_dd))

    return g_qf


def get_l_ho(m: float = 164.0 * constants.u_in_kg,
             w_x: float = 2.0 * np.pi * 30.0):
    l_ho = np.sqrt(constants.hbar / (m * w_x))
    return l_ho


def get_alphas(w_x: float = 2.0 * np.pi * 30.0,
               w_y: float = 2.0 * np.pi * 30.0,
               w_z: float = 2.0 * np.pi * 30.0):
    alpha_y = w_y / w_x
    alpha_z = w_z / w_x

    return alpha_y, alpha_z


def psi_gauss_2d_pdf(pos, mu=np.array(
    [0.0, 0.0]), var=np.array([[1.0, 0.0], [0.0, 1.0]])):
    """
    Gives values according to gaus dirstribution (2D)
    with meshgrid of x,y as input

    :param pos: stacked meshgrid of an x (1D) and y (1D)
    :param mu: Mean of gauss
    :param var: Variance of gauss

    :param z_mesh: values according to gaus dirstribution (2D)
        with meshgrid of x,y as input

    """
    cov = np.diag(var ** 2)
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

    :param a_x: Stretching factor in x direction (np.sqrt(2) * std_deviation)

    :param a_y: Stretching factor in y direction (np.sqrt(2) * std_deviation)

    :param x_0: Mean spatial x of pulse

    :param y_0: Mean spatial y of pulse

    :param k_0: Group velocity of pulse

    """

    return (
            (a_x * a_y * np.pi) ** -0.5
            * np.exp(-0.5 * (
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

    :param a_x: Stretching factor in x direction (np.sqrt(2) * std_deviation)

    :param a_y: Stretching factor in y direction (np.sqrt(2) * std_deviation)

    :param a_z: Stretching factor in z direction (np.sqrt(2) * std_deviation)

    :param x_0: Mean spatial x of pulse

    :param y_0: Mean spatial y of pulse

    :param z_0: Mean spatial z of pulse

    :param k_0: Group velocity of pulse

    """

    return ((a_x * a_y * a_z * np.pi ** (3.0 / 2.0)) ** -0.5
            * np.exp(-0.5 * (
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

    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x_0) * 1. / a) ** 2 + 1j * x * k_0))


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

    pulse = np.select([x < x_min, x < x_max, x_max < x], [0, a, 0])
    assert pulse.any(), ("Pulse is completely 0. Resolution is too small. "
                         "Resolution needs to be set, "
                         "as fft is used onto the pulse.")

    return pulse


def psi_gauss_solution(x):
    """
     Mathematical function of solution of non-linear Schroedinger for g=0

     :param x: mathematical variable

    """

    return np.exp(-x ** 2) / np.sqrt(np.pi)


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
        # print(np.sqrt(2 * mu))

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
        # print(np.sqrt(2 * mu))

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
        # print(np.sqrt(2 * mu))

        return mu * (1 - ((x ** 2 + y ** 2 + z ** 2) / (2 * mu))) / g

    else:
        print(f"Thomas-Fermi not possible for g=0.0. But you specified g={g}. Returning None.")
        return None


def mu_1d(g: float = 0.0):
    # mu is the chemical potential
    mu = ((3.0 * g) / (4.0 * np.sqrt(2.0))) ** (2.0 / 3.0)

    return mu


def mu_2d(g: float = 0.0):
    # mu is the chemical potential
    mu = np.sqrt(g / np.pi)

    return mu


def mu_3d(g: float = 0.0):
    # mu is the chemical potential
    mu = ((15 * g) / (16 * np.sqrt(2) * np.pi)) ** (2 / 5)

    return mu


def v_harmonic_1d(x):
    return 0.5 * x ** 2


def v_harmonic_2d(pos, alpha_y: float = 1.0):
    x = pos[:, :, 0]
    y = pos[:, :, 1]

    return v_2d(x, y, alpha_y=1.0)


def v_2d(x, y, alpha_y=1.0):
    return 0.5 * (x ** 2 + y ** 2)


def v_harmonic_3d(x, y, z, alpha_y: float = 1.0, alpha_z: float = 1.0):
    return 0.5 * (x ** 2 + (alpha_y * y) ** 2 + (alpha_z * z) ** 2)


def get_r_cut(k_mesh: np.ndarray, r_cut: float = 1.0):
    kr_singular = k_mesh * r_cut

    # remove known singularity at [0, 0, 0], for calculation
    if kr_singular[0, 0, 0] == 0.0:
        kr_singular[0, 0, 0] = 1.0
    else:
        print(f"WARNING: kr_singular[0, 0, 0] = {kr_singular[0, 0, 0]}, but expected 0.")

    # FFT of a symmetric box-function
    r_cut_mesh = (1.0
                  + (3.0 / kr_singular ** 2.0) * np.cos(kr_singular)
                  - (3.0 / kr_singular ** 3.0) * np.sin(kr_singular))

    # set known value at [0, 0, 0]
    if r_cut_mesh[0, 0, 0]:
        r_cut_mesh[0, 0, 0] = 0.0

    return r_cut_mesh


def dipol_dipol_interaction(kx_mesh: float,
                            ky_mesh: float,
                            kz_mesh: float,
                            r_cut: float = 1.0):
    k_squared = kx_mesh ** 2.0 + ky_mesh ** 2.0 + kz_mesh ** 2.0
    factor = 3.0 * (kz_mesh ** 2.0)
    # for [0, 0, 0] there is a singularity and factor/k_squared is 0/0, so we
    # arbitrary set the divisor to 1.0
    k_squared_singular_free = np.where(k_squared == 0.0, 1.0, k_squared)

    k_mesh: np.ndarray = np.sqrt(k_squared)
    r_cut_mesh: np.ndarray = get_r_cut(k_mesh, r_cut=r_cut)

    V_k_val = r_cut_mesh * ((factor / k_squared_singular_free) - 1.0)

    # Remove singularities (at this point there should not be any)
    # V_k_val[np.isnan(V_k_val)] = 0.0

    return V_k_val


def get_V_k_val_ddi(kx_mesh, ky_mesh, kz_mesh,
                    z_mesh, rho_cut: float = 1.0, z_cut: float = 1.0):
    """
    Explicit calculation of the Fourier transform with the cylindrical cut-off

    """
    k_rho2_mesh = kx_mesh ** 2.0 + ky_mesh ** 2.0
    k_rho_mesh = np.sqrt(k_rho2_mesh)
    k_r2_mesh = k_rho2_mesh + kz_mesh ** 2.0

    # remove artifical singularity
    k_r2_singular_free = np.where(k_r2_mesh == 0.0, 1.0, k_r2_mesh)
    cos2a = kz_mesh ** 2.0 / k_r2_singular_free

    sin2a = 1 - cos2a
    sinacosa = np.sqrt(sin2a * cos2a)
    term1 = cos2a - 1.0 / 3.0
    term2 = np.exp(-z_cut * k_rho_mesh) * (sin2a * np.cos(z_cut * kz_mesh)
                                           - sinacosa * np.sin(z_cut * kz_mesh)
                                           )
    term3 = quad(get_rho_integral(k_rho_mesh, kz_mesh, z_mesh, rho_cut), 0.0, z_cut)
    return 4.0 * np.pi * (term1 + term2 + term3)


def get_rho_integral(k_rho_mesh: float,
                     kz_mesh: float,
                     z_mesh: float,
                     rho_cut: float = 1.0,
                     ):
    shape = kz_mesh.shape

    r_bound = 2000.0 * rho_cut
    # r_bound = np.inf

    with run_time(name="quad bessel_func"):
        it = np.nditer([z_mesh, k_rho_mesh, kz_mesh, None], flags=['external_loop'])

        with it:
            for z1, k_rho1, kz1, out in it:
                for z, k_rho, kz in zip(z1, k_rho1, kz1):
                    out[...] = quad(bessel_func(z, k_rho, kz), rho_cut, r_bound)[0]

        result = np.array(out).reshape(shape)

    return result


def get_rho_integral1(kx_mesh: float,
                      ky_mesh: float,
                      kz_mesh: float,
                      z_mesh: float,
                      rho_cut: float = 1.0,
                      z_cut: float = 1.0):
    shape = kx_mesh.shape
    k_rho_mesh = np.sqrt(kx_mesh ** 2.0 + ky_mesh ** 2.0)

    result = []
    with run_time(name="quad bessel_func list"):
        it = np.nditer([z_mesh, kz_mesh, k_rho_mesh], flags=['multi_index'])
        for z, kz, k_rho in it:
            # print(f"triple: {z}, {kz}, {k_rho}")
            print(it.multi_index)
            result.append(quad(bessel_func(z, k_rho, kz), rho_cut, np.inf)[0])
        result = np.array(result).reshape(kz_mesh.shape)

    return result


def get_V_k_val_ddi2(x_mesh: float,
                     y_mesh: float,
                     z_mesh: float,
                     rho_cut: float = 1.0,
                     z_cut: float = 1.0):
    with run_time(name="fft V_ddi"):
        rho2_mesh = x_mesh ** 2.0 + y_mesh ** 2.0
        rho_mesh = np.sqrt(rho2_mesh)
        r_mesh = np.sqrt(rho2_mesh + z_mesh ** 2.0)
        cos_theta = z_mesh / r_mesh

        ddi = (1.0 - cos_theta ** 2.0) / (r_mesh ** 3.0)
        zeros = np.zeros(shape=ddi.shape)
        cond_cylinder = np.logical_and(np.abs(z_mesh) < z_cut, rho_mesh < rho_cut)
        ddi_cut = np.where(cond_cylinder, ddi, zeros)

        V_k_val = np.fft.fftn(ddi_cut)

        V_k_val_real = V_k_val.real

    return V_k_val_real


def get_V_k_val_ddi3(x_mesh: float,
                     y_mesh: float,
                     z_mesh: float,
                     x_cut: List[float],
                     y_cut: List[float],
                     z_cut: List[float]):
    with run_time(name="fft V_ddi xyz cut"):
        r_mesh = np.sqrt(x_mesh ** 2.0 + y_mesh ** 2.0 + z_mesh ** 2.0)

        r_mesh_singular_free = np.where(r_mesh == 0.0, 1.0, r_mesh)
        cos_theta = z_mesh / r_mesh_singular_free

        ddi = (1.0 - 3.0 * cos_theta ** 2.0) / (r_mesh ** 3.0)
        zeros = np.zeros(shape=ddi.shape)
        cond_x = (x_cut[0] < x_mesh) & (x_mesh < x_cut[1])
        cond_y = (y_cut[0] < y_mesh) & (y_mesh < y_cut[1])
        cond_z = (z_cut[0] < z_mesh) & (z_mesh < z_cut[1])
        cond_xyz_cut = cond_x & cond_y & cond_z
        ddi_cut = np.where(cond_xyz_cut, ddi, zeros)

        V_k_val = np.fft.fftn(ddi_cut)

        V_k_val_real = V_k_val.real

    return V_k_val_real


def bessel_func(z, k_rho, kz):
    bessel = lambda rho: (
        - np.cos(kz * z) * rho
        * (rho ** 2.0 - 2.0 * z ** 2.0) / ((rho ** 2.0 + z ** 2.0) ** 2.5) * jv(0, rho * k_rho)
        )

    return bessel


def f_kappa(kappa: np.ndarray, epsilon: float = 10 ** -10) -> float:
    k2_1 = (kappa ** 2.0 - 1.0 + epsilon)
    result = ((2.0 * kappa ** 2.0 + 1.0) - (3.0 * kappa ** 2.0) * atan_special(
        k2_1)) / k2_1

    return result


@np.vectorize
def atan_special(x):
    if x > 0:
        result = np.arctan(np.sqrt(x)) / np.sqrt(x)
    elif x == 0:
        result = 0.0
    else:
        result = np.arctanh(np.sqrt(-x)) / np.sqrt(-x)

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
    kappa_array: np.ndarray = np.linspace(x_min, x_max, res,
                                          endpoint=False)
    y = func_125(kappa_array, alpha_z, e_dd)
    if y[-1] > 0:
        kappa_root = min(kappa_array[y >= 0.0])
    else:
        kappa_root = min(kappa_array[y <= 0.0])

    return kappa_root


def density_in_trap(x: float, y: float, z: float,
                    R_r: float, R_z: float, g: float = 0.0):
    r = np.sqrt(x ** 2.0 + y ** 2.0)
    n_0 = 15.0 / (8.0 * np.pi * R_z * R_r ** 2.0)
    a = (r / R_r) ** 2.0 + (z / R_z) ** 2.0

    n_r = np.where(a > 1, 0.0, n_0 * (1.0 - a))

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
               shape: Tuple[int, int, int] = (64, 64, 64)) -> np.ndarray:
    noise = min + (max - min) * np.random.rand(*shape)

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
    psi_0_2d = functools.partial(psi_gauss_2d_pdf, mu=np.array(
        [0.0, 0.0]), var=np.array([1.0, 1.0]))
    psi_0_3d = functools.partial(psi_gauss_3d, a=1, x_0=0, y_0=0, z_0=0, k_0=0)

    # testing for 2d plot
    L = 10
    x = np.linspace(-L, L, resolution, endpoint=False)
    y = np.linspace(-L, L, resolution, endpoint=False)
    x_mesh, y_mesh, pos = get_meshgrid(x, y)
