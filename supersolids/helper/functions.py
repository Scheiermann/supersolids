#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Functions for Potential and initial wave function :math:`\psi_0`

"""

import functools

import numpy as np
from scipy import stats
from typing import Tuple, Callable, Optional, List

from supersolids.helper import constants


class Resolution:
    """
    Specifies the resolution of the simulation in x, y, z directions (1D, 2D, 3D).

    """
    def __init__(self,
                 x: float,
                 y: Optional[float] = None,
                 z: Optional[float] = None):
        dim = 1
        if y is not None:
            dim = dim + 1
        if z is not None:
            dim = dim + 1

        self.dim = dim
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> List[Optional[float]]:
        return str([self.x, self.y, self.z])


class Box:
    """
    Specifies the ranges in which the simulation is calculated (1D, 2D or 3D).
    Needs to be given in pairs (x0, x1), (y0, y1), (z0, z1).

    """
    def __init__(self,
                 x0: float, x1: float,
                 y0: Optional[float] = None, y1: Optional[float] = None,
                 z0: Optional[float] = None, z1: Optional[float] = None):
        dim = 1
        if (y0 is None) or (y1 is None):
            assert (y0 is None) or (y1 is None) is None, "y0 and y1 needs to be given in combination."
        else:
            dim = dim + 1
        if (z0 is None) or (z1 is None):
            assert (z0 is None) or (z1 is None) is None, "z0 and z1 needs to be given in combination."
        else:
            dim = dim + 1

        self.dim = dim
        self.x0: float = x0
        self.x1: float = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def __str__(self) -> List[Optional[float]]:
        return str([self.x0, self.x1, self.y0, self.y1, self.z0, self.z1])

    def lengths(self) -> List[float]:
        """
        Calculates the box lengths in the directions available in order [x, y, z]

        :return: List of the box length in the directions available in order [x, y, z]
        """
        if (self.y0 is None) and (self.z0 is None):
            box_lengths = [(self.x1 - self.x0)]
        elif self.z0 is None:
            box_lengths = [(self.x1 - self.x0), (self.y1 - self.y0)]
        else:
            box_lengths = [(self.x1 - self.x0),
                           (self.y1 - self.y0),
                           (self.z1 - self.z0)]

        return box_lengths

    def min_length(self):
        return min(self.lengths())


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
    e_dd = a_dd / a_s
    a_s_l_ho_ratio = a_s / l_ho

    return a_s_l_ho_ratio, e_dd


def get_g_qf(N: int, a_s_l_ho_ratio: float, epsilon_dd: float):
    g_qf = (32.0 / (3.0 * np.sqrt(np.pi))
            * 4.0 * np.pi * a_s_l_ho_ratio ** (5.0 / 2.0)
            * N ** (3.0 / 2.0)
            * (1.0 + (3.0 / 2.0) * epsilon_dd ** 2.0))

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


def psi_gauss_2d(x, y, a: float = 1.0, x_0: float = 0.0,
                 y_0: float = 0.0, k_0: float = 0.0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0

    :param x: mathematical variable

    :param y: mathematical variable

    :param a: Amplitude of pulse

    :param x_0: Mean spatial x of pulse

    :param y_0: Mean spatial y of pulse

    :param k_0: Group velocity of pulse

    """

    return (a * np.sqrt(np.pi) ** (-0.5)
            * np.exp(-0.5 * (((x - x_0) ** 2 + (y - y_0) ** 2) / (a ** 2))
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

    :param a_x: Stretching factor in x direction

    :param a_y: Stretching factor in y direction

    :param a_z: Stretching factor in z direction

    :param x_0: Mean spatial x of pulse

    :param y_0: Mean spatial y of pulse

    :param z_0: Mean spatial z of pulse

    :param k_0: Group velocity of pulse

    """

    return ((a_x * a_y * a_z * np.pi ** (3.0 / 2.0)) ** (-0.5)
            * np.exp(-0.5 * (
                    ((x - x_0) / a_x) ** 2
                    + ((y - y_0) / a_y) ** 2
                    + ((z - z_0) / a_z) ** 2)
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
    kr = np.where(kr_singular == 0.0, 10 ** -8, kr_singular)

    r_cut = (1.0
             + (3.0 / kr ** 2.0) * np.cos(kr)
             - (3.0 / kr ** 3.0) * np.sin(kr)
             )

    return r_cut


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
    r_cut: np.ndarray = get_r_cut(k_mesh, r_cut=r_cut)

    r_cut[np.isnan(r_cut)] = 0.0

    V_k_val = r_cut * ((factor / k_squared_singular_free) - 1.0)

    # Remove singularities (at this point there should not be any)
    V_k_val[np.isnan(V_k_val)] = 0.0

    return V_k_val


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
    kappa_array: np.ndarray = np.linspace(x_min, x_max, res)
    y = func_125(kappa_array, alpha_z, e_dd)
    if y[-1] > 0:
        kappa_root = min(kappa_array[y >= 0.0])
    else:
        kappa_root = min(kappa_array[y <= 0.0])

    return kappa_root


def density_in_trap(x: float, y: float, z: float,
                    R_r: float, R_z: float, g: float = 0.0):
    r = np.sqrt(x ** 2 + y ** 2)
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
    x = np.linspace(-L, L, resolution)
    y = np.linspace(-L, L, resolution)
    x_mesh, y_mesh, pos = get_meshgrid(x, y)
