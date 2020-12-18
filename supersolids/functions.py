#!/usr/bin/env python

"""
Functions for Potential and initial wave function psi_0

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from scipy import stats


def psi_gauss_2d(x, y, mu=np.array([0.0, 0.0]), var=np.array([1.0, 1.0])):
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    print(pos.shape)
    cov = np.diag(var ** 2)
    rv = stats.multivariate_normal(mean=mu, cov=cov)
    Z = rv.pdf(pos)

    return Z

def psi_gauss_3d(x, y, z, a, x_0=0.0, y_0=0.0, z_0=0.0, k_0=0.0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0

     Parameters
     ----------
     x : sympy.symbol
         mathematical variable

     y : sympy.symbol
         mathematical variable

     z : sympy.symbol
         mathematical variable

     a : float
        Amplitude of pulse

     x_0 : float
           Mean spatial x of pulse

     y_0 : float
           Mean spatial y of pulse

     z_0 : float
           Mean spatial z of pulse

     k_0 : float
           Group velocity of pulse
    """

    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * (((x - x_0) * 1.0) ** 2
                            + ((y - y_0) * 1.0) ** 2
                            + ((z - z_0) * 1.0) ** 2) / (a ** 2) + 1j * x * k_0))


def psi_gauss(x, a, x_0=0.0, k_0=0.0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0

     Parameters
     ----------
     x : sympy.symbol
         mathematical variable

     a : float
        Amplitude of pulse

     x_0 : float
           Mean spatial x of pulse

     k_0 : float
           Group velocity of pulse
    """

    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x_0) * 1. / a) ** 2 + 1j * x * k_0))


def psi_pdf(x, loc=0.0, scale=1.0):
    """
    Mathematical function of gauss pulse

    Parameters
    ----------
    x: sympy.symbol
        mathematical variable

    loc: float
        Localization of pulse centre

    scale: float
        Scale of pulse
    """
    return stats.norm.pdf(x, loc=loc, scale=scale)


def psi_rect(x, x_min, x_max, a):
    """
    Mathematical function of rectengular pulse between x_min and x_max with amplitude a

    Parameters
    ----------
    x: sympy.symbol
        mathematical variable

    x_min: float
        Minimum x value of pulse (spatial)

    x_max: float
        Maximum x value of pulse (spatial)

    a: float
        Amplitude of pulse
    """

    pulse = np.select([x < x_min, x < x_max, x_max < x], [0, a, 0])
    assert pulse.any(), ("Pulse is completely 0. Resolution is too small. "
        "Resolution needs to be set as fft is used onto the pulse.")

    return pulse


def psi_gauss_solution(x):
    """
     Mathematical function of solution of non-linear Schroedinger for g=0

     Parameters
     ----------
     x: sympy.symbol
        mathematical variable
    """

    return np.exp(-x ** 2) / np.sqrt(np.pi)


def thomas_fermi(x, g):
    """
     Mathematical function of Thomas-Fermi distribution with coupling constant g

     Parameters
     ----------
     x : sympy.symbol
         mathematical variable

     g : float
        coupling constant
    """

    # mu is the chemical potential
    mu = ((3 * g) / (4 * np.sqrt(2))) ** (2/3)
    # print(mu)

    # this needs to be >> 1, e.g 5.3
    # print(np.sqrt(2 * mu))

    return mu * (1 - ((x ** 2) / (2 * mu))) / g


def v_harmonic_1d(x):
    return 0.5 * x ** 2


def v_harmonic_2d(x, y):
    return 0.5 * (x ** 2 + y ** 2)


def v_harmonic_3d(x, y, z):
    return 0.5 * (x ** 2 + y ** 2 + z ** 2)
