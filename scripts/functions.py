#!/usr/bin/env python

"""
Functions for Potential and initial wave function psi_0

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from scipy.stats import norm


def psi_0_gauss(x, a, x_0, k_0):
    """
    Gaussian wave packet of width a and momentum k_0, centered at x_0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x_0) * 1. / a) ** 2 + 1j * x * k_0))


def psi_0_pdf(x, L):
    return norm.pdf(x, loc=-0.1 * L, scale=1.0)


def psi_0_rect(x, x_min, x_max, a):
    pulse = np.select([x < x_min, x < x_max, x_max < x], [0, a, 0])
    assert pulse.any(), ("Pulse is completely 0. Resolution is too small. "
        "Resolution needs to be set as fft is used onto the pulse.")

    return pulse
