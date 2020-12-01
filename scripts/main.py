#!/usr/bin/env python

from scipy.constants import hbar
from scipy.constants import pi
from scipy.constants import mu_0
from scipy.constants import m_u
import scipy.constants as sci_const

if __name__ == "__main__":
    a_0 = sci_const.physical_constants["Bohr radius"][0]
    mu_b = sci_const.physical_constants["Bohr magneton"][0]
    a = 100 * a_0
    mu = 6 * mu_b
    m_chromium = 52 * m_u
    g = 4 * pi * hbar ** 2 * a / m_chromium

    print(mu_0 * mu ** 2 / (3 * g))
