#!/usr/bin/env python

"""
Numerical Solver for non-linear Time-Dependent Schrodinger's equation.

author: Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
from sympy.physics.quantum.operator import DifferentialOperator
from sympy import Symbol, symbols, Function, Piecewise, pi, N
from sympy.functions import sqrt, sin
from sympy.physics.units import hbar
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.state import Wavefunction
from sympy.physics.quantum import TimeDepBra


class Schroedinger(object):
    """
    Implements a numerical solution of the time-dependent
    non-linear Schrodinger equation for an arbitrary potential:
    i \hbar \frac{\partial}{\partial t} \psi(r,t) = \left(\frac{-\hbar^{2}}{2 m} \nabla^{2}
                                                    + V(r) + g |\psi(x,t)|^{2} \right) \psi(x,t)

    For the momemnt we aim to solve:
    \mu \phi_{0}(x) = \left(\frac{-1}{2} \frac{\partial^{2}}{\partial x^{2}}
                      + \frac{1}{2} x^{2} + \tilde{g} |\phi_{0}(x)|^{2} \right) \phi_{0}(x)

    We will first implement the split operator without commutator relation ($H = H_{pot} + H_{kin}$)
    WARNING: We don't use Baker-Campell-Hausdorff formula, hence the accuracy is small. This is just a draft.
    WARNING: Normalization of $\psi$ at every step needs to be checked, but is NOT implemented.
    """

    def __init__(self, x, psi_x0, V_x, k0 = None, hbar=1, m=1, t0=0.0):
        """
        Parameters
        ----------
        x : array_like, float
            description
    """
if __name__ == '__main__':
    # Script runs, if script is run as main script (called by python split_operator.py)
    x_real = Symbol('x', real=True)
    x, y, z, t = symbols('x y z t')
    k, m = symbols('k m', integer=True)
    f, g, h = symbols('f g h', cls=Function)



