#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@itp.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information.

"""
Numerical solver for non-linear time-dependent Schrodinger equation.

"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from supersolids.SchroedingerSummary import SchroedingerSummary


class SchroedingerMixtureSummary(SchroedingerSummary):
    """
    Saves the properties of a Schroedinger system without the arrays,
    to save disk space when saving it with dill (pickle).
    """

    def __init__(self, SystemMixture) -> None:
        """
        Schrödinger equations for the specified system.

        """
        super().__init__(SystemMixture)

        self.a_11_bohr = SystemMixture.a_11_bohr
        self.a_12_bohr = SystemMixture.a_12_bohr
        self.a_22_bohr = SystemMixture.a_22_bohr
        self.N2 = SystemMixture.N2
        self.m1 = SystemMixture.m1
        self.m2 = SystemMixture.m2
        self.mu_1 = SystemMixture.mu_1
        self.mu_2 = SystemMixture.mu_2
        self.psi2_0 = SystemMixture.psi2_0
        self.psi2_0_noise = SystemMixture.psi2_0_noise
        self.input_path = SystemMixture.input_path

    def copy_to(self, System):
        System.a_11_bohr: float = self.a_11_bohr
        System.a_12_bohr: float = self.a_12_bohr
        System.a_22_bohr: float = self.a_22_bohr
        System.N2: float = self.N2
        System.m1: float = self.m1
        System.m2: float = self.m2
        System.mu_1: float = self.mu_1
        System.mu_2: float = self.mu_2
        System.psi2_0: Callable = self.psi2_0
        System.psi2_0_noise: np.ndarray = self.psi2_0_noise
        System.input_path: Path = self.input_path
