#!/usr/bin/env python
"""
Script to test sympy.physics

author: Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import sympy as sypy
from scipy.stats import norm
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sympy import Symbol, symbols, Function, Piecewise, pi, N, fft, ifft, lambdify
from sympy.functions import sqrt, sin
from sympy.physics.quantum.state import Wavefunction
from sympy.physics.quantum import TimeDepBra
from sympy.stats import Normal, density

if __name__ == '__main__':
    # Script runs, if script is run as main script (called by python *.py)

    L = 1
    n = 1
    x_real = Symbol('x', real=True)
    x, y, z, t = symbols('x y z t')
    k, m = symbols('k m', integer=True)
    f, g, h = symbols('f g h', cls=Function)
    psi = TimeDepBra('psi', 't')

    mu = Symbol("mu")
    sigma = Symbol("sigma", positive=True)
    f = Normal("lo", mu, sigma)

    g = Piecewise((0, x_real < 0), (0, x_real > L), (sqrt(2 // L) * sin(n * pi * x_real / L), True))
    w = Wavefunction(g, x_real)
    print(N(w.norm))
    print(w.is_normalized)
    p = w.prob()
    print(N(p(0.85 * L)))

    x_arr = np.linspace(0, L, 2 ** 5)
    y_arr = [N(p(i)) for i in x_arr]
    y_arr_fft = fft(y_arr, dps=15)
    y_arr_fft_cycle = ifft(y_arr_fft, dps=15)

    gauss_function = density(f)(x)
    gauss_function_standard = lambdify(x, density(f)(x).subs(mu, 0).subs(sigma, 1), "numpy")

    gauss_function_standard_y_arr = gauss_function_standard(x_arr)
    gauss_fft = fft(gauss_function_standard_y_arr)
    gauss_fft_cycle = ifft(gauss_fft)

    print(gauss_function)
    print(gauss_function_standard)
    print(gauss_function_standard_y_arr)
    print(sypy.latex(gauss_function))

    gauss = norm.pdf(x_arr, loc=0.5*L, scale=0.1)
    gauss_fft_scipy = sp.fft.fft(gauss)
    gauss_fft_scipy_cycle = sp.fft.ifft(gauss_fft_scipy)

    # extremely slow (as it is a symbolic fft)
    # func = np.vectorize(gauss_function_standard)
    # func_y = fft(func(x_arr))
    # func_cycle = ifft(func_y)
    # print(func_cycle)

    # Plot
    plot_rows = 2
    fig, axs = plt.subplots(plot_rows, 1, figsize=(25, 10))
    fig.tight_layout()

    colors = [
        "tab:cyan",
        "tab:pink",
        "tab:green",
        "tab:red",
        "tab:orange",
        "tab:blue",
        "tab:purple",
        "tab:gray",
    ]

    axs[0].set_title(r"harmnonic solution in 1D Box $0 < L = 10$")
    axs[0].plot(x_arr, y_arr, ".--", label="before fft", color=colors[0])
    axs[0].plot(x_arr, np.abs(y_arr_fft_cycle), "x", label="after double fft", color=colors[1])

    axs[1].set_title(r"gauss in 1D Box $0 < L = 10$")
    # axs[1].plot(x_arr, gauss_function_standard_y_arr, "x", label="gauss lambdified with x_arr", color=colors[0])
    # axs[1].plot(x_arr, np.abs(gauss_fft_cycle), "x", label="gaus lambdified fft cycle", color=colors[1])
    axs[1].plot(x_arr, gauss, '.--', lw=1, alpha=1.0, label='before fft', color=colors[0])
    axs[1].plot(x_arr, np.abs(gauss_fft_scipy_cycle), 'x-', lw=1, alpha=1.0, label='after double fft',
                color=colors[1])

    for i in range(0, 2):
        axs[i].legend()
        axs[i].grid()

    plt.savefig("double_fft.png")
    plt.show()
