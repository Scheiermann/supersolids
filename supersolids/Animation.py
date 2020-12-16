#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from os import sep

from supersolids import functions
from supersolids import parallel
from supersolids import Schroedinger

"""
Implements animation

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""


class Animation:
    def __init__(self):
        """
        Creates an Animation for a Schroedinger equation
        Methods need the object Schroedinger with the parameters of the equation
        """

        self.fig, self.axs = plt.subplots(nrows=1, ncols=1, squeeze=False)

        for ax in plt.gcf().get_axes():
            self.psi_line, = ax.plot([], [], "x--", c="r", label=r'$|\psi(x)|^2$')
            self.V_line, = ax.plot([], [], ".-", c="k", label=r'$V(x)$')
            self.psi_exact, = ax.plot([], [], ".-", c="blue", label=r'$\psi_{sol(x)}$')
            self.thomas_fermi, = ax.plot([], [], ".-", c="green", label=r'$n(x)$')

            self.title = ax.set_title("")
            ax.set_xlabel('$x$')
            ax.set_ylabel(r'$E$')
            ax.legend(prop=dict(size=12))
            ax.grid()

    def set_limits(self, row, col, x_min, x_max, y_min, y_max):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        row : int, index
              row of the subplot for the animation
            
        col : int, index
              column of the subplot for the animation

        x_min : float, index
               minimum x value of subplot

        x_max : float, index
               maximum x value of subplot

        y_min : float, index
               minimum y value of subplot

        y_max : float, index
               maximum y value of subplot
        """

        y_lim = (y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))
        self.axs[row, col].set_xlim(x_min, x_max)
        self.axs[row, col].set_ylim(y_lim)

    def set_limits_smart(self, row, col, System: Schroedinger.Schroedinger):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        row: int, index
            row of the subplot for the animation

        col: int, index
            column of the subplot for the animation

        System: Schroedinger, object
                Defines the Schroedinger equation for a given problem
        """

        assert type(System) is Schroedinger.Schroedinger, ("System"
                                                           "needs to be {}, but it is {}".format(
            Schroedinger.Schroedinger, type(System)))

        x_min = -System.L
        x_max = System.L

        # Save calculations in variable to shortcut repeated calculations
        psi_abs = np.abs(System.psi)
        psi_prob = psi_abs ** 2

        # Use named expressions to shortcut repeated calculations
        # Basic check if the initial wave function is normalized, and ensures to th whole function display if not
        if (psi_prob_max := psi_prob.max()) < (psi_abs_max := psi_abs.max()):
            y_min = psi_abs.min()
            y_max = psi_abs_max
        else:
            y_min = psi_prob.min()
            y_max = psi_prob_max

        self.set_limits(row, col, x_min, x_max, y_min, y_max)

    def init_func(self):
        """
        Initializes lines for animation
        """

        self.psi_line.set_data([], [])
        self.V_line.set_data([], [])
        self.psi_exact.set_data([], [])
        self.thomas_fermi.set_data([], [])
        self.title.set_text("")

        return self.psi_line, self.V_line, self.title, self.psi_exact

    def animate(self, frame_index, System: Schroedinger.Schroedinger):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        frame_index: int, index
                     Current index of frame

        System: Schroedinger, object
                Defines the Schroedinger equation for a given problem
        """

        assert type(System) is Schroedinger.Schroedinger, ("System needs to be class Schroedinger,"
                                                           "but it is {}".format(type(System)))

        System.time_step()
        if frame_index % 10 == 0:
            print(f"Round {frame_index}")

        x_V = np.linspace(System.x.min(), System.x.max(), 5 * System.resolution)
        self.V_line.set_data(x_V, System.V(x_V))
        self.psi_line.set_data(System.x, np.abs(System.psi) ** 2)
        self.psi_exact.set_data(x_V, functions.psi_gauss_solution(x_V))
        if System.g:
            self.thomas_fermi.set_data(x_V, functions.thomas_fermi(x_V, System.g))

        self.title.set_text(("g = {:.2}, dt = {:.6}, timesteps = {:d}, "
                             "imag_time = {}, t = {:02.05f}").format(System.g,
                                                                     System.dt,
                                                                     System.timesteps,
                                                                     System.imag_time,
                                                                     System.t,
                                                                     ))

        return self.psi_line, self.V_line, self.psi_exact, self.thomas_fermi, self.title

    def start(self, System: Schroedinger.Schroedinger, file_name):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        System: Schroedinger, object
                Defines the Schroedinger equation for a given problem
        """

        assert type(System) is Schroedinger.Schroedinger, ("System needs to be class Schroedinger,"
                                                           "but it is {}".format(type(System)))

        # blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_func,
                                       fargs=(System,), frames=System.timesteps, interval=30, blit=True)

        # requires either mencoder or ffmpeg to be installed on your system
        anim.save("results" + sep + file_name, fps=15, extra_args=['-vcodec', 'libx264'])


def simulate_case(resolution, timesteps, L, dt, g, imag_time=False,
                  psi_0=functions.psi_0_pdf, V=functions.v_harmonic, file_name="split.mp4"):

    with parallel.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, timesteps, L, dt, g=g,
                                             imag_time=imag_time, psi_0=psi_0, V=V)

    ani = Animation()
    ani.set_limits(0, 0, -L, L, 0, 0.2)
    # ani.set_limits_smart(0, Harmonic)

    with parallel.run_time():
        ani.start(Harmonic, file_name)
