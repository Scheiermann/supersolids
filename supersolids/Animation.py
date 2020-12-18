#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib import colors
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
    def __init__(self, dim=1):
        """
        Creates an Animation for a Schroedinger equation
        Methods need the object Schroedinger with the parameters of the equation
        """

        self.dim = dim
        self.fig, self.axs = plt.subplots(nrows=1, ncols=1, squeeze=False)

        if self.dim == 1:
            # TODO: Currently all subplots have the same plot, change that!
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

        elif self.dim == 2:
            cmap = cm.coolwarm
            levels = 20

            # TODO: Currently all subplots have the same plot, change that!
            for ax in plt.gcf().get_axes():
                plot_args = {"label": r"$|\psi(x)|^2$","cmap": cm.viridis, "linewidth": 5,
                             "rstride": 8, "cstride": 8, "alpha": 0.3}
                self.psi_line, = ax.plot_surface([], [], [], **plot_args)
                print("lol")
                # self.V_line, = ax.plot_surface([], [], [], label=r'$V(x)$')
                # self.psi_exact, = ax.plot_surface([], [], [], label=r'$\psi_{sol(x)}$')
                # self.thomas_fermi, = ax.plot([], [], [], label=r'$n(x)$')

                cbaxes = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
                self.fig.colorbar(self.psi_line, cax=cbaxes)

                self.title = ax.set_title("")
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                ax.set_zlabel(r'$E$')
                ax.legend(prop=dict(size=12))
                ax.grid()
                print(f"init done: {self.psi_line}")

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
        psi_abs = np.abs(System.psi_val)
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

        if self.dim == 1:
            self.psi_line.set_data([], [])
            self.V_line.set_data([], [])
            self.psi_exact.set_data([], [])
            self.thomas_fermi.set_data([], [])
            self.title.set_text("")
        elif self.dim == 2:
            self.psi_line.set_data([], [], [])
            # self.V_line.set_data([], [], [])
            # self.psi_exact.set_data([], [], [])
            # self.thomas_fermi.set_data([], [], [])
            self.title.set_text("")

        if self.dim == 1:
            return self.psi_line, self.V_line, self.psi_exact, self.thomas_fermi, self.title
        else:
            return self.psi_line, self.title

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
        assert type(System) is Schroedinger.Schroedinger, ("System"
                                                           "needs to be {}, but it is {}".format(
            Schroedinger.Schroedinger, type(System)))

        System.time_step()
        if frame_index % 10 == 0:
            print(f"Round {frame_index}")
        #
        # x_V = np.linspace(System.x.min(), System.x.max(), 5 * System.resolution)
        # if System.dim == 1:
        #     self.V_line.set_data(x_V, System.V(x_V))
        #     self.psi_exact.set_data(x_V, functions.psi_gauss_solution(x_V))
        # elif System.dim == 2:
        #     self.V_line.set_data(x_V, System.V(x_V, x_V))
        #     self.psi_exact.set_data(x_V, functions.psi_gauss_solution(x_V, x_V))
        # elif System.dim == 3:
        #     self.V_line.set_data(x_V, System.V(x_V, x_V, x_V))
        #     self.psi_exact.set_data(x_V, functions.psi_gauss_solution(x_V, x_V, x_V))
        # else:
        #     print("Spatial dimension over 3. This is not implemented.", file=sys.stderr)
        #     sys.exit(1)

        if System.dim == 1:
            self.psi_line.set_data(System.x, np.abs(System.psi_val) ** 2)
        elif System.dim == 2:
            self.psi_line.set_data(System.x, System.y, np.abs(System.psi_val) ** 2)
        elif System.dim == 3:
            self.psi_line.set_data(System.x, System.y, System.z, np.abs(System.psi_val) ** 2)
        #
        # if System.g:
        #     self.thomas_fermi.set_data(x_V, functions.thomas_fermi(x_V, System.g))

        self.title.set_text(("g = {:.2}, dt = {:.6}, timesteps = {:d}, "
                             "imag_time = {}, t = {:02.05f}").format(System.g,
                                                                     System.dt,
                                                                     System.timesteps,
                                                                     System.imag_time,
                                                                     System.t,
                                                                     ))

        if System.dim == 1:
            return self.psi_line, self.V_line, self.psi_exact, self.thomas_fermi, self.title
        else:
            return self.psi_line, self.title

    def start(self, System: Schroedinger.Schroedinger, file_name):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        System: Schroedinger, object
                Defines the Schroedinger equation for a given problem
        """

        assert type(System) is Schroedinger.Schroedinger, ("System"
                                                           "needs to be {}, but it is {}".format(
            Schroedinger.Schroedinger, type(System)))

        # blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_func,
                                       fargs=(System,), frames=System.timesteps, interval=30, blit=True)

        # requires either mencoder or ffmpeg to be installed on your system
        anim.save("results" + sep + file_name, fps=15, extra_args=['-vcodec', 'libx264'])


def simulate_case(resolution, timesteps, L, dt, g, imag_time=False,
                  psi_0=functions.psi_pdf, V=functions.v_harmonic_1d, dim=1, file_name="split.mp4"):
    with parallel.run_time():
        Harmonic = Schroedinger.Schroedinger(resolution, timesteps, L, dt, g=g,
                                             imag_time=imag_time, psi_0=psi_0, V=V, dim=dim)

    ani = Animation(dim=dim)
    print("Harmonic")
    ani.set_limits(0, 0, -L, L, 0, 0.2)
    # ani.set_limits_smart(0, Harmonic)

    with parallel.run_time():
        ani.start(Harmonic, file_name)


def plot_2d(X, Y, Z, L):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=5, rstride=8, cstride=8, alpha=0.3)

    cmap = cm.coolwarm
    levels = 20

    ax.contourf(X, Y, Z, zdir='z', offset=0.0, cmap=cmap, levels=levels)
    ax.contourf(X, Y, Z, zdir='x', offset=-L, cmap=cmap, levels=levels)
    p = ax.contourf(X, Y, Z, zdir='y', offset=L, cmap=cmap, levels=levels)
    cbaxes = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(p, cax=cbaxes)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
