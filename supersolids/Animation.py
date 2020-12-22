#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from os import sep

from supersolids import Schroedinger, functions

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

        if self.dim == 1:
            self.fig, self.axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
            # TODO: Currently all subplots have the same plot, change that!
            for ax in plt.gcf().get_axes():
                self.psi_line, = ax.plot([], [], "x--", c="r", label=r'$|\psi(x)|^2$')
                self.V_line, = ax.plot([], [], ".-", c="k", label=r'$V(x)$')
                self.psi_sol_line, = ax.plot([], [], ".-", c="blue", label=r'$\psi_{sol(x)}$')

                self.title = ax.set_title("")
                ax.set_xlabel('$x$')
                ax.set_ylabel(r'$E$')
                ax.legend(prop=dict(size=12))
                ax.grid()

        elif self.dim == 2:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

            cmap = cm.coolwarm
            levels = 20
            # TODO: Currently all subplots have the same plot, change that!
            plot_args = {"label": r"$|\psi(x)|^2$", "cmap": cm.viridis, "linewidth": 5,
                         "rstride": 8, "cstride": 8, "alpha": 0.3}
            self.psi_line, = self.ax.plot([], [], [], label=r"$|\psi(x)|^2$")
            self.V_line, = self.ax.plot([], [], [], label=r'$V(x)$')
            # self.psi_sol, = self.ax.plot([], [], [], label=r'$n(x)$')

            # cbaxes = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
            # self.fig.colorbar(self.psi_line, cax=cbaxes)

            self.title = self.ax.set_title("")
            self.ax.set_xlabel(r'$x$')
            self.ax.set_ylabel(r'$y$')
            # self.ax.set_zlabel('r$E$')
            # self.ax.legend(prop=dict(size=12))
            self.ax.grid()
            # print(f"init done: {self.psi_line}")

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
        # checks if the initial wave function is normalized, if not it ensures to display the whole function
        if (psi_prob_max := psi_prob.max()) < (psi_abs_max := psi_abs.max()):
            y_min = psi_abs.min()
            y_max = psi_abs_max
        else:
            y_min = psi_prob.min()
            y_max = psi_prob_max

        self.set_limits(row, col, x_min, x_max, y_min, y_max)

    def get_V_plot_values(self, i, j, System: Schroedinger.Schroedinger):
        if System.dim == 1:
            ylim = self.axs[i, j].get_ylim()
            # as the plot should be completely shown in the box (we choose a reserve here: 1.5)
            reserve = 1.5
            range_in_box = System.x[(System.V_val < ylim[1] * reserve) & (System.V_val > ylim[0] * reserve)]

            V_pos = np.linspace(range_in_box[0], range_in_box[-1], System.resolution)
            V_plot_val = System.V(V_pos)

        elif System.dim == 2:
            zlim = self.ax.get_zlim()
            # as the plot should be completely shown in the box (we choose a reserve here: 1.5)
            reserve = 1.5
            range_in_box = System.pos[(System.V_val < zlim[1] * reserve) & (System.V_val > zlim[0] * reserve)]

            x = np.linspace(range_in_box[:, 0].min(), range_in_box[:, 0].max(), System.resolution)
            y = np.linspace(range_in_box[:, 1].min(), range_in_box[:, 1].max(), System.resolution)
            xx, yy, V_pos = functions.get_meshgrid(x, y)
            V_plot_val = System.V(V_pos)

        return V_pos, V_plot_val

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

        if frame_index == 0:
            if System.dim == 1:
                self.V_pos, self.V_plot_val = self.get_V_plot_values(0, 0, System)
            elif System.dim == 2:
                self.V_pos, self.V_plot_val = self.get_V_plot_values(0, 0, System)
                self.V_line = self.ax.plot_surface(self.V_pos[:, :, 0], self.V_pos[:, :, 1], self.V_plot_val,
                                                   cmap=cm.Blues, linewidth=5, rstride=8, cstride=8, alpha=0.7)

        # System.time_step()
        if frame_index % 10 == 0:
            print(f"Round {frame_index}")

        if System.dim == 1:
            self.psi_line.set_data(System.x, np.abs(System.psi_val) ** 2)
            self.V_line.set_data(self.V_pos, self.V_plot_val)
            self.psi_sol_line.set_data(System.x, System.psi_sol_val)
        elif System.dim == 2:
            if frame_index >= 2:
                test_factor = (1.0/(float(frame_index) + 1.0))
                self.psi_line = self.ax.plot_surface(System.x_mesh, System.y_mesh,
                                                     test_factor * np.abs(System.psi_val) ** 2,
                                                     cmap=cm.Greys, linewidth=5, rstride=8, cstride=8, alpha=0.3)

            cmap = cm.coolwarm
            levels = 20

            # self.psi_z = self.ax.contourf(System.x_mesh, System.y_mesh, System.z_mesh, zdir='z', offset=0.0, cmap=cmap, levels=levels)
            # self.psi_x = self.ax.contourf(System.x_mesh, System.y_mesh, System.z_mesh, zdir='x', offset=-System.L, cmap=cmap, levels=levels)
            # self.psi_y = self.ax.contourf(System.x_mesh, System.y_mesh, System.z_mesh, zdir='y', offset=System.L, cmap=cmap, levels=levels)
            # cbaxes = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
            # cbaxes = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])
            # self.fig.colorbar(p, cax=cbaxes)

        elif System.dim == 3:
            # TODO: z needs to be meshgrid too, how to use 3d meshgrids?
            self.psi_line.set_data(System.x_mesh, System.y_mesh, System.z, np.abs(System.psi_val) ** 2)

        self.title.set_text(("g = {:.2}, dt = {:.6}, timesteps = {:d}, "
                             "imag_time = {}, t = {:02.05f}").format(System.g,
                                                                     System.dt,
                                                                     System.timesteps,
                                                                     System.imag_time,
                                                                     System.t,
                                                                     ))

        if System.dim == 1:
            return self.psi_line, self.V_line, self.psi_sol_line, self.title
        else:
            return self.psi_line, self.V_line, self.title

    def start(self, System: Schroedinger.Schroedinger, file_name):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        file_name : String
                    Name of file including file type to save the animation to (tested with mp4)
        System: Schroedinger, object
                Defines the Schroedinger equation for a given problem
        """

        assert type(System) is Schroedinger.Schroedinger, ("System"
                                                           "needs to be {}, but it is {}".format(
                                                            Schroedinger.Schroedinger, type(System)))

        # blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       fargs=(System,), frames=System.timesteps, interval=30, blit=True)

        # requires either mencoder or ffmpeg to be installed on your system
        anim.save("results" + sep + file_name, fps=15, extra_args=['-vcodec', 'libx264'])


def plot_2d(L=1, resolution=32, alpha=0.6, x_lim = (-1, 1), y_lim = (-1, 1),  z_lim = (0, 1), **kwargs):
    """

    Parameters
    ----------
    X : meshgrid
    Y : meshgrid
    Z : meshgrid
    L : Project on this axis

    Returns
    -------

    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)

    cmap = cm.coolwarm
    levels = 20
    for key, values in kwargs.items():
        if key == "pos":
            if type(values) == list:
                pos = values
            else:
                pos = values

        elif key == "func":
            if type(values) == list:
                psi_pos_adjusted, psi_val_adjusted = get_V_plot_values(ax, pos[0], values[0], resolution)
                ax.plot_surface(psi_pos_adjusted[:, :, 0], psi_pos_adjusted[:, :, 1], psi_val_adjusted,
                                cmap=cm.viridis, linewidth=5, rstride=8, cstride=8, alpha=alpha[0])

                ax.contourf(psi_pos_adjusted[:, :, 0], psi_pos_adjusted[:, :, 1], psi_val_adjusted,
                            zdir='z', offset=0.0, cmap=cmap, levels=levels)
                ax.contourf(psi_pos_adjusted[:, :, 0], psi_pos_adjusted[:, :, 1], psi_val_adjusted,
                            zdir='x', offset=-L, cmap=cmap, levels=levels)
                p = ax.contourf(psi_pos_adjusted[:, :, 0], psi_pos_adjusted[:, :, 1], psi_val_adjusted,
                                zdir='y', offset=L, cmap=cmap, levels=levels)
                color_bar_axes = fig.add_axes([0.9, 0.1, 0.03, 0.8])

                fig.colorbar(p, cax=color_bar_axes)

                for i, func in enumerate(values[1:]):
                    pos_adjusted, V_val_adjusted = get_V_plot_values(ax, pos[i], func, resolution)
                    # ax.plot_surface(pos[i][:, :, 0], pos[i][:, :, 1], func(pos[i]),
                    ax.plot_surface(pos_adjusted[:, :, 0], pos_adjusted[:, :, 1], V_val_adjusted,
                                    cmap=cm.viridis, linewidth=5, rstride=8, cstride=8, alpha=alpha[i])
            else:
                ax.plot_surface(pos[:, :, 0], pos[:, :, 1], values(pos),
                                cmap=cm.viridis, linewidth=5, rstride=8, cstride=8, alpha=alpha)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def get_V_plot_values(ax, pos, V, resolution):
    Z = V(pos)
    zlim = ax.get_zlim()
    # as the plot should be completely shown in the box (we choose a reserve here: 1.5)
    reserve = 1.5
    range_in_box = pos[(Z < zlim[1] * reserve) & (Z > zlim[0] * reserve)]

    print(f"{Z[Z < 0.04]}")
    print(f"{range_in_box}")
    x = np.linspace(range_in_box[:, 0].min(), range_in_box[:, 0].max(), resolution)
    y = np.linspace(range_in_box[:, 1].min(), range_in_box[:, 1].max(), resolution)
    xx, yy, V_pos = functions.get_meshgrid(x, y)
    V_plot_val = V(V_pos)

    return V_pos, V_plot_val

