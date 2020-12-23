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

            self.title = self.ax.set_title("")
            self.ax.set_xlabel(r'$x$')
            self.ax.set_ylabel(r'$y$')
            self.ax.set_zlabel('r$z$')
            self.ax.grid()

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

        # As V is constant, calculate and plot it just one time (at first frame)
        if frame_index == 0:
            if System.dim == 1:
                self.V_pos, self.V_plot_val = self.get_V_plot_values(0, 0, System)
            elif System.dim == 2:
                color_bar_axes = self.fig.add_axes([0.9, 0.1, 0.03, 0.8])

                self.V_pos, self.V_plot_val = self.get_V_plot_values(0, 0, System)
                System.V_line = self.ax.plot_surface(self.V_pos[:, :, 0], self.V_pos[:, :, 1], self.V_plot_val,
                                                   cmap=cm.Blues, linewidth=5, rstride=8, cstride=8, alpha=0.8)
        else:
            # Delete old plot, if it exists
            System.psi_line.remove()
            System.psi_x_line.remove()
            System.psi_y_line.remove()
            System.psi_z_line.remove()

        # System.time_step()
        if frame_index % 10 == 0:
            print(f"Round {frame_index}")

        if System.dim == 1:
            self.psi_line.set_data(System.x, np.abs(System.psi_val) ** 2)
            self.V_line.set_data(self.V_pos, self.V_plot_val)
            self.psi_sol_line.set_data(System.x, System.psi_sol_val)
        elif System.dim == 2:
            psi_pos, psi_val = crop_pos_to_limits(self.ax, System.pos, System.psi, func_val=System.psi_val)
            psi_prob = (0.8 ** frame_index) * np.abs(psi_val) ** 2
            print(f"frame: {frame_index}")
            print(f"prob: {psi_prob.max()}")
            System.psi_line = self.ax.plot_surface(psi_pos[:, :, 0], psi_pos[:, :, 1],
                                                 psi_prob,
                                                 cmap=cm.viridis, linewidth=5,
                                                 rstride=1, cstride=1, alpha=0.2)

            cmap = cm.coolwarm
            levels = 20

            System.psi_x_line = self.ax.contourf(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_prob,
                                                 zdir='x', offset=self.ax.get_xlim()[0], cmap=cmap, levels=levels)
            System.psi_y_line = self.ax.contourf(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_prob,
                                                 zdir='y', offset=self.ax.get_ylim()[0], cmap=cmap, levels=levels)
            System.psi_z_line = self.ax.contourf(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_prob,
                                                 zdir='z', offset=self.ax.get_zlim()[0], cmap=cmap, levels=levels)
            self.fig.colorbar(System.psi_x_line, cax=color_bar_axes)

        elif System.dim == 3:
            # TODO: z needs to be meshgrid too, how to use 3d meshgrids?
            System.psi_line.set_data(System.x_mesh, System.y_mesh, System.z, np.abs(System.psi_val) ** 2)

        self.title.set_text(("g = {:.2}, dt = {:.6}, timesteps = {:d}, imag_time = {},\n"
                             "t = {:02.05f}").format(System.g,
                                                     System.dt,
                                                     System.timesteps,
                                                     System.imag_time,
                                                     System.t,
                                                     ))

        if System.dim == 1:
            return self.psi_line, self.V_line, self.psi_sol_line, self.title
        else:
            # return System.psi_line, System.V_line, System.psi_x_line, System.psi_y_line, System.psi_z_line, self.title
            return System.psi_line, System.V_line, self.title

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
                                       fargs=(System,), frames=System.timesteps, interval=30,
                                       blit=True, cache_frame_data=False)

        # requires either mencoder or ffmpeg to be installed on your system
        anim.save("results" + sep + file_name, fps=15, dpi=300, extra_args=['-vcodec', 'libx264'])


def plot_2d(resolution=32, x_lim=(-1, 1), y_lim=(-1, 1),  z_lim=(0, 1), alpha=0.6, **kwargs):
    """

    Parameters
    ----------
    X : meshgrid
    Y : meshgrid
    Z : meshgrid

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
                psi_pos = values

        elif key == "func":
            if type(values) == list:
                psi_pos, psi_val = crop_pos_to_limits(ax, pos[0], values[0])
                ax.plot_surface(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_val,
                                cmap=cm.viridis, linewidth=5, rstride=1, cstride=1, alpha=alpha[0])

                for i, func in enumerate(values[1:], 1):
                    pos_adjusted, V_val_adjusted = get_V_plot_values(ax, pos[i], func, resolution)
                    ax.plot_surface(pos_adjusted[:, :, 0], pos_adjusted[:, :, 1], V_val_adjusted,
                                    cmap=cm.Blues, linewidth=5, rstride=1, cstride=1, alpha=alpha[i])
            else:
                psi_val = values(psi_pos)
                ax.plot_surface(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_val,
                                cmap=cm.viridis, linewidth=5, rstride=1, cstride=1, alpha=alpha)

            ax.contourf(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_val,
                        zdir='z', offset=z_lim[0], cmap=cmap, levels=levels)
            ax.contourf(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_val,
                        zdir='x', offset=x_lim[0], cmap=cmap, levels=levels)
            p = ax.contourf(psi_pos[:, :, 0], psi_pos[:, :, 1], psi_val,
                            zdir='y', offset=y_lim[0], cmap=cmap, levels=levels)
            color_bar_axes = fig.add_axes([0.9, 0.1, 0.03, 0.8])

            fig.colorbar(p, cax=color_bar_axes)

    ax.view_init(20.0, 45.0)
    ax.dist = 10.0

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def round_z_to_0(pos, func, tol=1.0e-5):
    z = func(pos)
    z.real[abs(z.real) < tol] = 0.0

    return pos, z


def crop_pos_to_limits(ax, pos, func, func_val=None):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    x = pos[:, :, 0][0]
    y = pos[:, :, 1][:, 0]
    if func_val is None:
        z = func(pos)
    else:
        z = func_val

    x_cropped = x[(x_lim[0] < x) & (x < x_lim[1])]
    y_cropped = y[(y_lim[0] < y) & (y < y_lim[1])]
    xx_cropped, yy_cropped, pos_cropped = functions.get_meshgrid(x_cropped, y_cropped)
    z_corresponding = func(pos_cropped)

    return pos_cropped, z_corresponding


def get_V_plot_values(ax, pos, V, resolution):
    Z = V(pos)
    zlim = ax.get_zlim()

    # as the plot should be completely shown in the box (we choose a reserve here: 1.5)
    reserve = 1.5
    range_in_box = pos[(Z < zlim[1] * reserve) & (Z > zlim[0] * reserve)]

    x = np.linspace(range_in_box[:, 0].min(), range_in_box[:, 0].max(), resolution)
    y = np.linspace(range_in_box[:, 1].min(), range_in_box[:, 1].max(), resolution)
    xx, yy, V_pos = functions.get_meshgrid(x, y)
    V_plot_val = V(V_pos)

    return V_pos, V_plot_val