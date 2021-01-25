#!/usr/bin/env python

# author: Daniel Scheiermann
# email: daniel.scheiermann@stud.uni-hannover.de
# license: MIT
# Please feel free to use and modify this, but keep the above information. Thanks!

"""
Implements animation

"""
import sys
from os import sep
from typing import Callable, Tuple, List

import numpy as np
from matplotlib import animation
from matplotlib import cm
from matplotlib import pyplot as plt

from supersolids import Schroedinger, functions



class Animation:
    def __init__(self,
                 dim=2,
                 camera_r_func=None,
                 camera_phi_func=None,
                 camera_z_func=None,
                 ):
        """
        Creates an Animation for a Schroedinger equation for the 1D or 2D case.
        Methods need the object Schroedinger with the parameters of the equation

        Parameters

        dim : int
            Dimension of the underlying Schroedinger to animate.

        camera_r_func : Callable, function
            r component of the movement of the camera.

        camera_phi_func : Callable, function
            phi component of the movement of the camera.

        camera_z_func : Callable, function
            z component of the movement of the camera.

        camera_r_0 : float
            r component of the starting point of the camera movement.

        camera_phi_0 : float
            phi component of the starting point of the camera movement.

        camera_z_0 : float
            z component of the starting point of the camera movement.

        """
        self.dim = dim
        assert 1 <= self.dim <= 2, ("Spatial dimension needs to be 1 or 2, "
                                    f"but it is {self.dim}."
                                    "This is not implemented.")

        # matplotlib
        if self.dim == 1:
            self.fig, self.axs = plt.subplots(nrows=1, ncols=1, squeeze=False)
            # TODO: Currently all subplots have the same plot, change that!
            for ax in plt.gcf().get_axes():
                self.psi_line, = ax.plot([], [],
                                         "x--", c="r", label=r'$|\psi(x)|^2$')
                self.V_line, = ax.plot([], [],
                                       ".-", c="k", label=r'$V(x)$')
                self.psi_sol_line, = ax.plot([], [],
                                             ".-", c="blue",
                                             label=r'$\psi_{sol(x)}$')

                self.title = ax.set_title("")
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$E$')
                ax.legend(prop=dict(size=12))
                ax.grid()

        elif self.dim == 2:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

            self.title = self.ax.set_title("")
            self.ax.set_xlabel(r'$x$')
            self.ax.set_ylabel(r'$y$')
            self.ax.set_zlabel(r'$z$')
            self.ax.grid()

            self.r_func = camera_r_func
            self.phi_func = camera_phi_func
            self.z_func = camera_z_func

            camera_r_0, camera_phi_0, camera_z_0 = functions.camera_3d_trajectory(
                0,
                r_func=self.r_func,
                phi_func=self.phi_func,
                z_func=self.z_func)

            self.ax.dist = camera_r_0
            self.ax.azim = camera_phi_0
            self.ax.elev = camera_z_0

    def set_limits(self, row: int, col: int,
                   x_min: float, x_max: float,
                   y_min: float, y_max: float):
        """
        Sets the plot limits appropriate,
        even if the initial wave function :math:`\psi_0` is not normalized.

        Parameters

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

    def set_limits_smart(self, row: int, col: int,
                         System: Schroedinger.Schroedinger):
        """
        Sets the plot limits appropriate,
        even if the initial wave function :math:`\psi_0` is not normalized.

        Parameters

        row: int, index
            row of the subplot for the animation

        col: int, index
            column of the subplot for the animation

        System: Schroedinger, object
            Defines the Schroedinger equation for a given problem
        """

        assert isinstance(System, Schroedinger.Schroedinger), (
            f"System needs to be {Schroedinger.Schroedinger},"
            f"but it is {type(System)}")
        assert self.dim == System.dim, ("Spatial dimension Animation and "
                                        "Schroedinger needs to be equal, "
                                        "but Animation.dim is {self.dim} "
                                        f"and Schroedinger.dim is {System.dim}")

        try:
            x_min = System.box["x0"]
            x_max = System.box["x1"]
        except KeyError:
            sys.exit(f"Keys x0, x1 of box needed, "
                     f"but it has the keys: {System.box.keys()}.")

        # Save calculations in variable to shortcut repeated calculations
        psi_abs = np.abs(System.psi_val)
        psi_prob = psi_abs ** 2

        # No Assignment expressions for versions < python3.8
        # (readthedocs can't parse them)

        # checks if the initial wave function is normalized,
        # if not it ensures to display the whole function
        psi_prob_max = psi_prob.max()
        psi_abs_max = psi_abs.max()
        if psi_prob_max < psi_abs_max:
            y_min = psi_abs.min()
            y_max = psi_abs_max
        else:
            y_min = psi_prob.min()
            y_max = psi_prob_max

        self.set_limits(row, col, x_min, x_max, y_min, y_max)

    def get_V_plot_values(self, i: int, j: int,
                          System: Schroedinger.Schroedinger,
                          reserve: float = 1.0):
        if System.dim == 1:
            ylim = self.axs[i, j].get_ylim()
            # as the plot should be completely shown in the box
            # (we choose a reserve here: 1.5)
            range_in_box = System.x[(System.V_val < ylim[1] * reserve)
                                    & (System.V_val > ylim[0] * reserve)]

            V_pos = np.linspace(range_in_box[0],
                                range_in_box[-1],
                                System.res["x"])
            V_plot_val = System.V(V_pos)

        elif System.dim == 2:
            zlim = self.ax.get_zlim()
            # as the plot should be completely shown in the box
            # (we choose a reserve here: 1.5)
            reserve = 1.5
            range_in_box = System.pos[(System.V_val < zlim[1] * reserve)
                                      & (System.V_val > zlim[0] * reserve)]

            x = np.linspace(range_in_box[:, 0].min(),
                            range_in_box[:, 0].max(),
                            System.res["x"])
            y = np.linspace(range_in_box[:, 1].min(),
                            range_in_box[:, 1].max(),
                            System.res["y"])
            _, _, V_pos = functions.get_meshgrid(x, y)
            V_plot_val = System.V(V_pos)

        return V_pos, V_plot_val

    def animate(self, frame_index: int,
                System: Schroedinger.Schroedinger,
                accuracy: float = 10 ** -6,
                plot_psi_sol: bool = False,
                plot_V: bool = True,
                ):
        """
        Sets the plot limits appropriate,
        even if the initial wave function :math:`\psi_0` is not normalized.

        Parameters

        frame_index: int, index
            Current index of frame

        System: Schroedinger, object
            Defines the Schroedinger equation for a given problem

        accuracy : float
            Convergence is reached when relative error of mu is smaller
            than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

        plot_psi_sol : bool
            Condition if :math:`\psi_{sol}` should be plotted.

        plot_V : bool
            Condition if V should be plotted.

        """
        assert isinstance(System, Schroedinger.Schroedinger), (
            f"System needs to be {Schroedinger.Schroedinger},"
            f"but it is {type(System)}")
        assert self.dim == System.dim, ("Spatial dimension Animation and "
                                        "Schroedinger needs to be equal, "
                                        "but Animation.dim is {self.dim} "
                                        f"and Schroedinger.dim is {System.dim}")

        # As V is constant, calculate and plot it just one time (at first
        # frame)
        if frame_index == 0:
            if System.dim == 1:
                if plot_V:
                    self.V_pos, self.V_plot_val = self.get_V_plot_values(
                        0, 0, System, reserve=1.0)
            elif System.dim == 2:
                if plot_V:
                    self.V_pos, self.V_plot_val = self.get_V_plot_values(
                        0, 0, System, reserve=1.0)
                    self.V_line = self.ax.plot_surface(self.V_pos[:, :, 0],
                                                       self.V_pos[:, :, 1],
                                                       self.V_plot_val,
                                                       cmap=cm.Blues,
                                                       linewidth=5,
                                                       rstride=8, cstride=8,
                                                       alpha=System.alpha_V
                                                       )

                    self.V_z_line = self.ax.contourf(self.V_pos[:, :, 0],
                                                     self.V_pos[:, :, 1],
                                                     self.V_plot_val,
                                                     zdir='z',
                                                     offset=self.ax.get_zlim()[
                                                            0],
                                                     cmap=cm.Blues,
                                                     levels=20,
                                                     alpha=System.alpha_V
                                                     )

        # FuncAnimation calls animate 3 times with frame_index=0,
        # causing problems with removing corresponding plot_lines
        # so we want to start plotting with frame_index=1,
        # and delete from upon the next frame, hence frame_index>=2
        if frame_index >= 2:
            if System.dim == 2:
                # Delete old plot, if it exists
                self.psi_line.remove()

                # Delete old contours, if they exists.
                # psi_x_line is a ContourPlotSet without remove,
                # collections gives a list of PolyColletion with remove
                for contour in self.psi_x_line.collections:
                    contour.remove()
                for contour in self.psi_y_line.collections:
                    contour.remove()
                for contour in self.psi_z_line.collections:
                    contour.remove()

            mu_old = System.mu
            System.time_step()
            mu_rel = np.abs((System.mu - mu_old) / System.mu)
            print(f"mu_rel: {mu_rel}")
            if mu_rel < accuracy:
                print(f"accuracy reached: {mu_rel}")
                self.anim.event_source.stop()

        if frame_index % 10 == 0:
            print(f"Round {frame_index}")

        if System.dim == 1:
            self.psi_line.set_data(System.x, np.abs(System.psi_val) ** 2.0)
            if plot_V:
                self.V_line.set_data(self.V_pos, self.V_plot_val)
            if plot_psi_sol:
                self.psi_sol_line.set_data(System.x, System.psi_sol_val)
        elif System.dim == 2:
            if frame_index >= 1:
                # rotate camera
                camera_r, camera_phi, camera_z = functions.camera_3d_trajectory(
                    frame_index,
                    r_func=self.r_func,
                    phi_func=self.phi_func,
                    z_func=self.z_func,
                    )

                self.ax.dist = camera_r
                self.ax.azim = camera_phi
                self.ax.elev = camera_z

                # here we crop the calculated mesh to the viewable mesh,
                # but the rest is still calculated to not change
                # the boundary conditions. Essentially we just zoom.
                psi_pos, psi_val = crop_pos_to_limits(self.ax,
                                                      System.pos,
                                                      System.psi,
                                                      func_val=System.psi_val)
                psi_prob = np.abs(psi_val) ** 2.0
                self.psi_line = self.ax.plot_surface(psi_pos[:, :, 0],
                                                     psi_pos[:, :, 1],
                                                     psi_prob,
                                                     cmap=cm.viridis,
                                                     linewidth=5,
                                                     rstride=1,
                                                     cstride=1,
                                                     alpha=System.alpha_psi
                                                     )

                cmap = cm.coolwarm
                levels = 20

                self.psi_x_line = self.ax.contourf(psi_pos[:, :, 0],
                                                   psi_pos[:, :, 1],
                                                   psi_prob,
                                                   zdir='x',
                                                   offset=self.ax.get_xlim()[
                    0],
                    cmap=cmap,
                    levels=levels)
                self.psi_y_line = self.ax.contourf(psi_pos[:, :, 0],
                                                   psi_pos[:, :, 1],
                                                   psi_prob,
                                                   zdir='y',
                                                   offset=self.ax.get_ylim()[
                    0],
                    cmap=cmap,
                    levels=levels)
                self.psi_z_line = self.ax.contourf(psi_pos[:, :, 0],
                                                   psi_pos[:, :, 1],
                                                   psi_prob,
                                                   zdir='z',
                                                   offset=self.ax.get_zlim()[
                    0],
                    cmap=cmap,
                    levels=levels
                )
                if frame_index == 1:
                    color_bar_axes = self.fig.add_axes([0.85, 0.1, 0.03, 0.8])
                    self.fig.colorbar(self.psi_x_line, cax=color_bar_axes)

        self.title.set_text(f"g = {System.g:.2}, dt = {System.dt:.6}, "
                            f"max_timesteps = {System.max_timesteps:d}, "
                            f"imag_time = {System.imag_time},\n"
                            f"t = {System.t:02.05f}")

        if System.dim == 1:
            return self.psi_line, self.V_line, self.psi_sol_line, self.title
        else:
            if plot_V:
                if frame_index == 0:
                    return self.V_line, self.title
                else:
                    return self.psi_line, self.V_line, self.title
            else:
                if frame_index == 0:
                    return self.title,
                else:
                    return self.psi_line, self.title

    def start(self, System: Schroedinger.Schroedinger,
              filename: str = "anim.mp4",
              accuracy: float = 10 ** -6,
              plot_psi_sol: bool = False,
              plot_V: bool = True
              ):
        """
        Sets the plot limits appropriate,
        even if the initial wave function :math:`\psi_0` is not normalized

        Parameters

        filename : String
            Name of file including file type to save the animation to
            (tested with mp4)

        accuracy : float
            Convergence is reached when relative error of mu is smaller
            than accuracy, where :math:`\mu = - \\log(\psi_{normed}) / (2 dt)`

        System: Schroedinger, object
            Defines the Schroedinger equation for a given problem

        plot_psi_sol :
            Condition if :math:`\psi_{sol}` should be plotted.

        plot_V : bool
            Condition if V should be plotted.

        """
        assert isinstance(System, Schroedinger.Schroedinger), (
            f"System needs to be {Schroedinger.Schroedinger},"
            f"but it is {type(System)}")
        assert self.dim == System.dim, ("Spatial dimension Animation and "
                                        "Schroedinger needs to be equal, "
                                        "but Animation.dim is {self.dim} "
                                        f"and Schroedinger.dim is {System.dim}")

        # blit=True means only re-draw the parts that have changed.
        self.anim = animation.FuncAnimation(self.fig, self.animate,
                                            fargs=(System,
                                                   accuracy,
                                                   plot_psi_sol,
                                                   plot_V),
                                            frames=System.max_timesteps,
                                            interval=30,
                                            blit=True,
                                            cache_frame_data=False)

        # requires either mencoder or ffmpeg to be installed on your system
        self.anim.save("results" + sep + filename,
                       fps=15, dpi=300, extra_args=['-vcodec', 'libx264'])


def plot_2d(resolution=32,
            x_lim: Tuple[float, float] = (-1, 1),
            y_lim: Tuple[float, float] = (-1, 1),
            z_lim: Tuple[float, float] = (0, 1),
            alpha: List[float] = [0.6],
            **kwargs):
    """

    Parameters

    resolution : int
        number of grid points in one direction

    x_lim : Tuple[float, float]
        Limits of plot in x direction

    y_lim : Tuple[float, float]
        Limits of plot in y direction

    z_lim : Tuple[float, float]
        Limits of plot in z direction

    alpha : float
        alpha value for plot transparency

    Returns


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
            if isinstance(values, list):
                pos = values
            else:
                psi_pos = values

        elif key == "func":
            if isinstance(values, list):
                psi_pos, psi_val = crop_pos_to_limits(ax, pos[0], values[0])
                ax.plot_surface(psi_pos[:, :, 0],
                                psi_pos[:, :, 1],
                                psi_val,
                                cmap=cm.viridis,
                                linewidth=5,
                                rstride=1,
                                cstride=1,
                                alpha=alpha[0])

                for i, func in enumerate(values[1:], 1):
                    pos_adjusted, V_val_adjusted = get_V_plot_values(ax,
                                                                     pos[i],
                                                                     func,
                                                                     resolution,
                                                                     reserve=1.0
                                                                     )
                    ax.plot_surface(pos_adjusted[:, :, 0],
                                    pos_adjusted[:, :, 1],
                                    V_val_adjusted,
                                    cmap=cm.Blues,
                                    linewidth=5,
                                    rstride=1,
                                    cstride=1,
                                    alpha=alpha[i])
            else:
                psi_val = values(psi_pos)
                ax.plot_surface(psi_pos[:, :, 0],
                                psi_pos[:, :, 1],
                                psi_val,
                                cmap=cm.viridis,
                                linewidth=5,
                                rstride=1,
                                cstride=1,
                                alpha=alpha)

            ax.contourf(psi_pos[:, :, 0],
                        psi_pos[:, :, 1],
                        psi_val,
                        zdir='z',
                        offset=z_lim[0],
                        cmap=cmap,
                        levels=levels)
            ax.contourf(psi_pos[:, :, 0],
                        psi_pos[:, :, 1],
                        psi_val,
                        zdir='x',
                        offset=x_lim[0],
                        cmap=cmap,
                        levels=levels)
            p = ax.contourf(psi_pos[:, :, 0],
                            psi_pos[:, :, 1],
                            psi_val,
                            zdir='y',
                            offset=y_lim[0],
                            cmap=cmap,
                            levels=levels)
            color_bar_axes = fig.add_axes([0.9, 0.1, 0.03, 0.8])

            fig.colorbar(p, cax=color_bar_axes)

    ax.view_init(20.0, 45.0)
    ax.dist = 10.0

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def round_z_to_0(pos, func, tol: float = 1.0e-5):
    z = func(pos)
    z.real[abs(z.real) < tol] = 0.0

    return pos, z


def crop_pos_to_limits(ax, pos, func, func_val=None):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    x = pos[:, :, 0][0]
    y = pos[:, :, 1][:, 0]

    # crop x and y 1D arrays to plot axis limits
    x_cropped = x[(x_lim[0] <= x) & (x <= x_lim[1])]
    y_cropped = y[(y_lim[0] <= y) & (y <= y_lim[1])]
    xx_cropped, _, pos_cropped = functions.get_meshgrid(x_cropped, y_cropped)

    if func_val is None:
        z_cropped = func(pos_cropped)
    else:
        # As func_val is calculated for whole pos,
        # which may be larger than the plot axis limits
        # (i.e to see one specific area,
        # but don't change the boundary conditions), it needs to be cropped too
        x_bol = np.where((x_lim[0] <= x) & (x <= x_lim[1]), True, False)
        y_bol = np.where((y_lim[0] <= y) & (y <= y_lim[1]), True, False)
        xx_bol, yy_bol, _ = functions.get_meshgrid(x_bol, y_bol)

        # Reshape is needed as func_val[xx_bol & yy_bol] is a 1D array (flat)
        z_cropped = np.reshape(func_val[xx_bol & yy_bol], xx_cropped.shape)

    return pos_cropped, z_cropped


def get_V_plot_values(ax, pos, V, resolution: int, reserve: float = 1.0):
    Z = V(pos)
    zlim = ax.get_zlim()

    # as the plot should be completely shown in the box
    # (we choose a reserve here: 1.5)
    range_in_box = pos[(Z < zlim[1] * reserve) & (Z > zlim[0] * reserve)]

    x = np.linspace(range_in_box[:, 0].min(),
                    range_in_box[:, 0].max(),
                    resolution)
    y = np.linspace(range_in_box[:, 1].min(),
                    range_in_box[:, 1].max(),
                    resolution)
    _, _, V_pos = functions.get_meshgrid(x, y)
    V_plot_val = V(V_pos)

    return V_pos, V_plot_val
