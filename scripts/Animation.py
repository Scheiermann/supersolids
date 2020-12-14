#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

"""
Implements animation

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""


class Animation():
    def __init__(self):
        """
        Creates an Animation for a Schroedinger equation
        Methods need the object Schroedinger with the parameters of the equation
        """
        self.fig, self.axs = plt.subplots(nrows=1, ncols=1, squeeze=False)

        for ax in plt.gcf().get_axes():
            self.psi_line, = ax.plot([], [], "x--", c="r", label=r'$|\psi(x)|^2$')
            self.V_line, = ax.plot([], [], ".-", c="k", label=r'$V(x)$')

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

    def set_limits_smart(self, row, col, Schroedinger):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        row : int, index
              row of the subplot for the animation

        col : int, index
              column of the subplot for the animation

        Schroedinger : Schroedinger, object
                       Defines the Schroedinger equation for a given problem
        """
        x_min = -Schroedinger.L
        x_max = Schroedinger.L

        # Save calculations in variable to shortcut repeated calculations
        psi_abs = np.abs(Schroedinger.psi)
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
        self.title.set_text("")

        return self.psi_line, self.V_line, self.title

    def animate(self, frame_index, Schoredinger):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        frame_index: int, index
                     Current index of frame

        Schroedinger : Schroedinger, object
                       Defines the Schroedinger equation for a given problem
        """

        Schoredinger.time_step()
        if frame_index % 10 == 0:
            print(f"Round {frame_index}")

        x_V = np.linspace(Schoredinger.x.min(), Schoredinger.x.max(), 5 * Schoredinger.resolution)
        self.V_line.set_data(x_V, Schoredinger.V(x_V))
        self.psi_line.set_data(Schoredinger.x, np.abs(Schoredinger.psi) ** 2)

        self.title.set_text("t = %.2f" % Schoredinger.t)

        return self.psi_line, self.V_line, self.title

    def start(self, Schroedinger):
        """
        Sets the plot limits appropriate even if the initial wave function psi_0 is not normalized

        Parameters
        ----------
        Schroedinger : Schroedinger, object
                       Defines the Schroedinger equation for a given problem
        """

        # blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_func,
                                       fargs=(Schroedinger,), frames=Schroedinger.timesteps, interval=30, blit=True)

        # requires either mencoder or ffmpeg to be installed on your system
        anim.save('results/split.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
