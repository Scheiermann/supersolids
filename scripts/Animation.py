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

        Parameters
        ----------
        Harmonic :  Schroedinger object from spli_operator.py
        """
        self.fig, self.axs = plt.subplots()

        for ax in plt.gcf().get_axes():
            self.psi_line, = ax.plot([], [], "x--", c="r", label=r'$|\psi(x)|^2$')
            self.V_line, = ax.plot([], [], ".-", c="k", label=r'$V(x)$')

            self.title = ax.set_title("")
            ax.set_xlabel('$x$')
            ax.set_ylabel(r'$E$')
            ax.legend(prop=dict(size=12))
            ax.grid()

    def set_limits(self, i, x_min, x_max, y_min, y_max):
        y_lim = (y_min - 0.2 * (y_max - y_min), y_max + 0.2 * (y_max - y_min))

        plt.gcf().get_axes()[i].set_xlim(x_min, x_max)
        plt.gcf().get_axes()[i].set_ylim(y_lim)


    def set_limits_smart(self, i, Harmonic):
        x_min = -Harmonic.L
        x_max = Harmonic.L

        # Save calculations in variable to shortcut repeated calculations
        psi_abs = np.abs(Harmonic.psi)
        psi_prob = psi_abs ** 2

        # Use named expressions to shortcut repeated calculations
        if (psi_prob_max := psi_prob.max()) < (psi_abs_max := psi_abs.max()):
            y_min = psi_abs.min()
            y_max = psi_abs_max
        else:
            y_min = psi_prob.min()
            y_max = psi_prob_max

        self.set_limits(i, x_min, x_max, y_min, y_max)

    def init_func(self):
        self.psi_line.set_data([], [])
        self.V_line.set_data([], [])
        self.title.set_text("")

        return self.psi_line, self.V_line, self.title

    def animate(self, i, Harmonic):
        Harmonic.time_step()
        if i % 10 == 0:
            print(f"Round {i}")
        # print(f"Round {i}:\n {Harmonic.psi}")
        x_V = np.linspace(Harmonic.x.min(), Harmonic.x.max(), 5 * Harmonic.resolution)
        self.V_line.set_data(x_V, Harmonic.V(x_V))
        self.psi_line.set_data(Harmonic.x, np.abs(Harmonic.psi) ** 2)

        self.title.set_text("t = %.2f" % Harmonic.t)

        return self.psi_line, self.V_line, self.title

    def start(self, Harmonic):
        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_func,
                                       fargs=(Harmonic,), frames=Harmonic.timesteps, interval=30, blit=True)

        # requires either mencoder or ffmpeg to be installed on your system
        anim.save('split.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
