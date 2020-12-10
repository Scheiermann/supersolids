#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

"""
Implements animation

author: Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""


# TODO: Build class for easy use animation
class Animation():
    def __init__(self, Harmonic):
        """

        Parameters
        ----------
        Harmonic :  Schroedinger object from spli_operator.py
        """
        self.fig = plt.figure()

        xlim = (0, Harmonic.L)
        ymin = abs(Harmonic.psi).min()
        ymax = abs(Harmonic.psi).max()

        ax1 = self.fig.add_subplot(111, xlim=xlim,
                              ylim=(ymin - 0.2 * (ymax - ymin),
                                    ymax + 0.2 * (ymax - ymin)))

        self.title = ax1.set_title("")
        # ax1.legend(prop=dict(size=12))
        ax1.set_xlabel('$x$')
        ax1.set_ylabel(r'$|\psi(x)|$')

        self.psi_x_line, = ax1.plot([], [], "x--", c="r", label=r'$|\psi(x)|$')
        self.V_x_line, = ax1.plot([], [], ".-", c="k", label=r'$V(x)$')
        self.Harmonic = Harmonic

    def build_animation(self):

        self.V_x_line.set_data(self.Harmonic.x, self.Harmonic.V(self.Harmonic.x))

        # call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, frames=self.Harmonic.timesteps,
                                       interval=30, blit=True)

        # requires either mencoder or ffmpeg to be installed on your system
        # anim.save('split.mp4', fps=15, extra_args=['-vcodec', 'libx264'])


    def init(self):
        self.psi_x_line.set_data([], [])
        self.V_x_line.set_data([], [])

        self.title.set_text("")
        print(f"Init:\n {self.Harmonic.psi}")
        return (self.psi_x_line, self.V_x_line, self.title)


    def animate(self, i):
        self.Harmonic.time_step()
        print(f"Round {i}")
        # print(f"Round {i}:\n {Harmonic.psi}")
        self.psi_x_line.set_data(self.Harmonic.x, np.abs(self.Harmonic.psi))
        self.V_x_line.set_data(self.Harmonic.x, self.Harmonic.V(self.Harmonic.x))

        self.title.set_text("t = %.2f" % self.Harmonic.t)
        return (psi_x_line, V_x_line, title)
