Supersolids
===========
Package to simulate and animate supersolids.
This is done by solving the dimensionless time-dependent
non-linear Schrodinger equation for an arbitrary potential.
The split operator method with the Trotter-Suzuki approximation is used.

Documentation
-------------
https://supersolids.readthedocs.io/en/latest/

Installing
----------
For the animation to work, **ffmpeg** needs to be installed on your system.

pip
---
 * pip install supersolids

Archlinux
---------
It is provided in the AUR
 * git clone https://aur.archlinux.org/python-supersolids.git
 * Run "makepkg -sic", where the PKGBUILD lies

Windows
-------
You need to add python to your path (if you didn't do it, when installing python/anaconda).
 * Then continue with pip installation

Source
---------------------------
Go to the directory, where the "setup.py" lies.
 * For **Linux** use "python setup.py install --user" from console to **build** and **install** the package

Usage
-----
The package uses __main__.py, so it can be run as module.
To get help for the flags, run:
 * python -m supersolids -h
 * python -m supersolids.tools.load_npz -h
 * python -m supersolids.tools.simulate_npz -h

To actually run (example):
 * python -m supersolids -Res='{"x": 16, "y": 32, "z": 62}' -Box='{"x0": -10, "x1": 10, "y0": -6, "y1": 5, "z0": -8, "z1": 8}'
 * python -m supersolids.tools.load_npz -frame_start=79000
 * python -m supersolids.tools.simulate_npz -dir_name=movie004 -filename_npz=step_079000.npz

If you use an IDE and your script parameter includes double quotes, escape the double quotes with backslashes, for example:
 * "-Res={\\"x\\": 256}" "-Box={\\"x0\\": -10, \\"x1\\": 10}" "-a={\\"a_x\\": 2.0}" -max_timesteps=51 "-V=lambda x, y, z: 100.0 * np.exp(-(x ** 2 + y ** 2)/ 1.0 ** 2)"

The default path for the results is ~/supersolids/results

Issues
------
1. Please read the **README.md** closely.
2. If the issue persist please **open an "Issue" in git**:
    * Click on "New Issue" on https://github.com/Scheiermann/supersolids/issues.
    * Assign a suitable label.
    * Follow the steps on git the to create the issue.
      Please **describe your issue closely** (what are your configurations, did it work before,
      what have you changed, what is the result, what have you expected as a result?).
    * Try to include screenshots.
    * Describe what you think causes the issue and if you have **suggestions how to solve** it,
      mention it!
    * **Close the issue**, if you accidentally did something wrong (but mention that before closing).

Contributing
------------
Please read the **CONTRIBUTING.rst**.
