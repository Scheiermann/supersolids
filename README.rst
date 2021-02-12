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

For **python3.9** currently there is no vtk wheel for python3.9, so you need to build it from source or use my build:
git clone https://github.com/Scheiermann/vtk_python39_wheel. Go to the directory vtk_python39_wheel/,
where the wheel lies (\*.whl).
Use this wheel to install, e.g:
pip install vtk-9.0.20210105-cp39-cp39-linux_x86_64.whl
Then install mayavi (pip install mayavi or also build it from source, as there could be incapabilities with vtk9).

Archlinux
---------
It is provided in the AUR under https://aur.archlinux.org/python-supersolids.git
For **python3.9** follow the instructions above,
then remove mayavi from the dependencies and run "makepkg -sic".
For **python3.8** the remove the dependecies in the PKGBUILD and uncomment
the pip install lines instead.

pip
---
For **python3.9** follow the instructions above, then continue with (else do the following):
Go to the directory supersolids/dist/, where the wheel lies (\*.whl).
Use this wheel to install, e.g:
pip install supersolids-0.1.21-py3-none-any.whl

Source
---------------------------
Go to the directory, where the "setup.py" lies.
For **Linux** use "python setup.py install --user" from console to **build** and **install** the package

Windows
-------
You need to add python to your path (if you didn't do it, when installing python/anaconda).
1. Open Anaconda Prompt. Use commands "where python", "where pip", "where conda".
2. Use the output path (without \*.exe, we call the output here AX, BX, CX) in the following command::
SETX PATH "%PATH%; AX; BX; CX"
For example, where the user is dr-angry:
SETX PATH "%PATH%; C:\Users\dr-angry\anaconda3\Scripts; C:\Users\dr-angry\anaconda3"
3. Now restart/open gitbash.
4. Use "python setup.py install" in gitbash from the path where "setup.py" lies.

Usage
-----
The package uses __main__.py, so it can be run as module.
To get help for the flags, run:
python -m supersolids -h

To actually run (example):
python -m supersolids -dt=0.0001 -Res=[2 ** 6, 2 ** 5, 2 ** 4]

The default path for the results is ~/supersolids/results

Issues
------
1. Please read the **README.md** closely.
2. If you have please check every step again.
3. If the issue persist please **open an "Issue" in git**:
    * Click on "New Issue" on https://github.com/Scheiermann/supersolids/issues.
    * Assign a suitable label.
    * Follow the steps on git the to create the issue.
      Please **describe your issue closely** (what are your configurations, did it work before,
      what have you changed, what is the result, what have you expected as a result?).
    * Try to include screenshots (to the question in 3b).
    * Describe what you think causes the issue and if you have **suggestions how to solve** it,
      mention it! (to the question in 3b).
    * **Close the issue**, if you accidentally did something wrong (but mention that before closing).

Contributing
------------
Please read the **CONTRIBUTING.rst**.
