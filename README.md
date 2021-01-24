# Supersolids  
Package to simulate and animate supersolids.  
This is done by solving the dimensionless time-dependent  
non-linear Schrodinger equation for an arbitrary potential.  
The split operator method with the Trotter-Suzuki approximation is used.  

## Documentation
https://supersolids.readthedocs.io/en/latest/

## Installing 

For the animation to work, ffmpeg needs to be installed on your system.

For **python3.9** currently there is no vtk wheel for python3.9, so you need to build it from source or use my build:  
git clone https://github.com/Scheiermann/vtk_python39_wheel. Go to the directory vtk_python39_wheel/,
where the wheel lies (*.whl).  
Use this wheel to install, e.g:  
pip install vtk-9.0.20210105-cp39-cp39-linux_x86_64.whl  
Then install mayavi (pip install mayavi or also build it from source, as there could be incapabilities with vtk9).  

### Arch based systems  
For **python3.9** follow the instructions above, then continue with (else do the following):  
Go to the directory, where you build your AUR packages.  
Run the following in console:  
"mkdir python-supersolids"  
"cd python-supersolids"  
Download the PKGBUILD and place it there.  
Run "makepkg -sic" in console, where the PKGBUILD lies.  
This downloads the packages, extract it, gets the dependencies and installs it automatically  
(you can do it manually for example with "sudo pacman -U python-supersolids-0.1.15-1-any.pkg.tar.zst")  

### With pip  
For **python3.9** follow the instructions above, then continue with (else do the following):  
Go to the directory supersolids/dist/, where the wheel lies (*.whl).  
Use this wheel to install, e.g:  
pip install supersolids-0.1.21-py3-none-any.whl  


### With setup.py  
Go to the directory, where the "setup.py" lies.  

For **Linux** use "python setup.py install --user" from console to **build** and **install** the package  

For **Windows**:  
You need to add python to your path (if you didn't do it, when installing python/anaconda).  
1. Open Anaconda Prompt. Use commands "where python", "where pip", "where conda".  
2. Use the output path (without *.exe, we call the output here AX, BX, CX) in the following command:  
   SETX PATH "%PATH%; AX; BX; CX"  
   For example, where the user is dr-angry:  
   SETX PATH "%PATH%; C:\Users\dr-angry\anaconda3\Scripts; C:\Users\dr-angry\anaconda3"  
3. Now restart/open gitbash.  
4. Use "python setup.py install" in gitbash from the path where "setup.py" lies.  

Or you can follow the guide here:  
https://www.magicmathmandarin.org/2017/12/07/setting-up-python-after-installing-or-re-installing-anaconda/  

## Issues  
1. Please read the **README.md** closely.  
2. If you have please check every step again.  
3. If the issue persist please **open a "Issue" in git**:  
a) Click on "New Issue" on https://github.com/Scheiermann/supersolids/issues.  
b) Assign a suitable label.  
c) Follow the steps on git the to create the issue.  
Please **describe your issue closely** (what are your configurations, did it work before,  
what have you changed, what is the result, what have you expected as a result?).  
d) Try to include screenshots (to the question in 3b).  
e) Describe what you think causes the issue and if you have **suggestions how to solve** it,  
mention it! (to the question in 3b).  
f) **Close the issue**, if you accidentally did something wrong (but mention that before closing).  

## Contributing  
Please read the **CONTRIBUTING.md**.  
