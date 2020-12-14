# Supersolids
Notes and script to supersolids

## How to contribute
Please read the **CONTRIBUTING.md**.  

## How to install
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

## Somethings does not work (Issues)
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
