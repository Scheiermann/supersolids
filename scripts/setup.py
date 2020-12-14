from setuptools import setup, find_packages
# create a wheel with python setup.py bdist_wheel
# from Cython.Build import cythonize
# To compile: python setup.py build_ext --inplace
setup(
    name='supersolids_notes',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/Scheiermann/supersolids_notes',
    license='MIT',
    author='Scheiermann',
    author_email='daniel.scheiermann@stud.uni-hannover.de',
    install_requires=["scipy", "matplotlib", "numpy", "sympy"],
    scripts=["Animation.py", "constants.py", "functions.py", "main.py", "parallel.py", "Schroedinger.py",
             "sympy_physics_test.py"],
    # ext_modules=cythonize("*.pyx", language_level=3),
    description='Notes and script to supersolids'
)
