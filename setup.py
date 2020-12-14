from setuptools import setup, find_packages
# create a wheel with python setup.py bdist_wheel
# from Cython.Build import cythonize
# To compile: python setup.py build_ext --inplace
# To upload to TestPyPi python -m twine upload --repository testpypi dist/*
setup(
    name='supersolids',
    version='0.1.2',
    packages=find_packages(),
    url='https://github.com/Scheiermann/supersolids',
    license='MIT',
    author='Scheiermann',
    author_email='daniel.scheiermann@stud.uni-hannover.de',
    install_requires=["matplotlib", "numpy", "scipy", "sympy"],
    # scripts=["Animation.py", "constants.py", "functions.py", "main.py", "parallel.py", "Schroedinger.py",
    #          "sympy_physics_test.py"],
    # ext_modules=cythonize("*.pyx", language_level=3),
    python_requires='>=3.8',
    description='Notes and script to supersolids'
)
