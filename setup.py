from setuptools import setup

# from Cython.Build import cythonize
# To compile: python setup.py build_ext --inplace

# create a source distribution and a wheel with python setup.py sdist bdist_wheel
# To upload to TestPyPi python -m twine upload --repository testpypi dist/*

setup(
    name="supersolids",
    version="0.1.17",
    packages=["", "supersolids"],
    package_data={"supersolids": ["results/anim.mp4",
                                  ]},
    url="https://github.com/Scheiermann/supersolids",
    license="MIT",
    author="Scheiermann",
    author_email="daniel.scheiermann@stud.uni-hannover.de",
    install_requires=["apptools", "envisage", "ffmpeg-python", "matplotlib", "mayavi", "numpy", "psutil", "PyQt5",
                      "scipy", "sympy", "traits", "traitsui", "vtk"],
    # ext_modules=cythonize("*.pyx", language_level=3),
    python_requires=">=3.8",
    description="Notes and script to supersolids"
)
