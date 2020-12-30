from setuptools import setup

# from Cython.Build import cythonize
# To compile: python setup.py build_ext --inplace

# create a source distribution and a wheel with python setup.py sdist bdist_wheel
# To upload to TestPyPi python -m twine upload --repository testpypi dist/*

setup(
    name="supersolids",
    version="0.1.9",
    packages=["", "supersolids"],
    package_data={"supersolids": ["results/split_time_imag.mp4",
                                  "results/split_time_real.mp4",
                                  "results/2d_gauss.mp4",
                                  "mayavi_tests/gauss_2d/anim.mp4",
                                  "mayavi_tests/gauss_3d/anim.mp4"
                                  ]},
    url="https://github.com/Scheiermann/supersolids",
    license="MIT",
    author="Scheiermann",
    author_email="daniel.scheiermann@stud.uni-hannover.de",
    install_requires=["apptools", "envisage", "ffmpeg-python", "matplotlib", "numpy", "psutil", "scipy",
                      "sympy", "traits", "traitsui"],
    # ext_modules=cythonize("*.pyx", language_level=3),
    python_requires=">=3.8",
    description="Notes and script to supersolids"
)
