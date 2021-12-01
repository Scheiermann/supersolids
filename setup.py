from os import path

from setuptools import setup, find_packages
from supersolids.helper.numba_compiled import cc

# from Cython.Build import cythonize
# To compile: python setup.py build_ext --inplace

# create a source distribution and a wheel with python setup.py sdist bdist_wheel
# To upload to TestPyPi python -m twine upload --repository testpypi dist/*

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="supersolids",
    version="0.1.34rc18",
    packages=find_packages(),
    package_data={"supersolids": []},
    url="https://github.com/Scheiermann/supersolids",
    license="MIT",
    author="Scheiermann",
    author_email="daniel.scheiermann@itp.uni-hannover.de",
    install_requires=["dill>0.3.1",
                      "fabric",
                      "ffmpeg-python",
                      "invoke",
                      "matplotlib",
                      "numpy",
                      "numba",
                      "mayavi",
                      "psutil",
                      "pyqt5",
                      "scipy",
                      "sphinx-autoapi",
                      "sphinx-rtd-theme",
                      ],
    # ext_modules=[cc.distutils_extension()],
    # ext_modules=cythonize("*.pyx", language_level=3),
    python_requires=">=3.6",
    description="Simulate and animate supersolids.",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
