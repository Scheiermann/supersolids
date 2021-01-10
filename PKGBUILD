# Maintainer: Daniel Scheiermann  <daniel.scheiermann@stud.uni-hannover.de>
_name=supersolids
pkgname=python-${_name}
pkgver=0.1.17
pkgrel=1
pkgdesc="Notes and script to supersolids"
url="https://github.com/Scheiermann/${_name}"
arch=(any)
license=("MIT")
# depends=("ffmpeg" "python-apptools" "python-envisage" "python-ffmpeg" "python-matplotlib"
#          "python-mayavi" "python-numpy" "python-traits" "python-traitsui" "python-scipy" "python-sympy" "vtk")
makedepends=("python-setuptools")
optdepends=("")
source=(${_name}-$pkgver.tar.gz::"https://test-files.pythonhosted.org/packages/source/${_name::1}/$_name/${_name}-$pkgver.tar.gz")
sha256sums=("SKIP")

build() {
  cd "$srcdir/${_name}-$pkgver"
  python setup.py build
}

check_disabled() { #ERROR: TypeError None is not callable
  cd "$srcdir/${_name}-$pkgver"
  python setup.py test
}

package() {
  cd "$srcdir/${_name}-$pkgver"
  # alternatively install dependencies with pip
  # for python3.8
  python -m pip install -U apptools envisage ffmpeg-python mayavi matplotlib numpy scipy sympy traits traitsui vtk
  # for python3.9
  # (watch out: currently there is no vtk wheel for python3.9, so you need to build it from source
  #  or take a unofficial build (provided by the project creator of this package), then install mayavi)
  # python -m pip install -U apptools envisage ffmpeg-python matplotlib numpy scipy sympy traits traitsui
  python setup.py install --skip-build --root="$pkgdir" --optimize=1

}
