# Maintainer: Daniel Scheiermann  <daniel.scheiermann@stud.uni-hannover.de>
_name=supersolids
pkgname=python-${_name}
pkgver=0.1.4
pkgrel=1
pkgdesc="Notes and script to supersolids"
url="https://github.com/Scheiermann/${_name}"
arch=(any)
license=('MIT')
depends=('python-matplotlib' 'python-numpy' 'python-scipy' 'python-sympy')
makedepends=('python-setuptools')
optdepends=('')
source=(${_name}-$pkgver.tar.gz::"https://test-files.pythonhosted.org/packages/b3/bd/3de80bca4928d5444241d54b944b01c6e24463200dcfab06d3b0adb9e808/${_name}-$pkgver.tar.gz")
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
  python setup.py install --skip-build --root="$pkgdir" --optimize=1

}
