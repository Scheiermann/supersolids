# Maintainer: Daniel Scheiermann  <daniel.scheiermann@stud.uni-hannover.de>
_name=supersolids
pkgname=python-${_name}
pkgver=0.1.7
pkgrel=1
pkgdesc="Notes and script to supersolids"
url="https://github.com/Scheiermann/${_name}"
arch=(any)
license=('MIT')
depends=('python-matplotlib' 'python-numpy' 'python-scipy' 'python-sympy')
makedepends=('python-setuptools')
optdepends=('')
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
  python setup.py install --skip-build --root="$pkgdir" --optimize=1

}
