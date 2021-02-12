.PHONY: bild clean doc help

.DEFAULT: help
help:
	@echo "make clean"
	@echo "          Clean all files from last build"
	@echo "make doc"
	@echo "         Rebuilds sphinx documentation"
	@echo "make build"
	@echo "         Builds python sdist and wheel, then rebuilds docs"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .mypy_cache/

doc:
	cd docs; make clean; make html

build:
	python setup.py sdist bdist_wheel
	make doc


