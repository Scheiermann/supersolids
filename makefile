.PHONY: bild clean doc help

.DEFAULT: help
help:
	@echo "make clean"
	@echo "          Clean all files from last build"
	@echo "make doc"
	@echo "         Rebuilds sphinx documentation"
	@echo "make build"
	@echo "         Builds python sdist and wheel, then rebuilds docs"
	@echo "make upload_test"
	@echo "         Builds python sdist and wheel, then rebuilds docs and uploads to testpypi"
	@echo "make upload"
	@echo "         Builds python sdist and wheel, then rebuilds docs and uploads to pypi"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	rm -rf supersolids/helper/*.so
	cd docs; make clean;

doc:
	cd docs; make clean; make html

build:
	python setup.py sdist bdist_wheel
	make doc
	git add dist/*
	git add docs/build/html/autoapi/*

build_test:
	rm -rf supersolids/helper/*.so
	python setup.py sdist bdist_wheel

upload_test:
	make build
	python -m twine upload --repository testpypi dist/*

upload:
	make build
	python -m twine upload --repository pypi dist/*

conda_build:
	conda config --set anaconda_upload no
	conda build .

conda_install_local:
	conda install --use-local supersolids
	conda install numba
	conda install cupy

conda_install_test:
	conda install -c scheiermann/label/testing supersolids
	conda install numba
	conda install cupy

conda_install:
	conda install -c scheiermann supersolids
	conda install numba
	conda install cupy

conda_upload_test:
	make conda_build
	anaconda upload $(shell conda build . --output) --label test

conda_upload:
	make conda_build
	path_package=$(conda build . --output)
	anaconda upload ${path_package}
