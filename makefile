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
	cd docs; make clean;

doc:
	cd docs; make clean; make html

build:
	python setup.py sdist bdist_wheel
	make doc
	git add dist/*
	git add docs/build/html/autoapi/*

upload_test:
	make build
	python -m twine upload --repository testpypi dist/*

upload:
	make build
	python -m twine upload --repository pypi dist/*
