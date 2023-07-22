#* Variables
PYTHON := python3
PYTHONPATH := `pwd`
#* Formatters
.PHONY: black
black:
	black --version
	black examples hsa_planar_control test

.PHONY: black-check
black-check:
	black --version
	black --diff --check examples hsa_planar_control test

.PHONY: flake8
flake8:
	flake8 --version
	flake8 examples hsa_planar_control test

.PHONY: format-codestyle
format-codestyle: black flake8

.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

.PHONY: check-codestyle
check-codestyle: black-check flake8

.PHONY: formatting
formatting: format-codestyle

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

all: format-codestyle cleanup test

ci: check-codestyle
