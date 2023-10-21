.PHONY: clean
.PHONY: installNecessaryPackagesPythonPip
.PHONY: jupyter python_format python_lint python_clean_notebook python_run_notebook

NECESSARY_PACKAGES_PYTHON_PIP = black jupyter matplotlib notebook numpy mypy pandas pandas-stubs pylint types-Pillow

PYTHON?=python

MYPY_FLAGS = --config-file=mypy.ini --enable-incomplete-feature=Unpack

clean:
	find "$(shell pwd -P)" -name "venv*" -exec rm -rf "{}" \;
	find "$(shell pwd -P)" -name ".ipynb_checkpoints" -exec rm -rf "{}" \;

installNecessaryPackagesPythonPip:
	$(PYTHON) -m pip install $(NECESSARY_PACKAGES_PYTHON_PIP)
	$(PYTHON) -m pip freeze > requirements.txt

jupyter:
	$(PYTHON) -m jupyter notebook

python_format:
	$(PYTHON) -m black $(wildcard *.py) $(wildcard lib/*.py)
	$(PYTHON) -m ipynb_format_code_cells $(wildcard *.ipynb)

python_lint:
	$(PYTHON) -m mypy $(MYPY_FLAGS) $(wildcard lib/*.py)
	$(PYTHON) -m pylint $(wildcard lib/*.py)

python_clean_notebook:
	$(foreach file, $(wildcard *.ipynb), $(PYTHON) -m jupyter nbconvert --clear-output --inplace "$(file)";)

python_run_notebook:
	$(foreach file, $(wildcard *.ipynb), $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace "$(file)";)
