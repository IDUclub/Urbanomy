PYTHON ?= python3
PACKAGE_NAME = urbanomy
SOURCE_DIR = $(PACKAGE_NAME)
TEST_DIR = tests

# code formatting and linting

lint:
	pylint ${SOURCE_DIR}

format:
	isort ${SOURCE_DIR}
	black ${SOURCE_DIR}

# installing

venv: # затем необходимо активировать: source .venv/bin/activate
	$(PYTHON) -m venv .venv

install: # простая установка зависимостей без editable режима
	$(PYTHON) -m pip install .

install-dev: # установка зависимостей для разработки
	$(PYTHON) -m pip install -e '.[dev]'

install-docs: # установка зависимостей для тестирования
	$(PYTHON) -m pip install -e '.[docs]'

install-test: # установка зависимостей для сборки документации
	$(PYTHON) -m pip install -e '.[test]'

# building the package

build:
	$(PYTHON) -m build .

clean:
	rm -rf ./build ./dist ./$(PACKAGE_NAME).egg-info

# pypi

pypi: clean build # команда для сборки и загрузки в pypi
	$(PYTHON) -m twine upload dist/*

test-pypi: clean build # команда для сборки и загрузки в тестовый pypi
	$(PYTHON) -m twine upload --repository testpypi dist/*

# testing

test: # тестирование
	pytest ${TEST_DIR}

test-cov: # тестирование с выводом процента покрытия кода тестами
	pytest ${TEST_DIR} --cov
