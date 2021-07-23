clean: clean-pyc clean-ipynb clean-catboost clean-build clean-test

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-ipynb:
	find . -name '*.ipynb_checkpoints' -exec rm -fr {} +

clean-catboost:
	find . -name 'catboost_info' -exec rm -fr {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-test:
	cd tests/; \
		rm -f .coverage; \
		rm -fr htmlcov/

test-search_space:
	cd tests/; \
		i=0; while [ "$$i" -le 100 ]; do \
			i=$$((i + 1));\
			pytest -q test_search_spaces.py; \
	done

test:
	python -m pytest -x -p no:warnings -rfEX tests/; \
	python ./tests/test_dashboards/_test_progress_board.py \

test-local:
	cd tests/local; \
		pytest _test_packages.py -p no:warnings; \
		pytest _test_performance.py -p no:warnings

test-examples:
	cd tests; \
		python test_examples.py

test-extensive: test test-local test-examples

push: test
	git push

release: reinstall test-extensive
	python -m twine upload dist/*

dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y hyperactive
	rm -fr build dist hyperactive.egg-info
	python setup.py bdist_wheel
	pip install dist/*
