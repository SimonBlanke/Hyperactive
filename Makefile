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

test:
	cd tests/; \
		pytest test_general_optimizer_api.py -p no:warnings; \
		pytest test_hyperactive_api.py -p no:warnings; \
		pytest test_checks.py -p no:warnings

test-local:
	cd tests/local; \
		pytest _test_packages.py -p no:warnings; \
		pytest _test_performance.py -p no:warnings

test-examples:
	cd examples/machine_learning; \
		python sklearn_example.py; \
		python xgboost_example.py; \
		python lightgbm_example.py; \
		python catboost_example.py; \
		python rgf_example.py; \
		python mlxtend_example.py
	cd examples/deep_learning; \
		python tensorflow_example.py; \
		python keras_example.py
	cd examples/distribution; \
		python multiprocessing_example.py; \
		python ray_example.py
	cd examples/memory_example; \
		python memory_example.py; \
		python scatter_init_example.py; \
		python warm_start_example.py
	cd examples/test_functions; \
		python himmelblau_function_example.py; \
		python rosenbrock_function_example.py
	cd examples/use_cases; \
		python SklearnPreprocessing.py; \
		python SklearnPipeline.py; \
		python Stacking.py; \
		python NeuralArchitectureSearch.py; \
		python ENAS.py; \
		python TransferLearning.py; \
		python MetaOptimization.py

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
