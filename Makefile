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
		pytest test_hyperactive_api.py -p no:warnings; \
		pytest test_optimizers.py -p no:warnings; \
		pytest test_packages.py -p no:warnings

test-examples:
	cd examples/machine_learning; \
		pytest sklearn_example.py -p no:warnings; \
		pytest xgboost_example.py -p no:warnings; \
		pytest lightgbm_example.py -p no:warnings; \
		pytest catboost_example.py -p no:warnings; \
		pytest rgf_example.py -p no:warnings; \
		pytest mlxtend_example.py -p no:warnings
	cd examples/deep_learning; \
		pytest tensorflow_example.py -p no:warnings; \
		pytest keras_example.py -p no:warnings
	cd examples/distribution; \
		pytest multiprocessing_example.py -p no:warnings; \
		pytest ray_example.py -p no:warnings
	cd examples/memory_example; \
		pytest memory_example.py -p no:warnings; \
		pytest scatter_init_example.py -p no:warnings; \
		pytest warm_start_example.py -p no:warnings
	cd examples/test_functions; \
		pytest himmelblau_function_example.py -p no:warnings; \
		pytest rosenbrock_function_example.py -p no:warnings
	cd examples/use_cases; \
		pytest SklearnPreprocessing.py -p no:warnings; \
		pytest SklearnPipeline.py -p no:warnings; \
		pytest Stacking.py -p no:warnings; \
		pytest NeuralArchitectureSearch.py -p no:warnings; \
		pytest ENAS.py -p no:warnings; \
		pytest TransferLearning.py -p no:warnings; \
		pytest MetaOptimization.py -p no:warnings

push: test
	git push

release: reinstall
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
