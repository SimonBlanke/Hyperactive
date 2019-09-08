pytest test_classes.py -p no:warnings
pytest test_arguments_api.py -p no:warnings
pytest test_arguments_optimizers.py -p no:warnings
pytest test_data.py -p no:warnings

pytest test_methods.py -p no:warnings
pytest test_optimizers.py -p no:warnings

pytest test_sklearn.py -p no:warnings
pytest test_xgboost.py -p no:warnings
pytest test_lightgbm.py -p no:warnings
pytest test_catboost.py -p no:warnings
pytest test_keras_mlp.py -p no:warnings
pytest ./local/_test_keras_cnn.py -p no:warnings

pytest test_performance.py -p no:warnings

pytest test_meta_learn.py -p no:warnings

python ../plots/plot_search.py

python ../examples/machine_learning/sklearn_.py
python ../examples/machine_learning/xgboost_.py
python ../examples/machine_learning/lightgbm_.py
python ../examples/machine_learning/catboost_.py

python ../examples/deep_learning/mlp_classification.py
python ../examples/deep_learning/mlp_regression.py
python ../examples/deep_learning/cnn_mnist.py
python ../examples/deep_learning/cnn_cifar10.py

python ../examples/distribution/multiprocessing_.py

python ../examples/advanced_features/memory_.py
python ../examples/advanced_features/scatter_init.py
python ../examples/advanced_features/transfer_learning.py
python ../examples/advanced_features/warm_start_keras.py
python ../examples/advanced_features/warm_start_sklearn.py
