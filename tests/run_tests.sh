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
pytest _test_keras_cnn.py -p no:warnings

pytest test_performance.py -p no:warnings

pytest test_meta_learn.py -p no:warnings
