#!/bin/sh

python ./machine_learning/Scikit-learn.py
python ./machine_learning/XGBoost.py
python ./machine_learning/LightGBM.py
python ./machine_learning/CatBoost.py
python ./machine_learning/RGF.py
python ./machine_learning/Mlxtend.py


python ./deep_learning/Tensorflow.py
python ./deep_learning/Keras.py

python ./distribution/Multiprocessing.py
python ./distribution/Ray.py

python ./extensions/Scatter-initialization.py
python ./extensions/Warm-start.py
python ./extensions/Memory.py


# python ./test_functions/Himmelblau's function.py
# python ./test_functions/Rosenbrock function.py


python ./use_cases/ENAS.py
python ./use_cases/Meta-Optimization.py
python ./use_cases/Neural Architecture Search.py
python ./use_cases/Sklearn Pipeline.py
python ./use_cases/Sklearn Preprocessing.py
python ./use_cases/Stacking.py
python ./use_cases/Transfer Learning.py
