import sys

# input_path = sys.argv[1]
output_path = "../docs/examples/"


def py2md(dict_, file_name):
    md_total_content = ""

    for heading in dict_.keys():
        py_file_name = dict_[heading]
        with open(py_file_name, "r") as py_file:
            py_file_content = py_file.read()

        front_str = "```python\n"
        back_str = "```"
        md_file_content = front_str + py_file_content + back_str

        md_total_content = (
            md_total_content + "## " + heading + "\n\n" + md_file_content + "\n\n"
        )

    md_file_name = output_path + "/" + file_name + ".md"
    md_file = open(md_file_name, "w+")
    md_file.write(md_total_content)
    md_file.close()


dict_ml = {
    "Sklearn": "machine_learning/sklearn_example.py",
    "XGBoost": "machine_learning/xgboost_example.py",
    "LightGBM": "machine_learning/lightgbm_example.py",
    "CatBoost": "machine_learning/catboost_example.py",
    "RGF": "machine_learning/rgf_example.py",
    "Mlxtend": "machine_learning/mlxtend_example.py",
}

dict_dl = {
    "Tensorflow": "deep_learning/tensorflow_example.py",
    "Keras": "deep_learning/keras_example.py",
}

dict_dist = {
    "Multiprocessing": "distribution/multiprocessing_example.py",
    "Ray": "distribution/ray_example.py",
}

dict_extensions = {
    "Memory": "extensions/memory_example.py",
    "Scatter initialization": "extensions/scatter_init_example.py",
    "Warm start": "extensions/warm_start_example.py",
}

dict_test_functions = {
    "Himmelblau’s function": "test_functions/himmelblau_function_example.py",
    "Rosenbrock function": "test_functions/rosenbrock_function_example.py",
}

dict_use_cases = {
    "Sklearn Preprocessing": "use_cases/SklearnPreprocessing.py",
    "Sklearn Pipeline": "use_cases/SklearnPipeline.py",
    "Stacking": "use_cases/Stacking.py",
    "Neural Architecture Search": "use_cases/NeuralArchitectureSearch.py",
    "ENAS": "use_cases/ENAS.py",
    "Transfer Learning": "use_cases/TransferLearning.py",
    "Meta-Optimization": "use_cases/MetaOptimization.py",
}

examples_dict = {
    "machine_learning": dict_ml,
    "deep_learning": dict_dl,
    "distribution": dict_dist,
    "extensions": dict_extensions,
    "test_functions": dict_test_functions,
    "use_cases": dict_use_cases,
}

for file_name in examples_dict.keys():
    py2md(examples_dict[file_name], file_name)
    print("Wrote", file_name, "example")
