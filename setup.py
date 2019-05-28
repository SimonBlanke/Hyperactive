import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hyperactive",
    version="0.1",
    author="Simon Blanke",
    author_email="simon.blanke@yahoo.com",
    license="MIT",
    description="Meta heuristic optimization techniques for scikit-learn models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "tqdm", "scikit-learn>=0.18"],
)
