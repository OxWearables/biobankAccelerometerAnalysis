import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="accelerometer",
    version="2.0",
    author="Aiden Doherty",
    author_email="aiden.doherty@bdi.ox.ac.uk",
    description="A package to extract meaningful health information from large accelerometer datasets e.g. how much time individuals spend in sleep, sedentary behaviour, walking and moderate intensity physical activity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/activityMonitoring/biobankAccelerometerAnalysis",
    packages=setuptools.find_packages(),
    install_requires=[
        'argparse',
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'sklearn',
        'sphinx',
        'sphinx-rtd-theme',
        'statsmodels',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
    ],
)
