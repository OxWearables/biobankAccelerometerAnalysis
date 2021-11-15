import setuptools
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_string(string, rel_path="accelerometer/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="accelerometer",
    python_requires=">=3.7",
    version=get_string("__version__"),
    author=get_string("__author__"),
    author_email=get_string("__email__"),
    description="A package to extract meaningful health information from large accelerometer datasets e.g. how much time individuals spend in sleep, sedentary behaviour, walking and moderate intensity physical activity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/activityMonitoring/biobankAccelerometerAnalysis",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas>=1.2.5',
        'scikit-learn>=0.24.2',
        'joblib==1.0.1',
        'statsmodels>=0.12.2',
        'tqdm>=4.59.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
    ],
    entry_points={
        "console_scripts": [
            "accProcess=accelerometer.accProcess:main",
            "accPlot=accelerometer.accPlot:main",
            "accWriteCmds=accelerometer.accWriteCmds:main",
            "accCollateSummary=accelerometer.accCollateSummary:main"
        ]
    },
)
