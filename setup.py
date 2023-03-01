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
        'tqdm>=4.59.0',
        'statsmodels>=0.12.2',
        'imbalanced-learn==0.8.1',
        'scikit-learn==1.0.1',
        'joblib==1.1.0',
        'tqdm>=4.59.0',
    ],
    extras_require={
        "dev": [
            "flake8",
            "autopep8",
            "ipython",
            "ipdb",
            "twine",
        ],
        "docs": [
            "sphinx>=4.2",
            "sphinx_rtd_theme>=1.0",
            "readthedocs-sphinx-search>=0.1",
            "sphinxcontrib-programoutput>=0.17",
            "docutils<0.18",
        ],
    },
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
