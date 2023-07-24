import sys
import os.path
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.insert(0, os.path.dirname(__file__))

import setuptools
import codecs

import versioneer


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_string(string, rel_path="src/accelerometer/__init__.py"):
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
    python_requires=">=3.7, <3.11",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A package to extract meaningful health information from large accelerometer datasets e.g. how much time individuals spend in sleep, sedentary behaviour, walking and moderate intensity physical activity",
    keywords="wearables, accelerometer, health data science, human activity recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/activityMonitoring/biobankAccelerometerAnalysis",
    download_url="https://github.com/activityMonitoring/biobankAccelerometerAnalysis",
    author=get_string("__author__"),
    maintainer=get_string("__maintainer__"),
    maintainer_email=get_string("__maintainer_email__"),
    license=get_string("__license__"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=setuptools.find_packages(where="src", exclude=("test", "tests")),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        'numpy==1.21.*',
        'scipy==1.7.*',
        'matplotlib==3.5.*',
        'pandas==1.3.*',
        'tqdm==4.65.*',
        'statsmodels==0.13.*',
        'joblib==1.1.*',
        'imbalanced-learn==0.8.1',
        'scikit-learn==1.0.2',
    ],
    extras_require={
        "dev": [
            "flake8",
            "autopep8",
            "ipython",
            "ipdb",
            "twine",
            "tomli",
        ],
        "docs": [
            "sphinx>=4.2",
            "sphinx_rtd_theme>=1.0",
            "readthedocs-sphinx-search>=0.1",
            "sphinxcontrib-programoutput>=0.17",
            "docutils<0.18",
        ],
    },
    entry_points={
        "console_scripts": [
            "accProcess=accelerometer.accProcess:main",
            "accPlot=accelerometer.accPlot:main",
            "accWriteCmds=accelerometer.accWriteCmds:main",
            "accCollateSummary=accelerometer.accCollateSummary:main"
        ]
    },
)
