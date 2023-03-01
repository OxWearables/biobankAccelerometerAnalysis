#!/bin/bash

# You must have Java Development Kit (JDK) 8 (1.8). If higher (>8) then it must
# support --release flag to pin down the version when compiling.
# Always compile with 8 (1.8) to keep backward compatibility.
# In conda, you can get a JDK version that supports --release flag:
# conda install openjdk
javac --version &&  # java version
javac -cp accelerometer/java/JTransforms-3.1-with-dependencies.jar accelerometer/java/*.java --release 8 &&  # compile java files (using release 8)
python setup.py sdist bdist_wheel &&  # setuptools
twine check dist/* &&
printf "\nTo upload to Test PyPI:\n> twine upload --repository-url https://test.pypi.org/legacy/ dist/*\n" &&
printf "\nTo upload to PyPI:\n> twine upload dist/*\n\n"

# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
