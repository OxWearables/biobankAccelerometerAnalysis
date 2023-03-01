#!/bin/bash

# Note: Be sure package already in PyPI

conda install anaconda-client &&
conda install conda-build &&
# TODO: allow user-specified version and append this to next line: --version x.x.x
conda skeleton pypi accelerometer --output-dir conda-recipe &&
conda build -c conda-forge conda-recipe/accelerometer

printf "\nNext steps:\n-----------\n"
printf "Login to Anaconda:\n> anaconda login\n"
printf "\nUpload package (path is printed in previous steps):\n> anaconda upload --user oxwear /path/to/package.tar.bz2\n\n"

# anaconda login
# anaconda upload --user oxwear /path/to/package.tar.bz2
