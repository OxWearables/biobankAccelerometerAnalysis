![Accelerometer data processing overview](https://raw.githubusercontent.com/OxWearables/biobankAccelerometerAnalysis/master/docs/source/accelerometerLogo.png)

[![Github all releases](https://img.shields.io/github/release/activityMonitoring/biobankAccelerometerAnalysis.svg)](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/releases/)
![install](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/install/badge.svg)
![flake8](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/flake8/badge.svg)
![junit](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/junit/badge.svg)
![gt3x](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/gt3x/badge.svg)
![cwa](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/cwa/badge.svg)

A tool to extract meaningful health information from large accelerometer datasets. The software generates time-series and summary metrics useful for answering key questions such as how much time is spent in sleep, sedentary behaviour, or doing physical activity.

## Install

*Minimum requirements*: Python>=3.7, Java 8 (1.8)

The following instructions make use of Anaconda to meet the minimum requirements:

1. Download & install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (light-weight version of Anaconda).
1. (Windows) Once installed, launch the **Anaconda Prompt**.
1. Create a virtual environment:
    ```console
    $ conda create -n accelerometer python=3.9 openjdk pip
    ```
    This creates a virtual environment called `accelerometer` with Python version 3.9, OpenJDK, and Pip.
1. Activate the environment:
    ```console
    $ conda activate accelerometer
    ```
    You should now see `(accelerometer)` written in front of your prompt.
1. Install `accelerometer`:
    ```console
    $ pip install accelerometer
    ```

You are all set! The next time that you want to use `accelerometer`, open the Anaconda Prompt and activate the environment (step 4). If you see `(accelerometer)` in front of your prompt, you are ready to go!

## Usage
To extract summary movement statistics from an Axivity file (.cwa):

```console
$ accProcess data/sample.cwa.gz

 <output written to data/sample-outputSummary.json>
 <time series output written to data/sample-timeSeries.csv.gz>
```

Movement statistics will be stored in a JSON file:
```json
{
    "file-name": "sample.cwa.gz",
    "file-startTime": "2014-05-07 13:29:50",
    "file-endTime": "2014-05-13 09:49:50",
    "acc-overall-avg(mg)": 32.78149,
    "wearTime-overall(days)": 5.8,
    "nonWearTime-overall(days)": 0.04,
    "quality-goodWearTime": 1
}
```

See [Data Dictionary](https://biobankaccanalysis.readthedocs.io/en/latest/datadict.html) for the list of output variables.

Actigraph and GENEActiv files are also supported, as well as custom CSV files. See [Usage](https://biobankaccanalysis.readthedocs.io/en/latest/usage.html#basic-usage) for more details.

To plot the activity profile:
```console
$ accPlot data/sample-timeSeries.csv.gz
 <output plot written to data/sample-timeSeries-plot.png>
```
![Time series plot](https://raw.githubusercontent.com/OxWearables/biobankAccelerometerAnalysis/master/docs/source/samplePlot.png)

### Troubleshooting 
Some systems may face issues with Java when running the script. If this is your case, try fixing OpenJDK to version 8:
```console
$ conda install -n accelerometer openjdk=8
```

## Under the hood
Interpreted levels of physical activity can vary, as many approaches can be
taken to extract summary physical activity information from raw accelerometer
data. To minimise error and bias, our tool uses published methods to calibrate,
resample, and summarise the accelerometer data.

![Accelerometer data processing overview](https://raw.githubusercontent.com/OxWearables/biobankAccelerometerAnalysis/master/docs/source/accMethodsOverview.png)
![Activity classification](https://raw.githubusercontent.com/OxWearables/biobankAccelerometerAnalysis/master/docs/source/accClassification.png)

See [Methods](https://biobankaccanalysis.readthedocs.io/en/latest/methods.html) for more details.


## Citing our work
When using this tool, please consider the works listed in [CITATION.md](https://github.com/OxWearables/biobankAccelerometerAnalysis/blob/master/CITATION.md).
    

## Licence
See [LICENSE.md](https://github.com/OxWearables/biobankAccelerometerAnalysis/blob/master/LICENSE.md).


## Acknowledgements
We would like to thank all our code contributors and manuscript co-authors.

[Contributors Graph](https://github.com/OxWearables/biobankAccelerometerAnalysis/graphs/contributors)
