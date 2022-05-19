![Accelerometer data processing overview](docs/source/accelerometerLogo.png)

[![Github all releases](https://img.shields.io/github/release/activityMonitoring/biobankAccelerometerAnalysis.svg)](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/releases/)
![install](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/install/badge.svg)
![flake8](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/flake8/badge.svg)
![junit](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/junit/badge.svg)
![gt3x](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/gt3x/badge.svg)
![cwa](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/cwa/badge.svg)

A tool to extract meaningful health information from large accelerometer datasets. The software generates time-series and summary metrics useful for answering key questions such as how much time is spent in sleep, sedentary behaviour, or doing physical activity.

## Installation

```bash
pip install accelerometer
```

You also need Java 8 (1.8.0) or greater. Check with the following:

```bash
java -version
```

You can try the following to check that everything works properly:
```bash
# Create an isolated environment
$ mkdir test_baa/ ; cd test_baa/
$ python -m venv baa
$ source baa/bin/activate

# Install and test
$ pip install accelerometer
$ wget -P data/ http://gas.ndph.ox.ac.uk/aidend/accModels/sample.cwa.gz  # download a sample file
$ accProcess data/sample.cwa.gz
$ accPlot data/sample-timeSeries.csv.gz
```


## Usage
To extract summary movement statistics from an Axivity file (.cwa):

```bash
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

See [here](https://biobankaccanalysis.readthedocs.io/en/latest/datadict.html) for the list of output variables.

Actigraph and GENEActiv files are also supported, as well as custom CSV files.  See the [documentation](https://biobankaccanalysis.readthedocs.io/en/latest/index.html) for more details.

To visualise the activity profile:
```bash
$ accPlot data/sample-timeSeries.csv.gz
 <output plot written to data/sample-timeSeries-plot.png>
```
![Time series plot](docs/source/samplePlot.png)

## Under the hood
Interpreted levels of physical activity can vary, as many approaches can be
taken to extract summary physical activity information from raw accelerometer
data. To minimise error and bias, our tool uses published methods to calibrate,
resample, and summarise the accelerometer data.
<!-- [Click here for detailed information on the data processing methods on our wiki.](https://biobankaccanalysis.readthedocs.io/en/latest/methods.html) -->

![Accelerometer data processing overview](docs/source/accMethodsOverview.png)
![Activity classification](docs/source/accClassification.png)


## Citing our work
When describing or using the *UK Biobank accelerometer dataset*, please cite [Doherty2017].
When using *this tool* to extract sleep duration and physical activity behaviours from your accelerometer data, please cite:


1. [Doherty2017] Doherty A, Jackson D, et al. (2017)
Large scale population assessment of physical activity using wrist worn
accelerometers: the UK Biobank study. PLOS ONE. 12(2):e0169649

1. [Willetts2018] Willetts M, Hollowell S, et al. (2018)
Statistical machine learning of sleep and physical activity phenotypes from
sensor data in 96,220 UK Biobank participants. Scientific Reports. 8(1):7961

1. [Doherty2018] Doherty A, Smith-Byrne K, et al. (2018)
GWAS identifies 14 loci for device-measured physical activity and sleep
duration. Nature Communications. 9(1):5257

1. [Walmsley2021] Walmsley R, Chan S, Smith-Byrne K, et al. (2021)
Reallocation of time between device-measured movement behaviours and risk
of incident cardiovascular disease. British Journal of Sports Medicine.
Published Online First. DOI: 10.1136/bjsports-2021-104050

###### Licence
See [license](LICENSE.md) before using this software.
