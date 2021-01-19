![Accelerometer data processing overview](docs/source/accelerometerLogo.png)

A tool to extract meaningful health information from large accelerometer datasets. The software generates time-series and summary metrics useful for answering key questions such as how much time is spent in sleep, sedentary behaviour, or doing physical activity.

![Build status](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/workflows/CI/badge.svg)

## Installation
Dependencies include: unix, java 8 ([Java 8 JDK](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)) and python 3.7 ([Anaconda's Python 3](https://www.anaconda.com/download/) or installation via [Brew](https://docs.python-guide.org/starting/install3/osx/) should do the trick).
```
$ git clone https://github.com/activityMonitoring/biobankAccelerometerAnalysis.git
$ bash utilities/downloadDataModels.sh 
$ pip install --upgrade pip
$ pip3 install --upgrade -r requirements.txt # Installs a known working set of dependencies, other package versions may also work. 
$ javac -cp java/JTransforms-3.1-with-dependencies.jar java/*.java
```
**Note a new dependency was introduced in January 2021, making the models compatible with the newest versions of dependency packages. You therefore need to download the updated files to achieve this**.
```
$ git pull
$ bash utilities/downloadDataModels.sh
$ pip install --upgrade pip
$ pip3 install --upgrade -r requirements.txt.
$ javac -cp java/JTransforms-3.1-with-dependencies.jar java/*.java
```


## Usage
To extract a summary of movement (average sample vector magnitude) and
(non)wear time from raw Axivity .CWA (or gzipped .cwa.gz) accelerometer files:

```
$ python3 accProcess.py data/sample.cwa.gz
 <output written to data/sample-outputSummary.json>
 <time series output written to data/sample-timeSeries.csv.gz>
```

The main JSON output will look like:
```
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

To visualise the time series and activity classification output:
```
$ python3 accPlot.py data/sample-timeSeries.csv.gz data/sample-plot.png
 <output plot written to data/sample-plot.png>
```
![Time series plot](docs/source/samplePlot.png)

The underlying modules can also be called in custom python scripts:
```Python
    from accelerometer import summariseEpoch
    summary = {}
    epochData, labels = summariseEpoch.getActivitySummary("sample-epoch.csv.gz", 
            "sample-nonWear.csv.gz", summary)
    # <nonWear file written to "sample-nonWear.csv.gz" and dict "summary" updated
    # with outcomes>
```

## Under the hood
Interpreted levels of physical activity can vary, as many approaches can be 
taken to extract summary physical activity information from raw accelerometer 
data. To minimise error and bias, our tool uses published methods to calibrate, 
resample, and summarise the accelerometer data. [Click here for detailed 
information on the 
data processing methods on our wiki.](https://biobankaccanalysis.readthedocs.io/en/latest/methods.html)

![Accelerometer data processing overview](docs/source/accMethodsOverview.png)
![Activity classification](docs/source/accClassification.png)



## Citing our work
When describing or using the UK Biobank accelerometer dataset, or using this tool
to extract overall activity from your accelerometer data, please cite [Doherty2017].

When using this tool to extract sleep duration and physical activity behaviours
from your accelerometer data, please cite [Willetts2018], [Doherty2018], and 
[Walmsley2020]

```
[Doherty2017] Doherty A, Jackson D, et al. (2017) 
Large scale population assessment of physical activity using wrist worn 
accelerometers: the UK Biobank study. PLOS ONE. 12(2):e0169649

[Willetts2018] Willetts M, Hollowell S, et al. (2018) 
Statistical machine learning of sleep and physical activity phenotypes from 
sensor data in 96,220 UK Biobank participants. Scientific Reports. 8(1):7961

[Doherty2018] Doherty A, Smith-Bryne K, et al. (2018) 
GWAS identifies 14 loci for device-measured physical activity and sleep 
duration. Nature Communications. 9(1):5257

[Walmsley2020] Walmsley R, Chan S, et al. (2020)
Reallocating time from machine-learned sleep, sedentary behaviour or light 
physical activity to moderate-to-vigorous physical activity is associated with 
lower cardiovascular disease risk (preprint https://doi.org/10.1101/2020.11.10.20227769)
```

###### Licence
This project is released under a [BSD 2-Clause Licence](http://opensource.org/licenses/BSD-2-Clause) (see LICENCE file)
