biobankAccelerometerAnalysis
======================

A tool to extract meaningful health information from large accelerometer
datasets e.g. how much time individuals spend in sleep, sedentary behaviour,
and physical activity.


<h2>Usage</h2>
To extract a summary of movement (average sample vector magnitude) and
(non)wear time from raw Axivity .CWA accelerometer files:

```
python ActivitySummary.py [input_file.CWA] [options]
python ActivitySummary.py sample.cwa
```

[Click here for a sample .CWA file.](http://gas.ndph.ox.ac.uk/aidend/sample.cwa)

The output will look like:
```
{
    "file-name": "sample.cwa", 
    "file-startTime": "2014-05-07 13:29:50", 
    "file-endTime": "2014-05-13 09:50:25", 
    "pa-overall-avg(mg)": "33.01", 
    "wearTime-overall(days)": "5.80", 
    "nonWearTime-overall(days)": "0.04"
}
```
[Click here for customised usage options on our wiki.](https://github.com/aidendoherty/biobankAcceleromerAnalysis/wiki/1.-Usage)


<h2>Installation</h2>
Dependancies include: java and python (numpy and pandas).

[Click here for detailed information on installing this software on our wiki.](https://github.com/aidendoherty/biobankAcceleromerAnalysis/wiki/2.-Installation)

<h2>Under the hood</h2>
We are using a combination of published methods to extract meaningful health
information from accelerometer data. [Click here for detailed information on the 
data processing methods on our wiki](https://github.com/aidendoherty/biobankAccelerometerAnalysis/wiki/3.-Methods-Overview).

![Accelerometer data processing overview](http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0169649.g001&type=large)


<h6>Licence</h6>
This project is released under a [BSD 2 Clause Licence](http://opensource.org/licenses/BSD-2-Clause)
