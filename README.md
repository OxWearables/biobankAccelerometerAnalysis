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

[Click here for a sample .CWA file.]
(http://users.fmrib.ox.ac.uk/~adoherty/sample.cwa)

The output will look like:
```
timestamp [input_file.CWA], [summary physical activity score], [start time] - [end time], [wear time], [non wear time]
2015-05-15 09:45:27 sample.cwa, 33.44 mg, 2014-05-07 13:29:50 - 2014-05-13 09:50:25, 8357 mins wear, 62     mins nonWear
```
[Click here for customised usage options on our wiki.]
(https://github.com/aidendoherty/biobankAcceleromerAnalysis/wiki/1. Usage)


<h2>Installation</h2>
Dependancies include: matlab, java, and python (numpy and pandas).
```
javac *.java
```

[Click here for detailed information on installing this software on our wiki.]
(https://github.com/aidendoherty/biobankAcceleromerAnalysis/wiki/2. Installation)

<h2>Under the hood</h2>
We are using a combination of published methods to extract meaningful health
information from accelerometer data. [Click here for detailed information on the 
data processing methods on our wiki.]
(https://github.com/aidendoherty/biobankAccelerometerAnalysis/wiki/3.-Methods-Overview)

![Accelerometer data processing overview]
(http://users.fmrib.ox.ac.uk/~adoherty/accProcessingOverviewDec2014.png)


<h6>Licence</h6>
This project is released under a [BSD 2-Clause Licence](http://opensource.org/licenses/BSD-2-Clause) (see LICENCE file)
