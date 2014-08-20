biobankAccelerometerAnalysis
======================

A tool to extract meaningful health information from large accelerometer
datasets e.g. how much time individuals spend in sleep, sedentary behaviour,
and physical activity.

This software will allow non-specialists worldwide to benefit from the
information currently locked within the accelerometer data of large health
studies e.g. UK Biobank are asking 100,000 people to wear wrist-worn
accelerometers for 1 week each, and will then monitor their health for the
rest of their lives.


<h2>Usage</h2>
To extract a summary of movement (average sample vector magnitude) and
(non)wear time from raw GENEActiv/Axivity .bin/.CWA accelerometer files ([click here for sample CWA file](http://users.fmrib.ox.ac.uk/~adoherty/CWA-DATA.CWA)):

```
python ActivitySummaryFromEpochs.py [input_file.CWA] [options]
python ActivitySummaryFromEpochs.py p001.cwa
```

For customised options, please refer to our wiki.


<h2>Installation</h2>
Dependancies include numpy and pandas python libraries.
```
javac BandpassFilter.java
javac AxivityAx3Epochs.java
```


<h2>Under the hood</h2>
We are using a combination of published methods to extract meaningful health
information from accelerometer data. For more information, please refer to our
wiki.

![Accelerometer data processing overview]
(http://users.fmrib.ox.ac.uk/~adoherty/accProcessingOverview.png)

<h3>Matlab implementation for reading, interpolating and calibrating CWA files</h3>
In the folder matlab you can find code for the first step in the processing pipeline (first box in image above). Most parameters are hard-coded at the moment for improved speed. On a typical cwa file this script should run within around 100 seconds, although that may depend on the speed of your machine. You can either run the code from the matlab shell or invoke MATLAB from the command line (e.g. in linux, mac os x).

In MATLAB, change to the matlab directory of this repository and run:
```
D = readInterpolateCalibrate('/path/to/CWA-DATA.CWA','/path/to/output.wav');
```
From the command line you can do the same (change the paths according to your MATLAB installation):
```
/Applications/MATLAB_R2014a.app/bin/matlab -nosplash -nodisplay -r "readInterpolateCalibrate('/path/to/CWA-DATA.CWA','/path/to/output.wav');exit;"
```
Both of these commands will write an output file in .wav format that contains the calibrated accelerometer readings (you can also use .flac for compression). The first three track correspond to X,Y,Z scaled between -8 and 8 [g]. Invalid entries are zero across all tracks. The fourth track contains the temperature readings (positive entries between 0 and 100) and light readings (negative entries between 0 and 1000). The position along the track indicates when these were sampled.

Reading in the cwa-files relies on some compiled c-code. There are some binaries included which should do the trick. If not, you can compile these yourself. In matlab, just change to 'matlab/lib/AX3_readFile_v0.1' and run these commands (requires a suitable build-environment, e.g. gcc):
```
mex parseDate.c; mex parseValueBlock.c;
```
If this does not work for you the code can be changed to use native matlab code, which works fine but is significantly slower. In calls to AX3_readFile (in readInterpolateCalibrate.m) simply change the 'useC' flag from '1' to '0'.

Why should you calibrate your data? Check the example plot in the repository: calibrationExample.png


<h6>Licence</h6>
This project is released under a BSD Licence (see LICENCE)
