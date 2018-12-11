#####
Usage
#####

Our tool uses published methods to extract summary sleep and activity statistics from raw binary accelerometer data files.



***********
Basic usage
***********
To extract a summary of movement (average sample vector magnitude) and
(non)wear time from raw Axivity .CWA accelerometer files:
::
    $ python3 accProcess.py data/sample.cwa
    $ <summary output written to data/sample-outputSummary.json>
    $ <time series output written to data/sample-timeSeries.csv.gz>
    $ <non wear episodes output written to data/sample-nonWearEpisodes.csv.gz>

This may take a few minutes. When done, there will be four files (default in the same folder as the input .cwa file) containing the extracted data.

+--------------------+--------------------------------------------------------+
| File               | Description                                            |
+====================+========================================================+
| OutputSummary.json | Summary statistics for the entire input file, such as  |
|                    | data quality, acceleration and non-wear time grouped   |
|                    | hour of day, and histograms of acceleration levels.    |
|                    | Download a sample file.                                |
+--------------------+--------------------------------------------------------+
| TimeSeries.csv     | Acceleration magnitude for each epoch, and whether the |
|                    | data was imputed or not.                               |
+--------------------+--------------------------------------------------------+
| Epoch.csv          | Acceleration data grouped in epochs (default = 5sec).  |
|                    | Detailed information about XYZ acceleration, standard  |
|                    | deviation, temperature, and data errors can be found   |
|                    | in this file.                                          |
+--------------------+--------------------------------------------------------+
| NonWearBouts.csv   | Start and end times for any non-wear bouts, and the    |
|                    | detected (presumably low) acceleration levels for each |
|                    | bout.                                                  |
+--------------------+--------------------------------------------------------+

To visualise the time output:
::
  python3 accPlot.py data/sample-timeSeries.csv.gz data/sample-plot.png
    <output plot written to data/sample-plot.png>

.. figure:: samplePlot.png

    Output plot of overall activity and class predictions for each 30sec time window



*************************
Processing multiple files
*************************

To process multiple files, we recommend the following directory structure be used:
::
    <studyName>/
        files.csv #listing all files in rawData directory
        rawData/ #all raw .cwa .bin .gt3x files
        summary/ #to store outputSummary.json
        epoch/ #to store feature output for 30sec windows
        timeSeries/ #simple csv time series output (VMag, activity binary predictions)
        nonWear/ #bouts of nonwear episodes
        stationary/ #temp store for features of stationary data for calibration
        clusterLogs/ #to store terminal output for each processed file

This can be created calling our utility script:
::
    $ bash utilities/createStudyDir.sh /myStudy/

Next move relevant raw accelerometer files to the rawData folder:
::
    $ mv *myAccelerometerFiles.cwa /myStudy/rawData/

Then use our python utility function to write processing cmds for all files:
::
    >>> from accelerometer import accUtils
    >>> accUtils.writeStudyAccProcessCmds("/myStudy/", "process-cmds.txt")
    <list of processing commands written to "process-cmds.txt">

    >>> # if for some reason we wanted to use different thresholds for moderate
    >>> # and vigorous intensity activities, we could go with
    >>> accUtils.writeStudyAccProcessCmds("/myStudy/", "process-cmds.txt", \
    >>>     cmdOptions="--mgMVPA 90 --mvVPA 435")
    <list of processing commands written to "process-cmds.txt">

We can then kick-start the processing of all accelerometer files. More advanced
users will probably want to parallelise the below script using their HPC
architecture of choice:
::
    $ bash process-cmds.txt

Finally, using our python utility function, we would like to collate all 
individual processed .json summary files into a single large csv for subsequent 
health analses:
::
    >>> from accelerometer import accUtils
    >>> accUtils.collateJSONfilesToSingleCSV("/myStudy/summary/", "myStudy/summary-info.csv")
    <summary CSV for all participants written to "/myStudy/sumamry-info.csv">
    """



************************************
Classifying different activity types
************************************
Different activity classification models can be specified to identify different 
activity types. For example to use activity states from the Willetts 2018 
Scientific Reports paper:
::
    python3 accProcess.py --activityModel activityModels/willetts2018.tar

To visualise the time series and new activity classification output:
::
  python3 accPlot.py data/sample-timeSeries.csv.gz data/sample-plot.png 
     --activityModel activityModels/willetts2018.tar
    <output plot written to data/sample-plot.png>

.. figure:: samplePlotWilletts.png
    
    Output plot of class predictions using Willetts 2018 classification model. 
    Note different set of activity classes.


**************
Advanced usage
**************
To list all available processing options and their defaults, simply type:
::
    python3 accProcess.py -h

Some example usages:

Specify file in another folder (note: use "" for file names with spaces):
::
    $ python3 accProcess.py "/otherPath/other file.cwa" 

Change epoch length to 60 seconds:
::
    $ python3 accProcess.py data/sample.cwa --epochPeriod 60 

Manually set calibration coefficients:
::
    $ python3 accProcess.py data/sample.cwa --skipCalibration True
        --calOffset -0.2 -0.4 1.5  --calSlope 0.7 0.8 0.7
        --calTemperature 0.2 0.2 0.2 --meanTemp 20.2


The underlying modules can also be called in custom python scripts:
::
    >>> from accelerometer import summariseEpoch
    >>> summary = {}
    >>> epochData, labels = summariseEpoch.getActivitySummary("data/sample-epoch.csv.gz", 
            "data/sample-nonWear.csv.gz", summary)
    <nonWear file written to "data/sample-nonWear.csv.gz" and dict "summary" updated
    with outcomes>
