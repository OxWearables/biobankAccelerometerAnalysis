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
    $ python3 accProcess.py data/sample.cwa.gz
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
  $ python3 accPlot.py data/sample-timeSeries.csv.gz data/sample-plot.png
    <output plot written to data/sample-plot.png>

.. figure:: samplePlot.png

    Output plot of overall activity and class predictions for each 30sec time window




**************
Input file types
**************

========================
GENEActiv
========================
Process data from raw `GENEActiv <https://49wvycy00mv416l561vrj345-wpengine.netdna-ssl.com/wp-content/uploads/2019/06/geneactiv_instruction_manual_v1.4.pdf>`_ .bin files:
::
    $ python3 accProcess.py data/sample.bin


========================
Actigraph
========================
Process data from raw `Actigraph <https://github.com/actigraph/GT3X-File-Format>`_ .gt3x files (both versions 1 and 2):
::
    $ python3 accProcess.py data/sample.gt3x --sampleRate 80

An example Actigraph file can be obtained from the `AGread <https://github.com/paulhibbing/AGread>`_ gitHub page:
::
    $ wget "https://github.com/paulhibbing/AGread/raw/master/data-raw/119AGBPFLW%20(2016-03-08).gt3x"
    $ mv 119AGBPFLW\ \(2016-03-08\).gt3x data/actigraph-example.gt3x
    $ python3 accProcess.py data/sample.gt3x --sampleRate 80


========================
CSV
========================
Process data from raw gzipped CSV files:
::
    $ python3 accProcess.py data/sample.csv.gz

It is very unwise to store accelerometer data in .csv format. However, if one
were to unzip and view .csv.gz file it would ideally be in this format:
::
    $ wget "http://gas.ndph.ox.ac.uk/aidend/accModels/sample-small.csv.gz"
    $ mv sample-small.csv.gz data/
    $ gunzip data/sample.csv.gz
    $ head -3 data/sample.csv
    time,x,y,z
    2014-05-07 13:29:50.439+0100 [Europe/London],-0.514,0.07,1.671
    2014-05-07 13:29:50.449+0100 [Europe/London],-0.089,-0.805,-0.59

If your CSV is in a different format, there are options to flexibly parse these.
Consider the below file with a different time format and the x/y/z columns having
different index positions
::
    $ head data/awkwardfile.csv
    time,temperature,z,y,x
    2014-05-07 13:29:50.439,20,0.07,1.671,-0.514
    2014-05-07 13:29:50.449,20,-0.805,-0.59,-0.089

The above file can be processed as follows:
::
    $ python3 accProcess.py data/awkwardFile.csv \
    --csvTimeFormat 'yyyy-MM-dd HH:mm:ss.SSS' --csvTimeXYZTempColsIndex 0,4,2,3


If your CSV also has temperature values, it is also possible to include these:
::
    $ python3 accProcess.py data/awkwardFile.csv \
    --csvTimeFormat 'yyyy-MM-dd HH:mm:ss.SSS' --csvTimeXYZTempColsIndex 0,4,2,3,1



*************************
Processing multiple files
*************************

Suppose we want to process hundreds of accelerometer files:
::
    studyName/
        files.csv  # listing files to be processed (optional)
        subject001.cwa
        subject002.cwa
        subject003.cwa
        ...

We provide a python utility function to facilitate generating the list of
commands to process each file:
::
    from accelerometer import accUtils
    accUtils.writeStudyAccProcessCmds(
        "myStudy/",
        outDir="myStudyResults/",
        cmdsFile="process-cmds.txt"
    )
    # <list of processing commands written to "process-cmds.txt">

    # If we need to pass arguments to the processing commands, use 'cmdsOptions'
    # e.g. if for some reason we wanted to use different thresholds for moderate
    # and vigorous intensity activities, we could go with
    accUtils.writeStudyAccProcessCmds(
        "myStudy/",
        outDir="myStudyResults/",
        cmdOptions="--mgCutPointMVPA 90 --mgCutPointVPA 435",
        cmdsFile="process-cmds.txt",
    )
    # <list of processing commands written to "process-cmds.txt">

In the example above, a `process-cmds.txt` text file is created, listing the
processing commands for each file listed in `files.csv`. If `files.csv` is
not present, all the accelerometer files in `myStudy/` will be processed, and
a `files.csv` will be created in place listing all the files. For
this to work, we need to specify which file type to use by setting the
`accExt` parameter, e.g., cwa, CWA, bin, BIN, gt3x. We can also directly
create our own `files.csv` with a column whose column name needs to be
'fileName'.

We can then kick-start the processing of all accelerometer files. More advanced
users will probably want to parallelise the below script using their HPC
architecture of choice:
::
    $ bash process-cmds.txt

The results of the processing are stored in `myStudyResults/`. The output
directory has the following structure (which is automatically created):
::
    myStudyResults/
        summary/ #to store outputSummary.json
            subject001-summary.json
            subject002-summary.json
            subject003-summary.json
            ...
        epoch/ #to store feature output for 30sec windows
            subject001-epoch.csv
            subject002-epoch.csv
            subject003-epoch.csv
            ...
        timeSeries/ #simple csv time series output (VMag, activity binary predictions)
            subject001-timeSeries.csv
            subject002-timeSeries.csv
            subject003-timeSeries.csv
            ...
        nonWear/ #bouts of nonwear episodes
            ...
        stationary/ #temp store for features of stationary data for calibration
            ...
        clusterLogs/ #to store terminal output for each processed file
            ...

Next, using our python utility function, we would like to collate all
individual processed .json summary files into a single large csv for subsequent
health analses:
::
    from accelerometer import accUtils
    accUtils.collateJSONfilesToSingleCSV("myStudyResults/summary/", \
        "myStudyResults/summary-info.csv")
    # <summary CSV for all participants written to "myStudyResults/sumamry-info.csv">

===============
Quality control
===============
If is often necessary to check that all files have successfully processed. Our
python utility function can write to file all participants' data that was not
successfully processed:
::
    from accelerometer import accUtils
    accUtils.identifyUnprocessedFiles("myStudy/files.csv", "myStudyResults/summary-info.csv", \
          "myStudyResults/files-unprocessed.csv")
    # <Output CSV listing files to be reprocessed written to "myStudyResults/files-unprocessed.csv">


On other occasions some participants' data may not have been calibrated properly.
Our python utility function can assigns the calibration coefs from a previous
good use of a given device in the same study dataset:
::
    from accelerometer import accUtils
    accUtils.updateCalibrationCoefs("myStudyResults/summary-info.csv", \
           "myStudyResults/files-recalibration.csv")
    # <CSV of files to be reprocessed written to "myStudyResults/files-recalibration.csv">


Our python utility function can then re-write processing cmds as follows:
::
    from accelerometer import accUtils
    accUtils.writeStudyAccProcessCmds("myStudy/", cmdsFile="process-cmds-recalibration.txt", \
       outDir="myStudyResults/", filesID="myStudyResults/files-calibration.csv", cmdOptions="--skipCalibration True")
    # <list of processing commands written to "process-cmds-recalibration.txt">

These 'reprocessed' files can then be processed as outlined in the section above.




************************************
Classifying different activity types
************************************
**Note that a major fix/improvement was introduced in April 2020. You therefore need to download the updated files to achieve this**.
::
	$ git pull
        $ bash utilities/downloadDataModels.sh
        $ pip3 install --user .
        $ javac -cp java/JTransforms-3.1-with-dependencies.jar java/*.java


Different activity classification models can be specified to identify different
activity types. For example, to use activity states from the Willetts 2018
Scientific Reports paper:
::
    $ python3 accProcess.py data/sample.cwa.gz \
        --activityModel activityModels/willetts2018-apr20Update.tar


To visualise the time series and new activity classification output:
::
    $ python3 accPlot.py data/sample-timeSeries.csv.gz data/sample-plot.png \
        --activityModel activityModels/willetts2018-apr20Update.tar
    <output plot written to data/sample-plot.png>

.. figure:: samplePlotWilletts.png

    Output plot of class predictions using Willetts 2018 classification model.
    Note different set of activity classes.

========================
Training a bespoke model
========================
It is also possible to train a bespoke activity classification model. This
requires a labelled dataset (.csv file) and a list of features (.txt file) to
include from the epoch file.

First we need to evaluate how well the model works on unseen data. We therefore
train a model on a 'training set' of participants, and then test how well that
model works on a 'test set' of participant. The command below allows us to achieve
this by specifying the test participant IDs (all other IDs will automatically go
to the training set). This will output <participant, time, actual, predicted>
predictions for each instance of data in the test set to a CSV file to help
assess the model:
::
    import accelerometer
    accelerometer.accClassification.trainClassificationModel( \
        "activityModels/labelled-acc-epochs.csv", \
        featuresTxt="activityModels/features.txt", \
        testParticipants="4,5", \
        outputPredict="activityModels/test-predictions.csv", \
        rfTrees=1000, rfThreads=1)
    # <Test predictions written to:  activityModels/test-predictions.csv>

A number of `metrics <https://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation>`_
can then be calculated from the test predictions csv file:
::
    import pandas as pd
    from accelerometer import accClassification

    # load data
    d = pd.read_csv("test-predictions.csv")

    # print summary to HTML file
    htmlFile = "classificationReport.html"
    yTrueCol = 'label'
    yPredCol = 'predicted'
    participantCol = 'participant'
    accClassification.perParticipantSummaryHTML(d, yTrueCol, yPredCol,
        participantCol, htmlFile)

After evaluating the performance of our model on unseen data, we then re-train
a final model that includes all possible data. We therefore specify the
outputModel parameter, and also set testParticipants to 'None' so as to maximise
the amount of training data for the final model. This results in an output .tar model:
::
    import accelerometer
    accelerometer.accClassification.trainClassificationModel( \
        "activityModels/labelled-acc-epochs.csv", \
        featuresTxt="activityModels/features.txt", \
        rfTrees=1000, rfThreads=1, \
        testParticipants=None, \
        outputModel="activityModels/sample-model.tar")
    # <Model saved to activityModels/sample-model.tar>


This new model can be deployed as follows:
::
    $ python3 accProcess.py --activityModel activityModels/sample-model.tar \
        data/sample.cwa.gz

============================
Leave one out classification
============================
To rigorously test a model with training data from <200 participants, leave one
participant out evaluation can be helpful. Building on the above
examples of training a bespoke model, we use python to create a list of commands
to test the performance of a model trained on unseen data for each participant:
::
    import pandas as pd
    trainingFile = "activityModels/labelled-acc-epochs.csv"
    d = pd.read_csv(trainingFile, usecols=['participant'])
    pts = sorted(d['participant'].unique())

    w = open('training-cmds.txt','w')
    for p in pts:
        cmd = "import accelerometer;"
        cmd += "accelerometer.accClassification.trainClassificationModel("
        cmd += "'" + trainingFile + "', "
        cmd += "featuresTxt='activityModels/features.txt',"
        cmd += "testParticipants='" + str(p) + "',"
        cmd += "labelCol='label',"
        cmd += "outputPredict='activityModels/testPredict-" + str(p) + ".csv',"
        cmd += "rfTrees=100, rfThreads=1)"
        w.write('python3 -c $"' + cmd + '"\n')
    w.close()
    # <list of processing commands written to "training-cmds.txt">

These commands can be executed as follows:
::
    $ bash training-cmds.txt

After processing the train/test commands, the resulting predictions for each
test participant can be collated as follows:
::
    $ head -1 activityModels/testPredict-1.csv > header.csv
    $ awk 'FNR > 1' activityModels/testPredict-*.csv > tmp.csv
    $ cat header.csv tmp.csv > test-predictions.csv
    $ rm header.csv
    $ rm tmp.csv

As indicated just above (under 'Training a bespoke model'), a number of metrics
can be calculated for the 'testPredict-all.csv' file.





**************
Advanced usage
**************
To list all available processing options and their defaults, simply type:
::
    $ python3 accProcess.py -h

Some example usages:

Specify file in another folder (note: use "" for file names with spaces):
::
    $ python3 accProcess.py "/otherPath/other file.cwa"

Change epoch length to 60 seconds:
::
    $ python3 accProcess.py data/sample.cwa.gz --epochPeriod 60

Manually set calibration coefficients:
::
    $ python3 accProcess.py data/sample.cwa.gz --skipCalibration True \
        --calOffset -0.2 -0.4 1.5  --calSlope 0.7 0.8 0.7 \
        --calTemperature 0.2 0.2 0.2 --meanTemp 20.2

Extract calibrated and resampled raw data .csv.gz file from raw .cwa file:
::
    $ python3 accProcess.py data/sample.cwa.gz --rawOutput True \
        --activityClassification False

The underlying modules can also be called in custom python scripts:
::
    from accelerometer import summariseEpoch
    summary = {}
    epochData, labels = summariseEpoch.getActivitySummary( \
        "data/sample-epoch.csv.gz", "data/sample-nonWear.csv.gz", summary)
    # <nonWear file written to "data/sample-nonWear.csv.gz" and dict "summary" \
    #    updated with outcomes>
