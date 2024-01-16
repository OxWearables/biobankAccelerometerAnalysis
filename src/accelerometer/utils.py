"""Module to provide generic utilities for other accelerometer modules."""

from collections import OrderedDict
import datetime
import json
import math
import os
import pandas as pd
import re
from tqdm.auto import tqdm

DAYS = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
TIME_SERIES_COL = 'time'


def meanSDstr(mean, std, numDecimalPlaces):
    """
    Return str of mean and stdev numbers formatted to number of decimalPlaces

    :param float mean: Mean number to be formatted.
    :param float std: Standard deviation number to be formatted.
    :param int decimalPlaces: Number of decimal places for output format
    :return: String formatted to number of decimalPlaces
    :rtype: str

    .. code-block:: python

        import accUtils
        accUtils.meanSDstr(2.567, 0.089, 2)
    """
    outStr = str(formatNum(mean, numDecimalPlaces))
    outStr += ' ('
    outStr += str(formatNum(std, numDecimalPlaces))
    outStr += ')'
    return outStr


def meanCIstr(mean, std, n, numDecimalPlaces):
    """
    Return str of mean and 95% confidence interval numbers formatted

    :param float mean: Mean number to be formatted.
    :param float std: Standard deviation number to be formatted.
    :param int n: Number of observations
    :param int decimalPlaces: Number of decimal places for output format
    :return: String formatted to number of decimalPlaces
    :rtype: str

    .. code-block:: python

        import accUtils
        accUtils.meanSDstr(2.567, 0.089, 2, 2)
    """
    stdErr = std / math.sqrt(n)
    lowerCI = mean - 1.96 * stdErr
    upperCI = mean + 1.96 * stdErr
    outStr = str(formatNum(mean, numDecimalPlaces))
    outStr += ' ('
    outStr += str(formatNum(lowerCI, numDecimalPlaces))
    outStr += ' - '
    outStr += str(formatNum(upperCI, numDecimalPlaces))
    outStr += ')'
    return outStr


def toScreen(msg):
    """
    Print msg str prepended with current time

    :param str mgs: Message to be printed to screen
    :return: None. Prints msg str prepended with current time.

    .. code-block:: python

        import accUtils
        accUtils.toScreen("hello")
    """

    timeFormat = '%Y-%m-%d %H:%M:%S'
    print(f"\n{datetime.datetime.now().strftime(timeFormat)}\t{msg}")


def writeCmds(accDir, outDir, cmdsFile='list-of-commands.txt', accExt="cwa", cmdOptions="", filesCSV=None):
    """
    Generate a text file listing processing commands for files found under accDir/

    :param str accDir: Directory with accelerometer files to process
    :param str outDir: Output directory to be created containing the processing results
    :param str cmdsFile: Output .txt file listing all processing commands
    :param str accExt: Acc file type e.g. cwa, CWA, bin, BIN, gt3x...
    :param str cmdOptions: String of processing options e.g. "--epochPeriod 10"
        Type 'python3 accProccess.py -h' for full list of options

    :return: None. New file written to <cmdsFile>.

    .. code-block:: python

        import accUtils
        accUtils.writeProcessingCommands("myAccDir/", "myResults/", "myProcessCmds.txt")
    """

    # Use filesCSV if provided, else retrieve all accel files under accDir/
    if filesCSV in os.listdir(accDir):
        filesCSV = pd.read_csv(os.path.join(accDir, filesCSV), index_col="fileName")
        filesCSV.index = accDir.rstrip("/") + "/" + filesCSV.index.astype('str')
        filePaths = filesCSV.index.to_numpy()

    else:
        filesCSV = None
        # List all accelerometer files under accDir/
        filePaths = []
        accExt = accExt.lower()
        for root, dirs, files in os.walk(accDir):
            for file in files:
                if file.lower().endswith((accExt,
                                          accExt + ".gz",
                                          accExt + ".zip",
                                          accExt + ".bz2",
                                          accExt + ".xz")):
                    filePaths.append(os.path.join(root, file))

    with open(cmdsFile, 'w') as f:
        for filePath in filePaths:

            # Use the file name as the output folder name for the process,
            # keeping the same directory structure of accDir/
            # Example: If filePath is {accDir}/group0/subject123.cwa then
            # outputFolder will be {outDir}/group0/subject123/
            outputFolder = filePath.replace(accDir.rstrip("/"), outDir.rstrip("/")).split(".")[0]

            # Enclose with single quotes to handle spaces
            filePath = "'" + filePath + "'"
            outputFolder = "'" + outputFolder + "'"

            cmd = f"accProcess {filePath} --outputFolder {outputFolder} {cmdOptions}"

            if filesCSV is not None:
                # Grab additional options provided in filesCSV (e.g. calibration params)
                cmdOptionsCSV = ' '.join(['--{} {}'.format(col, filesCSV.loc[filePath, col])
                                          for col in filesCSV.columns])
                cmd += " " + cmdOptionsCSV

            f.write(cmd)
            f.write('\n')

    print('List of commands written to ', cmdsFile)


def collateSummary(resultsDir, outputCsvFile="all-summary.csv"):
    """
    Read all -summary.json files under <resultsDir> and merge into one CSV file.
    Each json file represents summary data for one participant.
    Therefore output CSV file contains summary for all participants.

    :param str resultsDir: Directory containing JSON files.
    :param str outputCsvFile: Output CSV filename.

    :return: None. A new file is written to <outputCsvFile>.

    .. code-block:: python

        import accUtils
        accUtils.collateSummary("data/", "data/all-summary.csv")
    """

    print(f"Scanning {resultsDir} for summary files...")
    # Load all *-summary.json files under resultsDir/
    sumfiles = []
    jdicts = []
    for root, dirs, files in os.walk(resultsDir):
        for file in files:
            if file.lower().endswith("-summary.json"):
                sumfiles.append(os.path.join(root, file))

    print(f"Found {len(sumfiles)} summary files...")
    for file in tqdm(sumfiles):
        with open(file, 'r') as f:
            jdicts.append(json.load(f, object_pairs_hook=OrderedDict))

    summary = pd.DataFrame.from_dict(jdicts)  # merge to a dataframe
    summary['eid'] = summary['file-name'].str.split('/').str[-1].str.split('.').str[0]  # infer ID from filename
    summary.to_csv(outputCsvFile, index=False)
    print('Summary of', str(len(summary)), 'participants written to:', outputCsvFile)


def identifyUnprocessedFiles(filesCsv, summaryCsv, outputFilesCsv):
    """
    Identify files that have not been processed.
    Look through all processed accelerometer files, and find participants who do
    not have records in the summary csv file. This indicates there was a problem
    in processing their data. Therefore, output will be a new .csv file to
    support reprocessing of these files.

    :param str filesCsv: CSV listing acc files in study directory.
    :param str summaryCsv: Summary CSV of processed dataset.
    :param str outputFilesCsv: Output csv listing files to be reprocessed.

    :return: None. A new file is written to <outputFilesCsv>.

    .. code-block:: python

        import accUtils
        accUtils.identifyUnprocessedFiles("study/files.csv", "study/summary-all-files.csv", "study/files-reprocess.csv")
    """

    fileList = pd.read_csv(filesCsv)
    summary = pd.read_csv(summaryCsv)

    output = fileList[~fileList['fileName'].isin(list(summary['file-name']))]
    output = output.rename(columns={'Unnamed: 1': ''})
    output.to_csv(outputFilesCsv, index=False)

    print('Reprocessing for ', len(output), 'participants written to:',
          outputFilesCsv)


def updateCalibrationCoefs(inputCsvFile, outputCsvFile):
    """
    Read summary .csv file and update coefs for those with poor calibration
    Look through all processed accelerometer files, and find participants that
    did not have good calibration data. Then assigns the calibration coefs from
    previous good use of a given device. Output will be a new .csv file to
    support reprocessing of uncalibrated files with new pre-specified calibration coefs.

    :param str inputCsvFile: Summary CSV of processed dataset
    :param str outputCsvFile: Output CSV of files to be reprocessed with new
        calibration info

    :return: None. New file written to <outputCsvFile>

    .. code-block:: python

        import accUtils
        accUtils.updateCalibrationCoefs("data/summary-all-files.csv", "study/files-recalibration.csv")

    CSV of files to be reprocessed written to "study/files-recalibration.csv"
    """

    d = pd.read_csv(inputCsvFile)
    # select participants with good spread of stationary values for calibration
    goodCal = d.loc[(d['quality-calibratedOnOwnData'] == 1) & (d['quality-goodCalibration'] == 1)]
    # now only select participants whose data was NOT calibrated on a good spread of stationary values
    badCal = d.loc[(d['quality-calibratedOnOwnData'] == 1) & (d['quality-goodCalibration'] == 0)]

    # sort files by start time, which makes selection of most recent value easier
    goodCal = goodCal.sort_values(['file-startTime'])
    badCal = badCal.sort_values(['file-startTime'])

    calCols = ['calibration-xOffset(g)', 'calibration-yOffset(g)', 'calibration-zOffset(g)',
               'calibration-xSlope(g)', 'calibration-ySlope(g)', 'calibration-zSlope(g)',
               'calibration-xTemp(C)', 'calibration-yTemp(C)', 'calibration-zTemp(C)',
               'calibration-meanDeviceTemp(C)']

    # print output CSV file with suggested calibration parameters
    noOtherUses = 0
    nextUses = 0
    previousUses = 0
    f = open(outputCsvFile, 'w')
    f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
    for ix, row in badCal.iterrows():
        # first get current 'bad' file
        participant, device, startTime = row[['file-name', 'file-deviceID', 'file-startTime']]
        device = int(device)
        # get calibration values from most recent previous use of this device
        # (when it had a 'good' calibration)
        prevUse = goodCal[calCols][(goodCal['file-deviceID'] == device) &
                                   (goodCal['file-startTime'] < startTime)].tail(1)
        try:
            ofX, ofY, ofZ, slpX, slpY, slpZ, tmpX, tmpY, tmpZ, calTempAvg = prevUse.iloc[0]
            previousUses += 1
        except Exception:
            nextUse = goodCal[calCols][(goodCal['file-deviceID'] == device) &
                                       (goodCal['file-startTime'] > startTime)].head(1)
            if len(nextUse) < 1:
                print('no other uses for this device at all: ', str(device),
                      str(participant))
                noOtherUses += 1
                continue
            nextUses += 1
            ofX, ofY, ofZ, slpX, slpY, slpZ, tmpX, tmpY, tmpZ, calTempAvg = nextUse.iloc[0]

        # now construct output
        out = participant + ','
        out += str(ofX) + ' ' + str(ofY) + ' ' + str(ofZ) + ','
        out += str(slpX) + ' ' + str(slpY) + ' ' + str(slpZ) + ','
        out += str(tmpX) + ' ' + str(tmpY) + ' ' + str(tmpZ) + ','
        out += str(calTempAvg)
        f.write(out + '\n')
    f.close()
    print('previousUses', previousUses)
    print('nextUses', nextUses)
    print('noOtherUses', noOtherUses)

    print('Reprocessing for ', str(previousUses + nextUses),
          'participants written to:', outputCsvFile)


def writeFilesWithCalibrationCoefs(inputCsvFile, outputCsvFile):
    """
    Read summary .csv file and write files.csv with calibration coefs.
    Look through all processed accelerometer files, and write a new .csv file to
    support reprocessing of files with pre-specified calibration coefs.

    :param str inputCsvFile: Summary CSV of processed dataset
    :param str outputCsvFile: Output CSV of files to process with calibration info

    :return: None. New file written to <outputCsvFile>

   .. code-block:: python

        import accUtils
        accUtils.writeFilesWithCalibrationCoefs("data/summary-all-files.csv", "study/files-calibrated.csv")
    """

    d = pd.read_csv(inputCsvFile)

    calCols = ['calibration-xOffset(g)', 'calibration-yOffset(g)', 'calibration-zOffset(g)',
               'calibration-xSlope(g)', 'calibration-ySlope(g)', 'calibration-zSlope(g)',
               'calibration-xTemp(C)', 'calibration-yTemp(C)', 'calibration-zTemp(C)',
               'calibration-meanDeviceTemp(C)']

    # print output CSV file with suggested calibration parameters
    f = open(outputCsvFile, 'w')
    f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
    for ix, row in d.iterrows():
        # first get current file information
        participant = str(row['file-name'])
        ofX, ofY, ofZ, slpX, slpY, slpZ, tmpX, tmpY, tmpZ, calTempAvg = row[calCols]
        # now construct output
        out = participant + ','
        out += str(ofX) + ' ' + str(ofY) + ' ' + str(ofZ) + ','
        out += str(slpX) + ' ' + str(slpY) + ' ' + str(slpZ) + ','
        out += str(tmpX) + ' ' + str(tmpY) + ' ' + str(tmpZ) + ','
        out += str(calTempAvg)
        f.write(out + '\n')
    f.close()

    print('Files with calibration coefficients for ', str(len(d)),
          'participants written to:', outputCsvFile)


def date_parser(t):
    """
    Parse date a date string of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    """
    tz = re.search(r'(?<=\[).+?(?=\])', t)
    if tz is not None:
        tz = tz.group()
    t = re.sub(r'\[(.*?)\]', '', t)
    return pd.to_datetime(t, utc=True).tz_convert(tz)


def date_strftime(t):
    """
    Convert to time format of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    """
    tz = t.tz
    return t.strftime(f'%Y-%m-%d %H:%M:%S.%f%z [{tz}]')


def writeTimeSeries(e, labels, tsFile):
    """
    Write activity timeseries file

    :param pandas.DataFrame e: Pandas dataframe of epoch data. Must contain
        activity classification columns with missing rows imputed.
    :param list(str) labels: Activity state labels
    :param dict tsFile: output CSV filename

    """

    cols = ['acc'] + labels
    if 'MET' in e.columns:
        cols.append('MET')
    if 'imputed' in e.columns:
        cols.append('imputed')

    e = e[cols]

    # make output time format contain timezone
    # e.g. 2020-06-14 19:01:15.123000+0100 [Europe/London]
    e.index = e.index.to_series().apply(date_strftime)

    e.to_csv(tsFile, compression='gzip')
