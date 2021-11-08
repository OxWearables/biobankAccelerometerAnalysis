"""Module to provide generic utilities for other accelerometer modules."""

from collections import OrderedDict
import datetime
import glob
import json
import math
import os
import pandas as pd
import re

DAYS = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
TIME_SERIES_COL = 'time'


def formatNum(num, decimalPlaces):
    """return str of number formatted to number of decimalPlaces

    When writing out 10,000's of files, it is useful to format the output to n
    decimal places as a space saving measure.

    :param float num: Float number to be formatted.
    :param int decimalPlaces: Number of decimal places for output format
    :return: Number formatted to number of decimalPlaces
    :rtype: str

    :Example:
    >>> import accUtils
    >>> accUtils.formatNum(2.567, 2)
    2.57
    """

    fmt = '%.' + str(decimalPlaces) + 'f'
    return float(fmt % num)


def meanSDstr(mean, std, numDecimalPlaces):
    """return str of mean and stdev numbers formatted to number of decimalPlaces

    :param float mean: Mean number to be formatted.
    :param float std: Standard deviation number to be formatted.
    :param int decimalPlaces: Number of decimal places for output format
    :return: String formatted to number of decimalPlaces
    :rtype: str

    :Example:
    >>> import accUtils
    >>> accUtils.meanSDstr(2.567, 0.089, 2)
    2.57 (0.09)
    """
    outStr = str(formatNum(mean, numDecimalPlaces))
    outStr += ' ('
    outStr += str(formatNum(std, numDecimalPlaces))
    outStr += ')'
    return outStr


def meanCIstr(mean, std, n, numDecimalPlaces):
    """return str of mean and 95% confidence interval numbers formatted

    :param float mean: Mean number to be formatted.
    :param float std: Standard deviation number to be formatted.
    :param int n: Number of observations
    :param int decimalPlaces: Number of decimal places for output format
    :return: String formatted to number of decimalPlaces
    :rtype: str

    :Example:
    >>> import accUtils
    >>> accUtils.meanSDstr(2.567, 0.089, 2)
    2.57 (0.09)
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
    """Print msg str prepended with current time

    :param str mgs: Message to be printed to screen
    :return: Print msg str prepended with current time
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.toScreen("hello")
    2018-11-28 10:53:18    hello
    """

    timeFormat = '%Y-%m-%d %H:%M:%S'
    print(f"\n{datetime.datetime.now().strftime(timeFormat)}\t{msg}")


def writeStudyAccProcessCmds(accDir, outDir, cmdsFile='processCmds.txt',
                             accExt="cwa", cmdOptions=None, filesCSV="files.csv"):
    """Read files to process and write out list of processing commands

    This creates the following output directory structure containing all
    processing results:
    <outDir>/
        summary/  #to store outputSummary.json
        epoch/  #to store feature output for 30sec windows
        timeSeries/  #simple csv time series output (VMag, activity binary predictions)
        nonWear/  #bouts of nonwear episodes
        stationary/  #temp store for features of stationary data for calibration
        clusterLogs/  #to store terminal output for each processed file

    If a filesCSV exists in accDir/, process the files listed there. If not,
    all files in accDir/ are processed

    Then an acc processing command is written for each file and written to cmdsFile

    :param str accDirs: Directory(s) with accelerometer files to process
    :param str outDir: Output directory to be created containing the processing results
    :param str cmdsFile: Output .txt file listing all processing commands
    :param str accExt: Acc file type e.g. cwa, CWA, bin, BIN, gt3x...
    :param str cmdOptions: String of processing options e.g. "--epochPeriod 10"
        Type 'python3 accProccess.py -h' for full list of options
    :param str filesCSV: Name of .csv file listing acc files to process

    :return: New file written to <cmdsFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.writeStudyAccProcessingCmds("myAccDir/", "myResults/", "myProcessCmds.txt")
    <cmd options written to "myProcessCmds.txt">
    """

    # Create output directory structure
    summaryDir = os.path.join(outDir, 'summary')
    epochDir = os.path.join(outDir, 'epoch')
    timeSeriesDir = os.path.join(outDir, 'timeSeries')
    nonWearDir = os.path.join(outDir, 'nonWear')
    stationaryDir = os.path.join(outDir, 'stationary')
    logsDir = os.path.join(outDir, 'clusterLogs')
    rawDir = os.path.join(outDir, 'raw')
    npyDir = os.path.join(outDir, 'npy')

    createDirIfNotExists(summaryDir)
    createDirIfNotExists(epochDir)
    createDirIfNotExists(timeSeriesDir)
    createDirIfNotExists(nonWearDir)
    createDirIfNotExists(stationaryDir)
    createDirIfNotExists(logsDir)
    createDirIfNotExists(rawDir)
    createDirIfNotExists(npyDir)
    createDirIfNotExists(outDir)

    # Use filesCSV if provided, else process everything in accDir (and create filesCSV)
    if filesCSV in os.listdir(accDir):
        fileList = pd.read_csv(os.path.join(accDir, filesCSV))
    else:
        fileList = pd.DataFrame(
            {'fileName': [f for f in os.listdir(accDir) if f.lower().endswith(accExt.lower())]}
        )
        fileList.to_csv(os.path.join(accDir, filesCSV), index=False)

    with open(cmdsFile, 'w') as f:
        for i, row in fileList.iterrows():

            cmd = [
                'accProcess "{:s}"'.format(os.path.join(accDir, row['fileName'])),
                '--summaryFolder "{:s}"'.format(summaryDir),
                '--epochFolder "{:s}"'.format(epochDir),
                '--timeSeriesFolder "{:s}"'.format(timeSeriesDir),
                '--nonWearFolder "{:s}"'.format(nonWearDir),
                '--stationaryFolder "{:s}"'.format(stationaryDir),
                '--rawFolder "{:s}"'.format(rawDir),
                '--npyFolder "{:s}"'.format(npyDir),
                '--outputFolder "{:s}"'.format(outDir)
            ]

            # Grab additional arguments provided in filesCSV (e.g. calibration params)
            cmdOptionsCSV = ' '.join(['--{} {}'.format(col, row[col]) for col in fileList.columns[1:]])

            if cmdOptions:
                cmd.append(cmdOptions)
            if cmdOptionsCSV:
                cmd.append(cmdOptionsCSV)

            cmd = ' '.join(cmd)
            f.write(cmd)
            f.write('\n')

    print('Processing list written to ', cmdsFile)
    print('Suggested dir for log files: ', logsDir)


def collateJSONfilesToSingleCSV(inputJsonDir, outputCsvFile):
    """read all summary *.json files and convert into one large CSV file

    Each json file represents summary data for one participant. Therefore output
    CSV file contains summary for all participants.

    :param str inputJsonDir: Directory containing JSON files
    :param str outputCsvFile: Output CSV filename

    :return: New file written to <outputCsvFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.collateJSONfilesToSingleCSV("data/", "data/summary-all-files.csv")
    <summary CSV of all participants/files written to "data/sumamry-all-files.csv">
    """

    jdicts = []
    for filename in glob.glob(os.path.join(inputJsonDir, "*.json")):
        with open(filename, 'r') as f:
            jdicts.append(json.load(f, object_pairs_hook=OrderedDict))

    df = pd.DataFrame.from_dict(jdicts)  # merge to a dataframe
    refColumnItem = next((item for item in jdicts if item['quality-goodWearTime'] == 1), None)
    df = df[list(refColumnItem.keys())]  # maintain intended column ordering
    df['eid'] = df['file-name'].str.split('/').str[-1].str.split('.').str[0]  # infer participant ID
    df = df.set_index('eid')
    df.to_csv(outputCsvFile)
    print('Summary of', str(len(df)), 'participants written to:', outputCsvFile)


def identifyUnprocessedFiles(filesCsv, summaryCsv, outputFilesCsv):
    """identify files that have not been processed

    Look through all processed accelerometer files, and find participants who do
    not have records in the summary csv file. This indicates there was a problem
    in processing their data. Therefore, output will be a new .csv file to
    support reprocessing of these files

    :param str filesCsv: CSV listing acc files in study directory
    :param str summaryCsv: Summary CSV of processed dataset
    :param str outputFilesCsv: Output csv listing files to be reprocessed

    :return: New file written to <outputCsvFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.identifyUnprocessedFiles("study/files.csv", study/summary-all-files.csv",
        "study/files-reprocess.csv")
    <Output csv listing files to be reprocessed written to "study/files-reprocess.csv">
    """

    fileList = pd.read_csv(filesCsv)
    summary = pd.read_csv(summaryCsv)

    output = fileList[~fileList['fileName'].isin(list(summary['file-name']))]
    output = output.rename(columns={'Unnamed: 1': ''})
    output.to_csv(outputFilesCsv, index=False)

    print('Reprocessing for ', len(output), 'participants written to:',
          outputFilesCsv)


def updateCalibrationCoefs(inputCsvFile, outputCsvFile):
    """read summary .csv file and update coefs for those with poor calibration

    Look through all processed accelerometer files, and find participants that
    did not have good calibration data. Then assigns the calibration coefs from
    previous good use of a given device. Output will be a new .csv file to
    support reprocessing of uncalibrated files with new pre-specified calibration coefs.

    :param str inputCsvFile: Summary CSV of processed dataset
    :param str outputCsvFile: Output CSV of files to be reprocessed with new
        calibration info

    :return: New file written to <outputCsvFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.updateCalibrationCoefs("data/summary-all-files.csv", "study/files-recalibration.csv")
    <CSV of files to be reprocessed written to "study/files-recalibration.csv">
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
    """read summary .csv file and write files.csv with calibration coefs

    Look through all processed accelerometer files, and write a new .csv file to
    support reprocessing of files with pre-specified calibration coefs.

    :param str inputCsvFile: Summary CSV of processed dataset
    :param str outputCsvFile: Output CSV of files to process with calibration info

    :return: New file written to <outputCsvFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.writeFilesWithCalibrationCoefs("data/summary-all-files.csv",
    >>>     "study/files-calibrated.csv")
    <CSV of files to be reprocessed written to "study/files-calibrated.csv">
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


def createDirIfNotExists(folder):
    """ Create directory if it doesn't currently exist

    :param str folder: Directory to be checked/created

    :return: Dir now exists (created if didn't exist before, otherwise untouched)
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.createDirIfNotExists("/myStudy/summary/dec18/")
        <folder "/myStudy/summary/dec18/" now exists>
    """

    if not os.path.exists(folder):
        os.makedirs(folder)


def date_parser(t):
    '''
    Parse date a date string of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    '''
    tz = re.search(r'(?<=\[).+?(?=\])', t)
    if tz is not None:
        tz = tz.group()
    t = re.sub(r'\[(.*?)\]', '', t)
    return pd.to_datetime(t, utc=True).tz_convert(tz)


def date_strftime(t):
    '''
    Convert to time format of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    '''
    tz = t.tz
    return t.strftime(f'%Y-%m-%d %H:%M:%S.%f%z [{tz}]')


def writeTimeSeries(e, labels, tsFile):
    """ Write activity timeseries file
    :param pandas.DataFrame e: Pandas dataframe of epoch data. Must contain
        activity classification columns with missing rows imputed.
    :param list(str) labels: Activity state labels
    :param dict tsFile: output CSV filename

    :return: None
    :rtype: void
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
