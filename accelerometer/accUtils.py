"""Module to provide generic utilities for other accelerometer modules."""

from collections import OrderedDict
import datetime
import glob
import json
import math
import numpy as np
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



def writeStudyAccProcessCmds(studyDir, cmdsFile, runName="default",
        accExt="cwa", cmdOptions=None, filesID="files.csv"):
    """Read files to process and write out list of processing commands

    This method assumes that a study directory structure has been created by the
    createStudyDir.sh script where there is a folder structure of
    <studyName>/
        files.csv #listing all files in rawData directory
        rawData/ #all raw .cwa .bin .gt3x files
        summary/ #to store outputSummary.json
        epoch/ #to store feature output for 30sec windows
        timeSeries/ #simple csv time series output (VMag, activity binary predictions)
        nonWear/ #bouts of nonwear episodes
        stationary/ #temp store for features of stationary data for calibration
        clusterLogs/ #to store terminal output for each processed file

    If files.csv exists, process files listed here. If not, all files in
    rawData/ are read and listed in files.csv

    Then an acc processing command is written for each file and written to cmdsFile

    :param str studyDir: Root directory of study
    :param str cmdsFile: Output .txt file listing acc processing commands
    :param str runName: Name to assign to this processing run. Supports processing
        dataset in multiple different ways.
    :param str accExt: Acc file type e.g. cwa, CWA, bin, BIN, gt3x...
    :param str cmdOptions: String of processing options e.g. "--epochPeriod 10"
        Type 'python3 accProccess.py -h' for full list of options
    :param str files: Name of .csv file listing acc files to process

    :return: New file written to <cmdsFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.writeStudyAccProcessingCmds("/myStudy/", "myStudy-cmds.txt")
    <cmd options written to "myStudy-cmds.txt">
    """

    # firstly check if files.csv list exists (if not, create it)
    filesCSV = studyDir + filesID
    if not os.path.exists(filesCSV):
        csvWriter = open(filesCSV, 'w')
        csvWriter.write('fileName,\n')
        for inputFile in glob.glob(studyDir+"rawData/*." + accExt):
            csvWriter.write(inputFile + ',\n')
        csvWriter.close()

    # then create runName output directories
    summaryDir = studyDir + 'summary/' + runName + '/'
    epochDir = studyDir + 'epoch/' + runName + '/'
    timeSeriesDir = studyDir + 'timeSeries/' + runName + '/'
    nonWearDir = studyDir + 'nonWear/' + runName + '/'
    stationaryDir = studyDir + 'stationary/' + runName + '/'
    logsDir = studyDir + 'clusterLogs/' + runName + '/'
    createDirIfNotExists(summaryDir)
    createDirIfNotExists(epochDir)
    createDirIfNotExists(timeSeriesDir)
    createDirIfNotExists(nonWearDir)
    createDirIfNotExists(stationaryDir)
    createDirIfNotExists(logsDir)

    # next read files.csv
    fileList = pd.read_csv(filesCSV)
    # remove unnamed columns
    fileList.drop(fileList.columns[fileList.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    # and write commands text
    with open(cmdsFile, 'w') as txtWriter:
        for ix, row in fileList.iterrows():

            cmd = [
                'python3 accProcess.py "{:s}"'.format(str(row['fileName'])),
                '--summaryFolder "{:s}"'.format(summaryDir),
                '--epochFolder "{:s}"'.format(epochDir),
                '--timeSeriesFolder "{:s}"'.format(timeSeriesDir),
                '--nonWearFolder "{:s}"'.format(nonWearDir),
                '--stationaryFolder "{:s}"'.format(stationaryDir)
            ]

            # grab additional arguments provided in filesCSV; cmdOptionsCSV is '' if nothing found
            cmdOptionsCSV = ' '.join(['--{} {}'.format(col, row[col]) for col in fileList.columns[1:]])

            if cmdOptions:
                cmd.append(cmdOptions)
            if cmdOptionsCSV:
                cmd.append(cmdOptionsCSV)

            cmd = ' '.join(cmd)
            txtWriter.write(cmd)
            txtWriter.write('\n')

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

    ### First combine into <tmpJsonFile> the processed outputs from <inputJsonDir>
    tmpJsonFile = outputCsvFile.replace('.csv','-tmp.json')
    count = 0
    with open(tmpJsonFile,'w') as fSummary:
        for fStr in glob.glob(inputJsonDir + "*.json"):
            if fStr == tmpJsonFile: continue
            with open(fStr) as f:
                if count == 0:
                    fSummary.write('[')
                else:
                    fSummary.write(',')
                fSummary.write(f.read().rstrip())
                count += 1
        fSummary.write(']')

    ### Convert temporary json file into csv file
    dict = json.load(open(tmpJsonFile,"r"), object_pairs_hook=OrderedDict) #read json
    df = pd.DataFrame.from_dict(dict) #create pandas object from json dict
    refColumnItem = next((item for item in dict if item['quality-goodWearTime'] == 1), None)
    dAcc = df[list(refColumnItem.keys())] #maintain intended column ordering
    # infer participant ID
    dAcc['eid'] = dAcc['file-name'].str.split('/').str[-1].str.replace('.CWA','.cwa').str.replace('.cwa','')
    dAcc.to_csv(outputCsvFile, index=False)
    #remove tmpJsonFile
    os.remove(tmpJsonFile)
    print('Summary of', str(len(dAcc)), 'participants written to:', outputCsvFile)



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
    #select participants with good spread of stationary values for calibration
    goodCal = d.loc[(d['quality-calibratedOnOwnData']==1) & (d['quality-goodCalibration']==1)]
    #now only select participants whose data was NOT calibrated on a good spread of stationary values
    badCal = d.loc[(d['quality-calibratedOnOwnData']==1) & (d['quality-goodCalibration']==0)]

    #sort files by start time, which makes selection of most recent value easier
    goodCal = goodCal.sort_values(['file-startTime'])
    badCal = badCal.sort_values(['file-startTime'])

    calCols = ['calibration-xOffset(g)','calibration-yOffset(g)','calibration-zOffset(g)',
               'calibration-xSlope(g)','calibration-ySlope(g)','calibration-zSlope(g)',
               'calibration-xTemp(C)','calibration-yTemp(C)','calibration-zTemp(C)',
               'calibration-meanDeviceTemp(C)']

    #print output CSV file with suggested calibration parameters
    noOtherUses = 0
    nextUses = 0
    previousUses = 0
    f = open(outputCsvFile,'w')
    f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
    for ix, row in badCal.iterrows():
        #first get current 'bad' file
        participant, device, startTime = row[['file-name','file-deviceID','file-startTime']]
        device = int(device)
        #get calibration values from most recent previous use of this device
        # (when it had a 'good' calibration)
        prevUse = goodCal[calCols][(goodCal['file-deviceID']==device) & (goodCal['file-startTime']<startTime)].tail(1)
        try:
            ofX, ofY, ofZ, slpX, slpY, slpZ, tmpX, tmpY, tmpZ, calTempAvg = prevUse.iloc[0]
            previousUses += 1
        except:
            nextUse = goodCal[calCols][(goodCal['file-deviceID']==device) & (goodCal['file-startTime']>startTime)].head(1)
            if len(nextUse)<1:
                print('no other uses for this device at all: ', str(device),
                    str(participant))
                noOtherUses += 1
                continue
            nextUses += 1
            ofX, ofY, ofZ, slpX, slpY, slpZ, tmpX, tmpY, tmpZ, calTempAvg = nextUse.iloc[0]

        #now construct output
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

    calCols = ['calibration-xOffset(g)','calibration-yOffset(g)','calibration-zOffset(g)',
               'calibration-xSlope(g)','calibration-ySlope(g)','calibration-zSlope(g)',
               'calibration-xTemp(C)','calibration-yTemp(C)','calibration-zTemp(C)',
               'calibration-meanDeviceTemp(C)']

    #print output CSV file with suggested calibration parameters
    f = open(outputCsvFile,'w')
    f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
    for ix, row in d.iterrows():
        #first get current file information
        participant = str(row['file-name'])
        ofX, ofY, ofZ, slpX, slpY, slpZ, tmpX, tmpY, tmpZ, calTempAvg = row[calCols]
        #now construct output
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
    cols = ['accImputed']
    cols_new = ['acc']

    labelsImputed = [l + 'Imputed' for l in labels]
    cols.extend(labelsImputed)
    cols_new.extend(labels)

    if 'MET' in e.columns:
        cols.append('METImputed')
        cols_new.append('MET')

    e_new = pd.DataFrame(index=e.index)
    e_new['imputed'] = e.isna().any(1).astype('int')
    e_new[cols_new] = e[cols]

    # make output time format contain timezone
    # e.g. 2020-06-14 19:01:15.123000+0100 [Europe/London]
    e_new.index = e_new.index.to_series().apply(date_strftime)

    e_new.to_csv(tsFile, compression='gzip')
