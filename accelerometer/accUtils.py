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
    print(datetime.datetime.now().strftime(timeFormat), '\t', msg)



def generateTimeSeries(epochPD, tsFile, timeSeriesDateColumn = False, 
    epochPeriod = 30, activityClassification = None, labels = []):
    """Write .csv.gz time series file of <e> activity epoch data to <tsFile>

    These time-series files describe a participant's accelerometer measured 
    physical activity intensity every <epochPeriod> seconds. Each measurement 
    represents the average vector magnitude across all samples recorded in each 
    <epochPeriod>. This file may be useful for those interested in subsequently
    analysing accelerometer measured physical activity by time.

    The first column contains the acceleration value where the units of 
    measurement are milli-gravity. If <activityClassification> is requested, the 
    next column indicates the most likely activity state for any given
    <epochPeriod>. The final column indicates whether this is an actual 
    measurement (value = <blank>) or imputed (value = 1). 

    Each files is typically 760KB in the raw state and compresses to around 170KB.

    :param pandas.DataFrame epochPD: Pandas dataframe of epoch data
    :param str tsFile: Output filename for .csv.gz file
    :param bool timeSeriesDateColumn: Write time associated with each reading
    :param datetime startTime: Remove data before this time in analysis
    :param datetime endTime: Remove data after this time in analysis 
    :param int epochPeriod: size of epoch time window (in seconds)
    :param bool activityClassification: Write out most likely activity state
    :param list(str) labels: Activity state labels
    
    :return: Write .csv.gz time series file of activity epoch data to <tsFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> import pandas as pd
    >>> e = pd.read_csv("epochFile.csv.gz")
    >>> accUtils.generateTimeSeries(e, "timeSeriesFile.csv.gz")
    <file written to timeSeriesFile.csv.gz>
    """
    
    startTime = pd.to_datetime(epochPD.index.values[0])
    endTime = pd.to_datetime(epochPD.index.values[-1])
    # resample index so we can add imputed data
    e = epochPD.reindex(pd.date_range(start=startTime, end=endTime,
        freq=str(epochPeriod)+'s'))
    e.index.name = TIME_SERIES_COL

    # prepare time series header
    tsHead = 'acceleration (mg) - '
    tsHead += e.index.min().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += e.index.max().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += 'sampleRate = ' + str(epochPeriod) + ' seconds'
    
    # write time series file
    if len(e['accImputed']) > 0:
        # fill with vm, imputed data used where needed, convert to mg units
        e['vmFinal'] = e['accImputed'] * 1000
        # highlight any imputed values
        e['imputed'] = np.isnan(e['acc']).astype(int)
        # add activity prediction labels
        if activityClassification:
            if 'sleepDetected' in e.columns:
                labels.append('sleepDetected')
            if 'MET' in e.columns:
                labels.append('MET')
            e[['vmFinal','imputed'] + labels].to_csv(tsFile,
                float_format='%.1f', index=timeSeriesDateColumn, 
                header=[tsHead,'imputed']+labels, compression='gzip')
        else:
            e[['vmFinal','imputed']].to_csv(tsFile, float_format='%.1f', 
                index=timeSeriesDateColumn, header=[tsHead,'imputed'],
                compression='gzip')
        
    else:
        f = open(tsFile,'w')
        f.write(tsHead + '\n')
        f.write('no wearTime data,1')
        f.close()



def loadTimeSeriesCSV(tsFile):
    """Load time series csv.gz file and append date/time column to it

    The time associated with each reading can be inferred from the very first 
    row, which describes the sample rate, start and end times of the data.

    For example header
    "acceleration (mg) - 2014-05-07 13:29:50 - 2014-05-13 09:50:25 - sampleRate = 5 seconds, imputed" 
    indicates that the first measurement time is at 2014-05-07 13:29:50, the second 
    at 2014-05-07 13:29:55, the third at 2014-05-07 13:30:00 ... and the last at 
    2014-05-13 09:50:25.

    :param str tsFile: Output filename for .csv.gz file
    
    :return: Pandas dataframe of epoch data
    :rtype: pandas.DataFrame

    :Example:
    >>> import accUtils
    >>> import pandas as pd
    >>> df = accUtils.loadTimeSeriesCSV("sample-timeSeries.csv.gz")
    <returns pd.DataFrame>
    """
    # get header
    header = pd.read_csv(tsFile, nrows=1, header=0, compression='gzip')
    headerInfo = header.columns[0]
    if header.columns[0] == TIME_SERIES_COL:
        headerInfo = header.columns[1]
        header.columns = [TIME_SERIES_COL, 'acc'] + header.columns[2:].tolist()
    else:
        header.columns = ['acc'] + header.columns[1:].tolist()
        # read start time, endtime, and sample rate
        startDate = headerInfo.split(" - ")[1]
        endDate = headerInfo.split(" - ")[2]
        sampleRate = headerInfo.split("sampleRate = ")[1].split(" ")[0]

    # read data
    tsData = pd.read_csv(tsFile, skiprows=1, header=None, names=header.columns, 
        compression='gzip')
    if header.columns[0] != TIME_SERIES_COL:
        tsData.index = pd.date_range(start=startDate, end=endDate, 
            freq=str(sampleRate) + 's')
    return tsData



def writeStudyAccProcessCmds(studyDir, cmdsFile, accExt="cwa", cmdOptions=""):
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
    :param str accExt: Acc file type e.g. cwa, CWA, bin, BIN, gt3x...
    :param str cmdOptions: String of processing options e.g. "--epochPeriod 10"
        Type 'python3 accProccess.py -h' for full list of options

    :return: New file written to <outputCsvFile>
    :rtype: void

    :Example:
    >>> import accUtils
    >>> accUtils.writeStudyAccProcessingCmds("/myStudy/", "myStudy-cmds.txt")
    <cmd options written to "myStudy-cmds.txt">
    """
    
    # firstly check if files.csv list exists (if not, create it)
    filesCSV = studyDir + "files.csv"
    if not os.path.exists(filesCSV):
        csvWriter = open(filesCSV, 'w')
        csvWriter.write('fileName,\n')
        for rawFile in glob.glob(studyDir+"rawData/*." + accExt):
            csvWriter.write(rawFile + ',\n')
        csvWriter.close()

    # next read files.csv
    fileList = pd.read_csv(filesCSV)

    # and write commands text
    txtWriter = open(cmdsFile, 'w')
    for ix, row in fileList.iterrows():
        txtWriter.write('python3 accProcess.py')
        txtWriter.write(' ' + row['fileName'])
        txtWriter.write(' --summaryFolder ' + studyDir + 'summary/')
        txtWriter.write(' --epochFolder ' + studyDir + 'epoch/')
        txtWriter.write(' --timeSeriesFolder ' + studyDir + 'timeSeries/')
        txtWriter.write(' --nonWearFolder ' + studyDir + 'nonWear/')
        txtWriter.write(' --stationaryFolder ' + studyDir + 'stationary/')
        if cmdOptions != "":
            txtWriter.write(' ' + cmdOptions)
    txtWriter.close()
    print('Processing list written to ', cmdsFile)



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