"""Module to process raw accelerometer files into epoch data."""

from accelerometer import accUtils
import gzip
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
import struct
from subprocess import call
import sys
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent


def processInputFileToEpoch(  # noqa: C901
    inputFile, timeZone, timeShift,
    epochFile, stationaryFile, summary,
    skipCalibration=False, stationaryStd=13, xyzIntercept=[0.0, 0.0, 0.0],
    xyzSlope=[1.0, 1.0, 1.0], xyzTemp=[0.0, 0.0, 0.0], meanTemp=20.0,
    rawDataParser="AccelerometerParser", javaHeapSpace=None,
    useFilter=True, sampleRate=100, resampleMethod="linear", epochPeriod=30,
    activityClassification=True,
    rawOutput=False, rawFile=None, npyOutput=False, npyFile=None,
    startTime=None, endTime=None,
    verbose=False,
    csvStartTime=None, csvSampleRate=None,
    csvTimeFormat="yyyy-MM-dd HH:mm:ss.SSSxxxx '['VV']'",
    csvStartRow=1, csvTimeXYZTempColsIndex=None
):
    """Process raw accelerometer file, writing summary epoch stats to file

    This is usually achieved by
        1) identify 10sec stationary epochs
        2) record calibrated axes scale/offset/temp vals + static point stats
        3) use calibration coefficients and then write filtered avgVm epochs
        to <epochFile> from <inputFile>

    :param str inputFile: Input <cwa/cwa.gz/bin/gt3x> raw accelerometer file
    :param str epochFile: Output csv.gz file of processed epoch data
    :param str stationaryFile: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics
    :param bool skipCalibration: Perform software calibration (process data twice)
    :param int stationaryStd: Gravity threshold (in mg units) for stationary vs not
    :param list(float) xyzIntercept: Calbiration offset [x, y, z]
    :param list(float) xyzSlope: Calbiration slope [x, y, z]
    :param list(float) xyzTemp: Calbiration temperature coefficient [x, y, z]
    :param float meanTemp: Calibration mean temperature in file
    :param str rawDataParser: External helper process to read raw acc file. If a
        java class, it must omit .class ending.
    :param str javaHeapSpace: Amount of heap space allocated to java subprocesses.
        Useful for limiting RAM usage.
    :param bool useFilter: Filter ENMOtrunc signal
    :param int sampleRate: Resample data to n Hz
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param bool activityClassification: Extract features for machine learning
    :param bool rawOutput: Output calibrated and resampled raw data to a .csv.gz
        file? requires ~50MB/day.
    :param str rawFile: Output raw data ".csv.gz" filename
    :param bool npyOutput: Output calibrated and resampled raw data to a .npy
        file? requires ~60MB/day.
    :param str npyFile: Output raw data ".npy" filename
    :param datetime startTime: Remove data before this time in analysis
    :param datetime endTime: Remove data after this time in analysis
    :param bool verbose: Print verbose output
    :param datetime csvStartTime: start time for csv file when time column is not available
    :param float csvSampleRate: sample rate for csv file when time column is not available
    :param str csvTimeFormat: time format for csv file when time column is available
    :param int csvStartRow: start row for accelerometer data in csv file
    :param str csvTimeXYZTempColsIndex: index of column positions for XYZT columns, e.g. "1,2,3,0"

    :return: Raw processing summary values written to dict <summary>
    :rtype: void

    :Example:
    >>> import device
    >>> summary = {}
    >>> device.processInputFileToEpoch('inputFile.cwa', 'epochFile.csv.gz',
            'stationary.csv.gz', summary)
    <epoch file written to "epochFile.csv.gz", and calibration points to
        'stationary.csv.gz'>
    """
    summary['file-size'] = os.path.getsize(inputFile)
    summary['file-deviceID'] = getDeviceId(inputFile)
    useJava = True
    javaClassPath = f"{ROOT_DIR}/java/:{ROOT_DIR}/java/JTransforms-3.1-with-dependencies.jar"
    staticStdG = stationaryStd / 1000.0  # java expects units of G (not mg)

    if xyzIntercept != [0, 0, 0] or xyzSlope != [1, 1, 1] or xyzTemp != [0, 0, 0]:
        skipCalibration = True
        print('\nSkipping calibration as input parameter supplied')

    if 'omconvert' in rawDataParser:
        useJava = False

    if useJava:
        if not skipCalibration:
            # identify 10sec stationary epochs
            accUtils.toScreen("=== Calibrating ===")
            commandArgs = ["java", "-classpath", javaClassPath,
                           "-XX:ParallelGCThreads=1", rawDataParser, inputFile,
                           "timeZone:" + timeZone,
                           "timeShift:" + str(timeShift),
                           "outputFile:" + stationaryFile,
                           "verbose:" + str(verbose),
                           "filter:" + str(useFilter),
                           "getStationaryBouts:true", "epochPeriod:10",
                           "stationaryStd:" + str(staticStdG),
                           "sampleRate:" + str(sampleRate)]
            if javaHeapSpace:
                commandArgs.insert(1, javaHeapSpace)
            if csvStartTime:
                commandArgs.append("csvStartTime:" + csvStartTime.strftime("%Y-%m-%dT%H:%M"))
            if csvSampleRate:
                commandArgs.append("csvSampleRate:" + str(csvSampleRate))
            if csvTimeFormat:
                commandArgs.append("csvTimeFormat:" + str(csvTimeFormat))
            if csvStartRow is not None:
                commandArgs.append("csvStartRow:" + str(csvStartRow))
            if csvTimeXYZTempColsIndex:
                javaStrCsvTXYZ = ','.join([str(i) for i in csvTimeXYZTempColsIndex])
                commandArgs.append("csvTimeXYZTempColsIndex:" + javaStrCsvTXYZ)
            # call process to identify stationary epochs
            exitCode = call(commandArgs)
            if exitCode != 0:
                print(commandArgs)
                print("Error: java calibration failed, exit ", exitCode)
                sys.exit(-6)
            # record calibrated axes scale/offset/temp vals + static point stats
            getCalibrationCoefs(stationaryFile, summary)
            xyzIntercept = [summary['calibration-xOffset(g)'],
                            summary['calibration-yOffset(g)'],
                            summary['calibration-zOffset(g)']]
            xyzSlope = [summary['calibration-xSlope(g)'],
                        summary['calibration-ySlope(g)'],
                        summary['calibration-zSlope(g)']]
            xyzTemp = [summary['calibration-xTemp(C)'],
                       summary['calibration-yTemp(C)'],
                       summary['calibration-zTemp(C)']]
            meanTemp = summary['calibration-meanDeviceTemp(C)']
        else:
            storeCalibrationParams(summary, xyzIntercept, xyzSlope, xyzTemp, meanTemp)
            summary['quality-calibratedOnOwnData'] = 0
            summary['quality-goodCalibration'] = 1

        accUtils.toScreen('=== Extracting features ===')
        commandArgs = ["java", "-classpath", javaClassPath,
                       "-XX:ParallelGCThreads=1", rawDataParser, inputFile,
                       "timeZone:" + timeZone,
                       "timeShift:" + str(timeShift),
                       "outputFile:" + epochFile, "verbose:" + str(verbose),
                       "filter:" + str(useFilter),
                       "sampleRate:" + str(sampleRate),
                       "resampleMethod:" + str(resampleMethod),
                       "xIntercept:" + str(xyzIntercept[0]),
                       "yIntercept:" + str(xyzIntercept[1]),
                       "zIntercept:" + str(xyzIntercept[2]),
                       "xSlope:" + str(xyzSlope[0]),
                       "ySlope:" + str(xyzSlope[1]),
                       "zSlope:" + str(xyzSlope[2]),
                       "xTemp:" + str(xyzTemp[0]),
                       "yTemp:" + str(xyzTemp[1]),
                       "zTemp:" + str(xyzTemp[2]),
                       "meanTemp:" + str(meanTemp),
                       "epochPeriod:" + str(epochPeriod),
                       "rawOutput:" + str(rawOutput),
                       "rawFile:" + str(rawFile),
                       "npyOutput:" + str(npyOutput),
                       "npyFile:" + str(npyFile),
                       "getFeatures:" + str(activityClassification)]
        if javaHeapSpace:
            commandArgs.insert(1, javaHeapSpace)
        if startTime:
            commandArgs.append("startTime:" + startTime.strftime("%Y-%m-%dT%H:%M"))
        if endTime:
            commandArgs.append("endTime:" + endTime.strftime("%Y-%m-%dT%H:%M"))
        if csvStartTime:
            commandArgs.append("csvStartTime:" + csvStartTime.strftime("%Y-%m-%dT%H:%M"))
        if csvSampleRate:
            commandArgs.append("csvSampleRate:" + str(csvSampleRate))
        if csvTimeFormat:
            commandArgs.append("csvTimeFormat:" + str(csvTimeFormat))
        if csvStartRow:
            commandArgs.append("csvStartRow:" + str(csvStartRow))
        if csvTimeXYZTempColsIndex:
            javaStrCsvTXYZ = ','.join([str(i) for i in csvTimeXYZTempColsIndex])
            commandArgs.append("csvTimeXYZTempColsIndex:" + javaStrCsvTXYZ)
        exitCode = call(commandArgs)
        if exitCode != 0:
            print(commandArgs)
            print("Error: Java epoch generation failed, exit ", exitCode)
            sys.exit(-7)

    else:
        if not skipCalibration:
            commandArgs = [rawDataParser, inputFile, timeZone, timeShift,
                           "-svm-file", epochFile, "-info", stationaryFile,
                           "-svm-extended", "3", "-calibrate", "1",
                           "-interpolate-mode", "2",
                           "-svm-mode", "1", "-svm-epoch", str(epochPeriod),
                           "-svm-filter", "2"]
        else:
            calArgs = str(xyzSlope[0]) + ','
            calArgs += str(xyzSlope[1]) + ','
            calArgs += str(xyzSlope[2]) + ','
            calArgs += str(xyzIntercept[0]) + ','
            calArgs += str(xyzIntercept[1]) + ','
            calArgs += str(xyzIntercept[2]) + ','
            calArgs += str(xyzTemp[0]) + ','
            calArgs += str(xyzTemp[1]) + ','
            calArgs += str(xyzTemp[2]) + ','
            calArgs += str(meanTemp)
            commandArgs = [rawDataParser, inputFile, timeZone, timeShift,
                           "-svm-file", epochFile, "-info", stationaryFile,
                           "-svm-extended", "3", "-calibrate", "0",
                           "-calibration", calArgs, "-interpolate-mode", "2",
                           "-svm-mode", "1", "-svm-epoch", str(epochPeriod),
                           "-svm-filter", "2"]
        call(commandArgs)
        getOmconvertInfo(stationaryFile, summary)


def getCalibrationCoefs(staticBoutsFile, summary):
    """Identify calibration coefficients from java processed file

    Get axes offset/gain/temp calibration coefficients through linear regression
    of stationary episodes

    :param str stationaryFile: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    # learning/research parameters
    maxIter = 1000
    minIterImprovement = 0.0001  # 0.1mg
    # use python NUMPY framework to store stationary episodes from epoch file
    if isinstance(staticBoutsFile, pd.DataFrame):

        axesVals = staticBoutsFile[['xMean', 'yMean', 'zMean']].values
        tempVals = staticBoutsFile[['temperature']].values
    else:
        cols = ['xMean', 'yMean', 'zMean', 'temp', 'dataErrors']
        d = pd.read_csv(staticBoutsFile, usecols=cols, compression='gzip')
        d = d.to_numpy()
        if len(d) <= 5:
            storeCalibrationInformation(summary, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                        [0.0, 0.0, 0.0], 20, np.nan, np.nan,
                                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(d))
            return
        stationaryPoints = d[d[:, 4] == 0]  # don't consider episodes with data errors
        axesVals = stationaryPoints[:, [0, 1, 2]]
        tempVals = stationaryPoints[:, [3]]
    meanTemp = np.mean(tempVals)
    tempVals = np.copy(tempVals - meanTemp)
    # store information on spread of stationary points
    xMin, yMin, zMin = np.amin(axesVals, axis=0)
    xMax, yMax, zMax = np.amax(axesVals, axis=0)
    # initialise intercept/slope variables to assume no error initially present
    intercept = np.array([0.0, 0.0, 0.0])
    slope = np.array([1.0, 1.0, 1.0])
    tempCoef = np.array([0.0, 0.0, 0.0])
    # variables to support model fitting
    bestError = 1e16
    bestIntercept = np.copy(intercept)
    bestSlope = np.copy(slope)
    bestTemp = np.copy(tempCoef)
    # record initial uncalibrated error
    curr = intercept + (np.copy(axesVals) * slope) + (np.copy(tempVals) * tempCoef)
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)
    errors = np.linalg.norm(curr - target, axis=1)
    initError = np.sqrt(np.mean(np.square(errors)))  # root mean square error
    # iterate through linear model fitting
    try:
        for i in range(1, maxIter):
            # iterate through each axis, refitting its intercept/slope vals
            for a in range(0, 3):
                x = np.concatenate([curr[:, [a]], tempVals], axis=1)
                x = sm.add_constant(x, prepend=True)  # add bias/intercept term
                y = target[:, a]
                newI, newS, newT = sm.OLS(y, x).fit().params
                # update values as part of iterative closest point fitting process
                # refer to wiki as there is quite a bit of math behind next 3 lines
                intercept[a] = newI + (intercept[a] * newS)
                slope[a] = newS * slope[a]
                tempCoef[a] = newT + (tempCoef[a] * newS)
            # update vals (and targed) based on new intercept/slope/temp coeffs
            curr = intercept + (np.copy(axesVals) * slope) + (np.copy(tempVals) * tempCoef)
            target = curr / np.linalg.norm(curr, axis=1, keepdims=True)
            errors = np.linalg.norm(curr - target, axis=1)
            rms = np.sqrt(np.mean(np.square(errors)))  # root mean square error
            # assess iterative error convergence
            improvement = (bestError - rms) / bestError
            if rms < bestError:
                bestIntercept = np.copy(intercept)
                bestSlope = np.copy(slope)
                bestTemp = np.copy(tempCoef)
                bestError = rms
            if improvement < minIterImprovement:
                break  # break if not largely converged
    except Exception as exceptStr:
        # highlight problem with regression, and exit
        xMin, yMin, zMin = float('nan'), float('nan'), float('nan')
        xMax, yMax, zMax = float('nan'), float('nan'), float('nan')
        sys.stderr.write('WARNING: Calibration error\n ' + str(exceptStr))
    # store output to summary dictionary
    storeCalibrationInformation(summary, bestIntercept, bestSlope,
                                bestTemp, meanTemp, initError, bestError, xMin, xMax, yMin, yMax, zMin,
                                zMax, len(axesVals))


def getOmconvertInfo(omconvertInfoFile, summary):
    """Identify calibration coefficients for omconvert processed file

    Get axes offset/gain/temp calibration coeffs from omconvert info file

    :param str omconvertInfoFile: Output information file from omconvert
    :param dict summary: Output dictionary containing all summary metrics

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    file = open(omconvertInfoFile, 'r')
    for line in file:
        elements = line.split(':')
        name, value = elements[0], elements[1]
        if name == 'Calibration':
            vals = value.split(',')
            bestIntercept = float(vals[3]), float(vals[4]), float(vals[5])
            bestSlope = float(vals[0]), float(vals[1]), float(vals[2])
            bestTemp = float(vals[6]), float(vals[7]), float(vals[8])
            meanTemp = float(vals[-1])
        elif name == 'Calibration-Stationary-Error-Pre':
            initError = float(value)
        elif name == 'Calibration-Stationary-Error-Post':
            bestError = float(value)
        elif name == 'Calibration-Stationary-Min':
            vals = value.split(',')
            xMin, yMin, zMin = float(vals[0]), float(vals[1]), float(vals[2])
        elif name == 'Calibration-Stationary-Max':
            vals = value.split(',')
            xMax, yMax, zMax = float(vals[0]), float(vals[1]), float(vals[2])
        elif name == 'Calibration-Stationary-Count':
            nStatic = int(value)
    file.close()
    # store output to summary dictionary
    storeCalibrationInformation(summary, bestIntercept, bestSlope,
                                bestTemp, meanTemp, initError, bestError, xMin, xMax, yMin, yMax, zMin,
                                zMax, nStatic)


def storeCalibrationInformation(summary, bestIntercept, bestSlope,
                                bestTemp, meanTemp, initError, bestError, xMin, xMax, yMin, yMax, zMin,
                                zMax, nStatic, calibrationSphereCriteria=0.3):
    """Store calibration information to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) bestIntercept: Best x/y/z intercept values
    :param list(float) bestSlope: Best x/y/z slope values
    :param list(float) bestTemperature: Best x/y/z temperature values
    :param float meanTemp: Calibration mean temperature in file
    :param float initError: Root mean square error (in mg) before calibration
    :param float initError: Root mean square error (in mg) after calibration
    :param float xMin: xMin information on spread of stationary points
    :param float xMax: xMax information on spread of stationary points
    :param float yMin: yMin information on spread of stationary points
    :param float yMax: yMax information on spread of stationary points
    :param float zMin: zMin information on spread of stationary points
    :param float zMax: zMax information on spread of stationary points
    :param int nStatic: number of stationary points used for calibration
    :param float calibrationSphereCriteria: Threshold to check how well file was
        calibrated

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    # store output to summary dictionary
    summary['calibration-errsBefore(mg)'] = accUtils.formatNum(initError * 1000, 2)
    summary['calibration-errsAfter(mg)'] = accUtils.formatNum(bestError * 1000, 2)
    storeCalibrationParams(summary, bestIntercept, bestSlope, bestTemp, meanTemp)
    summary['calibration-numStaticPoints'] = nStatic
    summary['calibration-staticXmin(g)'] = accUtils.formatNum(xMin, 2)
    summary['calibration-staticXmax(g)'] = accUtils.formatNum(xMax, 2)
    summary['calibration-staticYmin(g)'] = accUtils.formatNum(yMin, 2)
    summary['calibration-staticYmax(g)'] = accUtils.formatNum(yMax, 2)
    summary['calibration-staticZmin(g)'] = accUtils.formatNum(zMin, 2)
    summary['calibration-staticZmax(g)'] = accUtils.formatNum(zMax, 2)
    # check how well calibrated file was
    summary['quality-calibratedOnOwnData'] = 1
    summary['quality-goodCalibration'] = 1
    s = calibrationSphereCriteria
    try:
        if xMin > -s or xMax < s or yMin > -s or yMax < s or zMin > -s or zMax < s or \
                np.isnan(xMin) or np.isnan(yMin) or np.isnan(zMin):
            summary['quality-goodCalibration'] = 0
    except UnboundLocalError:
        summary['quality-goodCalibration'] = 0


def storeCalibrationParams(summary, xyzOff, xyzSlope, xyzTemp, meanTemp):
    """Store calibration parameters to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) xyzOff: intercept [x, y, z]
    :param list(float) xyzSlope: slope [x, y, z]
    :param list(float) xyzTemp: temperature [x, y, z]
    :param float meanTemp: Calibration mean temperature in file

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    # store output to summary dictionary
    summary['calibration-xOffset(g)'] = accUtils.formatNum(xyzOff[0], 4)
    summary['calibration-yOffset(g)'] = accUtils.formatNum(xyzOff[1], 4)
    summary['calibration-zOffset(g)'] = accUtils.formatNum(xyzOff[2], 4)
    summary['calibration-xSlope(g)'] = accUtils.formatNum(xyzSlope[0], 4)
    summary['calibration-ySlope(g)'] = accUtils.formatNum(xyzSlope[1], 4)
    summary['calibration-zSlope(g)'] = accUtils.formatNum(xyzSlope[2], 4)
    summary['calibration-xTemp(C)'] = accUtils.formatNum(xyzTemp[0], 4)
    summary['calibration-yTemp(C)'] = accUtils.formatNum(xyzTemp[1], 4)
    summary['calibration-zTemp(C)'] = accUtils.formatNum(xyzTemp[2], 4)
    summary['calibration-meanDeviceTemp(C)'] = accUtils.formatNum(meanTemp, 2)


def getDeviceId(inputFile):
    """Get serial number of device

    First decides which DeviceId parsing method to use for <inputFile>.

    :param str inputFile: Input raw accelerometer file

    :return: Device ID
    :rtype: int
    """

    if inputFile.lower().endswith('.bin'):
        return getGeneaDeviceId(inputFile)
    elif inputFile.lower().endswith('.cwa') or inputFile.lower().endswith('.cwa.gz'):
        return getAxivityDeviceId(inputFile)
    elif inputFile.lower().endswith('.gt3x'):
        return getGT3XDeviceId(inputFile)
    elif inputFile.lower().endswith('.csv') or inputFile.lower().endswith('.csv.gz'):
        return "unknown (.csv)"
    else:
        print("ERROR: Cannot get deviceId for file: " + inputFile)


def getAxivityDeviceId(cwaFile):
    """Get serial number of Axivity device

    Parses the unique serial code from the header of an Axivity accelerometer file

    :param str cwaFile: Input raw .cwa accelerometer file

    :return: Device ID
    :rtype: int
    """
    if cwaFile.lower().endswith('.cwa'):
        f = open(cwaFile, 'rb')
    elif cwaFile.lower().endswith('.cwa.gz'):
        f = gzip.open(cwaFile, 'rb')
    header = f.read(2)
    if header == b'MD':
        blockSize = struct.unpack('H', f.read(2))[0]
        performClear = struct.unpack('B', f.read(1))[0]
        deviceId = struct.unpack('H', f.read(2))[0]
    else:
        print("ERROR: in getDeviceId(\"" + cwaFile + "\")")
        print("""A deviceId value could not be found in input file header,
         this usually occurs when the file is not an Axivity .cwa accelerometer
         file. Exiting...""")
        sys.exit(-8)
    f.close()
    return deviceId


def getGeneaDeviceId(binFile):
    """Get serial number of GENEActiv device

    Parses the unique serial code from the header of a GENEActiv accelerometer file

    :param str binFile: Input raw .bin accelerometer file

    :return: Device ID
    :rtype: int
    """

    f = open(binFile, 'r')  # 'Universal' newline mode
    next(f)  # Device Identity
    deviceId = next(f).split(':')[1].rstrip()  # Device Unique Serial Code:011710
    f.close()
    return deviceId


def getGT3XDeviceId(gt3xFile):
    """Get serial number of Actigraph device

    Parses the unique serial code from the header of a GT3X accelerometer file

    :param str gt3xFile: Input raw .gt3x accelerometer file

    :return: Device ID
    :rtype: int
    """

    import zipfile
    if zipfile.is_zipfile(gt3xFile):
        with zipfile.ZipFile(gt3xFile, 'r') as z:
            contents = z.infolist()
            print("\n".join(map(lambda x: str(x.filename).rjust(20, " ") + ", "
                                + str(x.file_size), contents)))

            if 'info.txt' in map(lambda x: x.filename, contents):
                print('info.txt found..')
                info_file = z.open('info.txt', 'r')
                # print info_file.read()
                for line in info_file:
                    if line.startswith(b"Serial Number:"):
                        newline = line.decode("utf-8")
                        newline = newline.split("Serial Number: ")[1]
                        print("Serial Number: " + newline)
                        return newline
            else:
                print("Could not find info.txt file")

    print("ERROR: in getDeviceId(\"" + gt3xFile + "\")")
    print("""A deviceId value could not be found in input file header,
     this usually occurs when the file is not an Actigraph .gt3x accelerometer
     file. Exiting...""")
    sys.exit(-8)
