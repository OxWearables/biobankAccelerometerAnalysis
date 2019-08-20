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


def processRawFileToEpoch(rawFile, epochFile, stationaryFile, summary,
    skipCalibration=False, stationaryStd=13, xIntercept=0.0,
    yIntercept=0.0, zIntercept=0.0, xSlope=0.0, ySlope=0.0,
    zSlope=0.0, xTemp=0.0, yTemp=0.0, zTemp=0.0, meanTemp=20.0,
    rawDataParser="AxivityAx3Epochs", javaHeapSpace=None,
    skipFiltering=False, sampleRate=100, epochPeriod=30,
    useAbs=False, activityClassification=True,
    rawOutput=False, rawOutputFile=None, npyOutput=False, npyOutputFile=None,
    fftOutput=False, startTime=None, endTime=None,
    verbose=False):
    """Process raw accelerometer file, writing summary epoch stats to file

    This is usually achieved by
        1) identify 10sec stationary epochs
        2) record calibrated axes scale/offset/temp vals + static point stats
        3) use calibration coefficients and then write filtered avgVm epochs
        to <epochFile> from <rawFile>

    :param str rawFile: Input <cwa/cwa.gz/bin/gt3x> raw accelerometer file
    :param str epochFile: Output csv.gz file of processed epoch data
    :param str stationaryFile: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics
    :param bool skipCalibration: Perform software calibration (process data twice)
    :param int stationaryStd: Gravity threshold (in mg units) for stationary vs not
    :param float xIntercept: Calbiration offset x
    :param float yIntercept: Calbiration offset y
    :param float zIntercept: Calbiration offset z
    :param float xSlope: Calbiration slope x
    :param float ySlope: Calbiration slope y
    :param float zSlope: Calbiration slope z
    :param float xTemp: Calbiration temperature coefficient x
    :param float yTemp: Calbiration temperature coefficient y
    :param float zTemp: Calbiration temperature coefficient z
    :param float meanTemp: Calibration mean temperature in file
    :param str rawDataParser: External helper process to read raw acc file. If a
        java class, it must omit .class ending.
    :param str javaHeapSpace: Amount of heap space allocated to java subprocesses.
        Useful for limiting RAM usage.
    :param bool skipFiltering: Skip filtering stage
    :param int sampleRate: Resample data to n Hz
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param bool useAbs: Use abs(VM) instead of trunc(VM)
    :param bool activityClassification: Extract features for machine learning
    :param bool rawOutput: Output calibrated and resampled raw data to a .csv.gz
        file? requires ~50MB/day.
    :param str rawOutputFile: Output raw data ".csv.gz" filename
    :param bool npyOutput: Output calibrated and resampled raw data to a .npy
        file? requires ~60MB/day.
    :param str npyOutputFile: Output raw data ".npy" filename
    :param bool fftOutput: Output FFT epochs to a .csv.gz file? requires ~100MB/day.
    :param datetime startTime: Remove data before this time in analysis
    :param datetime endTime: Remove data after this time in analysis
    :param bool verbose: Print verbose output

    :return: Raw processing summary values written to dict <summary>
    :rtype: void

    :Example:
    >>> import device
    >>> summary = {}
    >>> device.processRawFileToEpoch('rawFile.cwa', 'epochFile.csv.gz',
            'stationary.csv.gz', summary)
    <epoch file written to "epochFile.csv.gz", and calibration points to
        'stationary.csv.gz'>
    """

    summary['file-size'] = os.path.getsize(rawFile)
    summary['file-deviceID'] = getDeviceId(rawFile)
    useJava = True
    javaClassPath = "java:java/JTransforms-3.1-with-dependencies.jar"
    staticStdG = stationaryStd / 1000.0 #java expects units of G (not mg)

    if 'omconvert' in rawDataParser:
        useJava = False
    if useJava:
        # calibrate axes scale/offset/temp values
        if not skipCalibration:
            # identify 10sec stationary epochs
            accUtils.toScreen('calibrating to file: ' + stationaryFile)
            commandArgs = ["java", "-classpath", javaClassPath,
                "-XX:ParallelGCThreads=1", rawDataParser, rawFile,
                "outputFile:" + stationaryFile,
                "verbose:" + str(verbose), "filter:true",
                "getStationaryBouts:true", "epochPeriod:10",
                "stationaryStd:" + str(staticStdG)]
            if javaHeapSpace:
                commandArgs.insert(1, javaHeapSpace)
            # call process to identify stationary epochs
            exitCode = call(commandArgs)
            if exitCode != 0:
                print(commandArgs)
                print("Error: java calibration failed, exit ", exitCode)
                sys.exit(-6)
            # record calibrated axes scale/offset/temp vals + static point stats
            #! TODO: *bug* getCalibrationCoefs returns stuff and summary doesn't get filled
            getCalibrationCoefs(stationaryFile, summary)
            xIntercept = summary['calibration-xOffset(g)']
            yIntercept = summary['calibration-yOffset(g)']
            zIntercept = summary['calibration-zOffset(g)']
            xSlope = summary['calibration-xSlope(g)']
            ySlope = summary['calibration-ySlope(g)']
            zSlope = summary['calibration-zSlope(g)']
            xTemp = summary['calibration-xTemp(C)']
            yTemp = summary['calibration-yTemp(C)']
            zTemp = summary['calibration-zTemp(C)']
            meanTemp = summary['calibration-meanDeviceTemp(C)']
        else:
            storeCalibrationParams(summary, xIntercept, yIntercept, zIntercept,
                    xSlope, ySlope, zSlope, xTemp, yTemp, zTemp, meanTemp)
            summary['quality-calibratedOnOwnData'] = 0
            summary['quality-goodCalibration'] = 1

        # calculate and write filtered avgVm epochs from raw file
        commandArgs = ["java", "-classpath", javaClassPath,
            "-XX:ParallelGCThreads=1", rawDataParser, rawFile,
            "outputFile:" + epochFile, "verbose:" + str(verbose),
            "filter:"+str(skipFiltering),
            "sampleRate:" + str(sampleRate),
            "xIntercept:" + str(xIntercept),
            "yIntercept:" + str(yIntercept),
            "zIntercept:" + str(zIntercept),
            "xSlope:" + str(xSlope),
            "ySlope:" + str(ySlope),
            "zSlope:" + str(zSlope),
            "xTemp:" + str(xTemp),
            "yTemp:" + str(yTemp),
            "zTemp:" + str(zTemp),
            "meanTemp:" + str(meanTemp),
            "epochPeriod:" + str(epochPeriod),
            "rawOutput:" + str(rawOutput),
            "rawOutputFile:" + str(rawOutputFile),
            "npyOutput:" + str(npyOutput),
            "npyOutputFile:" + str(npyOutputFile),
            "fftOutput:" + str(fftOutput),
            "getEpochCovariance:True",
            "getSanDiegoFeatures:" + str(activityClassification),
            "getMADFeatures:" + str(activityClassification),
            "getAxisMeans:" + str(activityClassification),
            "getUnileverFeatures:" + str(activityClassification),
            "getEachAxis:" + str(activityClassification),
            "get3DFourier:" + str(activityClassification),
            "useAbs:" + str(useAbs)]
        accUtils.toScreen('epoch generation')
        if javaHeapSpace:
            commandArgs.insert(1, javaHeapSpace)
        if startTime:
            commandArgs.append("startTime:" + startTime.strftime("%Y-%m-%dT%H:%M"))
        if endTime:
            commandArgs.append("endTime:" + endTime.strftime("%Y-%m-%dT%H:%M"))
        exitCode = call(commandArgs)
        if exitCode != 0:
            print(commandArgs)
            print("Error: java epoch generation failed, exit ", exitCode)
            sys.exit(-7)

    else:
        if not skipCalibration:
            commandArgs = [rawDataParser, rawFile, "-svm-file", epochFile,
                    "-info", stationaryFile, "-svm-extended", "3",
                    "-calibrate", "1", "-interpolate-mode", "2",
                    "-svm-mode", "1", "-svm-epoch", str(epochPeriod),
                    "-svm-filter", "2"]
        else:
            calArgs = str(xSlope) + ','
            calArgs += str(ySlope) + ','
            calArgs += str(zSlope) + ','
            calArgs += str(xIntercept) + ','
            calArgs += str(yIntercept) + ','
            calArgs += str(zIntercept) + ','
            calArgs += str(xTemp) + ','
            calArgs += str(yTemp) + ','
            calArgs += str(zTemp) + ','
            calArgs += str(meanTemp)
            commandArgs = [rawDataParser, rawFile, "-svm-file",
                epochFile, "-info", stationaryFile,
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
    minIterImprovement = 0.0001 #0.1mg
    # use python NUMPY framework to store stationary episodes from epoch file
    if isinstance(staticBoutsFile, pd.DataFrame):

        axesVals = staticBoutsFile[['xMean','yMean','zMean']].values
        tempVals = staticBoutsFile[['temperature']].values
    else:
        d = np.loadtxt(open(staticBoutsFile,"rb"),delimiter=",",skiprows=1,
                usecols=(2,3,4,11,13))
        if len(d)<=5:
            return [0.0,0.0,0.0], [1.0,1.0,1.0], [0.0,0.0,0.0], 20, np.nan, np.nan, \
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(d)
        stationaryPoints = d[d[:,4] == 0] # don't consider episodes with data errors
        axesVals = stationaryPoints[:,[0,1,2]]
        tempVals = stationaryPoints[:,[3]]
    meanTemp = np.mean(tempVals)
    tempVals = np.copy(tempVals-meanTemp)
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
    target = curr / np.sqrt(np.sum(np.square(curr), axis=1))[:, None]
    initError = np.sqrt(np.mean(np.square(curr-target)))  # root mean square error
    # iterate through linear model fitting
    try:
        for i in range(1, maxIter):
            # iterate through each axis, refitting its intercept/slope vals
            for a in range(0,3):
                x = np.concatenate([curr[:, [a]], tempVals], axis=1)
                x = sm.add_constant(x, prepend=True)  # add bias/intercept term
                y = target[:, a]
                newI, newS, newT = sm.OLS(y,x).fit().params
                # update values as part of iterative closest point fitting process
                # refer to wiki as there is quite a bit of math behind next 3 lines
                intercept[a] = newI + (intercept[a] * newS)
                slope[a] = newS * slope[a]
                tempCoef[a] = newT + (tempCoef[a] * newS)
            # update vals (and targed) based on new intercept/slope/temp coeffs
            curr = intercept + (np.copy(axesVals) * slope) + (np.copy(tempVals) * tempCoef)
            target = curr / np.sqrt(np.sum(np.square(curr), axis=1))[:,None]
            rms = np.sqrt(np.mean(np.square(curr-target)))  # root mean square error
            # assess iterative error convergence
            improvement = (bestError-rms)/bestError
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
        sys.stderr.write('WARNING: calibration error\n ' + exceptStr)
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

    file = open(omconvertInfoFile,'r')
    for line in file:
        elements = line.split(':')
        name, value = elements[0], elements[1]
        if name == 'Calibration':
            vals = value.split(',')
            bestIntercept = float(vals[3]), float(vals[4]), float(vals[5])
            bestSlope = float(vals[0]), float(vals[1]), float(vals[2])
            bestTemp = float(vals[6]), float(vals[7]),float(vals[8])
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
    summary['calibration-errsBefore(mg)'] = accUtils.formatNum(initError*1000, 2)
    summary['calibration-errsAfter(mg)'] = accUtils.formatNum(bestError*1000, 2)
    storeCalibrationParams(summary, bestIntercept[0], bestIntercept[1],
            bestIntercept[2], bestSlope[0], bestSlope[1], bestSlope[2],
            bestTemp[0], bestTemp[1], bestTemp[2], meanTemp)
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



def storeCalibrationParams(summary, xOff, yOff, zOff, xSlope, ySlope, zSlope,
        xTemp, yTemp, zTemp, meanTemp):
    """Store calibration parameters to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param float xOff: x intercept
    :param float yOff: y intercept
    :param float zOff: z intercept
    :param float xSlope: x slope
    :param float ySlope: y slope
    :param float zSlope: z slope
    :param float xTemp: x temperature
    :param float yTemp: y temperature
    :param float zTemp: z temperature
    :param float meanTemp: Calibration mean temperature in file

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    # store output to summary dictionary
    summary['calibration-xOffset(g)'] = accUtils.formatNum(xOff, 4)
    summary['calibration-yOffset(g)'] = accUtils.formatNum(yOff, 4)
    summary['calibration-zOffset(g)'] = accUtils.formatNum(zOff, 4)
    summary['calibration-xSlope(g)'] = accUtils.formatNum(xSlope, 4)
    summary['calibration-ySlope(g)'] = accUtils.formatNum(ySlope, 4)
    summary['calibration-zSlope(g)'] = accUtils.formatNum(zSlope, 4)
    summary['calibration-xTemp(C)'] = accUtils.formatNum(xTemp, 4)
    summary['calibration-yTemp(C)'] = accUtils.formatNum(yTemp, 4)
    summary['calibration-zTemp(C)'] = accUtils.formatNum(zTemp, 4)
    summary['calibration-meanDeviceTemp(C)'] = accUtils.formatNum(meanTemp, 2)



def getDeviceId(rawFile):
    """Get serial number of device

    First decides which DeviceId parsing method to use for <rawFile>.

    :param str rawFile: Input raw accelerometer file

    :return: Device ID
    :rtype: int
    """

    if rawFile.lower().endswith('.bin'):
        return getGeneaDeviceId(rawFile)
    elif rawFile.lower().endswith('.cwa') or rawFile.lower().endswith('.cwa.gz'):
        return getAxivityDeviceId(rawFile)
    elif rawFile.lower().endswith('.gt3x'):
        return getGT3XDeviceId(rawFile)
    elif rawFile.lower().endswith('.csv'):
        return "unknown (.csv)"
    else:
        print("ERROR: cannot get deviceId for file: " + rawFile)



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
        f = gzip.open(cwaFile,'rb')
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

    f = open(binFile, 'r') # 'Universal' newline mode
    next(f) # Device Identity
    deviceId = next(f).split(':')[1].rstrip() # Device Unique Serial Code:011710
    f.close()
    return deviceId



def getGT3XDeviceId(cwaFile):
    """Get serial number of Actigraph device

    Parses the unique serial code from the header of a GT3X accelerometer file

    :param str cwaFile: Input raw .gt3x accelerometer file

    :return: Device ID
    :rtype: int
    """

    import zipfile
    if zipfile.is_zipfile(cwaFile):
        with zipfile.ZipFile(cwaFile, 'r') as z:
            contents = z.infolist()
            print("\n".join(map(lambda x: str(x.filename).rjust(20, " ") + ", "
                + str(x.file_size), contents)))

            if 'info.txt' in map(lambda x: x.filename, contents):
                print('info.txt found..')
                info_file = z.open('info.txt','r')
                # print info_file.read()
                for line in info_file:
                    print(line.startswith("Serial Number:"), line)
                    if line.startswith("Serial Number:"):
                        return line.split("Serial Number:")
            else:
                print("could not find info.txt file")

    print("ERROR: in getDeviceId(\"" + cwaFile + "\")")
    print("""A deviceId value could not be found in input file header,
     this usually occurs when the file is not an Actigraph .gt3x accelerometer
     file. Exiting...""")
    sys.exit(-8)
