"""Module to process raw accelerometer files into epoch data."""

from accelerometer import utils
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
    xyzSlope=[1.0, 1.0, 1.0], xyzSlopeT=[0.0, 0.0, 0.0],
    rawDataParser="AccelerometerParser", javaHeapSpace=None,
    useFilter=True, sampleRate=100, resampleMethod="linear", epochPeriod=30,
    extractFeatures=True,
    rawOutput=False, rawFile=None, npyOutput=False, npyFile=None,
    startTime=None, endTime=None,
    verbose=False,
    csvStartTime=None, csvSampleRate=None,
    csvTimeFormat="yyyy-MM-dd HH:mm:ss.SSSxxxx '['VV']'",
    csvStartRow=1, csvTimeXYZTempColsIndex=None
):
    """
    Process raw accelerometer file, writing summary epoch stats to file. This is usually achieved by:
    1) identify 10sec stationary epochs
    2) record calibrated axes scale/offset/temp vals + static point stats
    3) use calibration coefficients and then write filtered avgVm epochs to <epochFile> from <inputFile>

    :param str inputFile: Input <cwa/cwa.gz/bin/gt3x> raw accelerometer file
    :param str epochFile: Output csv.gz file of processed epoch data
    :param str stationaryFile: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics
    :param bool skipCalibration: Perform software calibration (process data twice)
    :param int stationaryStd: Gravity threshold (in mg units) for stationary vs not
    :param list(float) xyzIntercept: Calibration offset [x, y, z]
    :param list(float) xyzSlope: Calibration slope [x, y, z]
    :param list(float) xyzTemp: Calibration temperature coefficient [x, y, z]
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

    :return: None. Writes raw processing summary values to dict <summary>.

    .. code-block:: python

        import device
        summary = {}
        device.processInputFileToEpoch('inputFile.cwa', 'epochFile.csv.gz',
                'stationary.csv.gz', summary)
    """

    summary['file-size'] = os.path.getsize(inputFile)
    summary['file-deviceID'] = getDeviceId(inputFile)
    useJava = True
    pathSeparator = ';' if os.name == 'nt' else ':'
    javaClassPath = f"{ROOT_DIR}/java/{pathSeparator}{ROOT_DIR}/java/JTransforms-3.1-with-dependencies.jar"
    staticStdG = stationaryStd / 1000.0  # java expects units of G (not mg)

    if 'omconvert' in rawDataParser:
        useJava = False

    if useJava:
        if not skipCalibration:
            # identify 10sec stationary epochs
            utils.toScreen("=== Calibrating ===")
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
            xyzSlope = [summary['calibration-xSlope'],
                        summary['calibration-ySlope'],
                        summary['calibration-zSlope']]
            xyzSlopeT = [summary['calibration-xSlopeTemp'],
                         summary['calibration-ySlopeTemp'],
                         summary['calibration-zSlopeTemp']]
        else:
            storeCalibrationParams(summary, xyzIntercept, xyzSlope, xyzSlopeT)
            summary['quality-calibratedOnOwnData'] = 0
            summary['quality-goodCalibration'] = 1

        utils.toScreen('=== Extracting features ===')
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
                       "xSlopeT:" + str(xyzSlopeT[0]),
                       "ySlopeT:" + str(xyzSlopeT[1]),
                       "zSlopeT:" + str(xyzSlopeT[2]),
                       "epochPeriod:" + str(epochPeriod),
                       "rawOutput:" + str(rawOutput),
                       "rawFile:" + str(rawFile),
                       "npyOutput:" + str(npyOutput),
                       "npyFile:" + str(npyFile),
                       "getFeatures:" + str(extractFeatures)]
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
            calArgs += str(xyzSlopeT[0]) + ','
            calArgs += str(xyzSlopeT[1]) + ','
            calArgs += str(xyzSlopeT[2]) + ','
            commandArgs = [rawDataParser, inputFile, timeZone, timeShift,
                           "-svm-file", epochFile, "-info", stationaryFile,
                           "-svm-extended", "3", "-calibrate", "0",
                           "-calibration", calArgs, "-interpolate-mode", "2",
                           "-svm-mode", "1", "-svm-epoch", str(epochPeriod),
                           "-svm-filter", "2"]
        call(commandArgs)
        getOmconvertInfo(stationaryFile, summary)


def getCalibrationCoefs(staticBoutsFile, summary):
    """
    Identify calibration coefficients from java processed file. Get axes
    offset/gain/temp calibration coefficients through linear regression of
    stationary episodes.

    :param str stationaryFile: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics

    :return: None. Calibration summary values written to dict <summary>
    """

    if isinstance(staticBoutsFile, pd.DataFrame):
        data = staticBoutsFile

    else:
        data = pd.read_csv(staticBoutsFile)

    data = data[data['dataErrors'] == 0].dropna()  # drop segments with errors
    xyz = data[['xMean', 'yMean', 'zMean']].to_numpy()
    if 'temp' in data:
        T = data['temp'].to_numpy()
    else:  # use a dummy
        T = np.zeros(len(xyz), dtype=xyz.dtype)

    # Remove any zero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]
    T = T[nonzero]

    intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
    slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
    slopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
    bestIntercept = np.copy(intercept)
    bestSlope = np.copy(slope)
    bestSlopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr - target, axis=1)
    err = np.mean(errors)  # MAE more robust than RMSE. This is different from the paper
    initErr = err
    bestErr = 1e16
    nStatic = len(xyz)

    MAXITER = 1000
    IMPROV_TOL = 0.0001  # 0.01%
    ERR_TOL = 0.01  # 10mg
    CALIB_CUBE = 0.3
    CALIB_MIN_SAMPLES = 50

    # Check that we have enough uniformly distributed points:
    # need at least one point outside each face of the cube
    if len(xyz) < CALIB_MIN_SAMPLES or (np.max(xyz, axis=0) < CALIB_CUBE).any() or (np.min(xyz, axis=0) > -CALIB_CUBE).any():
        goodCalibration = 0

    else:  # we do have enough uniformly distributed points

        for it in range(MAXITER):

            # Weighting. Outliers are zeroed out
            # This is different from the paper
            maxerr = np.quantile(errors, .995)
            weights = np.maximum(1 - errors / maxerr, 0)

            # Optimize params for each axis
            for k in range(3):

                inp = curr[:, k]
                out = target[:, k]
                inp = np.column_stack((inp, T))
                inp = sm.add_constant(inp, prepend=True, has_constant='add')
                params = sm.WLS(out, inp, weights=weights).fit().params
                # In the following,
                # intercept == params[0]
                # slope == params[1]
                # slopeT == params[2]
                intercept[k] = params[0] + (intercept[k] * params[1])
                slope[k] = params[1] * slope[k]
                slopeT[k] = params[2] + (slopeT[k] * params[1])

            # Update current solution and target
            curr = intercept + (xyz * slope)
            curr = curr + (T[:, None] * slopeT)
            target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

            # Update errors
            errors = np.linalg.norm(curr - target, axis=1)
            err = np.mean(errors)
            errImprov = (bestErr - err) / bestErr

            if err < bestErr:
                bestIntercept = np.copy(intercept)
                bestSlope = np.copy(slope)
                bestSlopeT = np.copy(slopeT)
                bestErr = err

            if errImprov < IMPROV_TOL:
                break

        goodCalibration = int(not ((bestErr > ERR_TOL) or (it + 1 == MAXITER)))

    if goodCalibration == 0:  # restore calibr params
        bestIntercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
        bestSlope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
        bestSlopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
        bestErr = initErr

    storeCalibrationInformation(
        summary,
        bestIntercept=bestIntercept,
        bestSlope=bestSlope,
        bestSlopeT=bestSlopeT,
        initErr=initErr,
        bestErr=bestErr,
        nStatic=nStatic,
        calibratedOnOwnData=1,
        goodCalibration=goodCalibration
    )

    return


def getOmconvertInfo(omconvertInfoFile, summary):
    """
    Identify calibration coefficients for omconvert processed file. Get axes
    offset/gain/temp calibration coeffs from omconvert info file.

    :param str omconvertInfoFile: Output information file from omconvert
    :param dict summary: Output dictionary containing all summary metrics

    :return: None. Calibration summary values written to dict <summary>
    """

    file = open(omconvertInfoFile, 'r')
    for line in file:
        elements = line.split(':')
        name, value = elements[0], elements[1]
        if name == 'Calibration':
            vals = value.split(',')
            bestIntercept = float(vals[3]), float(vals[4]), float(vals[5])
            bestSlope = float(vals[0]), float(vals[1]), float(vals[2])
            bestSlopeT = float(vals[6]), float(vals[7]), float(vals[8])
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
    storeCalibrationInformation(summary, bestIntercept, bestSlope, bestSlopeT,
                                initError, bestError, nStatic, None, None)


def storeCalibrationInformation(
        summary, bestIntercept, bestSlope, bestSlopeT,
        initErr, bestErr, nStatic, calibratedOnOwnData, goodCalibration
):
    """
    Store calibration information to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) bestIntercept: Best x/y/z intercept values
    :param list(float) bestSlope: Best x/y/z slope values
    :param list(float) bestSlopeT: Best x/y/z temperature slope values
    :param float initErr: Error (in mg) before calibration
    :param float bestErr: Error (in mg) after calibration
    :param int nStatic: number of stationary points used for calibration
    :param calibratedOnOwnData: Whether params were self-derived
    :param goodCalibration: Whether calibration succeded

    :return: None. Calibration summary values written to dict <summary>
    """

    # store output to summary dictionary
    storeCalibrationParams(summary, bestIntercept, bestSlope, bestSlopeT)
    summary['calibration-errsBefore(mg)'] = initErr * 1000
    summary['calibration-errsAfter(mg)'] = bestErr * 1000
    summary['calibration-numStaticPoints'] = nStatic
    summary['quality-calibratedOnOwnData'] = calibratedOnOwnData
    summary['quality-goodCalibration'] = goodCalibration


def storeCalibrationParams(summary, xyzOff, xyzSlope, xyzSlopeT):
    """
    Store calibration parameters to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) xyzOff: intercept [x, y, z]
    :param list(float) xyzSlope: slope [x, y, z]
    :param list(float) xyzSlopeT: temperature slope [x, y, z]

    :return: None. Calibration summary values written to dict <summary>
    """

    # store output to summary dictionary
    summary['calibration-xOffset(g)'] = xyzOff[0]
    summary['calibration-yOffset(g)'] = xyzOff[1]
    summary['calibration-zOffset(g)'] = xyzOff[2]
    summary['calibration-xSlope'] = xyzSlope[0]
    summary['calibration-ySlope'] = xyzSlope[1]
    summary['calibration-zSlope'] = xyzSlope[2]
    summary['calibration-xSlopeTemp'] = xyzSlopeT[0]
    summary['calibration-ySlopeTemp'] = xyzSlopeT[1]
    summary['calibration-zSlopeTemp'] = xyzSlopeT[2]


def getDeviceId(inputFile):
    """
    Get serial number of device. First decides which DeviceId parsing method to use for <inputFile>.

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
    """
    Get serial number of Axivity device. Parses the unique serial code from the
    header of an Axivity accelerometer file.

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
    """
    Get serial number of GENEActiv device. Parses the unique serial code from
    the header of a GENEActiv accelerometer file

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
    """
    Get serial number of Actigraph device. Parses the unique serial code from
    the header of a GT3X accelerometer file

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
