#BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
"""
This command line application calculates average daily activity from raw
accelerometer data as follows:
    1) Extract and filter sum vector magnitude values for <60>sec epochs
    2) Identify nonWear data in the epochs, and remove it
    3) Construct an avg movement value for each of 1440 minutes in an avg day
    4) Get overall average movement per second from step 3
=== === === ===
The application can be run as follows:
    python ActivitySummary.py <input_file.CWA> <options>
e.g.
    python ActivitySummary.py p001.CWA 
    python ActivitySummary.py p001.CWA min_freq:10 
"""

import collections
import datetime
import json
import os
import pandas as pd
import pytz
import numpy as np
import statsmodels.api as sm
import struct
from subprocess import call, Popen
import sys
import argparse

def main():
    """
    Application entry point responsible for parsing command line requests
    """
    # uncomment for example input
    # sys.argv = "sample -calibrationOffest 1.2 4 99999.342435 -processRawFile false -summaryFolder newdir".split()

    print sys.argv

    parser = argparse.ArgumentParser(
        description="""A tool to extract meaningful health information from large accelerometer
         datasets e.g. how much time individuals spend in sleep, sedentary behaviour, and 
         physical activity.
        """,
        add_help=False
    )


    # optionals
    parser.add_argument('-summaryFolder', metavar='filename',default="",
                            help="""folder for the %(default)s summary statistics""")
    parser.add_argument('-nonWearFolder', metavar='filename',default="",
                            help="""folder for the %(default)s file""")
    parser.add_argument('-epochFolder', metavar='filename',default="",
                            help="""folder for the epoch.json file, this must be an 
                                    existing file if "-processRawFile" is set to False %(default)s file""")
    parser.add_argument('-stationaryFolder', metavar='filename',default="",
                            help="""folder for the  %(default)s file""")
    parser.add_argument('-timeSeriesFolder', metavar='filename',default="",
                            help="""folder for the %(default)s file""")
    parser.add_argument('-skipCalibration', 
                            metavar='True/False',default=True, type=bool,
                            help="""skip calibration? (default : %(default)s)""")
    parser.add_argument('-verbose', 
                            metavar='True/False',default=False, type=bool,
                            help="""enable verbose logging? (default : %(default)s)""")
    parser.add_argument('-deleteIntermediateFiles', 
                            metavar='True/False',default=True, type=bool,
                            help="""True will remove extra "helper" files created by 
                                    the program (default : %(default)s)""")
    parser.add_argument('-processRawFile', 
                            metavar='True/False',default=True, type=bool,
                            help="""False will skip processing of the .cwa file (the
                                epoch.csv file must already exist for this to work) 
                                (default : %(default)s)""")
    parser.add_argument('-epochPeriod', 
                            metavar='length',default=5, type=int,
                            help="""length in seconds of a single epoch (default : %(default)ss)""")
    parser.add_argument('-calibrationOffset', 
                            metavar=('x','y','z'),default=[0.0,0.0,0.0], type=float, nargs=3,
                            help="""accelerometer calibration offset (default : %(default)s)""")
    parser.add_argument('-calibrationSlope', 
                            metavar=('x','y','z'),default=[1.0,1.0,1.0], type=float, nargs=3,
                            help="""accelerometer calibration slope linking offset to temperature (default : %(default)s)""")
    parser.add_argument('-calibrationTemperature', 
                            metavar=('x','y','z'),default=[0.0,0.0,0.0], type=float, nargs=3,
                            help="""mean temperature in degrees Celsius of stationary data for calibration (default : %(default)s)""")
    parser.add_argument('-meanTemperature', 
                            metavar="temp",default=20, type=float,
                            help="""mean calibration temperature in degrees Celsius (default : %(default)s)""")
    parser.add_argument('-javaHeapSpace', 
                            metavar="amount in MB",default="", type=str,
                            help="""amount of heap space allocated to the java subprocesses, useful for limiting RAM usage (default : unlimited)""")
    # TODO add options e.g. 0 None [assumes presence of Epoch.csv and set processRawFile false] 1 AxivityAx3Epochs [java]
    parser.add_argument('-rawDataParser', 
                            metavar="rawDataParser",default="AxivityAx3Epochs", type=str,
                            help="""file containing a java program to process raw .cwa binary file, must be [.class] type (default : %(default)s)""")
    # required
    parser.add_argument('rawFile', metavar='file', type=str, 
                       help='the .cwa file to process (e.g. sample.cwa)')

    #check that enough command line arguments are entered
    if len(sys.argv)<2:
            msg = "\nInvalid input, please enter at least 1 parameter, e.g."
            msg += "\npython ActivitySummary.py inputFile.CWA \n"
            print msg
            parser.print_help()
            sys.exit(0)

    print ""
    args = parser.parse_args()

    if (args.processRawFile is False):
        if len(args.rawFile.split('.'))<2:
            args.rawFile += ".cwa" # edge case since we still need a name?
    elif not os.path.isfile(args.rawFile): 
        print "error: file does not exist. Exiting.."
        sys.exit()

    print "rawFile = " + str(args.rawFile)
    # get file extension
    args.rawFileEnd = '.' + args.rawFile.split('.')[-1]
    args.rawFileBegin = args.rawFile[0:-len(args.rawFileEnd)]
    print (vars(args))

    # no way to deal with blank strings, so disabled for now
    # for path in [args.summaryFolder, args.nonWearFolder, args.epochFolder, args.stationaryFolder, args.timeSeriesFolder]:
    #     print path
    #     if not os.access(path, os.F_OK): 
    #         print "error: " + path + " is not a valid directory"
    #         sys.exit() 

    # could check if folder exists? probably not neccesary
    summaryFile     = os.path.join(args.summaryFolder,    args.rawFileBegin + "OutputSummary.json")
    nonWearFile     = os.path.join(args.nonWearFolder,    args.rawFileBegin + "NonWearBouts.csv")
    epochFile       = os.path.join(args.epochFolder,      args.rawFileBegin + "Epoch.json")
    stationaryFile  = os.path.join(args.stationaryFolder, args.rawFileBegin + "Stationary.csv")
    tsFile          = os.path.join(args.timeSeriesFolder, args.rawFileBegin + "AccTimeSeries.csv")


    # quick add to global namespace
    meanTemp = args.meanTemperature
    deleteHelperFiles = args.deleteIntermediateFiles
    skipCalibration = args.skipCalibration
    verbose = args.verbose
    epochPeriod = args.epochPeriod
    skipRaw = not args.processRawFile
    rawFile = args.rawFile
    rawFileEnd = args.rawFileEnd
    calSlope = args.calibrationSlope
    calTemp = args.calibrationTemperature
    rawFileBegin = args.rawFileBegin
    calOff = args.calibrationOffset
    epochProcess = args.rawDataParser
    javaHeapSpace = args.javaHeapSpace
    
    # print "\nmeanTemp:", meanTemp, "\ndeleteHelperFiles:", deleteHelperFiles, "\nskipCalibration:", skipCalibration, "\nverbose:", verbose, "\nepochPeriod:", epochPeriod, "\nskipRaw:", skipRaw, "\nrawFile:", rawFile, "\nrawFileEnd:", rawFileEnd, "\ncalSlope:", calSlope, "\ncalTemp:", calTemp, "\nrawFileBegin:", rawFileBegin, "\ncalOff:", calOff, "\nepochProcess:", epochProcess, "\njavaHeapSpace:", javaHeapSpace
    
    #check source cwa file exists
    if not skipRaw and not os.path.isfile(rawFile):
        msg = "\n Invalid input"
        msg += "\n File does not exist: " + rawFile + "\n"
        sys.stderr.write(toScreen(msg))
        sys.exit(0)

    fileSize = -1
    deviceId = -1
    if not skipRaw:
        fileSize = os.path.getsize(rawFile)
        deviceId = getDeviceId(rawFile)
        useJava = True
        if 'omconvert' in epochProcess:
            useJava = False
        if useJava:
            #calibrate axes scale/offset values
            if not skipCalibration:
                #identify 10sec stationary epochs
                print toScreen('calibrating')
                commandArgs = ["java", "-classpath", "java", "-XX:ParallelGCThreads=1", epochProcess,
                        rawFile, "outputFile:" + stationaryFile,
                        "verbose:" + str(verbose), "filter:true",
                        "getStationaryBouts:true", "epochPeriod:10",
                        "stationaryStd:0.013"]
                if len(javaHeapSpace) > 1:
                    commandArgs.insert(1,javaHeapSpace);
                call(commandArgs)
                #record calibrated axes scale/offset/temperature vals + static point stats
                calOff, calSlope, calTemp, meanTemp, errPreCal, errPostCal, xMin, \
                        xMax, yMin, yMax, zMin, zMax, \
                        nStatic = getCalibrationCoefs(stationaryFile)
                if verbose:
                    print calOff, calSlope, calTemp, meanTemp, errPreCal, \
                            errPostCal, xMin, xMax, yMin, yMax, zMin, zMax, nStatic
          
            #calculate and write filtered avgVm epochs from raw file
            commandArgs = ["java", "-classpath", "java", "-XX:ParallelGCThreads=1", epochProcess,
                    rawFile, "outputFile:" + epochFile, "verbose:" + str(verbose),
                    "filter:true", "xIntercept:" + str(calOff[0]),
                    "yIntercept:" + str(calOff[1]), "zIntercept:" + str(calOff[2]),
                    "xSlope:" + str(calSlope[0]), "ySlope:" + str(calSlope[1]),
                    "zSlope:" + str(calSlope[2]), "xTemp:" + str(calTemp[0]),
                    "yTemp:" + str(calTemp[1]), "zTemp:" + str(calTemp[2]),
                    "meanTemp:" + str(meanTemp), "epochPeriod:" + str(epochPeriod)]
            print toScreen('epoch generation')
            if len(javaHeapSpace) > 1:
                commandArgs.insert(1,javaHeapSpace);
            print commandArgs
            call(commandArgs)
        else:
            if not skipCalibration:
                commandArgs = [epochProcess, rawFile, "-svm-file", epochFile,
                        "-info", stationaryFile, "-svm-extended", "3",
                        "-calibrate", "1", "-interpolate-mode", "2",
                        "-svm-mode", "1", "-svm-epoch", str(epochPeriod),
                        "-svm-filter", "2"]
            else:
                calArgs = str(calSlope[0]) + ',' + str(calSlope[1]) + ','
                calArgs += str(calSlope[2]) + ',' + str(calOff[0]) + ','
                calArgs += str(calOff[1]) + ',' + str(calOff[2]) + ','
                calArgs += str(calTemp[0]) + ',' + str(calTemp[1]) + ','
                calArgs += str(calTemp[2]) + ',' + str(meanTemp)
                commandArgs = [epochProcess, rawFile, "-svm-file", epochFile,
                        "-info", stationaryFile, "-svm-extended", "3",
                        "-calibrate", "0", "-calibration", calArgs,
                        "-interpolate-mode", "2", "-svm-mode", "1",
                        "-svm-epoch", str(epochPeriod), "-svm-filter", "2"]
            call(commandArgs)
            calOff, calSlope, calTemp, meanTemp, errPreCal, errPostCal, xMin, \
                    xMax, yMin, yMax, zMin, zMax, \
                    nStatic = getOmconvertInfo(stationaryFile)

    
    #calculate average, median, stdev, min, max, count, & ecdf of sample score in
    #1440 min diurnally adjusted day. Also get overall wear time minutes across
    #each hour
    ecdf1, step = np.linspace(0.001, 0.020, 20, retstep=True) #1mg bins from 1-20mg
    ecdf2, step = np.linspace(0.025, 0.100, 16, retstep=True) #5mg bins from 25-100mg
    ecdf3, step = np.linspace(0.125, 0.500, 16, retstep=True) #25mg bins from 125-500mg
    ecdf4, step = np.linspace(0.6, 2.0, 15, retstep=True) #100mg bins from 500-2000mg
    ecdfXVals = np.concatenate([ecdf1, ecdf2, ecdf3, ecdf4])
    print toScreen('generate summary variables from epochs')
    startTime, endTime, daylightSavingsCrossover, wearTimeMins, \
            nonWearTimeMins, numNonWearEpisodes, wearDay, wear24, diurnalHrs, \
            diurnalMins, numInterrupts, interruptMins, numDataErrs, \
            clipsPreCalibrSum, clipsPreCalibrMax, clipsPostCalibrSum, \
            clipsPostCalibrMax, epochSamplesN, epochSamplesAvg, \
            epochSamplesStd, epochSamplesMin, epochSamplesMax, tempMean, \
            tempStd, tempMin, tempMax, accAvg, accStd, unadjustedAccAvg, \
            unadjustedAccStd, unadjustedAccMedian, unadjustedAccMin, \
            unadjustedAccMax, accDays, accHours, \
            accEcdf = getEpochSummary(epochFile, 0, 0, epochPeriod, ecdfXVals, 
                    nonWearFile, tsFile)
    
    #data integrity outputs
    maxErrorRate = 0.001
    lowErrorRate = 1
    norm = epochSamplesN*1.0
    if ( clipsPreCalibrSum/norm >= maxErrorRate or 
            clipsPostCalibrSum/norm >= maxErrorRate or 
            numDataErrs/norm >= maxErrorRate ):
        lowErrorRate = 0
    #min wear time
    minDiurnalHrs = 24
    minWearDays = 5
    goodWearTime = 1
    if diurnalHrs < minDiurnalHrs or wearTimeMins/1440.0 < minWearDays:
        goodWearTime = 0
    #good calibration
    goodCalibration = 1
    s = 0.3 #sphere criteria
    try:
        if xMin>-s or xMax<s or yMin>-s or yMax<s or zMin>-s or zMax<s or \
                np.isnan(xMin) or np.isnan(yMin) or np.isnan(zMin):
            goodCalibration = 0
    except:
        goodCalibration = 0
    #calibrated on own data
    calibratedOnOwnData = 1
    if skipCalibration or skipRaw:
        calibratedOnOwnData = 0
    
    #store variables to dictionary
    result = collections.OrderedDict()
    result['file-name'] = rawFile
    f = '%Y-%m-%d %H:%M:%S'
    result['file-startTime'] = startTime.strftime(f)
    result['file-endTime'] = endTime.strftime(f)
    #physical activity output variable (mg)
    result['acc-overall-avg(mg)'] = formatNum(accAvg*1000, 2)
    result['acc-overall-std(mg)'] = formatNum(accStd*1000, 2)
    #data integrity outputs
    result['quality-lowErrorRate'] = lowErrorRate
    result['quality-goodWearTime'] = goodWearTime
    result['quality-goodCalibration'] = goodCalibration
    result['quality-calibratedOnOwnData'] = calibratedOnOwnData
    result['quality-daylightSavingsCrossover'] = daylightSavingsCrossover
    #physical activity variation by day / hour
    result['acc-mon-avg(mg)'] = formatNum(accDays[0]*1000, 2)
    result['acc-tue-avg(mg)'] = formatNum(accDays[1]*1000, 2)
    result['acc-wed-avg(mg)'] = formatNum(accDays[2]*1000, 2)
    result['acc-thur-avg(mg)'] = formatNum(accDays[3]*1000, 2)
    result['acc-fri-avg(mg)'] = formatNum(accDays[4]*1000, 2)
    result['acc-sat-avg(mg)'] = formatNum(accDays[5]*1000, 2)
    result['acc-sun-avg(mg)'] = formatNum(accDays[6]*1000, 2)
    result['file-firstDay(0=mon,6=sun)'] = startTime.weekday()
    for i in range(0,24):
        result['acc-hourOfDay' + str(i) + '-avg(mg)'] = formatNum(accHours[i]*1000, 2)
    #wear time characteristics
    result['wearTime-overall(days)'] = formatNum(wearTimeMins/1440.0, 2)
    result['nonWearTime-overall(days)'] = formatNum(nonWearTimeMins/1440.0, 2)
    result['wearTime-mon(hrs)'] = formatNum(wearDay[0]/60.0, 2)
    result['wearTime-tue(hrs)'] = formatNum(wearDay[1]/60.0, 2)
    result['wearTime-wed(hrs)'] = formatNum(wearDay[2]/60.0, 2)
    result['wearTime-thur(hrs)'] = formatNum(wearDay[3]/60.0, 2)
    result['wearTime-fri(hrs)'] = formatNum(wearDay[4]/60.0, 2)
    result['wearTime-sat(hrs)'] = formatNum(wearDay[5]/60.0, 2)
    result['wearTime-sun(hrs)'] = formatNum(wearDay[6]/60.0, 2)
    for i in range(0,24):
        result['wearTime-hourOfDay' + str(i) + '-(hrs)'] = formatNum(wear24[i]/60.0, 2)
    result['wearTime-diurnalHrs'] = diurnalHrs
    result['wearTime-diurnalMins'] = diurnalMins
    try:
        result['wearTime-numNonWearEpisodes(>1hr)'] = numNonWearEpisodes
    except:
        result['wearTime-numNonWearEpisodes(>1hr)'] = -1
    #physical activity stats and intensity distribution (minus diurnalWeights)
    result['acc-noDiurnalAdjust-avg(mg)'] = formatNum(unadjustedAccAvg*1000, 2)
    result['acc-noDiurnalAdjust-std(mg)'] = formatNum(unadjustedAccStd*1000, 2)
    result['acc-noDiurnalAdjust-median(mg)'] = formatNum(unadjustedAccMedian*1000, 2)
    result['acc-noDiurnalAdjust-min(mg)'] = formatNum(unadjustedAccMin*1000, 2)
    result['acc-noDiurnalAdjust-max(mg)'] = formatNum(unadjustedAccMax*1000, 2)
    for x, ecdf in zip(ecdfXVals, accEcdf):
        result['acc-ecdf-' + str(x*1000) + 'mg'] = formatNum(ecdf, 4)
    try:
        #calibration metrics 
        result['calibration-errsBefore(mg)'] = formatNum(errPreCal*1000, 2)
        result['calibration-errsAfter(mg)'] = formatNum(errPostCal*1000, 2)
        result['calibration-xOffset(g)'] = formatNum(calOff[0], 4)
        result['calibration-yOffset(g)'] = formatNum(calOff[1], 4)
        result['calibration-zOffset(g)'] = formatNum(calOff[2], 4)
        result['calibration-xSlope(g)'] = formatNum(calSlope[0], 4)
        result['calibration-ySlope(g)'] = formatNum(calSlope[1], 4)
        result['calibration-zSlope(g)'] = formatNum(calSlope[2], 4)
        result['calibration-xTemp(C)'] = formatNum(calTemp[0], 4)
        result['calibration-yTemp(C)'] = formatNum(calTemp[1], 4)
        result['calibration-zTemp(C)'] = formatNum(calTemp[2], 4)
        result['calibration-meanDeviceTemp(C)'] = formatNum(meanTemp, 2)
        result['calibration-numStaticPoints'] = nStatic
        result['calibration-staticXmin(g)'] = formatNum(xMin, 2)
        result['calibration-staticXmax(g)'] = formatNum(xMax, 2)
        result['calibration-staticYmin(g)'] = formatNum(yMin, 2)
        result['calibration-staticYmax(g)'] = formatNum(yMax, 2)
        result['calibration-staticZmin(g)'] = formatNum(zMin, 2)
        result['calibration-staticZmax(g)'] = formatNum(zMax, 2)
    except:
        result['calibration-errsBefore(g)'] = -1
        result['calibration-errsAfter(g)'] = -1
        result['calibration-xOffset(g)'] = -1
        result['calibration-yOffset(g)'] = -1
        result['calibration-zOffset(g)'] = -1
        result['calibration-xSlope(g)'] = -1
        result['calibration-ySlope(g)'] = -1
        result['calibration-zSlope(g)'] = -1
        result['calibration-xTemp(C)'] = -1
        result['calibration-yTemp(C)'] = -1
        result['calibration-zTemp(C)'] = -1
        result['calibration-meanDeviceTemp(C)'] = -1
        result['calibration-numStaticPoints'] = -1
        result['calibration-staticXmin(g)'] = -1
        result['calibration-staticXmax(g)'] = -1
        result['calibration-staticYmin(g)'] = -1
        result['calibration-staticYmax(g)'] = -1
        result['calibration-staticZmin(g)'] = -1
        result['calibration-staticZmax(g)'] = -1
    #raw file data quality indicators
    result['file-size'] = fileSize
    result['file-deviceID'] = deviceId
    #other housekeeping variables
    result['errs-interrupts-num'] = numInterrupts
    result['errs-interrupt-mins'] = formatNum(interruptMins, 1)
    try:
        result['errs-data-num'] = int(numDataErrs)
        result['clips-beforeCalibration-num'] = int(clipsPreCalibrSum)
        result['clips-beforeCalibration-max(perEpoch)'] = int(clipsPreCalibrMax)
        result['clips-afterCalibration-num'] = int(clipsPostCalibrSum)
        result['clips-afterCalibration-max(perEpoch)'] = int(clipsPostCalibrMax)
        result['totalSamples'] = int(epochSamplesN)
    except:
        result['errs-data-num'] = -1
        result['clips-beforeCalibration-num'] = -1
        result['clips-beforeCalibration-max(perEpoch)'] = -1
        result['clips-afterCalibration-num'] = -1
        result['clips-afterCalibration-max(perEpoch)'] = -1
        result['totalSamples'] = -1
    result['sampleRate-avg(Hz)'] = formatNum(epochSamplesAvg / epochPeriod, 1)
    result['sampleRate-std(Hz)'] = formatNum(epochSamplesStd / epochPeriod, 1)
    result['sampleRate-min(Hz)'] = formatNum(epochSamplesMin / epochPeriod, 1)
    result['sampleRate-max(Hz)'] = formatNum(epochSamplesMax / epochPeriod, 1)
    result['deviceTemp-mean'] = formatNum(tempMean, 1)
    result['deviceTemp-std'] = formatNum(tempStd, 1)
    result['deviceTemp-min'] = formatNum(tempMin, 1)
    result['deviceTemp-max'] = formatNum(tempMax, 1)
    
    #print basic output
    summaryVals = ['file-name', 'file-startTime', 'file-endTime', 'acc-overall-avg(mg)','wearTime-overall(days)','nonWearTime-overall(days)']
    summaryDict = collections.OrderedDict([(i, result[i]) for i in summaryVals])
    print toScreen(json.dumps(summaryDict, indent=4))
    
    #write detailed output to file
    f = open(summaryFile,'w')
    json.dump(result, f, indent=4)
    f.close()
    if deleteHelperFiles:
        try:
            os.remove(stationaryFile)
            os.remove(epochFile)
        except:
            print 'could not delete helper file' 
    if verbose:
        print toScreen('see all variables at: ' + summaryFile)


def getEpochSummary(epochFile,
        headerSize,
        dateColumn,
        epochSec,
        ecdfXVals,
        nonWearFile,
        tsFile):
    """
    Calculate diurnally adjusted average movement per minute from epoch file
    which has had nonWear episodes removed from it
    """
    #use python PANDAS framework to read in and store epochs
    e = pd.read_csv(epochFile, index_col=dateColumn, parse_dates=['Time'],
                header=headerSize).sort_index()
    cols = ['enmoTrunc','xRange','yRange','zRange']
    cols += ['xStd','yStd','zStd','temp','samples']
    cols += ['dataErrors','clipsBeforeCalibr','clipsAfterCalibr','rawSamples']
    e.columns = cols
    #get start & end times
    startTime = pd.to_datetime(e.index.values[0])
    endTime = pd.to_datetime(e.index.values[-1])

    #get interrupt and data error summary vals
    e.dropna(subset=['enmoTrunc','xStd','yStd','zStd'],how='all',inplace=True)
    epochNs = epochSec * np.timedelta64(1,'s')
    interrupts = np.where(np.diff(np.array(e.index)) > epochNs)[0]
    #get duration of each interrupt in minutes
    interruptMins = []
    for i in interrupts:
        interruptMins.append( np.diff(np.array(e[i:i+2].index)) /
                np.timedelta64(1,'m') )

    #check if data occurs at a daylight savings crossover
    daylightSavingsCrossover = 0
    localTime = pytz.timezone('Europe/London')
    startTimeZone = localTime.localize(startTime)
    endTimeZone = localTime.localize(endTime)
    if startTimeZone.dst() != endTimeZone.dst():
        daylightSavingsCrossover = 1
        #find whether clock needs to go forward or back
        if endTimeZone.dst() > startTimeZone.dst():
            offset = 1
        else:
            offset = -1
        print 'different timezones, offset = ' + str(offset)
        #find actual crossover time
        for t in localTime._utc_transition_times:
            if t>startTime:
                transition = t
                break
        #if Autumn crossover time, adjust transition time plus remove 1hr chunk
        if offset == -1:
            #pytz stores dst crossover at 1am, but clocks change at 2am local
            transition = transition + pd.DateOffset(hours=1)
            #remove last hr before DST cut, which will be subsequently overwritten
            e = e[(e.index < transition - pd.DateOffset(hours=1)) | 
                    (e.index >= transition)]
        print 'day light savings transition at:' + str(transition)
        #now update datetime index to 'fix' values after DST crossover
        e['newTime'] = e.index
        e['newTime'][e.index >= transition] = e.index + np.timedelta64(offset,'h')
        e['newTime'] = e['newTime'].fillna(e.index)
        e = e.set_index('newTime')
        #reset startTime and endTime variables
        startTime = pd.to_datetime(e.index.values[0])
        endTime = pd.to_datetime(e.index.values[-1])

    #calculate nonWear (nw) time
    minDuration = 60 #minutes
    maxStd = 0.013
    e['nw'] = np.where((e['xStd']<maxStd) & (e['yStd']<maxStd) & 
            (e['zStd']<maxStd),1,0)
    starts = e.index[(e['nw']==True) & (e['nw'].shift(1).fillna(False)==False)]
    ends = e.index[(e['nw']==True) & (e['nw'].shift(-1).fillna(False)==False)]
    nonWearEpisodes = [(start, end) for start, end in zip(starts, ends)
            if end > start + np.timedelta64(minDuration,'m')]
    #set nonWear data to nan and record to nonWearBouts file
    f = open(nonWearFile,'w')
    f.write('start,end,xStdMax,yStdMax,zStdMax\n')
    timeFormat = '%Y-%m-%d %H:%M:%S'
    for episode in nonWearEpisodes:
        tmp = e[['xStd','yStd','zStd']][episode[0]:episode[1]]
        summary = episode[0].strftime(timeFormat) + ','
        summary += episode[1].strftime(timeFormat) + ','
        summary += str(tmp['xStd'].mean()) + ','
        summary += str(tmp['yStd'].mean()) + ','
        summary += str(tmp['zStd'].mean())
        f.write(summary + '\n')
        #set main dataframe values to nan
        e[episode[0]:episode[1]] = np.nan
    f.close()

    paCol = 'enmoTrunc'
    wearSamples = e[paCol].count()
    nonWearSamples = len(e[np.isnan(e[paCol])].index.values)
    wearTimeMin = wearSamples * epochSec / 60.0
    nonWearTimeMin = nonWearSamples * epochSec / 60.0
   
    #get wear time in each of 24 hours across week
    epochsInMin = 60 / epochSec
    wearDay = []
    for i in range(0,7):
        wearDay.append( e[paCol][e.index.weekday == i].count() / epochsInMin )
    wear24 = []
    for i in range(0,24):
        wear24.append( e[paCol][e.index.hour == i].count() / epochsInMin )
    diurnalHrs = e[paCol].groupby(e.index.hour).mean().count()
    diurnalMins = e[paCol].groupby([e.index.hour,e.index.minute]).mean().count()

    #calculate imputation values to replace nan PA metric values
    #i.e. replace with mean avgVm from same time in other days
    e['hour'] = e.index.hour
    e['minute'] = e.index.minute
    wearTimeWeights = e.groupby(['hour', 'minute'])[paCol].mean() #weartime weighted data
    e = e.join(wearTimeWeights, on=['hour','minute'], rsuffix='_imputed')
    
    unadjustedAccData = e[paCol] #raw data
    #calculate stat summaries
    unadjustedAccAvg = unadjustedAccData.mean()
    unadjustedAccStd = unadjustedAccData.std()
    unadjustedAccMedian = unadjustedAccData.median()
    unadjustedAccMin = unadjustedAccData.min()
    unadjustedAccMax = unadjustedAccData.max()
    
    #now wearTime weight values
    e[paCol+'Adjusted'] = e[paCol].fillna(e[paCol + '_imputed'])
    accData = e[paCol+'Adjusted'] #weartime weighted data
    accAvg = accData.mean()
    accStd = accData.std()
    accDays = []
    for i in range(0,7):
        accDays.append(accData[accData.index.weekday == i].mean())
    accHours = []
    for i in range(0,24):
        accHours.append(accData[accData.index.hour == i].mean())

    #calculate empirical cumulative distribution function of vector magnitudes
    ecdfData = e[['hour','minute','enmoTrunc']][~np.isnan(e['enmoTrunc'])] #remove NaNs (necessary for statsmodels.api)
    if len(ecdfData) > 0:
        #set column names for actual, imputed, and adjusted intensity dist. vals
        cols = []
        colsImputed = []
        colsAdjusted = []
        for xVal in ecdfXVals:
            col = 'ecdf' + str(xVal)
            cols.append(col)
            colsImputed.append(col + 'Imputed')
            colsAdjusted.append(col + 'Adjusted')
            ecdfData[col] = (ecdfData['enmoTrunc']<=xVal) *1.0
        #calculate imputation values to replace nan metric values
        wearTimeWeights = ecdfData.groupby(['hour', 'minute'])[cols].mean() #weartime weighted
        ecdfData = ecdfData.join(wearTimeWeights, on=['hour','minute'], rsuffix='Imputed')
        #for each ecdf xVal column, apply missing data imputation
        for col,imputed,adjusted in zip(cols,colsImputed,colsAdjusted):
            ecdfData[adjusted] = ecdfData[col].fillna(ecdfData[imputed])
        
        accEcdf = ecdfData[colsAdjusted].mean()
    else:
        accEcdf = np.empty(20 + 16 + 16 + 15)
    
    #prepare time series header
    e = e.reindex(pd.date_range(startTime, endTime, freq=str(epochSec)+'s'))
    tsHead = 'acceleration (mg) - '
    tsHead += e.index.min().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += e.index.max().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += 'sampleRate = ' + str(epochSec) + ' seconds'
    if len(accData) > 0:
        #write time series file
        #convert 'vm' to mg units, and highlight any imputed values
        e['vmFinal'] = e[paCol+'Adjusted'] * 1000
        e['imputed'] = np.isnan(e[paCol]).astype(int)
        e[['vmFinal','imputed']].to_csv(tsFile, float_format='%.1f',
                index=False,header=[tsHead,'imputed'])
    else:
        f = open(tsFile,'w')
        f.write(tsHead + '\n')
        f.write('no wearTime data,1')
        f.close()
   
    #return physical activity summary
    return startTime, endTime, daylightSavingsCrossover, wearTimeMin, \
            nonWearTimeMin, len(nonWearEpisodes), wearDay, wear24, diurnalHrs, \
            diurnalMins, len(interrupts), np.sum(interruptMins), \
            e['dataErrors'].sum(), e['clipsBeforeCalibr'].sum(), \
            e['clipsBeforeCalibr'].max(), e['clipsAfterCalibr'].sum(), \
            e['clipsAfterCalibr'].max(), e['rawSamples'].sum(), \
            e['rawSamples'].mean(), e['rawSamples'].std(), \
            e['rawSamples'].min(), e['rawSamples'].max(), e['temp'].mean(), \
            e['temp'].std(), e['temp'].min(), e['temp'].max(), accAvg, accStd, \
            unadjustedAccAvg, unadjustedAccStd, unadjustedAccMedian, \
            unadjustedAccMin, unadjustedAccMax, accDays, accHours, accEcdf


def getCalibrationCoefs(staticBoutsFile):
    """
    Get axes offset/gain/temp calibration coefficients through linear regression
    of stationary episodes
    """
    #learning/research parameters
    maxIter = 1000
    minIterImprovement = 0.0001 #0.1mg
    #use python NUMPY framework to store stationary episodes from epoch file
    d = np.loadtxt(open(staticBoutsFile,"rb"),delimiter=",",skiprows=1,
            usecols=(2,3,4,11,13))
    if len(d)<=5: 
        return [0.0,0.0,0.0], [1.0,1.0,1.0], [0.0,0.0,0.0], 20, np.nan, np.nan, \
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, len(d) 
    stationaryPoints = d[d[:,4] == 0] #don't consider episodes with data errors
    axesVals = stationaryPoints[:,[0,1,2]]
    tempVals = stationaryPoints[:,[3]]
    meanTemp = np.mean(tempVals)
    tempVals = np.copy(tempVals-meanTemp)
    #store information on spread of stationary points
    xMin, yMin, zMin = np.amin(axesVals, axis=0)
    xMax, yMax, zMax = np.amax(axesVals, axis=0)
    #initialise intercept/slope variables to assume no error initially present
    intercept = np.array([0.0, 0.0, 0.0])
    slope = np.array([1.0, 1.0, 1.0])
    tempCoef = np.array([0.0, 0.0, 0.0])
    #variables to support model fitting
    initError = float("inf")
    prevError = float("inf")
    bestError = float("inf")
    bestIntercept = np.copy(intercept)
    bestSlope = np.copy(slope)
    bestTemp = np.copy(tempCoef)
    #record initial uncalibrated error
    curr = intercept + (np.copy(axesVals) * slope) + (np.copy(tempVals) * tempCoef)
    target = curr / np.sqrt(np.sum(np.square(curr), axis=1))[:,None]
    initError = np.sqrt(np.mean(np.square(curr-target))) #root mean square error
    #iterate through linear model fitting
    try:
        for i in range(1, maxIter):
            #iterate through each axis, refitting its intercept/slope vals
            for a in range(0,3):
                x = np.concatenate([curr[:,[a]], tempVals], axis=1)
                x = sm.add_constant(x, prepend=True) #add bias/intercept term
                y = target[:,a]
                newI, newS, newT = sm.OLS(y,x).fit().params
                #update values as part of iterative closest point fitting process
                #refer to wiki as there is quite a bit of math behind next 3 lines
                intercept[a] = newI + (intercept[a] * newS)
                slope[a] = newS * slope[a]
                tempCoef[a] = newT + (tempCoef[a] * newS)
            #update vals (and targed) based on new intercept/slope/temp coeffs
            curr = intercept + (np.copy(axesVals) * slope) + (np.copy(tempVals) * tempCoef)
            target = curr / np.sqrt(np.sum(np.square(curr), axis=1))[:,None]
            rms = np.sqrt(np.mean(np.square(curr-target))) #root mean square error
            #assess iterative error convergence
            improvement = (bestError-rms)/bestError
            prevError=rms
            if rms < bestError:
                bestIntercept = np.copy(intercept)
                bestSlope = np.copy(slope)
                bestTemp = np.copy(tempCoef)
                bestError = rms
            if improvement < minIterImprovement:
                break #break if not largely converged
    except:
        #highlight problem with regression, and exit
        xMin, yMin, zMin = float('nan'), float('nan'), float('nan')
        xMax, yMax, zMax = float('nan'), float('nan'), float('nan')
        sys.stderr.write('WARNING: calibration error\n')
    return bestIntercept, bestSlope, bestTemp, meanTemp, initError, bestError, \
            xMin, xMax, yMin, yMax, zMin, zMax, len(axesVals)


def getOmconvertInfo(omconvertInfoFile):
    """
    Get axes offset/gain/temp calibration coeffs from omconvert info file
    """
    file = open(omconvertInfoFile,'rU')
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
    return bestIntercept, bestSlope, bestTemp, meanTemp, initError, bestError, \
            xMin, xMax, yMin, yMax, zMin, zMax, nStatic


def getDeviceId(cwaFile):
    f = open(cwaFile, 'rb')
    header = f.read(2)
    if header == 'MD':
        blockSize = struct.unpack('H', f.read(2))[0]
        performClear = struct.unpack('B', f.read(1))[0]
        deviceId = struct.unpack('H', f.read(2))[0]
    f.close()
    return deviceId

def formatNum(num, decimalPlaces):
    fmt = '%.' + str(decimalPlaces) + 'f'
    return float(fmt % num)

def toScreen(msg):
    timeFormat = '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.now().strftime(timeFormat) +  ' ' + msg

"""
Standard boilerplate to call the main() function to begin the program.
"""
if __name__ == '__main__': 
    main() #Standard boilerplate to call the main() function to begin the program.
