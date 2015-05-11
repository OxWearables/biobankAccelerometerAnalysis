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

import sys
import os
import datetime
import behaviourEpisode
import pandas as pd
import numpy as np
import statsmodels.api as sm
from subprocess import call, Popen
import struct

def main():
    """
    Application entry point responsible for parsing command line requests
    """
    print sys.argv
    #check that enough command line arguments are entered
    if len(sys.argv)<2:
        msg = "\n Invalid input, please enter at least 1 parameter, e.g."
        msg += "\n python ActivitySummary.py inputFile.CWA \n"
        print msg
        sys.exit(0)
    #store command line arguments to local variables
    rawFile = sys.argv[1]      
    funcParams = sys.argv[2:]
    rawFile = rawFile.replace(".CWA", ".cwa")
    summaryFile = rawFile.replace(".cwa","OutputSummary.csv")
    tsFile = rawFile.replace(".cwa","AccTimeSeries.csv")
    nonWearFile = rawFile.replace(".cwa","NonWearBouts.csv")
    epochFile = rawFile.replace(".cwa","Epoch.csv")
    stationaryFile = rawFile.replace(".cwa","Stationary.csv")
    javaEpochProcess = "AxivityAx3Epochs"
    javaHeapSpace = ""
    skipRaw = False
    skipCalibration = False
    deleteHelperFiles = True
    verbose = False
    epochSec = 5
    epochPeriodStr = "epochPeriod:" + str(epochSec)
    #update default values by looping through user parameters
    for param in funcParams:
        #example param -> 'matlab:/Applications/MATLAB_R2014a.app/bin/matlab'
        if param.split(':')[0] == 'summaryFolder':
            summaryFile = param.split(':')[1] + summaryFile.split('/')[-1]
        elif param.split(':')[0] == 'epochFolder':
            epochFile = param.split(':')[1] + epochFile.split('/')[-1]
        elif param.split(':')[0] == 'nonWearFolder':
            nonWearFile = param.split(':')[1] + nonWearFile.split('/')[-1]
        elif param.split(':')[0] == 'stationaryFolder':
            stationaryFile = param.split(':')[1] + stationaryFile.split('/')[-1]
        elif param.split(':')[0] == 'timeSeriesFolder':
            tsFile = param.split(':')[1] + tsFile.split('/')[-1]
        elif param.split(':')[0] == 'skipRaw':
            skipRaw = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'skipCalibration':
            skipCalibration = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'verbose':
            verbose = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'deleteHelperFiles':
            deleteHelperFiles = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'epochPeriod':
            epochPeriodStr = param
        elif param.split(':')[0] == 'javaHeapSpace':
            javaHeapSpace = param.split(':')[1]

    #check source cwa file exists
    if not skipRaw and not os.path.isfile(rawFile):
        msg = "\n Invalid input"
        msg += "\n File does not exist: " + rawFile + "\n"
        sys.stderr.write(msg)
        sys.exit(0)

    if not skipRaw:
        #calibrate axes scale/offset values
        if not skipCalibration:
            #identify 10sec stationary epochs
            print 'calibrating'
            commandArgs = ["java", "-XX:ParallelGCThreads=1", javaEpochProcess,
                    rawFile, "outputFile:" + stationaryFile, "verbose:" + str(verbose),
                    "filter:true", "getStationaryBouts:true", "epochPeriod:10",
                    "stationaryStd:0.013"]
            if len(javaHeapSpace) > 1:
                commandArgs.insert(1,javaHeapSpace);
            call(commandArgs)
            #record calibrated axes scale/offset/temperature vals + static point stats
            calOff, calSlope, calTemp, meanTemp, errPreCal, errPostCal, xMin, xMax, yMin, yMax, zMin, zMax, nStatic = getCalibrationCoefs(stationaryFile)
            if verbose:
                print calOff, calSlope, calTemp, meanTemp, errPreCal, errPostCal, xMin, xMax, yMin, yMax, zMin, zMax, nStatic
            commandArgs = ["java", "-XX:ParallelGCThreads=1", javaEpochProcess,
                    rawFile, "outputFile:" + epochFile, "verbose:" + str(verbose),
                    "filter:true", "xIntercept:" + str(calOff[0]),
                    "yIntercept:" + str(calOff[1]), "zIntercept:" + str(calOff[2]),
                    "xSlope:" + str(calSlope[0]), "ySlope:" + str(calSlope[1]),
                    "zSlope:" + str(calSlope[2]), "xTemp:" + str(calTemp[0]),
                    "yTemp:" + str(calTemp[1]), "zTemp:" + str(calTemp[2]),
                    "meanTemp:" + str(meanTemp), epochPeriodStr]
        else: 
            commandArgs = ["java", "-XX:ParallelGCThreads=1", javaEpochProcess,
                    rawFile, "outputFile:" + epochFile, "verbose:" + str(verbose), 
                    "filter:true", epochPeriodStr]
      
        #calculate and write filtered avgVm epochs from raw file
        print 'epoch generation'
        if len(javaHeapSpace) > 1:
            commandArgs.insert(1,javaHeapSpace);
        call(commandArgs)

        #identify and remove nonWear episodes
        print 'nonwear identification'
        numNonWearEpisodes = identifyAndRemoveNonWearTime(epochFile, nonWearFile,
                funcParams, epochSec)    
    
    #define PA metrics i.e. column names from java epoch process
    paMetrics = ['enmoTrunc', 'enmoAbs', 'en', 'enmoAbsBP']
    
    #calculate average, median, stdev, min, max, count, & ecdf of sample score in
    #1440 min diurnally adjusted day. Also get overall wear time minutes across
    #each hour
    print 'summary stats generation'
    startTime, endTime, wearTimeMins, nonWearTimeMins, wear24, avgDayMins, numInterrupts, interruptMins, numDataErrs, clipsPreCalibrSum, clipsPreCalibrMax, clipsPostCalibrSum, clipsPostCalibrMax, epochSamplesN, epochSamplesAvg, epochSamplesStd, epochSamplesMin, epochSamplesMax, tempMean, tempStd, paWAvg, paWStd, paAvg, paStd, paMedian, paMin, paMax, paEcdf1, paEcdf2, paEcdf3, paEcdf4 = getEpochSummary(epochFile, 0, 0, epochSec, tsFile, paMetrics)
    
    #print processed summary variables from accelerometer file
    fSummary = rawFile + ','
    cmdSummary = rawFile + ', '
    #physical activity output variable (mg)
    f = '%.2f'
    fSummary += f % (paWAvg[0]*1000) + ','
    fSummary += f % (paWStd[0]*1000) + ','
    cmdSummary += f % (paWAvg[0]*1000) + ' mg, '
    #wear time characteristics
    f = '%Y-%m-%d %H:%M:%S'
    fSummary += startTime.strftime(f) + ',' + endTime.strftime(f) + ','
    cmdSummary += startTime.strftime(f) + ' - ' + endTime.strftime(f) + ', '
    f = '%.0f'
    fSummary += f % wearTimeMins + ', ' + f % nonWearTimeMins + ', '
    cmdSummary += f % wearTimeMins + ' mins wear, '
    cmdSummary += f % nonWearTimeMins + ' mins nonWear'
    for i in range(0,24):
        fSummary += str(wear24[i]) + ','
    fSummary += str(avgDayMins) + ','
    try:
        fSummary += str(numNonWearEpisodes) + ','
        #calibration metrics 
        fSummary += str(errPreCal) + ',' + str(errPostCal) + ','
        fSummary += str(calOff[0]) + ',' + str(calOff[1]) + ','
        fSummary += str(calOff[2]) + ',' + str(calSlope[0]) + ','
        fSummary += str(calSlope[1]) + ',' + str(calSlope[2]) + ','
        fSummary += str(calTemp[0]) + ',' + str(calTemp[1]) + ','
        fSummary += str(calTemp[2]) + ',' + str(meanTemp) + ','
        fSummary += str(nStatic) + ','
        f = '%.2f'
        fSummary += f % xMin + ',' + f % xMax + ','
        fSummary += f % yMin + ',' + f % yMax + ','
        fSummary += f % zMin + ',' + f % zMax + ','
        #raw file data quality indicators
        fSummary += str(os.path.getsize(rawFile)) + ',' + str(getDeviceId(rawFile)) + ','
    except:
        fSummary += '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,'
    f = '%.1f'
    fSummary += str(numInterrupts) + ',' + f % interruptMins + ','
    fSummary += str(numDataErrs) + ','
    fSummary += str(clipsPreCalibrSum) + ',' + str(clipsPreCalibrMax) + ','
    fSummary += str(clipsPostCalibrSum) + ',' + str(clipsPostCalibrMax) + ','
    fSummary += str(epochSamplesN) + ',' + f % epochSamplesAvg + ','
    fSummary += f % epochSamplesStd + ',' + str(epochSamplesMin) + ','
    fSummary += str(epochSamplesMax) + ','
    fSummary += f % tempMean + ',' + f % tempStd + ','
    #PA variable stats
    for m in range(0,len(paWAvg)): # 'm' for (physical activity) metric
        f = '%.2f'
        fSummary += f % (paWAvg[m]*1000) + ','
        fSummary += f % (paWStd[m]*1000) + ','
        fSummary += f % (paAvg[m]*1000) + ','
        fSummary += f % (paStd[m]*1000) + ','
        fSummary += f % (paMedian[m]*1000) + ','
        fSummary += f % (paMin[m]*1000) + ','
        fSummary += f % (paMax[m]*1000) + ','
        f = '%.3f'
        fSummary += ','.join([f % v for v in paEcdf1[m]]) + ','
        fSummary += ','.join([f % v for v in paEcdf2[m]]) + ','
        fSummary += ','.join([f % v for v in paEcdf3[m]]) + ','
        fSummary += ','.join([f % v for v in paEcdf4[m]]) + ','
    fSummary = fSummary[:-1] #remove trailing comma
    #print basic output
    print cmdSummary
    #write detailed output to file
    f = open(summaryFile,'w')
    f.write(fSummary)
    f.close()
    if deleteHelperFiles:
        os.remove(stationaryFile)
        os.remove(epochFile)
    if verbose:
        print summaryFile
        print fSummary


def getEpochSummary(epochFile,
        headerSize,
        dateColumn,
        epochSec,
        tsFile,
        paMetrics):
    """
    Calculate diurnally adjusted average movement per minute from epoch file
    which has had nonWear episodes removed from it
    """
    #use python PANDAS framework to read in and store epochs
    e = pd.read_csv(epochFile, index_col=dateColumn, parse_dates=True,
                header=headerSize)
    #get start & end times, plus wear & nonWear minutes
    startTime = pd.to_datetime(e.index.values[0])
    endTime = pd.to_datetime(e.index.values[-1])
    wearSamples = e[paMetrics[0]].count()
    nonWearSamples = len(e[np.isnan(e[paMetrics[0]])].index.values)
    wearTimeMin = wearSamples * epochSec / 60.0
    nonWearTimeMin = nonWearSamples * epochSec / 60.0
    
    #get wear time in each of 24 hours across week
    epochsInMin = 60 / epochSec
    wear24 = []
    for i in range(0,24):
        wear24.append( e[paMetrics[0]][e.index.hour == i].count() / epochsInMin )
    
    #create arrays to store summary values for various physical activity metrics
    paWAvg = []
    paWStd = []
    paAvg = []
    paStd = []
    paMedian = []
    paMin = []
    paMax = []
    paEcdf1 = []
    paEcdf2 = []
    paEcdf3 = []
    paEcdf4 = []
    for m in paMetrics: # 'm' for (physical activity) metric
        pa = e[m] #raw data
        paW = pa.groupby([e.index.hour, e.index.minute]).mean() #weartime weighted data
      
        #calculate stat summaries
        paWAvg.append(paW.mean())
        paWStd.append(paW.std())
        paAvg.append(pa.mean())
        paStd.append(pa.std())
        paMedian.append(pa.median())
        paMin.append(pa.min())
        paMax.append(pa.max())

        #calculate empirical cumulative distribution function of vector magnitudes
        if m == 'en':
            ecdf = sm.distributions.ECDF(pa)
            x, step = np.linspace(0.905, 0.990, 18, retstep=True) #5mg bins from 901-990mg
            paEcdf1.append(ecdf(x))
            x, step = np.linspace(0.991, 1.010, 20, retstep=True) #1mg bins from 991-1010mg
            paEcdf2.append(ecdf(x))
            x, step = np.linspace(1.015, 1.100, 18, retstep=True) #5mg bins from 1015-1100mg
            paEcdf3.append(ecdf(x))
            x, step = np.linspace(1.2, 3.0, 19, retstep=True) #100mg bins from 1200-3000mg
            paEcdf4.append(ecdf(x))
        else:
            ecdf = sm.distributions.ECDF(pa)
            x, step = np.linspace(0.001, 0.020, 20, retstep=True) #1mg bins from 1-20mg
            paEcdf1.append(ecdf(x))
            x, step = np.linspace(0.025, 0.100, 16, retstep=True) #5mg bins from 25-100mg
            paEcdf2.append(ecdf(x))
            x, step = np.linspace(0.125, 0.500, 16, retstep=True) #25mg bins from 125-500mg
            paEcdf3.append(ecdf(x))
            x, step = np.linspace(0.6, 2.0, 15, retstep=True) #100mg bins from 500-2000mg
            paEcdf4.append(ecdf(x))

    #write time series file
    #replace nan avgVm vals with mean avgVm from same time in other days
    e['hour'] = e.index.hour
    e['minute'] = e.index.minute
    pa = paMetrics[0]
    ts = e.join(e.groupby(('hour','minute'))[pa].mean(), on=['hour','minute'], rsuffix='_imputed')
    ts['vm'] = ts[pa].fillna(ts[pa + '_imputed'])
    #convert 'vm' to mg units, and highlight any imputed values
    ts['vmFinal'] = ts['vm'] * 1000
    ts['imputed'] = np.isnan(ts[pa]).replace({True:'1',False:''})
    #prepare time series header
    tsHead = 'acceleration (mg) - '
    tsHead += ts.index.min().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += ts.index.max().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += 'sampleRate = ' + str(epochSec) + ' seconds'
    ts[['vmFinal','imputed']].to_csv(tsFile, float_format='%.1f',index=False,header=[tsHead,'imputed'])
   
    #get interrupt and data error summary vals
    epochNs = epochSec * np.timedelta64(1,'s')
    interrupts = np.where(np.diff(np.array(e.index)) > epochNs)[0]
    #get duration of each interrupt in minutes
    interruptMins = []
    for i in interrupts:
        interruptMins.append(np.diff(np.array(e[i:i+2].index)) / np.timedelta64(1,'m'))

    #return physical activity summary
    return startTime, endTime, wearTimeMin, nonWearTimeMin, wear24, paW.count(), len(interrupts), np.sum(interruptMins), e['dataErrors'].sum(), e['clipsBeforeCalibr'].sum(), e['clipsBeforeCalibr'].max(), e['clipsAfterCalibr'].sum(), e['clipsAfterCalibr'].max(), e['samples'].sum(), e['samples'].mean(), e['samples'].std(), e['samples'].min(), e['samples'].max(), e['temp'].mean(), e['temp'].std(), paWAvg, paWStd, paAvg, paStd, paMedian, paMin, paMax, paEcdf1, paEcdf2, paEcdf3, paEcdf4


def identifyAndRemoveNonWearTime(
            epochFile,
            nonWearEpisodesFile,
            funcParams,
            epochSec):
    """
    Identify and remove nonWear episodes from an epoch CSV file
    Inputs:
    - epochFile: an epoch .csv file
    - nonWearEpisodesFile: path to write nonWearBouts.csv file to
    - funcParams: an array of [<name>:<value>] items, specifically:
        [nonWearEpisodesOutputFile:<name.csv>], default = <epochFile>_mvpa_bout_list.csv
        [headerSize:<lines>], default = 1 
        [datetimeColumn:<int>], default = 0, index of datetime column
        [timeFormat:<python_timeFormat_string>], default = '%Y-%m-%d %H:%M:%S.%f'
        [xIndex:<int>], default = 8
        [yIndex:<int>], default = 9
        [zIndex:<int>], default = 10
        [targetWearTimeDays:<int>], default = 28
        [behavType:<string>], default = 'nonwear'
        [minFreq:<int>], default = 60, min num epochs in episode
        [maxRange:<float>], default = 0.013, movement below this indicates nonwear
        [graceMaxFreq:<int>], default = 0, max num "grace" epochs in episode outside <maxRange> thresholds
        [displayOutput:<bool>], default = False
    Output:
    - new file created (funcParams 'nonWearEpisodesOutputFile')
    """
    '''
    Firstly determine parameters to influence the calculation of epochs
    '''
    #variables to store default parameter options
    headerSize = 1
    datetimeColumn, xIndex, yIndex, zIndex = 0, 11, 12, 13
    timeFormat = '%Y-%m-%d %H:%M:%S.%f'
    targetWearTimeDays, behavType = 28, 'nonwear'
    minFreq = 3600 / epochSec
    maxRange, graceMaxFreq = 0.013, 0
    displayOutput = False
    #update default values by looping through available user parameters
    for param in funcParams:
        #param will look like 'nonWearEpisodesOutputFile:aidenNonWearBouts.csv'
        if param.split(':')[0] == 'nonWearEpisodesFile':
            nonWearEpisodesOutputFile = param.split(':')[1]
        elif param.split(':')[0] == 'headerSize':
            headerSize = int(param.split(':')[1])
        elif param.split(':')[0] == 'datetimeColumn':
            datetimeColumn = int(param.split(':')[1])
        elif param.split(':')[0] == 'timeFormat':
            timeFormat = param.replace('timeFormat:','')
        elif param.split(':')[0] == 'xIndex':
            xIndex = int(param.split(':')[1])
        elif param.split(':')[0] == 'yIndex':
            yIndex = int(param.split(':')[1])
        elif param.split(':')[0] == 'zIndex':
            zIndex = int(param.split(':')[1])
        elif param.split(':')[0] == 'targetWearTimeDays':
            targetWearTimeDays = int(param.split(':')[1])
        elif param.split(':')[0] == 'behavType':
            behavType = param.split(':')[1]
        elif param.split(':')[0] == 'minFreq':
            minFreq = int(param.split(':')[1])
        elif param.split(':')[0] == 'maxRange':
            maxRange = float(param.split(':')[1])
        elif param.split(':')[0] == 'graceMaxFreq':
            graceMaxFreq = int(param.split(':')[1])
        elif param.split(':')[0] == 'displayOutput':
            displayOutput = param.split(':')[1] in ['true', 'True']
    #now calculate nonwear episodes and store to list
    episodesList, firstDay, lastDay = behaviourEpisode.identifyNonWearEpisodes(
                    epochFile, headerSize, datetimeColumn, timeFormat, xIndex, yIndex,
                    zIndex, targetWearTimeDays, behavType, minFreq, maxRange, 
                    graceMaxFreq)
    #print summary of each nonwear episode detected, returning sum nonwear time
    sumNonWear, numNonWearEpisodes = behaviourEpisode.writeSummaryOfEpisodes(
                    nonWearEpisodesFile, episodesList, displayOutput)
    removeNonWearFromEpochFile(epochFile,episodesList,headerSize,timeFormat)
    return numNonWearEpisodes


def removeNonWearFromEpochFile(
            epochFile,
            nonWearEpisodes,
            headerSize,
            timeFormat):
    """
    Replace any nonWear episodes in the epochFile with null values
    """
    #only run if there is nonWear data to remove
    if len(nonWearEpisodes) > 0:
        f = open(epochFile,'rU')
        epochs = f.readlines() #read file into memory
        f.close()
        f = open(epochFile,'w')
        
        #rewrite header lines
        for headerLine in epochs[:headerSize]:
            f.write(headerLine)
        
        #rewrite all epochs that are periods of wear
        nans = ',,,,,,,,,,,,,,,,,,\n'
        episodeCounter = 0
        for epoch in epochs[headerSize:]:
            epochTime = datetime.datetime.strptime(epoch.split(',')[0],timeFormat)
            #write epoch if it is a period of wear i.e. it is not after the
            #   startTime of next nonWear episode, or it is after endTime of
            #   last nonWear episode
            if ( epochTime < nonWearEpisodes[episodeCounter].startTime or
                    (epochTime > nonWearEpisodes[episodeCounter].endTime and 
                    episodeCounter == len(nonWearEpisodes)-1 ) ):
                f.write(epoch)
            elif ( epochTime >= nonWearEpisodes[episodeCounter].startTime and 
                    epochTime <= nonWearEpisodes[episodeCounter].endTime ):
                f.write(epochTime.strftime(timeFormat) + nans)
            #move counter to next nonWear episode if at end of current episode
            if ( epochTime == nonWearEpisodes[episodeCounter].endTime and 
                    episodeCounter < len(nonWearEpisodes)-1 ):
                f.write(epochTime.strftime(timeFormat) + nans)
                episodeCounter += 1
        f.close()


def getCalibrationCoefs(staticBoutsFile):
    """
    Get axes offset/gain/temp calibration coefficients through linear regression
    of stationary episodes
    """
    #learning/research parameters
    maxIter = 1000
    minIterImprovement = 0.0001 #0.1mg
    #use python NUMPY framework to store stationary episodes from epoch file
    d = np.loadtxt(open(staticBoutsFile,"rb"),delimiter=",",skiprows=1,usecols=(5,6,7,14,16))
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
    return bestIntercept, bestSlope, bestTemp, meanTemp, initError, bestError, xMin, xMax, yMin, yMax, zMin, zMax, len(axesVals)

def getDeviceId(cwaFile):
    f = open(cwaFile, 'rb')
    header = f.read(2)
    if header == 'MD':
        blockSize = struct.unpack('H', f.read(2))[0]
        performClear = struct.unpack('B', f.read(1))[0]
        deviceId = struct.unpack('H', f.read(2))[0]
    f.close()
    return deviceId


"""
Standard boilerplate to call the main() function to begin the program.
"""
if __name__ == '__main__': 
    main() #Standard boilerplate to call the main() function to begin the program.
