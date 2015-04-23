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
    wavFile = rawFile
    stationaryFile = rawFile.replace(".cwa","Stationary.csv")
    epochFile = rawFile.replace(".cwa","Epoch.csv")
    nonWearFile = rawFile.replace(".cwa","NonWearBouts.csv")
    tsFile = rawFile.replace(".cwa","AccTimeSeries.csv")
    matlabPath = "matlab"
    javaEpochProcess = "AxivityAx3Epochs"
    skipRaw = False
    skipMatlab = True
    skipCalibration = False
    skipJava = False
    deleteWav = False
    deleteHelperFiles = True
    verbose = True
    epochSec = 5
    epochPeriodStr = "epochPeriod:" + str(epochSec)
    #update default values by looping through user parameters
    for param in funcParams:
        #example param -> 'matlab:/Applications/MATLAB_R2014a.app/bin/matlab'
        if param.split(':')[0] == 'matlabPath':
            matlabPath = param.split(':')[1]
        elif param.split(':')[0] == 'summaryFolder':
            summaryFile = param.split(':')[1] + summaryFile.split('/')[-1]
        elif param.split(':')[0] == 'wavFolder':
            wavFile = param.split(':')[1] + wavFile.split('/')[-1]
        elif param.split(':')[0] == 'epochFolder':
            epochFile = param.split(':')[1] + epochFile.split('/')[-1]
        elif param.split(':')[0] == 'nonWearFolder':
            nonWearFile = param.split(':')[1] + nonWearFile.split('/')[-1]
        elif param.split(':')[0] == 'stationaryFolder':
            stationaryFile = param.split(':')[1] + stationaryFile.split('/')[-1]
        elif param.split(':')[0] == 'skipRaw':
            skipRaw = param.split(':')[1] in ['true', 'True']
            if skipRaw:
                skipMatlab = True
                skipCalibration = True
                skipJava = True
        elif param.split(':')[0] == 'skipMatlab':
            skipMatlab = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'skipCalibration':
            skipCalibration = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'verbose':
            verbose = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'deleteWav':
            deleteWav = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'deleteHelperFiles':
            deleteHelperFiles = param.split(':')[1] in ['true', 'True']
        if param.split(':')[0] == 'skipJava':
            skipJava = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'epochPeriod':
            epochPeriodStr = param

    #check source cwa file exists
    if not skipRaw and not os.path.isfile(rawFile):
        msg = "\n Invalid input"
        msg += "\n File does not exist: " + rawFile + "\n"
        sys.stderr.write(msg)
        sys.exit(0)

    if not skipMatlab:
        wavFile = rawFile.replace(".cwa",".wav")
        #interpolate and calibrate raw .CWA file, writing output to .wav file
        commandArgs = [matlabPath, "-nosplash",
                "-nodisplay", "-r", "cd matlab;readInterpolateCalibrate('" + rawFile
                + "', '" + wavFile + "');exit;"]
        call(commandArgs)
        javaEpochProcess = "AxivityAx3WavEpochs"
    
    #calibrate axes scale/offset values
    if not skipCalibration:
        #identify 10sec stationary epochs
        commandArgs = ["java", "-XX:ParallelGCThreads=1", javaEpochProcess,
                rawFile, "outputFile:" + stationaryFile, "verbose:" + str(verbose),
                "filter:true", "getStationaryBouts:true", "epochPeriod:10",
                "stationaryStd:0.013"]
        call(commandArgs)
        #record calibrated axes scale/offset/temperature vals + static point stats
        calOff, calSlope, calTemp, meanTemp, errPreCal, errPostCal, xMin, xMax, yMin, yMax, zMin, zMax = getCalibrationCoefs(stationaryFile)
        if verbose:
            print calOff, calSlope, calTemp, meanTemp, errPreCal, errPostCal, xMin, xMax, yMin, yMax, zMin, zMax
        commandArgs = ["java", "-XX:ParallelGCThreads=1", javaEpochProcess,
                wavFile, "outputFile:" + epochFile, "verbose:" + str(verbose), "filter:true", 
                "xIntercept:" + str(calOff[0]), "yIntercept:" + str(calOff[1]),
                "zIntercept:" + str(calOff[2]), "xSlope:" + str(calSlope[0]),
                "ySlope:" + str(calSlope[1]), "zSlope:" + str(calSlope[2]),
                "xTemp:" + str(calTemp[0]), "yTemp:" + str(calTemp[1]),
                "zTemp:" + str(calTemp[2]), "meanTemp:" + str(meanTemp),
                epochPeriodStr]
    else: 
        commandArgs = ["java", "-XX:ParallelGCThreads=1", javaEpochProcess,
                wavFile, "outputFile:" + epochFile, "verbose:" + str(verbose), 
                "filter:true", epochPeriodStr]
  
    #calculate and write filtered avgVm epochs from .wav file
    if not skipJava:
        call(commandArgs)
    if deleteWav:
        os.remove(wavFile)

    #get stats on interrupts observed in epoch file (done before nonWear removal)
    numInterrupts, interruptMins, numDataErrs = getInterruptsSummary(epochFile, 0, 0, epochSec)

    #identify and remove nonWear episodes
    firstDay, lastDay, wearTime, sumNonWear, numNonWearEpisodes = identifyAndRemoveNonWearTime(
            epochFile, nonWearFile, funcParams, epochSec)    
    
    #calculate average, median, stdev, min, max, count, & ecdf of sample score in
    #1440 min diurnally adjusted day. Also get overall wear time minutes across
    #week in quadrants (0-6h, 6-12, 12-18, 18-24)
    avgSampleVm, medianVm, stdevVm, minVm, maxVm, countVm, q1Wear, q2Wear, q3Wear, q4Wear, wear24, clipsPreCalibrSum, clipsPreCalibrMax, clipsPostCalibrSum, clipsPostCalibrMax, samplesSum, samplesMean, samplesStd, tempMean, tempStd, ecdfLow, ecdfMid, ecdfHigh = getEpochSummary(epochFile, 0, 0, epochSec, tsFile)

    #print processed summary variables from accelerometer file
    outputSummary = rawFile + ','
    try:
        outputSummary += str(os.path.getsize(rawFile)) + ',' + str(getDeviceId(rawFile)) + ','
    except:
        outputSummary += '-1,-1,'
    outputSummary += str(avgSampleVm) + ',' + str(medianVm) + ','
    outputSummary += str(stdevVm) + ',' + str(minVm) +',' + str(maxVm) + ','
    outputSummary += str(countVm) + ','
    outputSummary += str(firstDay)[:-3] + ',' + str(lastDay)[:-3] + ','
    outputSummary += str(wearTime) + ',' + str(sumNonWear) + ','
    outputSummary += str(numNonWearEpisodes) + ',' + str(q1Wear) + ','
    outputSummary += str(q2Wear) + ',' + str(q3Wear) + ',' + str(q4Wear) + ','
    for i in range(0,24):
        outputSummary += str(wear24[i]) + ','
    try:
        outputSummary += str(errPreCal) + ',' + str(errPostCal) + ','
        outputSummary += str(calOff[0]) + ',' + str(calOff[1]) + ','
        outputSummary += str(calOff[2]) + ',' + str(calSlope[0]) + ','
        outputSummary += str(calSlope[1]) + ',' + str(calSlope[2]) + ','
        outputSummary += str(calTemp[0]) + ',' + str(calTemp[1]) + ','
        outputSummary += str(calTemp[2]) + ',' + str(meanTemp) + ','
        outputSummary += str(xMin) + ',' + str(xMax) + ',' + str(yMin) + ','
        outputSummary += str(yMax) + ',' + str(zMin) + ',' + str(zMax) + ','
    except:
        outputSummary += '-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,'
    outputSummary += str(numInterrupts) + ',' + str(interruptMins) + ','
    outputSummary += str(numDataErrs) + ','
    outputSummary += str(clipsPreCalibrSum) + ',' + str(clipsPreCalibrMax) + ','
    outputSummary += str(clipsPostCalibrSum) + ',' + str(clipsPostCalibrMax) + ','
    outputSummary += str(samplesSum) + ',' + str(samplesMean) + ','
    outputSummary += str(samplesStd) + ','
    outputSummary += str(tempMean) + ',' + str(tempStd) + ','
    outputSummary += ','.join(map(str,ecdfLow)) + ','
    outputSummary += ','.join(map(str,ecdfMid)) + ','
    outputSummary += ','.join(map(str,ecdfHigh))
    f = open(summaryFile,'w')
    f.write(outputSummary)
    f.close()
    if deleteHelperFiles:
        os.remove(stationaryFile)
        os.remove(epochFile)
    if verbose:
        print summaryFile
        print outputSummary


def getEpochSummary(epochFile, headerSize, dateColumn, epochSec, tsFile):
    """
    Calculate diurnally adjusted average movement per minute from epoch file
    which has had nonWear episodes removed from it
    """
    #use python PANDAS framework to read in and store epochs
    e = pd.read_csv(epochFile, index_col=dateColumn, parse_dates=True,
                header=headerSize)
    #diurnal adjustment: construct average 1440 minute day
    avgDay = e['avgVm'].groupby([e.index.hour, e.index.minute]).mean()
    #get wear time in each daily quadrant (0-6h,6-12,12-18,18-24) across week
    epochsInMin = 60 / epochSec
    q1Wear = e['avgVm'][e.index.hour<6].count() / epochsInMin
    q2Wear = e['avgVm'][(e.index.hour>=6) & (e.index.hour<12)].count() / epochsInMin
    q3Wear = e['avgVm'][(e.index.hour>=12) & (e.index.hour<18)].count() / epochsInMin
    q4Wear = e['avgVm'][e.index.hour>=18].count() / epochsInMin
    wear24 = []
    for i in range(0,24):
        wear24.append( e['avgVm'][e.index.hour == i].count() / epochsInMin )
    #calculate empirical cumulative distribution function of vector magnitudes
    ecdf = sm.distributions.ECDF(e['avgVm'])
    #1mg categories from 1-100mg
    x, step = np.linspace(0.001, .100, 100, retstep=True)
    ecdfLow = ecdf(x)
    #10mg categories from 110mg to 1g 
    x, step = np.linspace(.110, 1.0, 90, retstep=True)
    ecdfMid = ecdf(x)
    #100mg categories from 1g to 3g
    x, step = np.linspace(1.1, 3.0, 20, retstep=True)
    ecdfHigh = ecdf(x)
    #write time series file
    tsHead = 'acceleration (mg) - '
    tsHead += e.index.min().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += e.index.max().strftime('%Y-%m-%d %H:%M:%S') + ' - '
    tsHead += 'sampleRate = ' + str(epochSec) + ' seconds'
    e['acc']=e['avgVm']*1000
    e['acc'].to_csv(tsFile, float_format='%.1f',index=False,header=[tsHead])
    #return physical activity summary
    return avgDay.mean(), avgDay.median(), avgDay.std(), avgDay.min(), avgDay.max(), avgDay.count(), q1Wear, q2Wear, q3Wear, q4Wear, wear24, e['clipsBeforeCalibr'].sum(), e['clipsBeforeCalibr'].max(), e['clipsAfterCalibr'].sum(), e['clipsAfterCalibr'].max(), e['samples'].sum(), e['samples'].mean(), e['samples'].std(), e['temp'].mean(), e['temp'].std(), ecdfLow, ecdfMid, ecdfHigh


def getInterruptsSummary(epochFile, headerSize, dateColumn, epochSec):
    """
    Summaryise any interrupts in epoch file before nonWear episodes are removed
    """
    e = pd.read_csv(epochFile, index_col=dateColumn, parse_dates=True,
                header=headerSize)
    epochNs = epochSec * np.timedelta64(1,'s')
    interrupts = np.where(np.diff(np.array(e.index)) > epochNs)[0]
    #get duration of each interrupt in minutes
    dur = []
    for i in interrupts:
        dur.append(np.diff(np.array(e[i:i+2].index)) / np.timedelta64(1,'m'))
    return len(interrupts), np.sum(dur), e['dataErrors'].sum()


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
    datetimeColumn, xIndex, yIndex, zIndex = 0, 8, 9, 10
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
    #calculate max possible wear time in minutes (pre Python 2.7 compatible)
    wearTime = ((lastDay-firstDay).days*3600*24) + ((lastDay - firstDay).seconds)
    wearTime = wearTime / 60 #convert from seconds to minutes
    wearTime -= sumNonWear #total wear = max possible wear - nonWear
    removeNonWearFromEpochFile(epochFile,episodesList,headerSize,timeFormat)
    return firstDay, lastDay, wearTime, sumNonWear, numNonWearEpisodes


def removeNonWearFromEpochFile(
            epochFile,
            nonWearEpisodes,
            headerSize,
            timeFormat):
    """
    Remove any nonWear episodes from the epochFile
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
            #move counter to next nonWear episode if at end of current episode
            elif ( epochTime == nonWearEpisodes[episodeCounter].endTime and 
                    episodeCounter < len(nonWearEpisodes)-1 ):
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
    d = np.loadtxt(open(staticBoutsFile,"rb"),delimiter=",",skiprows=1,usecols=(2,3,4,11,13))
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
    return bestIntercept, bestSlope, bestTemp, meanTemp, initError, bestError, xMin, xMax, yMin, yMax, zMin, zMax

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
