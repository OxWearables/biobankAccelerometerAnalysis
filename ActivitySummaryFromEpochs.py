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
    python ActivitySummaryFromEpochs.py <input_file.CWA> <options>
e.g.
    python ActivitySummaryFromEpochs.py p001.CWA 
    python ActivitySummaryFromEpochs.py p001.CWA min_freq:10 
"""

import sys
import os
import datetime
import behaviourEpisode
import pandas as pd
from subprocess import call, Popen

def main():
    """
    Application entry point responsible for parsing command line requests
    """
    #check that enough command line arguments are entered
    if len(sys.argv)<2:
        msg = "\n Invalid input, please enter at least 1 parameter, e.g."
        msg += "\n python ActivitySummaryFromEpoch.py inputFile.CWA \n"
        print msg
        sys.exit(0)
    #store command line arguments to local variables
    rawFile = sys.argv[1]      
    funcParams = sys.argv[2:]
    wavFile = rawFile.replace(".CWA", ".wav").replace(".cwa",".wav")
    epochFile = wavFile.replace(".wav","Epoch.csv")
    matlabPath = "matlab"
    skipMatlab = False
    skipJava = False
    deleteWav = False
    epochPeriodStr = "epochPeriod:60"
    #update default values by looping through user parameters
    for param in funcParams:
        #example param -> 'matlab:/Applications/MATLAB_R2014a.app/bin/matlab'
        if param.split(':')[0] == 'matlab':
            matlabPath = param.split(':')[1]
        elif param.split(':')[0] == 'skipMatlab':
            skipMatlab = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'deleteWav':
            deleteWav = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'skipJava':
            skipJava = param.split(':')[1] in ['true', 'True']
        elif param.split(':')[0] == 'epochPeriod':
            epochPeriodStr = param

    #check source cwa file exists
    if not os.path.isfile(rawFile):
        msg = "\n Invalid input"
        msg += "\n File does not exist: " + rawFile + "\n"
        print msg
        sys.exit(0)

    #interpolate and calibrate raw .CWA file, writing output to .wav file
    commandArgs = [matlabPath, "-nosplash",
            "-nodisplay", "-r", "cd matlab;readInterpolateCalibrate('" + rawFile
            + "', '" + wavFile + "');exit;"]
    if not skipMatlab:
        call(commandArgs)
    
    #calculate and write filtered AvgVm epochs from .wav file
    commandArgs = ["java", "-mx256m", "AxivityAx3WavEpochs", wavFile, "outputFile:" + 
            epochFile, "filter:true", epochPeriodStr]
    if not skipJava:
        call(commandArgs)
    if deleteWav:
        os.remove(wavFile)

    #identify and remove nonWear episodes
    firstDay, lastDay, wearTime, sumNonWear, numNonWearEpisodes = identifyAndRemoveNonWearTime(
            epochFile, funcParams)    
    
    #calculate average sample score (diurnally adjusted)
    avgSampleVm = getAverageVmMinute(epochFile,0,0)

    #print processed summary variables from accelerometer file
    outputSummary = rawFile + ',' + str(avgSampleVm) + ',' + str(firstDay) + ','
    outputSummary += str(lastDay) + ',' + str(wearTime) + ','
    outputSummary += str(sumNonWear) + ',' + str(numNonWearEpisodes)
    f = open(rawFile.replace(".cwa","OutputSummary.csv"),'w')
    f.write(outputSummary)
    f.close()
    print outputSummary



def getAverageVmMinute(epochFile,headerSize,dateColumn):
    """
    Calculate diurnally adjusted average movement per minute from epoch file
    which has had nonWear episodes removed from it
    """
    #use python PANDAS framework to read in and store epochs
    e = pd.read_csv(epochFile, index_col=dateColumn, parse_dates=True,
                header=headerSize)
    #diurnal adjustment: construct average 1440 minute day
    avgDay = e[['AvgVm']].groupby([e.index.hour, e.index.minute]).mean()
    #return average minute score
    return avgDay.mean()[0]


def identifyAndRemoveNonWearTime(epochFile, funcParams):
    """
    Identify and remove nonWear episodes from an epoch CSV file
    Inputs:
    - epochFile: an epoch .csv file
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
    nonWearEpisodesOutputFile = epochFile.split('.')[0] + 'NonWearBouts.csv'
    headerSize = 1
    datetimeColumn, xIndex, yIndex, zIndex = 0, 8, 9, 10
    timeFormat = '%Y-%m-%d %H:%M:%S.%f'
    targetWearTimeDays, behavType = 28, 'nonwear'
    minFreq, maxRange, graceMaxFreq = 60, 0.013, 0
    displayOutput = False
    #update default values by looping through available user parameters
    for param in funcParams:
        #param will look like 'nonWearEpisodesOutputFile:aidenNonWearBouts.csv'
        if param.split(':')[0] == 'nonWearEpisodesOutputFile':
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
                    nonWearEpisodesOutputFile, episodesList, displayOutput)
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

"""
Standard boilerplate to call the main() function to begin the program.
"""
if __name__ == '__main__': 
    main() #Standard boilerplate to call the main() function to begin the program.
