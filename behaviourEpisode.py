#BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
"""
This python object:
    1) has attributes to represent an episode of behaviour.
        e.g. behavType, startTime, endTime, value 
    2) static methods to identify behaviour episodes from time series data
        e.g. to calculate nonWear 
e.g.
    nonWearBout = new behaviourEpisode('nonWear', '2013-11-16 02:28',
            '2013-11-16 02:48', 0.01, 0.01, 0.01)
    nonWearBouts = behaviourEpisode.identify_episodes(input.csv, 'nonWear', 60, 0, 15000, 2, 0, 1809)
"""

import sys
import datetime
from collections import deque

class behaviourEpisode:
        
    def __init__(self, behavType, startTime, endTime, xStd, yStd, zStd):
        """
        Constructor to initialise object attributes 
        """
        self.behavType = behavType #string
        self.startTime = startTime #datetime
        self.endTime = endTime
        self.xStd = xStd #float
        self.yStd = yStd
        self.zStd = zStd

    def __init__(self, behavType):
        """
        Simple constructor
        """
        self.behavType = behavType

    #class variable to store getSummaryNonWear() header information
    summaryHeaderNonWear = 'behaviourType,startTime,endTime,xStd,yStd,zStd'
        
    def getSummaryNonWear(self):
        output = self.behavType + ',' + str(self.startTime) +',' +str(self.endTime)
        output += ',' + str(self.xStd) + ',' + str(self.yStd)
        output += ',' + str(self.zStd)
        return output

'''
Static methods to identify behaviours of interest from epoch csv data
'''

def identifyNonWearEpisodes(epochFile, headerSize, timeCol,
                        timeFormat, xIndex, yIndex, zIndex, targetWeartimeDays,
                        behavType, minFreq, maxStd, minNumAxes, graceMaxFreq):
    """
    This method reads through <epochFile.csv> and identifies episodes of <behavType>
    as being a bout/episode of at least <minFreq> minutes with all 2/3 axes
    having a standard deviation below <maxStd> except for some allowed minutes of
    grace <graceMaxFreq>
    This processing method is based on that of van Hees, Ekelund, Brage and colleagues
    in PLos ONE 2011 6(7), "Estimation of Daily Energy Expenditure in Pregnant and
    Non-Pregnant Women Using a Wrist-Worn Tri-Axial Accelerometer"
    ===
    To identify episodes when reading each line/epoch of CSV, consider:
        1. Is this epoch a potential episode start point (need to see x items ahead)?
            a) i.e. is the value std, for at least <minNumAxes> out of three axes, < maxStd
            b) if so create a new 'behaviourEpisode'
        -----
        2. Is this epoch a potential episode ending point?
            a) i.e. is the value std, for at least <minNumAxes> out of three axes, > maxStd
            b) if so complete info for 'behaviourEpisode' (see 1b) and store it
        -----
        3. Is this epoch part of an existing episode?
            a) i.e. is there a start point and are current axes still < maxStd
            b) if so add the datetime & unit-count info to episode running total lists
    ===
    Inputs:
        epochFile, csv file, already summarised into epochs, e.g. input.csv
        headerSize, int, num lines in <epochFile> header
        timeCol, int, index of datetime column in CSV
        timeFormat, string, format of <epochFile> datetime values
        (x/y/z)Index, int, index of column in CSV relating to std of x/y/z values in epoch
        targetWeartimeDays, int,
        behavType, string, label episodes found e.g. 'MVPA'
        minFreq, int, how many consecutive instances each bout/episode should be e.g. 10
        maxStd, int, upper bound axis std e.g. 0.05 (50mg)
        minNumAxes, int
        graceMaxFreq, int, how many grace occurrences can be outside lower/upper bound e.g. 2
    Output:
        array of identified episodes, of type 'behaviourEpisode'
    """
    '''
    setup variables to help process & store newly detected episodes
    '''
    episodeList = [] #list of type 'behaviourEpisode' for method output
    #to detect start/end of episodes
    seekingNewEpisode = True #bool
    timesQ = deque() #datetime queue
    xQ = deque() #float queue
    yQ = deque()
    zQ = deque()
    #to store candidate episode info
    newEpisode = behaviourEpisode(behavType) #type 'behaviourEpisode'
    xVals = [] #float
    yVals = []
    zVals = []
    #to extract info from CSV <epochFile>
    lineParts = [] #string list
    epochTime = datetime.datetime.now()
    '''
    first determine start and end time of raw data to consider (i.e target days of wear)
    '''
    firstDay = datetime.datetime.now()
    lastDay = datetime.datetime.now()
    epochReader = open(epochFile,'rU') #open file reader...
    #ignore the first <headerSize> lines
    for counter in range(headerSize):
        next(epochReader)
    #get firstDay/epoch
    lineParts = next(epochReader).split(',')
    firstDay = datetime.datetime.strptime(lineParts[timeCol],timeFormat)
    #traverse through rest of CSV file
    for line in epochReader:
        lineParts = line.split(',')
    lastDay = datetime.datetime.strptime(lineParts[timeCol],timeFormat)
    #print 'total data: ', firstDay, lastDay
    #calculate if more days data collected than initially targetted
    dayDiff = (((lastDay-firstDay).days*3600*24) + ((lastDay-firstDay).seconds))
    dayDiff = dayDiff / 3600 / 24.0 #convert to days
    if dayDiff > targetWeartimeDays:
        #if we have more days than needed, ignore first day
        #then take next <targetWeartimeDays>
        firstDay = datetime.datetime(firstDay.year, firstDay.month, firstDay.day, 0, 0, 0)
        firstDay += datetime.timedelta(days=1)
        lastDay = firstDay + datetime.timedelta(days=targetWeartimeDays)
        lastDay += datetime.timedelta(microseconds=-1)
    #print 'considered data: ', firstDay, lastDay
    #finally close file reader as we're at the end of it
    epochReader.close()
    '''
    Now traverse through entire file, to identify episodes (i.e. start/end boundaries)
    '''
    xStd = 0.0
    yStd = 0.0
    zStd = 0.0
    epochReader = open(epochFile,'rU')
    #ignore the first <headerSize> lines
    for counter in range(headerSize):
        next(epochReader)
    #read in first <minFreq> instances from file to initialise queue
    while len(xQ) < minFreq:
        try:
            lineParts = next(epochReader).split(',') #read next from file
        except:
            sys.stderr.write('insufficient epoch data to identify episodes\n')
            break
        epochTime = datetime.datetime.strptime(lineParts[timeCol],timeFormat)
        if epochTime >= firstDay and epochTime <= lastDay:
            xStd = float(lineParts[xIndex])
            yStd = float(lineParts[yIndex])
            zStd = float(lineParts[zIndex])
            timesQ.append(epochTime)
            xQ.append(xStd)
            yQ.append(yStd)
            zQ.append(zStd)
    #now traverse through rest of CSV file, to identify episodes
    for line in epochReader:
        #file reading item: add the latest item to the end of the running queue
        lineParts = line.split(',')
        epochTime = datetime.datetime.strptime(lineParts[timeCol],timeFormat)
        if epochTime >= firstDay and epochTime <= lastDay:
            xStd = float(lineParts[xIndex])
            yStd = float(lineParts[yIndex])
            zStd = float(lineParts[zIndex])
            timesQ.append(epochTime)
            xQ.append(xStd)
            yQ.append(yStd)
            zQ.append(zStd)
            #remove first item from queue to maintain size
            timesQ.popleft()
            xQ.popleft()
            yQ.popleft()
            zQ.popleft()
            '''
            1. Calculate if this epoch starts a new episode by processing <minFreq>
                        items ahead to check it stays in bounds
            '''
            if seekingNewEpisode:          
                #check if each axis has a stdev of less than <maxStd>
                nonWearStart = [] #bool list on whether each axes meets start criteria
                nonWearStart.append(isNonWearStart(xQ, maxStd, graceMaxFreq))
                nonWearStart.append(isNonWearStart(yQ, maxStd, graceMaxFreq))
                nonWearStart.append(isNonWearStart(zQ, maxStd, graceMaxFreq))
                #it's a new nonwear episode if >=minNumAxes axes satisfy start episode criteria
                # and also if the proposed start time occurs after prev episode end-time
                if sum(nonWearStart) >=minNumAxes and (len(episodeList) == 0 or timesQ[0] > episodeList[-1].endTime):
                    newEpisode = behaviourEpisode(behavType)
                    newEpisode.startTime = timesQ[0]
                    del xVals[:] #reset value arrays
                    del yVals[:]
                    del zVals[:]
                    xVals.extend(list(xQ))
                    yVals.extend(list(yQ))
                    zVals.extend(list(zQ)) 
                    seekingNewEpisode = False #flag we no longer seek nonWear start 
                '''
                2. Find episode end-time, provided we not seeking to identify start time
                        End-time epoch value is out of bounds and not considered a "grace" value
                '''
            elif not seekingNewEpisode:            
                #check if each axis appears to end an episode (i.e. std >= maxStd)
                nonWearEnd = [] #bool list on whether each axes meets end criteria
                nonWearEnd.append(isNonWearEnd(xStd, False, xQ, maxStd, graceMaxFreq))
                nonWearEnd.append(isNonWearEnd(xStd, False, xQ, maxStd, graceMaxFreq))
                nonWearEnd.append(isNonWearEnd(xStd, False, xQ, maxStd, graceMaxFreq))
                #it's end of nonwear episode if minNumAxes satisfies end episode criteria
                if sum(nonWearEnd) >=minNumAxes:
                    seekingNewEpisode = True #overall loop: look for nonWear start again
                    #determine if any grace values at list end
                    numEndValsToTrim = numOutliersAtListEnd(xVals, 0, maxStd)
                    #if there are grace values at list end, trim them off
                    if numEndValsToTrim > 0:
                        del xVals[-numEndValsToTrim:] #trim from values list
                        del yVals[-numEndValsToTrim:]
                        del zVals[-numEndValsToTrim:]
                    #if still valid episode (duration), record it, and add to episodes list
                    if len(xVals) >= minFreq:
                        #epoch[-2] is true end-time, curr epoch[-1] not part of episode
                        newEpisode.endTime = timesQ[-2 - numEndValsToTrim]
                        newEpisode.xStd = max(xVals)
                        newEpisode.yStd = max(yVals)
                        newEpisode.zStd = max(zVals)
                        episodeList.append(newEpisode)
                    '''
                    3. Else epoch is part of ongoing nonwear episode, therefore log values
                    '''
                else:
                    xVals.append(xStd)
                    yVals.append(yStd)
                    zVals.append(zStd)
    '''
    In case there is an episode at the end of file, record it after loop above
    '''
    #if not seeking a new episode start time
    #   i.e. we determined a valid episode started towards end of file
    if not seekingNewEpisode:
        #if a valid episode end, record episode info, and add to list
        newEpisode.endTime = epochTime
        newEpisode.xStd = max(xVals)
        newEpisode.yStd = max(yVals)
        newEpisode.zStd = max(zVals)
        episodeList.append(newEpisode)
    '''
    finally, return list of identified episodes, and first/last considered readings
    '''
    return episodeList, firstDay, lastDay

def isNonWearStart(valsQ, maxStd, graceMaxFreq):
    """
    Method checks if <valsQ> starts a nonWear episode
    """
    return isEpisodeStart(valsQ, 0, maxStd, graceMaxFreq, maxStd+0.00001, float("inf"))

def isEpisodeStart(valsQ, minCount, maxCount, 
                        graceMaxFreq, graceMinCount, graceMaxCount):
    """
    Return true/false if <valsQ> starts an episode based on upcoming <valsQ>
            and other threshold parameters
    Inputs:
        valsQ, float queue, upcoming n values (equal to minimum episode duration)
        minCount, int, lower bound count/vector-magnitude score e.g. 210
        maxCount, int, upper bound count/vector-magnitude e.g. 690
        graceMaxFreq, int, how many grace occurrences can be outside lower/upper bound e.g. 2
        graceMinCount, int, lower bound count/vector-magnitude of grace time e.g. 0
        graceMaxCount, int, upper bound count/vector-magnitude of grace time e.g. 15000
    """
    #before processing whole queue, check if 1st item is a candidate episode start
    if valsQ[0] >= minCount and valsQ[0] <= maxCount:
        num_episode_items = 1
        numGrace = 0
        numInvalid = 0
        #now count episode, grace, & invalid candidate items in remaining queue items
        #***check if performance hit when converting queue to list
        # (would it be better to iterate through all n queue items 
        # (with num_episode_items=0 3 lines above) vs. n-1 list items + conversion 
        # of queue-to-list (with num_episode_items=1)
        for instance_value in list(valsQ)[1:]:
            if instance_value >= minCount and instance_value <= maxCount:
                num_episode_items += 1
            elif instance_value >= graceMinCount and instance_value <= graceMaxCount:
                numGrace += 1
            else: 
                numInvalid += 1
            #check to see if item is no longer a candidate activity start time
            if numInvalid > 0 or numGrace > graceMaxFreq:
                    break
        #check if current item is a candidate episode start time
        if numInvalid == 0 and numGrace <= graceMaxFreq:
            return True
    #Value isn't an episode start time if we get here
    return False

def isNonWearEnd(epochCount, isLastItem, valsQ, maxStd, graceMaxFreq):
    """
    Method checks if <valsQ> starts a nonWear episode
    """
    return isEpisodeEnd(epochCount, isLastItem, valsQ, 0, maxStd, graceMaxFreq,
                maxStd+0.00001, float("inf"))

def isEpisodeEnd(epochCount, isLastItem, valsQ, minCount, maxCount, graceMaxFreq, graceMinCount, graceMaxCount):
    """
    Return true/false if <epochCount> ends an episode based on recent <valsQ>
            and other threshold parameters
    Inputs:
        epochCount, string, candidate end-of-episode count value
        isLastItem, bool, is this the last item in list?
        valsQ, float queue, previous n values (equal to minimum episode duration)
        minCount, int, lower bound count/vector-magnitude score e.g. 210
        maxCount, int, upper bound count/vector-magnitude e.g. 690
        graceMaxFreq, int, how many grace occurrences can be outside lower/upper bound e.g. 2
        graceMinCount, int, lower bound count/vector-magnitude of grace time e.g. 0
        graceMaxCount, int, upper bound count/vector-magnitude of grace time e.g. 15000
    """
    #check current epochCount value could be an episode end time candidate
    if (epochCount < minCount or epochCount > maxCount) or isLastItem:
        numGrace = 0
        numInvalid = 0
        #now look if previous n values (i.e. all queue items) were grace or invalid 
        for val in valsQ:
            if val < minCount or val > maxCount and (val >= graceMinCount and val <= graceMaxCount):
                numGrace += 1
            elif val < minCount or val > maxCount:
                numInvalid += 1
            #check if complete outlier exists or too many grace values
            if numInvalid > 0 or numGrace > graceMaxFreq:
                return True #we've detected episode endtime
    #This value isn't an episode end time if we reach here
    return False

def numOutliersAtListEnd(episodeCountVals, minCount, maxCount):
    """
    Returns how many (if any) successive outliers are at the end of
    <episodeCountVals> e.g. a 15min episode may have 13 valid mins and 2min
    grace at the very end. However the 2min successively occurring at the end
    should be identified (so they can be removed elsewhere)
    Inputs:
        episodeCountVals, float list, list of episode's candidate epoch values
        minCount, int, lower bound count/vector-magnitude score e.g. 210
        maxCount, int, upper bound count/vector-magnitude e.g. 690
    """
    numEndOutliers = 0 
    #go through all list values to see how many are out of bounds
    for epoch_value in reversed(episodeCountVals):
        if epoch_value < minCount or epoch_value > maxCount:
            numEndOutliers += 1 #iterate if loop is still finding outliers
        else:
            break #break loop once we come across a valid value
    return numEndOutliers

def writeSummaryOfEpisodes(outputFile, episodesList, displayOutput):
    """
    Print a summary of each episode to <outputFile>
    Inputs:
        outputFile, string, name/path of file to write summary to
        episodesList, <behaviourEpisode> list,
        displayOutput, bool, write to screen or not (in addition to file write)
    Output:
        total_duration, int, num minutes in total across all episodes
        num episodes
    """
    #print out a summary of each episode detected
    boutWriter = open(outputFile, 'w')
    episodeDuration = 0
    #episode list header
    boutWriter.write(behaviourEpisode.summaryHeaderNonWear +'\n')  
    for episode in episodesList:
        #duration in minutes (pre Python 2.7 compatible too)
        episodeDuration += (((episode.endTime-episode.startTime).days*3600*24) +
                        ((episode.endTime - episode.startTime).seconds))/60
        boutWriter.write(episode.getSummaryNonWear() + '\n') #behav_type, start_time, end_time, xStd, yStd, zStd
        if displayOutput:
            print episode.getSummaryNonWear()
    boutWriter.close()
    return episodeDuration, len(episodesList)
