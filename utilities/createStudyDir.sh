#!/bin/bash
# ------------------------------------------------------------------
# [Aiden Doherty] Create dir structure to process acc files
# ------------------------------------------------------------------

rootDir=$1

# Structure will be created as follows
#    <studyName>/
#        files.csv #listing all files in rawData directory
#        rawData/ #all raw .cwa .bin .gt3x files
#        summary/ #to store outputSummary.json
#        epoch/ #to store feature output for 30sec windows
#        timeSeries/ #simple csv time series output (VMag, activity binary predictions)
#        nonWear/ #bouts of nonwear episodes
#        stationary/ #temp store for features of stationary data for calibration
#        clusterLogs/ #to store terminal output for each processed file

mkdir ${1}rawData/
mkdir ${1}summary/
mkdir ${1}epoch/
mkdir ${1}timeSeries/
mkdir ${1}nonWear/
mkdir ${1}stationary/
mkdir ${1}clusterLogs/

echo "Acc processing directory structure created under " $1
