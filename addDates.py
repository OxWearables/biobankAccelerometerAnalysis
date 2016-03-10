import datetime
import os
import pandas as pd
import time
import sys
import copy
import re
import math

print """usage: 'python addDates.py sampleAccTimeSeries.csv
you can also drag .csv files onto this script to launch it.
This script adds a date/time column to any AccTimeSeries.csv file."""
if len(sys.argv)>1:
    tsFile = sys.argv[1]
    if os.path.isfile(sys.argv[1]):
        print "found file: " + str(sys.argv[1])
    else:
        print str(sys.argv[1]) + " is not a file"
        sys.exit()
else:
    print "please enter the path of a file to process.."
    # tsFile = "sampleAccTimeSeries.csv"
    sys.exit()
tsData = pd.read_csv(tsFile, header=0)
accCol = next(x for x in tsData.dtypes.index if x.startswith('acceleration'))
imputationCol = next(x for x in tsData.dtypes.index if x.startswith('imputed'))
print accCol, imputationCol
dates = re.findall(r'(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)',accCol)
startDate  = dates[0]
endDate    = dates[1]
print "found start and end date", startDate, endDate
startDate  = datetime.datetime.strptime(startDate,"%Y-%m-%d %H:%M:%S")
endDate    = datetime.datetime.strptime(endDate,"%Y-%m-%d %H:%M:%S")
print "parsed start and end date", startDate, endDate
sampleRate = re.search(r'sampleRate = (\d+) seconds', accCol).group(1)
sampleRate = datetime.timedelta(seconds=int(sampleRate))
print "found sample rate (seconds)", sampleRate
numrows = tsData.shape[0]
print "processing " + str(numrows) + " rows"
# print tsData
# this should be the same as the date for the last row 
# print 'computed endDate', startDate + sampleRate * (tsData.shape[0]-1)
temp=[]

for row in tsData.iterrows():
    index, data = row
    # print index, numrows, index % (numrows/10)
    if (index % (numrows/10)) == 0:
        print index, numrows, str(math.floor(100*index/numrows)) + "%"
    temp.append(startDate + sampleRate * index)

tsData['datetime'] = temp
print tsData
print tsData.shape
tsData.to_csv(tsFile.split(".csv")[0]+"_datecol.csv", index=False)
# print endDate
