import argparse
import sys
import os

# sys.argv = "sample -calOff 1.2 4 99999.342435 -skipRaw true -summaryFolder newdir".split()


parser = argparse.ArgumentParser(
    description="""A tool to extract meaningful health information from large accelerometer
     datasets e.g. how much time individuals spend in sleep, sedentary behaviour, and 
     physical activity.
    """,
    add_help=False
)

# optionals
parser.add_argument('-summaryFolder', metavar='filename',default="",
                        help='folder for the %(default)s summary statistics')
parser.add_argument('-nonWearFolder', metavar='filename',default="",
                        help='folder for the %(default)s file')
parser.add_argument('-epochFolder', metavar='filename',default="",
                        help='folder for the %(default)s file')
parser.add_argument('-stationaryFolder', metavar='filename',default="",
                        help='folder for the  %(default)s file')
parser.add_argument('-timeSeriesFolder', metavar='filename',default="",
                        help='folder for the %(default)s file')
parser.add_argument('-skipCalibration', 
                        metavar='True/False',default=True, type=bool,
                        help='skip calibration? (default : %(default)s)')
parser.add_argument('-verbose', 
                        metavar='True/False',default=False, type=bool,
                        help='enable verbose logging? (default : %(default)s)')
parser.add_argument('-deleteHelperFiles', 
                        metavar='True/False',default=True, type=bool,
                        help='skip calibration? (default : %(default)s)')
parser.add_argument('-skipRaw', 
                        metavar='True/False',default=False, type=bool,
                        help='skipRaw? (default : %(default)s)')
parser.add_argument('-epochPeriod', 
                        metavar='length',default=5, type=int,
                        help='length in seconds of a single epoch (default : %(default)ss)')
parser.add_argument('-calOff', 
                        metavar=('x','y','z'),default=[0.0,0.0,0.0], type=float, nargs=3,
                        help='calibration offset (default : %(default)s)')
parser.add_argument('-calSlope', 
                        metavar=('x','y','z'),default=[1.0,1.0,1.0], type=float, nargs=3,
                        help='calibration slope linking offset to temperature (default : %(default)s)')
parser.add_argument('-calTemp', 
                        metavar=('x','y','z'),default=[0.0,0.0,0.0], type=float, nargs=3,
                        help='calibration temperature (default : %(default)s)')
parser.add_argument('-meanTemp', 
                        metavar="temp",default=20, type=float,
                        help='mean calibration temperature in degrees Celsius (default : %(default)s)')
parser.add_argument('-javaHeapSpace', 
                        metavar="amount in MB",default="", type=str,
                        help='amount of heap space allocated to the java subprocesses, useful for limiting RAM (default : %(default)s)')
parser.add_argument('-epochProcess', 
                        metavar="epochProcess",default="AxivityAx3Epochs", type=str,
                        help='epochProcess (default : %(default)s)')

# required
parser.add_argument('rawFile', metavar='file', type=str, 
                   help='the .cwa file to process (e.g. sample.cwa)')

if len(sys.argv)<2:
        msg = "\nInvalid input, please enter at least 1 parameter, e.g."
        msg += "\npython ActivitySummary.py inputFile.CWA \n"
        print msg
        parser.print_help()
        sys.exit(0)

print
args = parser.parse_args(sys.argv)
print args

if (args.skipRaw is True):
    if len(args.rawFile.split('.'))<2:
        args.rawFile += ".cwa" # edge case since we still need a name?
elif not os.path.isfile(args.rawFile): 
    print "error: file does not exist. Exiting.."
    sys.exit()

print "rawFile = " + str(args.rawFile)
# get file extension
args.rawFileEnd = '.' + args.rawFile.split('.')[-1]
args.rawFileBegin = args.rawFile[0:-len(args.rawFileEnd)]
print (args.rawFileBegin)


# could check if folder exists? probably not neccesary
args.summaryFolder    = os.path.join(args.summaryFolder,    args.rawFileBegin + "OutputSummary.json")
args.nonWearFolder    = os.path.join(args.nonWearFolder,    args.rawFileBegin + "NonWearBouts.csv")
args.epochFolder      = os.path.join(args.epochFolder,      args.rawFileBegin + "Epoch.json")
args.stationaryFolder = os.path.join(args.stationaryFolder, args.rawFileBegin + "Stationary.csv")
args.timeSeriesFolder = os.path.join(args.timeSeriesFolder, args.rawFileBegin + "AccTimeSeries.csv")


print "args.summaryFolder: " ,args.summaryFolder
print "args.nonWearFolder: " ,args.nonWearFolder
print "args.epochFolder: " ,args.epochFolder
print "args.stationaryFolder: " ,args.stationaryFolder
print "args.timeSeriesFolder: " ,args.timeSeriesFolder
