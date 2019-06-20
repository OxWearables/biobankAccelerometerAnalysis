"""Command line tool to extract meaningful health info from accelerometer data."""

import accelerometer.accUtils
import accelerometer.accClassification
import argparse
import collections
import datetime
import accelerometer.device
import json
import os
import accelerometer.summariseEpoch
import sys
import pandas as pd


def main():
    """
    Application entry point responsible for parsing command line requests
    """

    parser = argparse.ArgumentParser(
        description="""A tool to extract physical activity information from
            raw accelerometer files.""", add_help=True
    )
    # required
    parser.add_argument('rawFile', metavar='input file', type=str,
                            help="""the <.cwa/.cwa.gz> file to process
                            (e.g. sample.cwa.gz). If the file path contains
                            spaces,it must be enclosed in quote marks 
                            (e.g. \"../My Documents/sample.cwa\")
                            """)

    #optional inputs
    parser.add_argument('--startTime',
                            metavar='e.g. 1991-01-01T23:59', default=None,
                            type=str2date, help="""removes data before this
                            time in the final analysis
                            (default : %(default)s)""")
    parser.add_argument('--endTime',
                            metavar='e.g 1991-01-01T23:59', default=None,
                            type=str2date, help="""removes data after this
                            time in the final analysis
                            (default : %(default)s)""")
    parser.add_argument('--timeSeriesDateColumn',
                            metavar='True/False', default=False, type=str2bool,
                            help="""adds a date/time column to the timeSeries
                            file, so acceleration and imputation values can be
                            compared easily. This increases output filesize
                            (default : %(default)s)""")
    parser.add_argument('--processRawFile',
                            metavar='True/False', default=True, type=str2bool,
                            help="""False will skip processing of the .cwa file
                             (the epoch.csv file must already exist for this to
                             work) (default : %(default)s)""")
    parser.add_argument('--epochPeriod',
                            metavar='length', default=30, type=int,
                            help="""length in seconds of a single epoch (default
                             : %(default)ss, must be an integer)""")
    parser.add_argument('--sampleRate',
                            metavar='Hz, or samples/second', default=100,
                            type=int, help="""resample data to n Hz (default
                             : %(default)ss, must be an integer)""")
    parser.add_argument('--useAbs',
                            metavar='useAbs', default=False, type=str2bool,
                            help="""use abs(VM) instead of trunc(VM)
                            (default : %(default)s)""")
    parser.add_argument('--skipFiltering',
                            metavar='True/False', default=False, type=str2bool,
                            help="""Skip filtering stage
                             (default : %(default)s)""")
    # calibration parameters
    parser.add_argument('--skipCalibration',
                            metavar='True/False', default=False, type=str2bool,
                            help="""skip calibration? (default : %(default)s)""")
    parser.add_argument('--calOffset',
                            metavar=('x', 'y', 'z'),default=[0.0, 0.0, 0.0],
                            type=float, nargs=3,
                            help="""accelerometer calibration offset (default :
                             %(default)s)""")
    parser.add_argument('--calSlope',
                            metavar=('x', 'y', 'z'), default=[1.0, 1.0, 1.0],
                            type=float, nargs=3,
                            help="""accelerometer calibration slope linking
                            offset to temperature (default : %(default)s)""")
    parser.add_argument('--calTemp',
                            metavar=('x', 'y', 'z'), default=[0.0, 0.0, 0.0],
                            type=float, nargs=3,
                            help="""mean temperature in degrees Celsius of
                            stationary data for calibration
                            (default : %(default)s)""")
    parser.add_argument('--meanTemp',
                            metavar="temp", default=20.0, type=float,
                            help="""mean calibration temperature in degrees
                            Celsius (default : %(default)s)""")
    parser.add_argument('--stationaryStd',
                            metavar='mg', default=13, type=int,
                            help="""stationary mg threshold (default
                             : %(default)s mg))""")
    parser.add_argument('--calibrationSphereCriteria',
                            metavar='mg', default=0.3, type=float,
                            help="""calibration sphere threshold (default
                             : %(default)s mg))""")
    # activity parameters
    parser.add_argument('--mgMVPA',
                            metavar="mg", default=100, type=int,
                            help="""MVPA threshold (default : %(default)s)""")
    parser.add_argument('--mgVPA',
                            metavar="mg", default=425, type=int,
                            help="""VPA threshold (default : %(default)s)""")
    # calling helper processess and conducting multi-threadings
    parser.add_argument('--rawDataParser',
                            metavar="rawDataParser", default="AxivityAx3Epochs",
                            type=str,
                            help="""file containing a java program to process
                            raw .cwa binary file, must end with .class (omitted)
                             (default : %(default)s)""")
    parser.add_argument('--javaHeapSpace',
                            metavar="amount in MB", default="", type=str,
                            help="""amount of heap space allocated to the java
                            subprocesses,useful for limiting RAM usage (default
                            : unlimited)""")
    # activity classification arguments
    parser.add_argument('--activityClassification',
                            metavar='True/False', default=True, type=str2bool,
                            help="""Use pre-trained random forest to predict
                            activity type
                            (default : %(default)s)""")
    parser.add_argument('--activityModel', type=str,
                            default="activityModels/doherty2018.tar",
                            help="""trained activity model .tar file""")
    parser.add_argument('--rawOutput',
                            metavar='True/False', default=False, type=str2bool,
                            help="""output calibrated and resampled raw data to 
                            a .csv.gz file? NOTE: requires ~50MB per day. 
                            (default : %(default)s)""")
    parser.add_argument('--npyOutput',
                            metavar='True/False', default=False, type=str2bool,
                            help="""output calibrated and resampled raw data to 
                            .npy file? NOTE: requires ~60MB per day. 
                            (default : %(default)s)""")
    parser.add_argument('--fftOutput',
                            metavar='True/False', default=False, type=str2bool,
                            help="""output FFT epochs to a .csv file? NOTE:
                            requires ~0.1GB per day. (default : %(default)s)""")
    # optional outputs
    parser.add_argument('--outputFolder', metavar='filename',default="",
                            help="""folder for all of the output files, \
                                unless specified using other options""")
    parser.add_argument('--summaryFolder', metavar='filename',default="",
                        help="folder for -summary.json summary stats")
    parser.add_argument('--epochFolder', metavar='filename', default="",
                            help="""folder -epoch.csv.gz - must be an existing
                            file if "-processRawFile" is set to False""")
    parser.add_argument('--timeSeriesFolder', metavar='filename', default="",
                            help="folder for -timeSeries.csv.gz file")
    parser.add_argument('--nonWearFolder', metavar='filename',default="",
                            help="folder for -nonWearBouts.csv.gz file")
    parser.add_argument('--stationaryFolder', metavar='filename', default="",
                            help="folder -stationaryPoints.csv.gz file")
    parser.add_argument('--rawFolder', metavar='filename', default="",
                            help="folder for raw .csv.gz file")
    parser.add_argument('--verbose',
                            metavar='True/False', default=False, type=str2bool,
                            help="""enable verbose logging? (default :
                            %(default)s)""")
    parser.add_argument('--deleteIntermediateFiles',
                            metavar='True/False', default=True, type=str2bool,
                            help="""True will remove extra "helper" files created
                                    by the program (default : %(default)s)""")
    parser.add_argument('--intensityDistribution',
                            metavar='True/False', default=False, type=str2bool,
                            help="""Save intensity distribution
                             (default : %(default)s)""")

    #
    # check that enough command line arguments are entered
    #
    if len(sys.argv) < 2:
        msg = "\nInvalid input, please enter at least 1 parameter, e.g."
        msg += "\npython accProcess.py data/sample.cwa.gz \n"
        accelerometer.accUtils.toScreen(msg)
        parser.print_help()
        sys.exit(-1)
    processingStartTime = datetime.datetime.now()
    args = parser.parse_args()

    ##########################
    # check input/output files/dirs exist and validate input args
    ##########################
    if args.processRawFile is False:
        if len(args.rawFile.split('.')) < 2:
            args.rawFile += ".cwa"  # TODO edge case since we still need a name?
    elif not os.path.isfile(args.rawFile):
        if args.rawFile:
            print("error: specified file " + args.rawFile + " does not exist. Exiting..")
        else:
            print("error: no file specified. Exiting..")
        sys.exit(-2)
    # get file extension
    rawFileNoExt = ('.').join(args.rawFile.split('.')[:-1])
    (rawFilePath, rawFileName) = os.path.split(rawFileNoExt)
    # check target output folders exist
    for path in [args.summaryFolder, args.nonWearFolder, args.epochFolder,
                 args.stationaryFolder, args.timeSeriesFolder, args.outputFolder]:
        if len(path) > 0 and not os.access(path, os.F_OK):
            print("error: " + path + " is not a valid directory")
            sys.exit(-3)
    # assign output file names
    if args.outputFolder == "" and rawFilePath != "":
        args.outputFolder = rawFilePath + '/'
    if args.summaryFolder == "":
        args.summaryFolder = args.outputFolder
    if args.nonWearFolder == "":
        args.nonWearFolder = args.outputFolder
    if args.epochFolder == "":
        args.epochFolder = args.outputFolder
    if args.stationaryFolder == "":
        args.stationaryFolder = args.outputFolder
    if args.timeSeriesFolder == "":
        args.timeSeriesFolder = args.outputFolder
    if args.rawFolder == "":
        args.rawFolder = args.outputFolder
    args.summaryFile = args.summaryFolder + rawFileName + "-summary.json"
    args.nonWearFile = args.nonWearFolder + rawFileName + "-nonWearBouts.csv.gz"
    args.epochFile = args.epochFolder + rawFileName + "-epoch.csv.gz"
    args.stationaryFile = args.stationaryFolder + rawFileName + "-stationaryPoints.csv"
    args.tsFile = args.timeSeriesFolder + rawFileName + "-timeSeries.csv.gz"
    args.rawOutputFile = args.rawFolder + rawFileName + ".csv.gz"
    args.npyOutputFile = args.rawFolder + rawFileName + ".npy"

    # check user specified end time is not before start time
    if args.startTime and args.endTime:
        if args.startTime >= args.endTime:
            print("start and end time arguments are invalid!")
            print("startTime:", args.startTime.strftime("%Y-%m-%dT%H:%M"))
            print("endTime:", args.endTime.strftime("%Y-%m-%dT%H:%M"))
            sys.exit(-4)

    # print processing options to screen
    print("processing file " + args.rawFile + "' with these arguments:\n")
    for key, value in sorted(vars(args).items()):
        if not (isinstance(value, str) and len(value)==0):
            print(key.ljust(15), ':', value)
    print("\n")


    ##########################
    # start processing file
    ##########################
    summary = {}
    # now process the .CWA file
    if args.processRawFile:
        summary['file-name'] = args.rawFile
        accelerometer.device.processRawFileToEpoch(args.rawFile, args.epochFile,
            args.stationaryFile, summary, skipCalibration=args.skipCalibration,
            stationaryStd=args.stationaryStd, xIntercept=args.calOffset[0],
            yIntercept=args.calOffset[1], zIntercept=args.calOffset[2],
            xSlope=args.calSlope[0], ySlope=args.calSlope[1],
            zSlope=args.calSlope[2], xTemp=args.calTemp[0],
            yTemp=args.calTemp[1], zTemp=args.calTemp[2],
            meanTemp=args.meanTemp, rawDataParser=args.rawDataParser,
            javaHeapSpace=args.javaHeapSpace, skipFiltering=args.skipFiltering,
            sampleRate=args.sampleRate, epochPeriod=args.epochPeriod,
            useAbs=args.useAbs, activityClassification=args.activityClassification,
            rawOutput=args.rawOutput, rawOutputFile=args.rawOutputFile,
            npyOutput=args.npyOutput, npyOutputFile=args.npyOutputFile,
            fftOutput=args.fftOutput, startTime=args.startTime,
            endTime=args.endTime, verbose=args.verbose)
    else:
        summary['file-name'] = args.epochFile

    # summarise epoch
    epochData, labels = accelerometer.summariseEpoch.getActivitySummary(
        args.epochFile, args.nonWearFile, summary,
        activityClassification=args.activityClassification, startTime=args.startTime,
        endTime=args.endTime, epochPeriod=args.epochPeriod,
        stationaryStd=args.stationaryStd, mgMVPA=args.mgMVPA,
        mgVPA=args.mgVPA, activityModel=args.activityModel,
        intensityDistribution=args.intensityDistribution,
        verbose=args.verbose)

    # generate time series file (note: this will also resample to epochData so do this last)
    accelerometer.accUtils.generateTimeSeries(epochData, args.tsFile,
        epochPeriod=args.epochPeriod,
        timeSeriesDateColumn=args.timeSeriesDateColumn,
        activityClassification=args.activityClassification, labels=labels)

    # print basic output
    summaryVals = ['file-name', 'file-startTime', 'file-endTime', \
            'acc-overall-avg','wearTime-overall(days)', \
            'nonWearTime-overall(days)', 'quality-goodWearTime']
    summaryDict = collections.OrderedDict([(i, summary[i]) for i in summaryVals])
    accelerometer.accUtils.toScreen(json.dumps(summaryDict, indent=4))

    # write detailed output to file
    f = open(args.summaryFile,'w')
    json.dump(summary, f, indent=4)
    f.close()
    accelerometer.accUtils.toScreen('Summary file written to: ' + args.summaryFile)


    ##########################
    # remove helper files and close program
    ##########################
    if args.deleteIntermediateFiles:
        try:
            os.remove(args.stationaryFile)
            os.remove(args.epochFile)
        except:
            accelerometer.accUtils.toScreen('could not delete helper file')
    # finally, print out processing summary message
    processingEndTime = datetime.datetime.now()
    processingTime = (processingEndTime - processingStartTime).total_seconds()
    accelerometer.accUtils.toScreen("in total, processing took " + \
        str(processingTime) + " seconds")



def str2bool(v):
    """
    Used to parse true/false values from the command line. E.g. "True" -> True
    """

    return v.lower() in ("yes", "true", "t", "1")


def str2date(v):
    """
    Used to parse date values from the command line. E.g. "1994-11-30T12:00" -> time.datetime
    """

    eg = "1994-11-30T12:00" # example date
    if v.count("-")!=eg.count("-"):
        print("ERROR: not enough dashes in date")
    elif v.count("T")!=eg.count("T"):
        print("ERROR: no T seperator in date")
    elif v.count(":")!=eg.count(":"):
        print("ERROR: no ':' seperator in date")
    elif len(v.split("-")[0])!=4:
        print("ERROR: year in date must be 4 numbers")
    elif len(v.split("-")[1])!=2 and len(v.split("-")[1])!=1:
        print("ERROR: month in date must be 1-2 numbers")
    elif len(v.split("-")[2].split("T")[0])!=2 and len(v.split("-")[2].split("T")[0])!=1:
        print("ERROR: day in date must be 1-2 numbers")
    else:
        return pd.datetime.strptime(v, "%Y-%m-%dT%H:%M")
    print("please change your input date:")
    print('"'+v+'"')
    print("to match the example date format:")
    print('"'+eg+'"')
    raise ValueError("date in incorrect format")

if __name__ == '__main__':
    main()  # Standard boilerplate to call the main() function to begin the program.
