"""Command line tool to extract meaningful health info from accelerometer data."""

import accelerometer.utils
import accelerometer.classification
import argparse
import collections
import datetime
import accelerometer.device
import json
import os
import accelerometer.summarisation
import pandas as pd
import atexit
import warnings


def main():  # noqa: C901
    """
    Application entry point responsible for parsing command line requests
    """

    parser = argparse.ArgumentParser(
        description="""A tool to extract physical activity information from
            raw accelerometer files.""", add_help=True
    )
    # required
    parser.add_argument('inputFile', metavar='input file', type=str,
                        help="""the <.cwa/.cwa.gz> file to process
                            (e.g. sample.cwa.gz). If the file path contains
                            spaces,it must be enclosed in quote marks
                            (e.g. \"../My Documents/sample.cwa\")
                            """)

    # optional inputs
    parser.add_argument('--timeZone',
                        metavar='e.g. Europe/London', default='Europe/London',
                        type=str, help="""timezone in country/city format to
                            be used for daylight savings crossover check
                            (default : %(default)s""")
    parser.add_argument('--timeShift',
                        metavar='e.g. 10 (mins)', default=0,
                        type=int, help="""time shift to be applied, e.g.
                            -15 will shift the device internal time by -15
                            minutes. Not to be confused with timezone offsets.
                            (default : %(default)s""")
    parser.add_argument('--startTime',
                        metavar='e.g. 1991-01-01T23:59', default=None,
                        type=str2date, help="""removes data before this
                            time (local) in the final analysis
                            (default : %(default)s)""")
    parser.add_argument('--endTime',
                        metavar='e.g 1991-01-01T23:59', default=None,
                        type=str2date, help="""removes data after this
                            time (local) in the final analysis
                            (default : %(default)s)""")
    parser.add_argument('--processInputFile',
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
    parser.add_argument('--resampleMethod',
                        metavar='linear/nearest', default="linear",
                        type=str, help="""Method to use for resampling
                            (default : %(default)s)""")
    parser.add_argument('--useFilter',
                        metavar='True/False', default=True, type=str2bool,
                        help="""Filter ENMOtrunc values?
                             (default : %(default)s)""")
    parser.add_argument('--csvStartTime',
                        metavar='e.g. 2020-01-01T00:01', default=None,
                        type=str2date, help="""start time for csv file
                            when time column is not available
                            (default : %(default)s)""")
    parser.add_argument('--csvSampleRate',
                        metavar='Hz, or samples/second', default=None,
                        type=float, help="""sample rate for csv file
                            when time column is not available (default
                             : %(default)s)""")
    parser.add_argument('--csvTimeFormat',
                        metavar='time format',
                        default="yyyy-MM-dd HH:mm:ss.SSSxxxx '['VV']'",
                        type=str, help="""time format for csv file
                            when time column is available (default
                             : %(default)s)""")
    parser.add_argument('--csvStartRow',
                        metavar='start row', default=1, type=int,
                        help="""start row for accelerometer data in csv file (default
                             : %(default)s, must be an integer)""")
    parser.add_argument('--csvTimeXYZTempColsIndex',
                        metavar='time,x,y,z,temperature',
                        default="0,1,2,3,4", type=str,
                        help="""index of column positions for time
                            and x/y/z/temperature columns, e.g. "0,1,2,3,4" (default
                             : %(default)s)""")
    # optional outputs
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
    # calibration parameters
    parser.add_argument('--skipCalibration',
                        metavar='True/False', default=False, type=str2bool,
                        help="""skip calibration? (default : %(default)s)""")
    parser.add_argument('--calOffset',
                        metavar=('x', 'y', 'z'), default=[0.0, 0.0, 0.0],
                        type=float, nargs=3,
                        help="""accelerometer calibration offset in g
                        (default : %(default)s)""")
    parser.add_argument('--calSlope',
                        metavar=('x', 'y', 'z'), default=[1.0, 1.0, 1.0],
                        type=float, nargs=3,
                        help="""accelerometer slopes for calibration
                        (default : %(default)s)""")
    parser.add_argument('--calTemp',
                        metavar=('x', 'y', 'z'), default=[0.0, 0.0, 0.0],
                        type=float, nargs=3,
                        help="""temperature slopes for calibration
                        (default : %(default)s)""")
    parser.add_argument('--meanTemp',
                        metavar="temp", default=None, type=float,
                        help="""(DEPRECATED) mean calibration temperature in degrees
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
    parser.add_argument('--mgCpLPA',
                        metavar="mg", default=45, type=int,
                        help="""LPA threshold for cut point based activity
                            definition (default : %(default)s)""")
    parser.add_argument('--mgCpMPA',
                        metavar="mg", default=100, type=int,
                        help="""MPA threshold for cut point based activity
                            definition (default : %(default)s)""")
    parser.add_argument('--mgCpVPA',
                        metavar="mg", default=400, type=int,
                        help="""VPA threshold for cut point based activity
                            definition (default : %(default)s)""")
    parser.add_argument('--intensityDistribution',
                        metavar='True/False', default=False, type=str2bool,
                        help="""Save intensity distribution
                             (default : %(default)s)""")
    parser.add_argument('--extractFeatures',
                        metavar='True/False', default=True, type=str2bool,
                        help="""Whether to extract signal features. Needed for
                            activity classification (default : %(default)s)""")
    # activity classification arguments
    parser.add_argument('--activityClassification',
                        metavar='True/False', default=True, type=str2bool,
                        help="""Use pre-trained random forest to predict
                            activity type (default : %(default)s)""")
    parser.add_argument('--activityModel', type=str,
                        default="walmsley",
                        help="""trained activity model .tar file""")

    # circadian rhythm options
    parser.add_argument('--psd',
                        metavar='True/False', default=False, type=str2bool,
                        help="""Calculate power spectral density for 24 hour
                                    circadian period
                             (default : %(default)s)""")
    parser.add_argument('--fourierFrequency',
                        metavar='True/False', default=False, type=str2bool,
                        help="""Calculate dominant frequency of sleep for circadian rhythm analysis
                             (default : %(default)s)""")
    parser.add_argument('--fourierWithAcc',
                        metavar='True/False', default=False, type=str2bool,
                        help="""True will do the Fourier analysis of circadian rhythms (for PSD and Fourier Frequency) with
                                    acceleration data instead of sleep signal
                             (default : %(default)s)""")
    parser.add_argument('--m10l5',
                        metavar='True/False', default=False, type=str2bool,
                        help="""Calculate relative amplitude of most and
                                    least active acceleration periods for circadian rhythm analysis
                             (default : %(default)s)""")
    # optional outputs
    parser.add_argument('--outputFolder', '-o', metavar='filename', default=None,
                        help="""folder for all of the output files (default : %(default)s)""")
    parser.add_argument('--verbose',
                        metavar='True/False', default=False, type=str2bool,
                        help="""enable verbose logging? (default :
                            %(default)s)""")
    parser.add_argument('--deleteIntermediateFiles',
                        metavar='True/False', default=True, type=str2bool,
                        help="""True will remove extra "helper" files created
                                    by the program (default : %(default)s)""")
    # calling helper processess and conducting multi-threadings
    parser.add_argument('--rawDataParser',
                        metavar="rawDataParser", default="AccelerometerParser",
                        type=str,
                        help="""file containing a java program to process
                            raw .cwa binary file, must end with .class (omitted)
                             (default : %(default)s)""")
    parser.add_argument('--javaHeapSpace',
                        metavar="amount in MB", default="", type=str,
                        help="""amount of heap space allocated to the java
                            subprocesses,useful for limiting RAM usage (default
                            : unlimited)""")

    args = parser.parse_args()

    processingStartTime = datetime.datetime.now()

    if args.calOffset != [0, 0, 0] or args.calSlope != [1, 1, 1] or args.calTemp != [0, 0, 0]:
        args.skipCalibration = True
        warnings.warn('Skipping calibration as coefficients supplied')

    if args.meanTemp is not None:
        warnings.warn("Passing --meanTemp is deprecated. Calibration will be performed (--skipCalibration False)")
        args.skipCalibration = False
        args.calOffset = [0, 0, 0]
        args.calSlope = [1, 1, 1]
        args.calTemp = [0, 0, 0]

    if args.activityClassification and not args.extractFeatures:
        args.extractFeatures = True
        warnings.warn('Setting --extractFeatures True: Required for activity classification')

    assert args.sampleRate >= 25, "sampleRate<25 currently not supported"

    if args.sampleRate <= 40:
        warnings.warn("Skipping lowpass filter (--useFilter False) as sampleRate too low (<= 40)")
        args.useFilter = False

    # Parent folder and basename of input file
    inputFileFolder = os.path.dirname(args.inputFile)
    inputFileName = os.path.basename(args.inputFile).split(".")[0]

    # Set default output folder if not specified
    if args.outputFolder is None:
        args.outputFolder = os.path.abspath(inputFileFolder)

    os.makedirs(args.outputFolder, exist_ok=True)

    assert os.access(args.outputFolder, os.W_OK), (
        f"Either folder '{args.outputFolder}' does not exist "
        "or you do not have write permission"
    )

    # Set default output filenames
    args.summaryFile = os.path.join(args.outputFolder, inputFileName + "-summary.json")
    args.nonWearFile = os.path.join(args.outputFolder, inputFileName + "-nonWearBouts.csv.gz")
    args.epochFile = os.path.join(args.outputFolder, inputFileName + "-epoch.csv.gz")
    args.stationaryFile = os.path.join(args.outputFolder, inputFileName + "-stationaryPoints.csv.gz")
    args.tsFile = os.path.join(args.outputFolder, inputFileName + "-timeSeries.csv.gz")
    args.rawFile = os.path.join(args.outputFolder, inputFileName + ".csv.gz")
    args.npyFile = os.path.join(args.outputFolder, inputFileName + ".npy")  # .gz?

    # Schedule to delete intermediate files at program exit
    if args.deleteIntermediateFiles:
        @atexit.register
        def deleteIntermediateFiles():
            try:
                if os.path.exists(args.stationaryFile):
                    os.remove(args.stationaryFile)
                if os.path.exists(args.nonWearFile):
                    os.remove(args.nonWearFile)
                if os.path.exists(args.epochFile):
                    os.remove(args.epochFile)
            except OSError:
                accelerometer.utils.toScreen('Could not delete intermediate files')

    # Check user-specified end time is not before start time
    if args.startTime and args.endTime:
        assert args.startTime <= args.endTime, (
            "startTime and endTime arguments are invalid!\n"
            f"startTime: {args.startTime.strftime('%Y-%m-%dT%H:%M')}\n"
            f"endTime:, {args.endTime.strftime('%Y-%m-%dT%H:%M')}\n"
        )

    # Print processing options to screen
    print(f"Processing file '{args.inputFile}' with these arguments:\n")
    for key, value in sorted(vars(args).items()):
        if not (isinstance(value, str) and len(value) == 0):
            print(key.ljust(25), ':', value)

    ##########################
    # Start processing file
    ##########################
    summary = {}
    # Now process the .CWA file
    if args.processInputFile:
        summary['file-name'] = args.inputFile
        accelerometer.device.processInputFileToEpoch(
            args.inputFile, args.timeZone,
            args.timeShift, args.epochFile, args.stationaryFile, summary,
            skipCalibration=args.skipCalibration,
            stationaryStd=args.stationaryStd, xyzIntercept=args.calOffset,
            xyzSlope=args.calSlope, xyzSlopeT=args.calTemp,
            rawDataParser=args.rawDataParser, javaHeapSpace=args.javaHeapSpace,
            useFilter=args.useFilter, sampleRate=args.sampleRate, resampleMethod=args.resampleMethod,
            epochPeriod=args.epochPeriod,
            extractFeatures=args.extractFeatures,
            rawOutput=args.rawOutput, rawFile=args.rawFile,
            npyOutput=args.npyOutput, npyFile=args.npyFile,
            startTime=args.startTime, endTime=args.endTime, verbose=args.verbose,
            csvStartTime=args.csvStartTime, csvSampleRate=args.csvSampleRate,
            csvTimeFormat=args.csvTimeFormat, csvStartRow=args.csvStartRow,
            csvTimeXYZTempColsIndex=list(map(int, args.csvTimeXYZTempColsIndex.split(',')))
        )
    else:
        summary['file-name'] = args.epochFile

    # Summarise epoch
    epochData, labels = accelerometer.summarisation.getActivitySummary(
        args.epochFile, args.nonWearFile, summary,
        activityClassification=args.activityClassification,
        timeZone=args.timeZone, startTime=args.startTime,
        endTime=args.endTime, epochPeriod=args.epochPeriod,
        stationaryStd=args.stationaryStd,
        mgCpLPA=args.mgCpLPA, mgCpMPA=args.mgCpMPA, mgCpVPA=args.mgCpVPA,
        activityModel=args.activityModel,
        intensityDistribution=args.intensityDistribution,
        psd=args.psd, fourierFrequency=args.fourierFrequency,
        fourierWithAcc=args.fourierWithAcc, m10l5=args.m10l5)

    # Generate time series file
    accelerometer.utils.writeTimeSeries(epochData, labels, args.tsFile)

    # Print short summary
    accelerometer.utils.toScreen("=== Short summary ===")
    summaryVals = ['file-name', 'file-startTime', 'file-endTime',
                   'acc-overall-avg', 'wearTime-overall(days)',
                   'nonWearTime-overall(days)', 'quality-goodWearTime']
    summaryDict = collections.OrderedDict([(i, summary[i]) for i in summaryVals])
    print(json.dumps(summaryDict, indent=4))

    # Write summary to file
    with open(args.summaryFile, 'w') as f:
        json.dump(summary, f, indent=4)
    print('Full summary written to: ' + args.summaryFile)

    ##########################
    # Closing
    ##########################
    processingEndTime = datetime.datetime.now()
    processingTime = (processingEndTime - processingStartTime).total_seconds()
    accelerometer.utils.toScreen(
        "In total, processing took " + str(processingTime) + " seconds"
    )


def str2bool(v):
    """
    Used to parse true/false values from the command line. E.g. "True" -> True
    """

    return v.lower() in ("yes", "true", "t", "1")


def str2date(v):
    """
    Used to parse date values from the command line. E.g. "1994-11-30T12:00" -> time.datetime
    """

    eg = "1994-11-30T12:00"  # example date
    if v.count("-") != eg.count("-"):
        print("ERROR: Not enough dashes in date")
    elif v.count("T") != eg.count("T"):
        print("ERROR: No T seperator in date")
    elif v.count(":") != eg.count(":"):
        print("ERROR: No ':' seperator in date")
    elif len(v.split("-")[0]) != 4:
        print("ERROR: Year in date must be 4 numbers")
    elif len(v.split("-")[1]) != 2 and len(v.split("-")[1]) != 1:
        print("ERROR: Month in date must be 1-2 numbers")
    elif len(v.split("-")[2].split("T")[0]) != 2 and len(v.split("-")[2].split("T")[0]) != 1:
        print("ERROR: Day in date must be 1-2 numbers")
    else:
        return pd.datetime.strptime(v, "%Y-%m-%dT%H:%M")
    print("Please change your input date:")
    print('"' + v + '"')
    print("to match the example date format:")
    print('"' + eg + '"')
    raise ValueError("Date in incorrect format")


if __name__ == '__main__':
    main()  # Standard boilerplate to call the main() function to begin the program.
