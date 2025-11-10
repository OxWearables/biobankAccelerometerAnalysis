"""Command line tool to extract meaningful health info from accelerometer data."""

import accelerometer.utils
from accelerometer.utils import str2bool, discover_files
import accelerometer.classification
import argparse
import datetime
import accelerometer.device
import json
import os
import accelerometer.summarisation
import pandas as pd
import sys
import warnings
import traceback
from accelerometer.exceptions import AccelerometerException


def process_single_file(input_file, args):  # noqa: C901
    """
    Process a single accelerometer file.

    Parameters
    ----------
    input_file : str or Path
        Path to the input file
    args : argparse.Namespace
        Command line arguments (with inputFile replaced)

    Returns
    -------
    tuple
        (success: bool, error_msg: str or None, processing_time: float)
    """
    processing_start_time = datetime.datetime.now()

    # Convert to string for compatibility with existing code
    input_file = str(input_file)

    # Update args.inputFile to the current file
    args.inputFile = input_file

    # Parent folder and basename of input file
    input_file_name = os.path.basename(input_file).split(".")[0]

    # Create subdirectory within output folder based on input file name
    output_folder = os.path.join(args.outputFolder, input_file_name)
    output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    if not os.access(output_folder, os.W_OK):
        raise PermissionError(
            f"Either folder '{output_folder}' does not exist "
            "or you do not have write permission"
        )

    # Set default output filenames
    summary_file = os.path.join(output_folder, input_file_name + "-summary.json")
    epoch_file = os.path.join(output_folder, input_file_name + "-epoch.csv.gz")
    stationary_file = os.path.join(output_folder, input_file_name + "-stationaryPoints.csv.gz")
    ts_file = os.path.join(output_folder, input_file_name + "-timeSeries.csv.gz")
    raw_file = os.path.join(output_folder, input_file_name + ".csv.gz")
    npy_file = os.path.join(output_folder, input_file_name + ".npy")

    try:
        # Print processing options to screen
        print(f"Processing file '{input_file}' with these arguments:\n")
        for key, value in sorted(vars(args).items()):
            if not (isinstance(value, str) and len(value) == 0):
                print(key.ljust(25), ':', value)

        ##########################
        # Start processing file
        ##########################
        summary = {}
        # Now process the .CWA file
        if args.processInputFile:
            summary['file-name'] = input_file
            accelerometer.device.process_input_file_to_epoch(
                input_file, args.timeZone,
                args.timeShift, epoch_file, stationary_file, summary,
                skip_calibration=args.skipCalibration,
                stationary_std=args.stationaryStd, xyz_intercept=args.calOffset,
                xyz_slope=args.calSlope, xyz_slope_t=args.calTemp,
                raw_data_parser=args.rawDataParser, java_heap_space=args.javaHeapSpace,
                use_filter=args.useFilter, sample_rate=args.sampleRate, resample_method=args.resampleMethod,
                epoch_period=args.epochPeriod,
                extract_features=args.extractFeatures,
                raw_output=args.rawOutput, raw_file=raw_file,
                npy_output=args.npyOutput, npy_file=npy_file,
                start_time=args.startTime, end_time=args.endTime, verbose=args.verbose,
                csv_start_time=args.csvStartTime, csv_sample_rate=args.csvSampleRate,
                csv_time_format=args.csvTimeFormat, csv_start_row=args.csvStartRow,
                csv_time_xyz_temp_cols_index=list(map(int, args.csvTimeXYZTempColsIndex.split(',')))
            )
        else:
            summary['file-name'] = epoch_file

        # Summarise epoch
        epoch_data, labels = accelerometer.summarisation.get_activity_summary(
            epoch_file, summary,
            activity_classification=args.activityClassification,
            time_zone=args.timeZone, start_time=args.startTime,
            end_time=args.endTime, epoch_period=args.epochPeriod,
            stationary_std=args.stationaryStd, min_non_wear_duration=args.minNonWearDuration,
            mg_cp_lpa=args.mgCpLPA, mg_cp_mpa=args.mgCpMPA, mg_cp_vpa=args.mgCpVPA,
            remove_spurious_sleep=args.removeSpuriousSleep, remove_spurious_sleep_tol=args.removeSpuriousSleepTol,
            activity_model=args.activityModel,
            intensity_distribution=args.intensityDistribution,
            psd=args.psd, fourier_frequency=args.fourierFrequency,
            fourier_with_acc=args.fourierWithAcc, m10l5=args.m10l5,
            min_wear_per_day=args.minWearPerDay)

        # Generate time series file
        accelerometer.utils.write_time_series(epoch_data, labels, ts_file)

        # Print short summary
        accelerometer.utils.to_screen("=== Short summary ===")
        summary_vals = ['file-name', 'file-startTime', 'file-endTime',
                        'acc-overall-avg', 'wearTime-overall(days)',
                        'nonWearTime-overall(days)', 'quality-goodWearTime']
        summary_dict = {i: summary[i] for i in summary_vals}
        print(json.dumps(summary_dict, indent=4))

        # Write summary to file
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f'Full summary written to: {summary_file}')

        ##########################
        # Closing
        ##########################
        processing_end_time = datetime.datetime.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()
        accelerometer.utils.to_screen(
            f"In total, processing took {processing_time} seconds"
        )

        return (True, None, processing_time)

    except AccelerometerException as e:
        # Catch all accelerometer-specific exceptions
        processing_end_time = datetime.datetime.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()

        error_msg = str(e)
        if args.verbose:
            error_msg = traceback.format_exc()

        return (False, error_msg, processing_time)

    except (IOError, OSError, PermissionError) as e:
        # Catch file system errors (permissions, disk space, etc.)
        processing_end_time = datetime.datetime.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()

        error_msg = f"File system error: {e}"
        if args.verbose:
            error_msg = traceback.format_exc()

        return (False, error_msg, processing_time)

    except (ValueError, KeyError, TypeError) as e:
        # Catch data parsing/validation errors
        processing_end_time = datetime.datetime.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()

        error_msg = f"Data error: {e}"
        if args.verbose:
            error_msg = traceback.format_exc()

        return (False, error_msg, processing_time)

    except SystemExit as e:
        # Catch any remaining sys.exit() calls (should be rare now)
        processing_end_time = datetime.datetime.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()

        error_msg = f"Process exited with code {e.code}"
        if args.verbose:
            error_msg = traceback.format_exc()

        return (False, error_msg, processing_time)

    except Exception as e:
        # Catch-all for unexpected errors (last resort)
        # This ensures we always return gracefully rather than crashing
        processing_end_time = datetime.datetime.now()
        processing_time = (processing_end_time - processing_start_time).total_seconds()

        error_msg = f"Unexpected error: {e}"
        if args.verbose:
            error_msg = traceback.format_exc()

        return (False, error_msg, processing_time)

    finally:
        # Clean up intermediate files (always runs, even on exception)
        if args.deleteIntermediateFiles:
            try:
                if os.path.exists(stationary_file):
                    os.remove(stationary_file)
                if os.path.exists(epoch_file):
                    os.remove(epoch_file)
            except OSError:
                accelerometer.utils.to_screen('Could not delete intermediate files')


def process_batch(input_folder, args):  # noqa: C901
    """
    Process multiple accelerometer files in a folder.

    Parameters
    ----------
    input_folder : str or Path
        Path to the folder containing files
    args : argparse.Namespace
        Command line arguments

    Returns
    -------
    None
        Prints summary to console
    """
    batch_start_time = datetime.datetime.now()

    # Parse file extensions (required, validated in main())
    # Strip whitespace and dots, filter out empty strings
    extensions = [ext.strip().strip('.') for ext in args.fileExtensions.split(',') if ext.strip()]

    # Validate that we have at least one extension after filtering
    if not extensions:
        print("Error: --fileExtensions contains no valid extensions after parsing")
        print("Please provide at least one valid extension (e.g., --fileExtensions cwa)")
        sys.exit(-1)

    # Discover files
    print("=" * 80)
    print("BATCH PROCESSING MODE")
    print("=" * 80)
    print()

    ext_display = ', '.join(extensions)
    print(f"Searching for files with extensions: {ext_display}")
    print(f"Recursive search: {args.recursive}")
    print(f"Input folder: {os.path.abspath(input_folder)}")
    print()

    files = discover_files(input_folder, extensions, args.recursive)

    if not files:
        print("No files found matching the criteria.")
        print("Exiting with error.")
        sys.exit(-1)

    print(f"Found {len(files)} file(s) to process:")
    print()
    for i, file_path in enumerate(files, 1):
        print(f"  {i}. {file_path}")
    print()

    # Check for duplicate basenames (outputs would overwrite each other)
    basenames = {}
    for file_path in files:
        basename = file_path.name.split('.')[0]
        if basename in basenames:
            print()
            print("=" * 80)
            print("WARNING: Duplicate file basenames detected!")
            print("=" * 80)
            print()
            print("The following files have the same basename and will overwrite")
            print("each other's outputs in the same directory:")
            print()
            # Collect all duplicates
            duplicate_groups = {}
            for fp in files:
                bn = fp.name.split('.')[0]
                if bn not in duplicate_groups:
                    duplicate_groups[bn] = []
                duplicate_groups[bn].append(fp)
            # Print only the duplicates
            for bn, fps in duplicate_groups.items():
                if len(fps) > 1:
                    print(f"  Basename '{bn}':")
                    for fp in fps:
                        print(f"    - {fp}")
                    print()
            print("Please rename files or process them separately to avoid data loss.")
            print("=" * 80)
            print()
            sys.exit(-1)
        basenames[basename] = file_path

    # Process each file
    results = []

    for i, file_path in enumerate(files, 1):
        print("=" * 80)
        print(f"Processing file {i}/{len(files)}: {file_path.name}")
        print("=" * 80)
        print()

        success, error_msg, proc_time = process_single_file(file_path, args)

        results.append({
            'file': file_path.name,
            'path': file_path,
            'success': success,
            'error': error_msg,
            'time': proc_time
        })

        print()
        if success:
            print(f"✓ Successfully processed in {proc_time:.1f} seconds")
        else:
            print(f"✗ FAILED after {proc_time:.1f} seconds")
            if error_msg:
                print(f"Error: {error_msg}")
        print()

    # Print summary
    batch_end_time = datetime.datetime.now()
    total_time = (batch_end_time - batch_start_time).total_seconds()

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print("=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print()
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f} seconds")
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average time per successful file: {avg_time:.1f} seconds")
    print()

    if failed:
        print("Failed files:")
        for r in failed:
            print(f"  ✗ {r['file']}")
            if r['error']:
                # Print error (full traceback if verbose, otherwise first line)
                if args.verbose:
                    # Indent each line of the error for readability
                    for line in r['error'].split('\n'):
                        if line:
                            print(f"    {line}")
                else:
                    # Print first line only for brevity
                    error_line = r['error'].split('\n')[0]
                    print(f"    Error: {error_line}")
        print()

    if successful:
        print("Successful files:")
        for r in successful:
            print(f"  ✓ {r['file']} ({r['time']:.1f}s)")
        print()

    print("=" * 80)

    # Exit with error code if any files failed
    if failed:
        sys.exit(1)


def main():  # noqa: C901
    """
    Application entry point responsible for parsing command line requests
    """

    parser = argparse.ArgumentParser(
        description="""A tool to extract physical activity information from
            raw accelerometer files.""", add_help=True
    )
    # required
    parser.add_argument('inputFile', metavar='input file or folder', type=str,
                        help="""the <.cwa/.cwa.gz> file to process
                            (e.g. sample.cwa.gz), or a folder containing multiple files.
                            If the file path contains spaces, it must be enclosed in quote marks
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
                        metavar='e.g. 2000-12-31 23:59:59', default=None,
                        type=str, help="""removes data before this
                            time (local) in the final analysis
                            (default : %(default)s)""")
    parser.add_argument('--endTime',
                        metavar='e.g 2000-12-31 23:59:59', default=None,
                        type=str, help="""removes data after this
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
                        metavar='e.g. 2000-12-31 23:59:59', default=None,
                        type=str, help="""start time for csv file
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
    parser.add_argument('--minNonWearDuration',
                        metavar='mins', default=60, type=int,
                        help="""minimum non-wear duration in minutes
                            (default : %(default)s mins))""")
    parser.add_argument('--minWearPerDay',
                        metavar="e.g. '20h', '1200m'", default=None, type=str,
                        help="""minimum wear time per day for a day to be included
                            in summary statistics. Days with less wear time will be
                            excluded. Supports formats: '20h' (hours), '1200m' (minutes),
                            '0.5d' (days), or '20' (hours by default).
                            (default : %(default)s (all days included))""")
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
    parser.add_argument('--removeSpuriousSleep',
                        metavar='True/False', default=True, type=str2bool,
                        help="""Remove spurious sleep periods from the
                            activity classification? (default : %(default)s)""")
    parser.add_argument('--removeSpuriousSleepTol',
                        metavar='mins', default=60, type=int,
                        help="""Sleep tolerance in minutes. If `--removeSpuriousSleep`
                            and a sleep streak is shorter than this, it will be replaced
                            with sedentary activity (default : %(default)s)""")

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
    parser.add_argument('--outputFolder', '-o', metavar='filename', default='outputs',
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
    # batch processing options
    parser.add_argument('--fileExtensions',
                        metavar='e.g. cwa,bin,csv', default=None, type=str,
                        help="""comma-separated list of file extensions to process
                            when input is a folder (e.g. "cwa,bin,csv").
                            Automatically includes compressed versions (.gz, .zip, etc.).
                            REQUIRED when processing a folder.
                            (default : %(default)s)""")
    parser.add_argument('--recursive',
                        metavar='True/False', default=False, type=str2bool,
                        help="""recursively search subdirectories when input is a folder
                            (default : %(default)s)""")

    args = parser.parse_args()

    # Parse minWearPerDay time string to hours
    if args.minWearPerDay is not None:
        try:
            args.minWearPerDay = accelerometer.utils.parse_time_string(args.minWearPerDay)
        except ValueError as e:
            print(f"Error parsing --minWearPerDay: {e}")
            sys.exit(-1)

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

    # Check user-specified end time is not before start time
    if args.startTime and args.endTime:
        assert pd.Timestamp(args.startTime) <= pd.Timestamp(args.endTime), (
            "startTime and endTime arguments are invalid!\n"
            f"startTime: {args.startTime}\n"
            f"endTime: {args.endTime}\n"
        )

    # Ensure output folder is absolute path
    args.outputFolder = os.path.abspath(args.outputFolder)

    # Determine if input is a file or directory
    if os.path.isdir(args.inputFile):
        # Batch processing mode - require fileExtensions to be specified
        if args.fileExtensions is None:
            print("Error: --fileExtensions must be specified when processing a folder")
            print("Example: accProcess /path/to/folder --fileExtensions cwa,bin,csv")
            sys.exit(-1)
        process_batch(args.inputFile, args)
    elif os.path.isfile(args.inputFile):
        # Single file processing mode
        # Note: outputFolder will be set inside process_single_file to include subdirectory
        success, error_msg, processing_time = process_single_file(args.inputFile, args)

        if not success:
            print(f"\n✗ Processing failed: {error_msg}")
            sys.exit(-1)
    else:
        print(f"Error: Input path '{args.inputFile}' is neither a file nor a directory")
        sys.exit(-1)



if __name__ == '__main__':
    main()  # Standard boilerplate to call the main() function to begin the program.
