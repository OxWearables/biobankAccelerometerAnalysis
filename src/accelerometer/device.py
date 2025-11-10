"""Module to process raw accelerometer files into epoch data."""

from accelerometer import utils
from accelerometer.exceptions import CalibrationError, DeviceError, ProcessingError
import gzip
import numpy as np
import os
import dateutil
import pandas as pd
import statsmodels.api as sm
import struct
from subprocess import call
import pathlib
from typing import Dict, List, Optional, Union

ROOT_DIR = pathlib.Path(__file__).parent

# ============================================================================
# CALIBRATION CONSTANTS
# ============================================================================
# These constants control the auto-calibration algorithm that eliminates
# factory calibration dependency. The algorithm uses stationary periods to
# optimize 9 parameters (offset, slope, temp coefficients) via iterative
# weighted least squares, targeting a unit gravity sphere (|acceleration| = 1g).
#
# These values are scientifically validated - DO NOT modify without proper
# justification and validation against published results (Doherty et al. 2017).
# ============================================================================

# Maximum iterations for calibration optimization
CALIBRATION_MAX_ITERATIONS = 1000

# Improvement tolerance: stop if improvement < 0.01% between iterations
CALIBRATION_IMPROVEMENT_TOLERANCE = 0.0001

# Error tolerance: calibration fails if mean error > 10mg (scientifically validated)
# This threshold ensures calibrated data meets accelerometer accuracy standards
CALIBRATION_ERROR_TOLERANCE = 0.01

# Calibration cube: minimum axis range (±0.3g) required for reliable calibration
# Ensures sufficient angular diversity in stationary positions
CALIBRATION_CUBE_THRESHOLD = 0.3

# Minimum number of stationary samples required for calibration
# Fewer samples may not provide sufficient coverage of orientation space
CALIBRATION_MIN_SAMPLES = 50


def process_input_file_to_epoch(  # noqa: C901
    input_file: str,
    time_zone: str,
    time_shift: int,
    epoch_file: str,
    stationary_file: str,
    summary: Dict,
    skip_calibration: bool = False,
    stationary_std: int = 13,
    xyz_intercept: Optional[List[float]] = None,
    xyz_slope: Optional[List[float]] = None,
    xyz_slope_t: Optional[List[float]] = None,
    raw_data_parser: str = "AccelerometerParser",
    java_heap_space: Optional[int] = None,
    use_filter: bool = True,
    sample_rate: int = 100,
    resample_method: str = "linear",
    epoch_period: int = 30,
    extract_features: bool = True,
    raw_output: bool = False,
    raw_file: Optional[str] = None,
    npy_output: bool = False,
    npy_file: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    verbose: bool = False,
    csv_start_time: Optional[str] = None,
    csv_sample_rate: Optional[float] = None,
    csv_time_format: str = "yyyy-MM-dd HH:mm:ss.SSSxxxx '['VV']'",
    csv_start_row: int = 1,
    csv_time_xyz_temp_cols_index: Optional[str] = None
) -> None:
    """
    Process raw accelerometer file, writing summary epoch stats to file. This is usually achieved by:
    1) identify 10sec stationary epochs
    2) record calibrated axes scale/offset/temp vals + static point stats
    3) use calibration coefficients and then write filtered avgVm epochs to <epoch_file> from <input_file>

    :param str input_file: Input <cwa/cwa.gz/bin/gt3x> raw accelerometer file
    :param str epoch_file: Output csv.gz file of processed epoch data
    :param str stationary_file: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics
    :param bool skip_calibration: Perform software calibration (process data twice)
    :param int stationary_std: Gravity threshold (in mg units) for stationary vs not
    :param list(float) xyz_intercept: Calibration offset [x, y, z]. Defaults to [0.0, 0.0, 0.0]
    :param list(float) xyz_slope: Calibration slope [x, y, z]. Defaults to [1.0, 1.0, 1.0]
    :param list(float) xyz_temp: Calibration temperature coefficient [x, y, z]. Defaults to [0.0, 0.0, 0.0]
    :param str raw_data_parser: External helper process to read raw acc file. If a
        java class, it must omit .class ending.
    :param str java_heap_space: Amount of heap space allocated to java subprocesses.
        Useful for limiting RAM usage.
    :param bool use_filter: Filter ENMOtrunc signal
    :param int sample_rate: Resample data to n Hz
    :param int epoch_period: Size of epoch time window (in seconds)
    :param bool activity_classification: Extract features for machine learning
    :param bool raw_output: Output calibrated and resampled raw data to a .csv.gz
        file? requires ~50MB/day.
    :param str raw_file: Output raw data ".csv.gz" filename
    :param bool npy_output: Output calibrated and resampled raw data to a .npy
        file? requires ~60MB/day.
    :param str npy_file: Output raw data ".npy" filename
    :param datetime start_time: Remove data before this time in analysis
    :param datetime end_time: Remove data after this time in analysis
    :param bool verbose: Print verbose output
    :param datetime csv_start_time: start time for csv file when time column is not available
    :param float csv_sample_rate: sample rate for csv file when time column is not available
    :param str csv_time_format: time format for csv file when time column is available
    :param int csv_start_row: start row for accelerometer data in csv file
    :param str csv_time_xyz_temp_cols_index: index of column positions for XYZT columns, e.g. "1,2,3,0"

    :return: None. Writes raw processing summary values to dict <summary>.

    .. code-block:: python

        import device
        summary = {}
        device.process_input_file_to_epoch('inputFile.cwa', 'epochFile.csv.gz',
                'stationary.csv.gz', summary)
    """
    # Initialize mutable default arguments to avoid shared state bugs
    if xyz_intercept is None:
        xyz_intercept = [0.0, 0.0, 0.0]
    if xyz_slope is None:
        xyz_slope = [1.0, 1.0, 1.0]
    if xyz_slope_t is None:
        xyz_slope_t = [0.0, 0.0, 0.0]

    summary['file-size'] = os.path.getsize(input_file)
    summary['file-deviceID'] = get_device_id(input_file)
    use_java = True
    path_separator = ';' if os.name == 'nt' else ':'
    java_class_path = f"{ROOT_DIR}/java/{path_separator}{ROOT_DIR}/java/JTransforms-3.1-with-dependencies.jar"
    static_std_g = stationary_std / 1000.0  # java expects units of G (not mg)

    if 'omconvert' in raw_data_parser:
        use_java = False

    if use_java:
        if not skip_calibration:
            # identify 10sec stationary epochs
            utils.to_screen("=== Calibrating ===")
            command_args = ["java", "-classpath", java_class_path,
                            "-XX:ParallelGCThreads=1", raw_data_parser, input_file,
                            f"timeZone:{time_zone}",
                            f"timeShift:{time_shift}",
                            f"outputFile:{stationary_file}",
                            f"verbose:{verbose}",
                            f"filter:{use_filter}",
                            "getStationaryBouts:true", "epochPeriod:10",
                            f"stationaryStd:{static_std_g}",
                            f"sampleRate:{sample_rate}"]
            if java_heap_space:
                command_args.insert(1, java_heap_space)
            if csv_start_time:
                command_args.append(f"csvStartTime:{to_iso_datetime(csv_start_time)}")
            if csv_sample_rate:
                command_args.append(f"csvSampleRate:{csv_sample_rate}")
            if csv_time_format:
                command_args.append(f"csvTimeFormat:{csv_time_format}")
            if csv_start_row is not None:
                command_args.append(f"csvStartRow:{csv_start_row}")
            if csv_time_xyz_temp_cols_index:
                java_str_csv_txyz = ','.join([str(i) for i in csv_time_xyz_temp_cols_index])
                command_args.append(f"csvTimeXYZTempColsIndex:{java_str_csv_txyz}")
            # call process to identify stationary epochs
            exit_code = call(command_args)
            if exit_code != 0:
                print(command_args)
                print("Error: java calibration failed, exit ", exit_code)
                raise CalibrationError(
                    f"Java calibration process failed with exit code {exit_code}. "
                    f"Command: {' '.join(command_args)}"
                )
            # record calibrated axes scale/offset/temp vals + static point stats
            get_calibration_coefs(stationary_file, summary)
            xyz_intercept = [summary['calibration-xOffset(g)'],
                             summary['calibration-yOffset(g)'],
                             summary['calibration-zOffset(g)']]
            xyz_slope = [summary['calibration-xSlope'],
                         summary['calibration-ySlope'],
                         summary['calibration-zSlope']]
            xyz_slope_t = [summary['calibration-xSlopeTemp'],
                           summary['calibration-ySlopeTemp'],
                           summary['calibration-zSlopeTemp']]
        else:
            store_calibration_params(summary, xyz_intercept, xyz_slope, xyz_slope_t)
            summary['quality-calibratedOnOwnData'] = 0
            summary['quality-goodCalibration'] = 1

        utils.to_screen('=== Extracting features ===')
        command_args = ["java", "-classpath", java_class_path,
                        "-XX:ParallelGCThreads=1", raw_data_parser, input_file,
                        f"timeZone:{time_zone}",
                        f"timeShift:{time_shift}",
                        f"outputFile:{epoch_file}", f"verbose:{verbose}",
                        f"filter:{use_filter}",
                        f"sampleRate:{sample_rate}",
                        f"resampleMethod:{resample_method}",
                        f"xIntercept:{xyz_intercept[0]}",
                        f"yIntercept:{xyz_intercept[1]}",
                        f"zIntercept:{xyz_intercept[2]}",
                        f"xSlope:{xyz_slope[0]}",
                        f"ySlope:{xyz_slope[1]}",
                        f"zSlope:{xyz_slope[2]}",
                        f"xSlopeT:{xyz_slope_t[0]}",
                        f"ySlopeT:{xyz_slope_t[1]}",
                        f"zSlopeT:{xyz_slope_t[2]}",
                        f"epochPeriod:{epoch_period}",
                        f"rawOutput:{raw_output}",
                        f"rawFile:{raw_file}",
                        f"npyOutput:{npy_output}",
                        f"npyFile:{npy_file}",
                        f"getFeatures:{extract_features}"]
        if java_heap_space:
            command_args.insert(1, java_heap_space)
        if start_time:
            command_args.append(f"startTime:{to_iso_datetime(start_time)}")
        if end_time:
            command_args.append(f"endTime:{to_iso_datetime(end_time)}")
        if csv_start_time:
            command_args.append(f"csvStartTime:{csv_start_time.strftime('%Y-%m-%dT%H:%M')}")
        if csv_sample_rate:
            command_args.append(f"csvSampleRate:{csv_sample_rate}")
        if csv_time_format:
            command_args.append(f"csvTimeFormat:{csv_time_format}")
        if csv_start_row:
            command_args.append(f"csvStartRow:{csv_start_row}")
        if csv_time_xyz_temp_cols_index:
            java_str_csv_txyz = ','.join([str(i) for i in csv_time_xyz_temp_cols_index])
            command_args.append(f"csvTimeXYZTempColsIndex:{java_str_csv_txyz}")
        exit_code = call(command_args)
        if exit_code != 0:
            print(command_args)
            print("Error: Java epoch generation failed, exit ", exit_code)
            raise ProcessingError(
                f"Java epoch generation process failed with exit code {exit_code}. "
                f"Command: {' '.join(command_args)}"
            )

    else:
        if not skip_calibration:
            command_args = [raw_data_parser, input_file, time_zone, time_shift,
                            "-svm-file", epoch_file, "-info", stationary_file,
                            "-svm-extended", "3", "-calibrate", "1",
                            "-interpolate-mode", "2",
                            "-svm-mode", "1", "-svm-epoch", str(epoch_period),
                            "-svm-filter", "2"]
        else:
            cal_args = str(xyz_slope[0]) + ','
            cal_args += str(xyz_slope[1]) + ','
            cal_args += str(xyz_slope[2]) + ','
            cal_args += str(xyz_intercept[0]) + ','
            cal_args += str(xyz_intercept[1]) + ','
            cal_args += str(xyz_intercept[2]) + ','
            cal_args += str(xyz_slope_t[0]) + ','
            cal_args += str(xyz_slope_t[1]) + ','
            cal_args += str(xyz_slope_t[2]) + ','
            command_args = [raw_data_parser, input_file, time_zone, time_shift,
                            "-svm-file", epoch_file, "-info", stationary_file,
                            "-svm-extended", "3", "-calibrate", "0",
                            "-calibration", cal_args, "-interpolate-mode", "2",
                            "-svm-mode", "1", "-svm-epoch", str(epoch_period),
                            "-svm-filter", "2"]
        call(command_args)
        get_omconvert_info(stationary_file, summary)


def get_calibration_coefs(static_bouts_file: Union[str, pd.DataFrame], summary: Dict) -> None:
    """
    Identify calibration coefficients from java processed file. Get axes
    offset/gain/temp calibration coefficients through linear regression of
    stationary episodes.

    :param str static_bouts_file: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics

    :return: None. Calibration summary values written to dict <summary>
    """

    if isinstance(static_bouts_file, pd.DataFrame):
        calibration_data = static_bouts_file

    else:
        calibration_data = pd.read_csv(static_bouts_file)

    calibration_data = calibration_data[calibration_data['dataErrors'] == 0].dropna()  # drop segments with errors
    xyz = calibration_data[['xMean', 'yMean', 'zMean']].to_numpy()
    if 'temp' in calibration_data:
        temperature = calibration_data['temp'].to_numpy()
    else:  # use a dummy
        temperature = np.zeros(len(xyz), dtype=xyz.dtype)

    # Remove any zero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]
    temperature = temperature[nonzero]

    intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
    slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
    slope_t = np.array([0.0, 0.0, 0.0], dtype=temperature.dtype)
    best_intercept = np.copy(intercept)
    best_slope = np.copy(slope)
    best_slope_t = np.copy(slope_t)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr - target, axis=1)
    err = np.mean(errors)  # MAE more robust than RMSE. This is different from the paper
    init_err = err
    best_err = 1e16
    n_static = len(xyz)

    # Check that we have enough uniformly distributed points:
    # need at least one point outside each face of the cube
    if (len(xyz) < CALIBRATION_MIN_SAMPLES or
            (np.max(xyz, axis=0) < CALIBRATION_CUBE_THRESHOLD).any() or
            (np.min(xyz, axis=0) > -CALIBRATION_CUBE_THRESHOLD).any()):
        good_calibration = 0

    else:  # we do have enough uniformly distributed points

        for it in range(CALIBRATION_MAX_ITERATIONS):

            # Weighting. Outliers are zeroed out
            # This is different from the paper
            maxerr = np.quantile(errors, .995)
            weights = np.maximum(1 - errors / maxerr, 0)

            # Optimize params for each axis
            for k in range(3):

                inp = curr[:, k]
                out = target[:, k]
                inp = np.column_stack((inp, temperature))
                inp = sm.add_constant(inp, prepend=True, has_constant='add')
                params = sm.WLS(out, inp, weights=weights).fit().params
                # In the following,
                # intercept == params[0]
                # slope == params[1]
                # slope_t == params[2]
                intercept[k] = params[0] + (intercept[k] * params[1])
                slope[k] = params[1] * slope[k]
                slope_t[k] = params[2] + (slope_t[k] * params[1])

            # Update current solution and target
            curr = intercept + (xyz * slope)
            curr = curr + (temperature[:, None] * slope_t)
            target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

            # Update errors
            errors = np.linalg.norm(curr - target, axis=1)
            err = np.mean(errors)
            err_improv = (best_err - err) / best_err

            if err < best_err:
                best_intercept = np.copy(intercept)
                best_slope = np.copy(slope)
                best_slope_t = np.copy(slope_t)
                best_err = err

            if err_improv < CALIBRATION_IMPROVEMENT_TOLERANCE:
                break

        good_calibration = int(not ((best_err > CALIBRATION_ERROR_TOLERANCE) or (it + 1 == CALIBRATION_MAX_ITERATIONS)))

    if good_calibration == 0:  # restore calibr params
        best_intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
        best_slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
        best_slope_t = np.array([0.0, 0.0, 0.0], dtype=temperature.dtype)
        best_err = init_err

    store_calibration_information(
        summary,
        best_intercept=best_intercept,
        best_slope=best_slope,
        best_slope_t=best_slope_t,
        init_err=init_err,
        best_err=best_err,
        n_static=n_static,
        calibrated_on_own_data=1,
        good_calibration=good_calibration
    )

    return


def get_omconvert_info(omconvert_info_file: str, summary: Dict) -> None:
    """
    Identify calibration coefficients for omconvert processed file. Get axes
    offset/gain/temp calibration coeffs from omconvert info file.

    :param str omconvert_info_file: Output information file from omconvert
    :param dict summary: Output dictionary containing all summary metrics

    :return: None. Calibration summary values written to dict <summary>
    """

    with open(omconvert_info_file, 'r') as info_file:
        for line in info_file:
            elements = line.split(':')
            name, value = elements[0], elements[1]
            if name == 'Calibration':
                vals = value.split(',')
                best_intercept = float(vals[3]), float(vals[4]), float(vals[5])
                best_slope = float(vals[0]), float(vals[1]), float(vals[2])
                best_slope_t = float(vals[6]), float(vals[7]), float(vals[8])
            elif name == 'Calibration-Stationary-Error-Pre':
                init_error = float(value)
            elif name == 'Calibration-Stationary-Error-Post':
                best_error = float(value)
            elif name == 'Calibration-Stationary-Min':
                vals = value.split(',')
                x_min, y_min, z_min = float(vals[0]), float(vals[1]), float(vals[2])
            elif name == 'Calibration-Stationary-Max':
                vals = value.split(',')
                x_max, y_max, z_max = float(vals[0]), float(vals[1]), float(vals[2])
            elif name == 'Calibration-Stationary-Count':
                n_static = int(value)
    # store output to summary dictionary
    store_calibration_information(summary, best_intercept, best_slope, best_slope_t,
                                  init_error, best_error, n_static, None, None)


def store_calibration_information(
        summary: Dict,
        best_intercept: Union[List[float], np.ndarray],
        best_slope: Union[List[float], np.ndarray],
        best_slope_t: Union[List[float], np.ndarray],
        init_err: float,
        best_err: float,
        n_static: int,
        calibrated_on_own_data: int,
        good_calibration: int
) -> None:
    """
    Store calibration information to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) best_intercept: Best x/y/z intercept values
    :param list(float) best_slope: Best x/y/z slope values
    :param list(float) best_slope_t: Best x/y/z temperature slope values
    :param float init_err: Error (in mg) before calibration
    :param float best_err: Error (in mg) after calibration
    :param int n_static: number of stationary points used for calibration
    :param calibrated_on_own_data: Whether params were self-derived
    :param good_calibration: Whether calibration succeded

    :return: None. Calibration summary values written to dict <summary>
    """

    # store output to summary dictionary
    store_calibration_params(summary, best_intercept, best_slope, best_slope_t)
    summary['calibration-errsBefore(mg)'] = init_err * 1000
    summary['calibration-errsAfter(mg)'] = best_err * 1000
    summary['calibration-numStaticPoints'] = n_static
    summary['quality-calibratedOnOwnData'] = calibrated_on_own_data
    summary['quality-goodCalibration'] = good_calibration


def store_calibration_params(
        summary: Dict,
        xyz_off: Union[List[float], np.ndarray],
        xyz_slope: Union[List[float], np.ndarray],
        xyz_slope_t: Union[List[float], np.ndarray]
) -> None:
    """
    Store calibration parameters to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) xyz_off: intercept [x, y, z]
    :param list(float) xyz_slope: slope [x, y, z]
    :param list(float) xyz_slope_t: temperature slope [x, y, z]

    :return: None. Calibration summary values written to dict <summary>
    """

    # store output to summary dictionary
    summary['calibration-xOffset(g)'] = xyz_off[0]
    summary['calibration-yOffset(g)'] = xyz_off[1]
    summary['calibration-zOffset(g)'] = xyz_off[2]
    summary['calibration-xSlope'] = xyz_slope[0]
    summary['calibration-ySlope'] = xyz_slope[1]
    summary['calibration-zSlope'] = xyz_slope[2]
    summary['calibration-xSlopeTemp'] = xyz_slope_t[0]
    summary['calibration-ySlopeTemp'] = xyz_slope_t[1]
    summary['calibration-zSlopeTemp'] = xyz_slope_t[2]


def get_device_id(input_file: str) -> Optional[str]:
    """
    Get serial number of device. First decides which DeviceId parsing method to use for <input_file>.

    :param str input_file: Input raw accelerometer file

    :return: Device ID
    :rtype: str
    """

    if input_file.lower().endswith('.bin'):
        return get_genea_device_id(input_file)
    elif input_file.lower().endswith('.cwa') or input_file.lower().endswith('.cwa.gz'):
        return get_axivity_device_id(input_file)
    elif input_file.lower().endswith('.gt3x'):
        return get_gt3x_device_id(input_file)
    elif input_file.lower().endswith('.csv') or input_file.lower().endswith('.csv.gz'):
        return "unknown (.csv)"
    else:
        print(f"ERROR: Cannot get deviceId for file: {input_file}")


def get_axivity_device_id(cwa_file: str) -> str:
    """
    Get serial number of Axivity device. Parses the unique serial code from the
    header of an Axivity accelerometer file.

    :param str cwa_file: Input raw .cwa accelerometer file

    :return: Device ID
    :rtype: str
    """
    if cwa_file.lower().endswith('.cwa'):
        open_func = open
    elif cwa_file.lower().endswith('.cwa.gz'):
        open_func = gzip.open

    with open_func(cwa_file, 'rb') as device_file:
        header = device_file.read(2)
        if header == b'MD':
            block_size = struct.unpack('H', device_file.read(2))[0]
            perform_clear = struct.unpack('B', device_file.read(1))[0]
            device_id = struct.unpack('H', device_file.read(2))[0]
        else:
            print(f"ERROR: in get_device_id(\"{cwa_file}\")")
            print("""A deviceId value could not be found in input file header,
             this usually occurs when the file is not an Axivity .cwa accelerometer
             file.""")
            raise DeviceError(
                f"Device ID not found in file header for '{cwa_file}'. "
                "This usually occurs when the file is not an Axivity .cwa accelerometer file."
            )
    return str(device_id)


def get_genea_device_id(bin_file: str) -> str:
    """
    Get serial number of GENEActiv device. Parses the unique serial code from
    the header of a GENEActiv accelerometer file

    :param str bin_file: Input raw .bin accelerometer file

    :return: Device ID
    :rtype: str
    """

    with open(bin_file, 'r') as genea_file:  # 'Universal' newline mode
        next(genea_file)  # Device Identity
        device_id = next(genea_file).split(':')[1].rstrip()  # Device Unique Serial Code:011710
    return device_id


def get_gt3x_device_id(gt3x_file: str) -> Optional[str]:
    """
    Get serial number of Actigraph device. Parses the unique serial code from
    the header of a GT3X accelerometer file

    :param str gt3x_file: Input raw .gt3x accelerometer file

    :return: Device ID
    :rtype: str
    """

    import zipfile
    if zipfile.is_zipfile(gt3x_file):
        with zipfile.ZipFile(gt3x_file, 'r') as z:
            contents = z.infolist()
            print("\n".join(map(lambda x: str(x.filename).rjust(20, " ") + ", "
                                + str(x.file_size), contents)))

            if 'info.txt' in map(lambda x: x.filename, contents):
                print('info.txt found..')
                info_file = z.open('info.txt', 'r')
                # print info_file.read()
                for line in info_file:
                    if line.startswith(b"Serial Number:"):
                        newline = line.decode("utf-8")
                        newline = newline.split("Serial Number: ")[1]
                        print(f"Serial Number: {newline}")
                        return newline
            else:
                print("Could not find info.txt file")

    print(f"ERROR: in get_device_id(\"{gt3x_file}\")")
    print("""A deviceId value could not be found in input file header,
     this usually occurs when the file is not an Actigraph .gt3x accelerometer
     file.""")
    raise DeviceError(
        f"Device ID not found in file header for '{gt3x_file}'. "
        "This usually occurs when the file is not an Actigraph .gt3x accelerometer file."
    )


def to_iso_datetime(dt: str) -> str:
    """ Given input string representing a datetime, return its ISO formatted
    datetime string. """
    return dateutil.parser.parse(dt).isoformat()
