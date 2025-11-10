"""Module to provide generic utilities for other accelerometer modules."""

import datetime
import json
import math
import os
import pandas as pd
import re
from tqdm.auto import tqdm

DAYS = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
TIME_SERIES_COL = 'time'


def str2bool(v):
    """Convert string to boolean for command-line argument parsing.

    Used to parse true/false values from the command line.

    :param str v: String value to convert
    :return: Boolean representation
    :rtype: bool

    Example
    -------
    >>> str2bool("True")
    True
    >>> str2bool("false")
    False
    >>> str2bool("1")
    True
    """
    return v.lower() in ("yes", "true", "t", "1")


def format_num(num, num_decimal_places):
    """Format a number to a specified number of decimal places."""
    return round(num, num_decimal_places)


def mean_sd_str(mean, std, num_decimal_places):
    """
    Return str of mean and stdev numbers formatted to number of decimalPlaces

    :param float mean: Mean number to be formatted.
    :param float std: Standard deviation number to be formatted.
    :param int num_decimal_places: Number of decimal places for output format
    :return: String formatted to number of decimalPlaces
    :rtype: str

    .. code-block:: python

        import accUtils
        accUtils.mean_sd_str(2.567, 0.089, 2)
    """
    out_str = str(format_num(mean, num_decimal_places))
    out_str += ' ('
    out_str += str(format_num(std, num_decimal_places))
    out_str += ')'
    return out_str


def mean_ci_str(mean, std, n, num_decimal_places):
    """
    Return str of mean and 95% confidence interval numbers formatted

    :param float mean: Mean number to be formatted.
    :param float std: Standard deviation number to be formatted.
    :param int n: Number of observations
    :param int num_decimal_places: Number of decimal places for output format
    :return: String formatted to number of decimalPlaces
    :rtype: str

    .. code-block:: python

        import accUtils
        accUtils.mean_ci_str(2.567, 0.089, 2, 2)
    """
    std_err = std / math.sqrt(n)
    lower_ci = mean - 1.96 * std_err
    upper_ci = mean + 1.96 * std_err
    out_str = str(format_num(mean, num_decimal_places))
    out_str += ' ('
    out_str += str(format_num(lower_ci, num_decimal_places))
    out_str += ' - '
    out_str += str(format_num(upper_ci, num_decimal_places))
    out_str += ')'
    return out_str


def to_screen(msg):
    """
    Print msg str prepended with current time

    :param str mgs: Message to be printed to screen
    :return: None. Prints msg str prepended with current time.

    .. code-block:: python

        import accUtils
        accUtils.to_screen("hello")
    """

    time_format = '%Y-%m-%d %H:%M:%S'
    print(f"\n{datetime.datetime.now().strftime(time_format)}\t{msg}")


def write_cmds(acc_dir, out_dir, cmds_file='list-of-commands.txt', acc_ext="cwa", cmd_options="", files_csv=None):
    """
    Generate a text file listing processing commands for files found under acc_dir/

    :param str acc_dir: Directory with accelerometer files to process
    :param str out_dir: Output directory to be created containing the processing results
    :param str cmds_file: Output .txt file listing all processing commands
    :param str acc_ext: Acc file type e.g. cwa, CWA, bin, BIN, gt3x...
    :param str cmd_options: String of processing options e.g. "--epochPeriod 10"
        Type 'python3 accProccess.py -h' for full list of options

    :return: None. New file written to <cmds_file>.

    .. code-block:: python

        import accUtils
        accUtils.write_cmds("myAccDir/", "myResults/", "myProcessCmds.txt")
    """

    # Use files_csv if provided, else retrieve all accel files under acc_dir/
    if files_csv in os.listdir(acc_dir):
        files_csv = pd.read_csv(os.path.join(acc_dir, files_csv), index_col="fileName")
        files_csv.index = acc_dir.rstrip("/") + "/" + files_csv.index.astype('str')
        file_paths = files_csv.index.to_numpy()

    else:
        files_csv = None
        # List all accelerometer files under acc_dir/
        file_paths = []
        acc_ext = acc_ext.lower()
        for root, dirs, files in os.walk(acc_dir):
            for file in files:
                if file.lower().endswith((acc_ext,
                                          acc_ext + ".gz",
                                          acc_ext + ".zip",
                                          acc_ext + ".bz2",
                                          acc_ext + ".xz")):
                    file_paths.append(os.path.join(root, file))

    with open(cmds_file, 'w') as f:
        for file_path in file_paths:

            # Use the file name as the output folder name for the process,
            # keeping the same directory structure of acc_dir/
            # Example: If file_path is {acc_dir}/group0/subject123.cwa then
            # output_folder will be {out_dir}/group0/subject123/
            output_folder = file_path.replace(acc_dir.rstrip("/"), out_dir.rstrip("/")).split(".")[0]

            # Enclose with single quotes to handle spaces
            file_path = "'" + file_path + "'"
            output_folder = "'" + output_folder + "'"

            cmd = f"accProcess {file_path} --outputFolder {output_folder} {cmd_options}"

            if files_csv is not None:
                # Grab additional options provided in files_csv (e.g. calibration params)
                cmd_options_csv = ' '.join(['--{} {}'.format(col, files_csv.loc[file_path, col])
                                            for col in files_csv.columns])
                cmd += " " + cmd_options_csv

            f.write(cmd)
            f.write('\n')

    print('List of commands written to ', cmds_file)


def collate_summary(results_dir, output_csv_file="all-summary.csv"):
    """
    Read all -summary.json files under <results_dir> and merge into one CSV file.
    Each json file represents summary data for one participant.
    Therefore output CSV file contains summary for all participants.

    :param str results_dir: Directory containing JSON files.
    :param str output_csv_file: Output CSV filename.

    :return: None. A new file is written to <output_csv_file>.

    .. code-block:: python

        import accUtils
        accUtils.collate_summary("data/", "data/all-summary.csv")
    """

    print(f"Scanning {results_dir} for summary files...")
    # Load all *-summary.json files under results_dir/
    summary_files = []
    json_dicts = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.lower().endswith("-summary.json"):
                summary_files.append(os.path.join(root, file))

    print(f"Found {len(summary_files)} summary files...")
    for file in tqdm(summary_files):
        with open(file, 'r') as f:
            json_dicts.append(json.load(f, object_pairs_hook=dict))

    summary = pd.DataFrame.from_dict(json_dicts)  # merge to a dataframe
    summary['eid'] = summary['file-name'].str.split('/').str[-1].str.split('.').str[0]  # infer ID from filename
    summary.to_csv(output_csv_file, index=False)
    print('Summary of', str(len(summary)), 'participants written to:', output_csv_file)


def identify_unprocessed_files(files_csv, summary_csv, output_files_csv):
    """
    Identify files that have not been processed.
    Look through all processed accelerometer files, and find participants who do
    not have records in the summary csv file. This indicates there was a problem
    in processing their data. Therefore, output will be a new .csv file to
    support reprocessing of these files.

    :param str files_csv: CSV listing acc files in study directory.
    :param str summary_csv: Summary CSV of processed dataset.
    :param str output_files_csv: Output csv listing files to be reprocessed.

    :return: None. A new file is written to <output_files_csv>.

    .. code-block:: python

        import accUtils
        accUtils.identify_unprocessed_files("study/files.csv", "study/summary-all-files.csv", "study/files-reprocess.csv")
    """

    file_list = pd.read_csv(files_csv)
    summary = pd.read_csv(summary_csv)

    output = file_list[~file_list['fileName'].isin(list(summary['file-name']))]
    output = output.rename(columns={'Unnamed: 1': ''})
    output.to_csv(output_files_csv, index=False)

    print('Reprocessing for ', len(output), 'participants written to:',
          output_files_csv)


def update_calibration_coefs(input_csv_file, output_csv_file):
    """
    Read summary .csv file and update coefs for those with poor calibration
    Look through all processed accelerometer files, and find participants that
    did not have good calibration data. Then assigns the calibration coefs from
    previous good use of a given device. Output will be a new .csv file to
    support reprocessing of uncalibrated files with new pre-specified calibration coefs.

    :param str input_csv_file: Summary CSV of processed dataset
    :param str output_csv_file: Output CSV of files to be reprocessed with new
        calibration info

    :return: None. New file written to <output_csv_file>

    .. code-block:: python

        import accUtils
        accUtils.update_calibration_coefs("data/summary-all-files.csv", "study/files-recalibration.csv")

    CSV of files to be reprocessed written to "study/files-recalibration.csv"
    """

    d = pd.read_csv(input_csv_file)
    # select participants with good spread of stationary values for calibration
    good_cal = d.loc[(d['quality-calibratedOnOwnData'] == 1) & (d['quality-goodCalibration'] == 1)]
    # now only select participants whose data was NOT calibrated on a good spread of stationary values
    bad_cal = d.loc[(d['quality-calibratedOnOwnData'] == 1) & (d['quality-goodCalibration'] == 0)]

    # sort files by start time, which makes selection of most recent value easier
    good_cal = good_cal.sort_values(['file-startTime'])
    bad_cal = bad_cal.sort_values(['file-startTime'])

    cal_cols = ['calibration-xOffset(g)', 'calibration-yOffset(g)', 'calibration-zOffset(g)',
                'calibration-xSlope(g)', 'calibration-ySlope(g)', 'calibration-zSlope(g)',
                'calibration-xTemp(C)', 'calibration-yTemp(C)', 'calibration-zTemp(C)',
                'calibration-meanDeviceTemp(C)']

    # print output CSV file with suggested calibration parameters
    no_other_uses = 0
    next_uses = 0
    previous_uses = 0
    with open(output_csv_file, 'w') as f:
        f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
        for ix, row in bad_cal.iterrows():
            # first get current 'bad' file
            participant, device, start_time = row[['file-name', 'file-deviceID', 'file-startTime']]
            device = int(device)
            # get calibration values from most recent previous use of this device
            # (when it had a 'good' calibration)
            prev_use = good_cal[cal_cols][(good_cal['file-deviceID'] == device) &
                                          (good_cal['file-startTime'] < start_time)].tail(1)
            try:
                of_x, of_y, of_z, slp_x, slp_y, slp_z, tmp_x, tmp_y, tmp_z, cal_temp_avg = prev_use.iloc[0]
                previous_uses += 1
            except (IndexError, KeyError):
                # No previous use found, try next use
                next_use = good_cal[cal_cols][(good_cal['file-deviceID'] == device) &
                                              (good_cal['file-startTime'] > start_time)].head(1)
                if len(next_use) < 1:
                    print('no other uses for this device at all: ', str(device),
                          str(participant))
                    no_other_uses += 1
                    continue
                next_uses += 1
                of_x, of_y, of_z, slp_x, slp_y, slp_z, tmp_x, tmp_y, tmp_z, cal_temp_avg = next_use.iloc[0]

            # now construct output
            out = participant + ','
            out += str(of_x) + ' ' + str(of_y) + ' ' + str(of_z) + ','
            out += str(slp_x) + ' ' + str(slp_y) + ' ' + str(slp_z) + ','
            out += str(tmp_x) + ' ' + str(tmp_y) + ' ' + str(tmp_z) + ','
            out += str(cal_temp_avg)
            f.write(out + '\n')
    print('previousUses', previous_uses)
    print('nextUses', next_uses)
    print('noOtherUses', no_other_uses)

    print('Reprocessing for ', str(previous_uses + next_uses),
          'participants written to:', output_csv_file)


def write_files_with_calibration_coefs(input_csv_file, output_csv_file):
    """
    Read summary .csv file and write files.csv with calibration coefs.
    Look through all processed accelerometer files, and write a new .csv file to
    support reprocessing of files with pre-specified calibration coefs.

    :param str input_csv_file: Summary CSV of processed dataset
    :param str output_csv_file: Output CSV of files to process with calibration info

    :return: None. New file written to <output_csv_file>

   .. code-block:: python

        import accUtils
        accUtils.write_files_with_calibration_coefs("data/summary-all-files.csv", "study/files-calibrated.csv")
    """

    d = pd.read_csv(input_csv_file)

    cal_cols = ['calibration-xOffset(g)', 'calibration-yOffset(g)', 'calibration-zOffset(g)',
                'calibration-xSlope(g)', 'calibration-ySlope(g)', 'calibration-zSlope(g)',
                'calibration-xTemp(C)', 'calibration-yTemp(C)', 'calibration-zTemp(C)',
                'calibration-meanDeviceTemp(C)']

    # print output CSV file with suggested calibration parameters
    with open(output_csv_file, 'w') as f:
        f.write('fileName,calOffset,calSlope,calTemp,meanTemp\n')
        for ix, row in d.iterrows():
            # first get current file information
            participant = str(row['file-name'])
            of_x, of_y, of_z, slp_x, slp_y, slp_z, tmp_x, tmp_y, tmp_z, cal_temp_avg = row[cal_cols]
            # now construct output
            out = participant + ','
            out += str(of_x) + ' ' + str(of_y) + ' ' + str(of_z) + ','
            out += str(slp_x) + ' ' + str(slp_y) + ' ' + str(slp_z) + ','
            out += str(tmp_x) + ' ' + str(tmp_y) + ' ' + str(tmp_z) + ','
            out += str(cal_temp_avg)
            f.write(out + '\n')

    print('Files with calibration coefficients for ', str(len(d)),
          'participants written to:', output_csv_file)


def date_parser(t):
    """
    Parse date a date string of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    """
    tz = re.search(r'(?<=\[).+?(?=\])', t)
    if tz is not None:
        tz = tz.group()
    t = re.sub(r'\[(.*?)\]', '', t)
    return pd.to_datetime(t, utc=True).tz_convert(tz)


def date_strftime(t):
    """
    Convert to time format of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    """
    tz = t.tz
    return t.strftime(f'%Y-%m-%d %H:%M:%S.%f%z [{tz}]')


def parse_time_string(time_str):
    """
    Parse a time string and return the value in hours.

    Supports formats like:
    - '20h' or '20H' -> 20 hours
    - '1200m' or '1200M' -> 20 hours (1200 minutes)
    - '0.5d' or '0.5D' -> 12 hours (0.5 days)
    - '20' -> 20 hours (default unit is hours)

    :param str time_str: Time string to parse
    :return: Time value in hours
    :rtype: float

    .. code-block:: python

        import accUtils
        hours = accUtils.parse_time_string('20h')  # returns 20.0
        hours = accUtils.parse_time_string('1200m')  # returns 20.0
    """
    if time_str is None:
        return None

    time_str = str(time_str).strip()

    # Extract number and unit
    match = re.match(r'^([0-9.]+)([hdmHDM]?)$', time_str)
    if not match:
        raise ValueError(f"Invalid time format: '{time_str}'. "
                         "Expected format: number followed by optional unit (h/m/d), "
                         "e.g., '20h', '1200m', '0.5d', or '20'")

    value = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else 'h'  # default to hours

    # Convert to hours
    if unit == 'h':
        return value
    elif unit == 'm':
        return value / 60.0
    elif unit == 'd':
        return value * 24.0
    else:
        raise ValueError(f"Unknown time unit: '{unit}'. Use 'h' (hours), 'm' (minutes), or 'd' (days)")


def write_time_series(epoch_data, labels, ts_file):
    """
    Write activity timeseries file

    :param pandas.DataFrame epoch_data: Pandas dataframe of epoch data. Must contain
        activity classification columns with missing rows imputed.
    :param list(str) labels: Activity state labels
    :param dict ts_file: output CSV filename

    """

    cols = ['acc'] + labels
    if 'MET' in epoch_data.columns:
        cols.append('MET')
    if 'imputed' in epoch_data.columns:
        cols.append('imputed')

    epoch_data = epoch_data[cols]

    # make output time format contain timezone
    # e.g. 2020-06-14 19:01:15.123000+0100 [Europe/London]
    epoch_data.index = epoch_data.index.to_series().apply(date_strftime)

    epoch_data.to_csv(ts_file, compression='gzip')
