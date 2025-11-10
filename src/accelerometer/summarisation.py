"""Module to generate overall activity summary from epoch data."""
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from accelerometer import utils
from accelerometer import classification
from accelerometer import circadian
from accelerometer.exceptions import DataError


def get_activity_summary(  # noqa: C901
    epoch_file, summary,
    activity_classification=True, time_zone='Europe/London',
    start_time=None, end_time=None,
    epoch_period=30, stationary_std=13, min_non_wear_duration=60,
    mg_cp_lpa=45, mg_cp_mpa=100, mg_cp_vpa=400,
    remove_spurious_sleep=True, remove_spurious_sleep_tol=60,
    activity_model="walmsley",
    intensity_distribution=False, imputation=True,
    psd=False, fourier_frequency=False, fourier_with_acc=False, m10l5=False,
    min_wear_per_day=None
):
    """
    Calculate overall activity summary from <epoch_file> data. Get overall
    activity summary from input <epoch_file>. This is achieved by:
    1) get interrupt and data error summary vals
    2) check if data occurs at a daylight savings crossover
    3) calculate wear-time statistics, and write nonWear episodes to file
    4) predict activity from features, and add label column
    5) calculate imputation values to replace nan PA metric values
    6) calculate empirical cumulative distribution function of vector magnitudes
    7) derive main movement summaries (overall, weekday/weekend, and hour)

    :param str epoch_file: Input csv.gz file of processed epoch data
    :param dict summary: Output dictionary containing all summary metrics
    :param bool activity_classification: Perform machine learning of activity states
    :param str time_zone: timezone in country/city format to be used for daylight savings crossover check
    :param datetime start_time: Remove data before this time in analysis
    :param datetime end_time: Remove data after this time in analysis
    :param int epoch_period: Size of epoch time window (in seconds)
    :param int stationary_std: Threshold (in mg units) for stationary vs not
    :param int min_non_wear_duration: Minimum duration of nonwear events (minutes)
    :param int mg_cp_lpa: Milli-gravity threshold for light intensity activity
    :param int mg_cp_mpa: Milli-gravity threshold for moderate intensity activity
    :param int mg_cp_vpa: Milli-gravity threshold for vigorous intensity activity
    :param bool remove_spurious_sleep: Remove spurious sleep epochs
    :param int remove_spurious_sleep_tol: Tolerance (in minutes) for spurious sleep removal
    :param str activity_model: Input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METS for each activity state
    :param bool intensity_distribution: Add intensity outputs to dict <summary>
    :param bool imputation: Impute missing data using data from other days around the same time
    :param bool verbose: Print verbose output
    :param float min_wear_per_day: Minimum wear time (in hours) required for a day to be
        included in summary statistics. Days with less wear time will be excluded.
        If None, all days are included.

    :return: A tuple containing a pandas dataframe of activity epoch data,
        activity prediction labels (empty if <activity_classification>==False), and
        movement summary values written to dict <summary>.
    :rtype: tuple

    .. code-block:: python

        from accelerometer import summarisation
        summary = {}
        epochData, labels = summarisation.get_activity_summary("epoch.csv.gz", summary)
    """

    utils.to_screen("=== Summarizing ===")

    if isinstance(epoch_file, pd.DataFrame):
        epoch_data = epoch_file
    else:
        epoch_data = pd.read_csv(epoch_file, index_col=['time'], parse_dates=['time'], date_parser=utils.date_parser)

    # Remove data before/after user specified start/end times
    rows = epoch_data.shape[0]
    if start_time:
        epoch_data = epoch_data.loc[pd.Timestamp(start_time, tz=time_zone):]
    if end_time:
        epoch_data = epoch_data.loc[:pd.Timestamp(end_time, tz=time_zone)]
    # Quit if no data left
    if epoch_data.shape[0] == 0:
        print("No rows remaining after start/end time removal")
        print(f"Previously there were {rows} rows, now shape: {epoch_data.shape}")
        raise DataError(
            f"No data remaining after start/end time filtering. "
            f"Original rows: {rows}, remaining: {epoch_data.shape[0]}"
        )

    # Get start & end times
    start_time = epoch_data.index[0]
    end_time = epoch_data.index[-1]
    summary['file-startTime'] = utils.date_strftime(start_time)
    summary['file-endTime'] = utils.date_strftime(end_time)
    summary['file-firstDay(0=mon,6=sun)'] = start_time.weekday()

    # Quality checks
    check_quality(epoch_data, summary)

    # enmo : Euclidean Norm Minus One
    # Trunc :  negative values truncated to zero (i.e never negative)
    # emmo = 1 - sqrt(x, y, z)
    # enmoTrunc = max(enmo, 0)
    epoch_data['acc'] = epoch_data['enmoTrunc'] * 1000  # convert enmoTrunc to milli-G units

    # Resolve interrupts
    epoch_data = resolve_interrupts(epoch_data, epoch_period, summary)
    # Resolve non-wear
    epoch_data = resolve_non_wear(epoch_data, stationary_std, min_non_wear_duration, summary)

    # Predict activity from features, and add label column
    labels = []
    if activity_classification:
        epoch_data, labels = classification.activity_classification(
            epoch_data,
            activity_model,
            mg_cp_lpa, mg_cp_mpa, mg_cp_vpa,
            remove_spurious_sleep, remove_spurious_sleep_tol
        )

    # Calculate empirical cumulative distribution function of vector magnitudes
    if intensity_distribution:
        calculate_ecdf(epoch_data['acc'], summary)

    # Calculate circadian metrics
    if psd:
        circadian.calculate_psd(impute_missing(epoch_data[['acc'] + labels]), epoch_period, fourier_with_acc, labels, summary)
    if fourier_frequency:
        circadian.calculate_fourier_freq(impute_missing(epoch_data[['acc'] + labels]), epoch_period, fourier_with_acc, labels, summary)
    if m10l5:
        circadian.calculate_m10l5(impute_missing(epoch_data[['acc'] + labels]), epoch_period, summary)

    # Main movement summaries
    write_movement_summaries(epoch_data, labels, summary, min_wear_per_day)

    # Return physical activity summary
    return epoch_data, labels


def check_quality(data, summary):
    summary['totalReads'] = data['rawSamples'].sum().item()
    # Check DST
    if data.index[0].dst() < data.index[-1].dst():
        summary['quality-daylightSavingsCrossover'] = 1
    elif data.index[0].dst() > data.index[-1].dst():
        summary['quality-daylightSavingsCrossover'] = -1
    else:
        summary['quality-daylightSavingsCrossover'] = 0
    # Check value clips
    summary['clipsBeforeCalibration'] = data['clipsBeforeCalibr'].sum().item()
    summary['clipsAfterCalibration'] = data['clipsAfterCalibr'].sum().item()


def resolve_interrupts(data, epoch_period, summary):
    """Fix any interrupts in the recording by resampling

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epoch_period: Size of epoch time window (in seconds)
    :param dict summary: Dictionary containing summary metrics

    :return: Write dict <summary> keys 'err-interrupts-num' & 'errs-interrupt-mins'
    :rtype: void
    """
    epoch_period = pd.Timedelta(epoch_period, unit='S')
    gaps = data.index.to_series().diff()
    gaps = gaps[gaps > epoch_period]
    summary['errs-interrupts-num'] = len(gaps)
    summary['errs-interrupt-mins'] = gaps.sum().total_seconds() / 60

    data = data.asfreq(epoch_period, normalize=False, fill_value=None)
    data['missing'] = data.isna().any(1)

    return data


def resolve_non_wear(data, std_tol, patience, summary):
    """Calculate nonWear time, write episodes to file, and return wear statistics

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int std_tol: Threshold (in mg units) for stationary vs not
    :param int patience: Minimum duration of nonwear events (minutes)
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> keys 'wearTime-numNonWearEpisodes(>1hr)',
        'wearTime-overall(days)', 'nonWearTime-overall(days)', 'wearTime-diurnalHrs',
        'wearTime-diurnalMins', 'quality-goodWearTime', 'wearTime-<day...>', and
        'wearTime-hourOfDay-<hr...>'
    :rtype: void

    :return: Write .csv.gz non wear episodes file to <nonWearFile>
    :rtype: void
    """

    std_tol = std_tol / 1000.0  # mg to g
    stationary = (data[['xStd', 'yStd', 'zStd']] < std_tol).all(1)
    stationary_group = ((stationary != stationary.shift(1))
                        .cumsum()
                        .where(stationary))
    stationary_len = (stationary_group.groupby(stationary_group, dropna=True)
                      .apply(lambda g: g.index[-1] - g.index[0]))
    if len(stationary_len) > 0:
        non_wear_len = stationary_len[stationary_len > pd.Timedelta(patience, 'm')]
    else:
        non_wear_len = pd.Series(dtype='timedelta64[ns]')  # empty series
    non_wear = stationary_group.isin(non_wear_len.index)
    missing = non_wear | data['missing']
    data = data.mask(missing)  # set non wear rows to nan
    data['missing'] = missing

    freq = to_offset(pd.infer_freq(data.index))
    epoch_in_days = pd.to_timedelta(freq).total_seconds() / (60 * 60 * 24)
    num_missing_rows = missing.sum()
    non_wear_time = num_missing_rows * epoch_in_days
    wear_time = (len(data) - num_missing_rows) * epoch_in_days
    is_good_coverage = not (missing  # check there's at least some data for each hour pocket
                            .groupby(missing.index.hour)
                            .all().any())
    is_good_wear_time = wear_time >= 3  # check there's at least 3 days of wear time

    summary['wearTime-numNonWearEpisodes(>1hr)'] = int(len(non_wear_len))
    summary['wearTime-overall(days)'] = wear_time
    summary['nonWearTime-overall(days)'] = non_wear_time
    summary['quality-goodWearTime'] = int(is_good_coverage and is_good_wear_time)

    return data


def impute_missing(data, extrapolate=True):
    """Impute missing/nonwear segments

    Impute non-wear data segments using the average of similar time-of-day values
    with one minute granularity on different days of the measurement. This
    imputation accounts for potential wear time diurnal bias where, for example,
    if the device was systematically less worn during sleep in an individual,
    the crude average vector magnitude during wear time would be a biased
    overestimate of the true average. See
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649#sec013

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param bool verbose: Print verbose output

    :return: Update DataFrame <e> columns nan values with time-of-day imputation
    :rtype: void
    """

    if extrapolate:
        # padding at the boundaries to have full 24h
        data = data.reindex(
            pd.date_range(
                data.index[0].floor('D'),
                data.index[-1].ceil('D'),
                freq=to_offset(pd.infer_freq(data.index)),
                closed='left',
                name='time',
            ),
            method='nearest',
            tolerance=pd.Timedelta('1m'),
            limit=1)

    def fillna(subframe):
        # Transform will first pass the subframe column-by-column as a Series.
        # After passing all columns, it will pass the entire subframe again as a DataFrame.
        # Processing the entire subframe is optional (return value can be omitted). See 'Notes' in transform doc.
        if isinstance(subframe, pd.Series):
            values = subframe.to_numpy()
            missing_mask = np.isnan(values)
            num_missing = len(values[missing_mask])
            if 0 < num_missing < len(values):  # check values contains a NaN and is not all NaN
                values[missing_mask] = np.nanmean(values)
                return values  # will be cast back to a Series automatically
            else:
                return subframe

    data = (
        data
        # first attempt imputation using same day of week
        .groupby([data.index.weekday, data.index.hour, data.index.minute])
        .transform(fillna)
        # then try within weekday/weekend
        .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute])
        .transform(fillna)
        # finally, use all other days
        .groupby([data.index.hour, data.index.minute])
        .transform(fillna)
    )

    return data


def calculate_ecdf(x, summary):
    """Calculate activity intensity empirical cumulative distribution

    The input data must not be imputed, as ECDF requires different imputation
    where nan/non-wear data segments are IMPUTED FOR EACH INTENSITY LEVEL. Here,
    the average of similar time-of-day values is imputed with one minute
    granularity on different days of the measurement. Following intensity levels
    are calculated:
    1mg bins from 1-20mg
    5mg bins from 25-100mg
    25mg bins from 125-500mg
    100mg bins from 500-2000mg

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param str inputCol: Column to calculate intensity distribution on
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> keys '<inputCol>-ecdf-<level...>mg'
    :rtype: void
    """

    levels = np.concatenate([
        np.linspace(1, 20, 20),  # 1mg bins from 1-20mg
        np.linspace(25, 100, 16),  # 5mg bins from 25-100mg
        np.linspace(125, 500, 16),  # 25mg bins from 125-500mg
        np.linspace(600, 2000, 15),  # 100mg bins from 500-2000mg
    ]).astype('int')

    whrnan = x.isna().to_numpy()
    ecdf = x.to_numpy().reshape(-1, 1) <= levels.reshape(1, -1)
    ecdf[whrnan] = np.nan

    ecdf = (pd.DataFrame(ecdf, index=x.index, columns=levels)
            .groupby([x.index.hour, x.index.minute])
            .mean()  # first average is across same time of day
            .mean()  # second average is within each level
            )

    # Write to summary
    for level, val in ecdf.iteritems():
        summary[f'{x.name}-ecdf-{level}mg'] = val


def write_movement_summaries(data, labels, summary, min_wear_per_day=None):  # noqa: C901
    """Write overall summary stats for each activity type to summary dict

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics
    :param float min_wear_per_day: Minimum wear time (in hours) required for a day to be
        included in summary statistics. Days with less wear time will be excluded.
        If None, all days are included.

    :return: Write dict <summary> keys for each activity type 'overall-<avg/sd>',
        'week<day/end>-avg', '<day..>-avg', 'hourOfDay-<hr..>-avg',
        'hourOfWeek<day/end>-<hr..>-avg'
    :rtype: void
    """

    data = data.copy()
    data['wearTime'] = ~data['missing']
    freq = to_offset(pd.infer_freq(data.index))

    # Hours of activity for each recorded day
    epoch_in_hours = pd.Timedelta(freq).total_seconds() / 3600
    cols = ['wearTime'] + labels
    daily_stats = (
        data[cols].astype('float')
        .groupby(data.index.date)
        .sum()
        * epoch_in_hours
    ).reset_index(drop=True)

    # Filter days based on minimum wear time threshold
    valid_days = pd.Series([True] * len(daily_stats))
    if min_wear_per_day is not None:
        valid_days = daily_stats['wearTime'] >= min_wear_per_day
        num_excluded_days = (~valid_days).sum()
        num_included_days = valid_days.sum()

        summary['wearTime-numDaysExcluded'] = int(num_excluded_days)
        summary['wearTime-numDaysIncluded'] = int(num_included_days)
        summary['wearTime-minWearPerDayThreshold(hrs)'] = min_wear_per_day

        if num_excluded_days > 0:
            utils.to_screen(f"Excluding {num_excluded_days} day(s) with wear time < {min_wear_per_day}h")
            utils.to_screen(f"Including {num_included_days} day(s) with wear time >= {min_wear_per_day}h")

    # Write per-day statistics (including excluded days for transparency)
    for i, row in daily_stats.iterrows():
        included = valid_days.iloc[i]
        for col in cols:
            summary[f'day{i}-recorded-{col}(hrs)'] = row.loc[col]
        summary[f'day{i}-includedInSummary'] = int(included)

    # Mark epochs from excluded days in the data
    data['excludeDay'] = False
    if min_wear_per_day is not None and num_excluded_days > 0:
        excluded_day_indices = daily_stats.index[~valid_days].tolist()
        for day_idx in excluded_day_indices:
            day_date = daily_stats.index[daily_stats.index == day_idx][0]
            # Get the actual date from data corresponding to this day index
            all_dates = data.index.date
            unique_dates = pd.Series(all_dates).unique()
            if day_idx < len(unique_dates):
                target_date = unique_dates[day_idx]
                data.loc[data.index.date == target_date, 'excludeDay'] = True

    # In the following, we resample, pad and impute the data so that we have a
    # multiple of 24h for the stats calculations
    t_start, t_end = data.index[0], data.index[-1]
    cols = ['acc', 'wearTime'] + labels
    if 'MET' in data.columns:
        cols.append('MET')

    # Filter out excluded days before imputation and summary calculation
    data_for_summary = data.copy()
    if min_wear_per_day is not None and num_excluded_days > 0:
        # Set excluded days to missing for imputation purposes
        data_for_summary.loc[data_for_summary['excludeDay'], cols] = np.nan
        data_for_summary.loc[data_for_summary['excludeDay'], 'missing'] = True

    data_for_summary = impute_missing(data_for_summary[cols].astype('float'))

    # Overall stats (no padding, i.e. only within recording period)
    # Only calculate on included days
    overall_stats = data_for_summary[t_start:t_end].apply(['mean', 'std'])
    for col in overall_stats:
        summary[f'{col}-overall-avg'] = overall_stats[col].loc['mean']
        summary[f'{col}-overall-sd'] = overall_stats[col].loc['std']

    day_of_week_stats = (
        data_for_summary
        .groupby([data_for_summary.index.weekday, data_for_summary.index.hour])
        .mean()
    )
    day_of_week_stats.index = day_of_week_stats.index.set_levels(
        day_of_week_stats
        .index.levels[0].to_series()
        .replace({0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'})
        .to_list(),
        level=0
    )
    day_of_week_stats.index.set_names(['DayOfWeek', 'Hour'], inplace=True)

    # Week stats
    for col, value in day_of_week_stats.mean().items():
        summary[f'{col}-week-avg'] = value

    # Stats by day of week (Mon, Tue, ...)
    for col, stats in day_of_week_stats.groupby(level=0).mean().to_dict().items():
        for day_of_week, value in stats.items():
            summary[f'{col}-{day_of_week}-avg'] = value

    # Stats by hour of day
    for col, stats in day_of_week_stats.groupby(level=1).mean().to_dict().items():
        for hour, value in stats.items():
            summary[f'{col}-hourOfDay-{hour}-avg'] = value

    # (not included but could be) Stats by hour of day AND day of week
    # for col, stats in day_of_week_stats.to_dict().items():
    #     for key, value in stats.items():
    #         day_of_week, hour = key
    #         summary[f'{col}-hourOf{day_of_week}-{hour}-avg'] = value

    weekday_or_weekend_stats = (
        day_of_week_stats
        .groupby([
            day_of_week_stats.index.get_level_values('DayOfWeek').str.contains('sat|sun'),
            day_of_week_stats.index.get_level_values('Hour')
        ])
        .mean()
    )
    weekday_or_weekend_stats.index = weekday_or_weekend_stats.index.set_levels(
        weekday_or_weekend_stats
        .index.levels[0].to_series()
        .replace({True: 'Weekend', False: 'Weekday'})
        .to_list(),
        level=0
    )
    weekday_or_weekend_stats.index.set_names(['WeekdayOrWeekend', 'Hour'], inplace=True)

    # Weekday/weekend stats
    for col, stats in weekday_or_weekend_stats.groupby(level=0).mean().to_dict().items():
        for weekday_or_weekend, value in stats.items():
            summary[f'{col}-{weekday_or_weekend.lower()}-avg'] = value

    # Stats by hour of day AND by weekday/weekend
    for col, stats in weekday_or_weekend_stats.to_dict().items():
        for key, value in stats.items():
            weekday_or_weekend, hour = key
            summary[f'{col}-hourOf{weekday_or_weekend}-{hour}-avg'] = value

    return
