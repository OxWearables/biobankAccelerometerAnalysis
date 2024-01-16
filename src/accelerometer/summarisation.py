"""Module to generate overall activity summary from epoch data."""
import sys
import pytz
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from accelerometer import utils
from accelerometer import classification
from accelerometer import circadian


def getActivitySummary(  # noqa: C901
    epochFile, nonWearFile, summary,
    activityClassification=True, timeZone='Europe/London',
    startTime=None, endTime=None,
    epochPeriod=30, stationaryStd=13, minNonWearDuration=60,
    mgCpLPA=45, mgCpMPA=100, mgCpVPA=400,
    activityModel="walmsley",
    intensityDistribution=False, imputation=True,
    psd=False, fourierFrequency=False, fourierWithAcc=False, m10l5=False
):
    """
    Calculate overall activity summary from <epochFile> data. Get overall
    activity summary from input <epochFile>. This is achieved by:
    1) get interrupt and data error summary vals
    2) check if data occurs at a daylight savings crossover
    3) calculate wear-time statistics, and write nonWear episodes to file
    4) predict activity from features, and add label column
    5) calculate imputation values to replace nan PA metric values
    6) calculate empirical cumulative distribution function of vector magnitudes
    7) derive main movement summaries (overall, weekday/weekend, and hour)

    :param str epochFile: Input csv.gz file of processed epoch data
    :param str nonWearFile: Output filename for non wear .csv.gz episodes
    :param dict summary: Output dictionary containing all summary metrics
    :param bool activityClassification: Perform machine learning of activity states
    :param str timeZone: timezone in country/city format to be used for daylight savings crossover check
    :param datetime startTime: Remove data before this time in analysis
    :param datetime endTime: Remove data after this time in analysis
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param int stationaryStd: Threshold (in mg units) for stationary vs not
    :param int minNonWearDuration: Minimum duration of nonwear events (minutes)
    :param int mgCutPointMVPA: Milli-gravity threshold for moderate intensity activity
    :param int mgCutPointVPA: Milli-gravity threshold for vigorous intensity activity
    :param str activityModel: Input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METS for each activity state
    :param bool intensityDistribution: Add intensity outputs to dict <summary>
    :param bool imputation: Impute missing data using data from other days around the same time
    :param bool verbose: Print verbose output

    :return: A tuple containing a pandas dataframe of activity epoch data,
        activity prediction labels (empty if <activityClassification>==False), and
        movement summary values written to dict <summary>. Also writes .csv.gz non
        wear episodes file to <nonWearFile>.
    :rtype: tuple

    .. code-block:: python

        import summariseEpoch
        summary = {}
        epochData, labels = summariseEpoch.getActivitySummary( "epoch.csv.gz",
                "nonWear.csv.gz", summary)
    """

    utils.toScreen("=== Summarizing ===")

    if isinstance(epochFile, pd.DataFrame):
        data = epochFile
    else:
        data = pd.read_csv(epochFile, index_col=['time'], parse_dates=['time'], date_parser=utils.date_parser)

    # Remove data before/after user specified start/end times
    rows = data.shape[0]
    tz = pytz.timezone(timeZone)
    if startTime:
        localStartTime = tz.localize(startTime)
        data = data[data.index >= localStartTime]
    if endTime:
        localEndTime = tz.localize(endTime)
        data = data[data.index <= localEndTime]
    # Quit if no data left
    if data.shape[0] == 0:
        print("No rows remaining after start/end time removal")
        print("Previously there were %d rows, now shape: %s" % (rows, str(data.shape)))
        sys.exit(-9)

    # Get start & end times
    startTime = data.index[0]
    endTime = data.index[-1]
    summary['file-startTime'] = utils.date_strftime(startTime)
    summary['file-endTime'] = utils.date_strftime(endTime)
    summary['file-firstDay(0=mon,6=sun)'] = startTime.weekday()

    # Quality checks
    checkQuality(data, summary)

    # enmo : Euclidean Norm Minus One
    # Trunc :  negative values truncated to zero (i.e never negative)
    # emmo = 1 - sqrt(x, y, z)
    # enmoTrunc = max(enmo, 0)
    data['acc'] = data['enmoTrunc'] * 1000  # convert enmoTrunc to milli-G units

    # Resolve interrupts
    data = resolveInterrupts(data, epochPeriod, summary)
    # Resolve non-wear
    data = resolveNonWear(data, stationaryStd, minNonWearDuration, summary)

    # Predict activity from features, and add label column
    labels = []
    if activityClassification:
        data, labels = classification.activityClassification(data, activityModel, mgCpLPA, mgCpMPA, mgCpVPA)

    # Calculate empirical cumulative distribution function of vector magnitudes
    if intensityDistribution:
        calculateECDF(data['acc'], summary)

    # Calculate circadian metrics
    if psd:
        circadian.calculatePSD(data, epochPeriod, fourierWithAcc, labels, summary)
    if fourierFrequency:
        circadian.calculateFourierFreq(data, epochPeriod, fourierWithAcc, labels, summary)
    if m10l5:
        circadian.calculateM10L5(data, epochPeriod, summary)

    # Main movement summaries
    writeMovementSummaries(data, labels, summary)

    # Return physical activity summary
    return data, labels


def checkQuality(data, summary):
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


def resolveInterrupts(data, epochPeriod, summary):
    """Fix any interrupts in the recording by resampling

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param dict summary: Dictionary containing summary metrics

    :return: Write dict <summary> keys 'err-interrupts-num' & 'errs-interrupt-mins'
    :rtype: void
    """
    epochPeriod = pd.Timedelta(epochPeriod, unit='S')
    gaps = data.index.to_series().diff()
    gaps = gaps[gaps > epochPeriod]
    summary['errs-interrupts-num'] = len(gaps)
    summary['errs-interrupt-mins'] = gaps.sum().total_seconds() / 60

    data = data.asfreq(epochPeriod, normalize=False, fill_value=None)
    data['missing'] = data.isna().any(1)

    return data


def resolveNonWear(data, stdTol, patience, summary):
    """Calculate nonWear time, write episodes to file, and return wear statistics

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int maxStd: Threshold (in mg units) for stationary vs not
    :param int minDuration: Minimum duration of nonwear events (minutes)
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> keys 'wearTime-numNonWearEpisodes(>1hr)',
        'wearTime-overall(days)', 'nonWearTime-overall(days)', 'wearTime-diurnalHrs',
        'wearTime-diurnalMins', 'quality-goodWearTime', 'wearTime-<day...>', and
        'wearTime-hourOfDay-<hr...>'
    :rtype: void

    :return: Write .csv.gz non wear episodes file to <nonWearFile>
    :rtype: void
    """

    stdTol = stdTol / 1000.0  # mg to g
    stationary = (data[['xStd', 'yStd', 'zStd']] < stdTol).all(1)
    stationaryGroup = ((stationary != stationary.shift(1))
                       .cumsum()
                       .where(stationary))
    stationaryLen = (stationaryGroup.groupby(stationaryGroup, dropna=True)
                     .apply(lambda g: g.index[-1] - g.index[0]))
    if len(stationaryLen) > 0:
        nonWearLen = stationaryLen[stationaryLen > pd.Timedelta(patience, 'm')]
    else:
        nonWearLen = pd.Series(dtype='timedelta64[ns]')  # empty series
    nonWear = stationaryGroup.isin(nonWearLen.index)
    missing = nonWear | data['missing']
    data = data.mask(missing)  # set non wear rows to nan
    data['missing'] = missing

    freq = to_offset(pd.infer_freq(data.index))
    epochInDays = pd.to_timedelta(freq).total_seconds() / (60 * 60 * 24)
    numMissingRows = missing.sum()
    nonWearTime = numMissingRows * epochInDays
    wearTime = (len(data) - numMissingRows) * epochInDays
    isGoodCoverage = not (missing  # check there's at least some data for each hour pocket
                          .groupby(missing.index.hour)
                          .all().any())
    isGoodWearTime = wearTime >= 3  # check there's at least 3 days of wear time

    summary['wearTime-numNonWearEpisodes(>1hr)'] = int(len(nonWearLen))
    summary['wearTime-overall(days)'] = wearTime
    summary['nonWearTime-overall(days)'] = nonWearTime
    summary['quality-goodWearTime'] = int(isGoodCoverage and isGoodWearTime)

    return data


def imputeMissing(data, extrapolate=True):
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
            x = subframe.to_numpy()
            nan = np.isnan(x)
            nanlen = len(x[nan])
            if 0 < nanlen < len(x):  # check x contains a NaN and is not all NaN
                x[nan] = np.nanmean(x)
                return x  # will be cast back to a Series automatically
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


def calculateECDF(x, summary):
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


def writeMovementSummaries(data, labels, summary):  # noqa: C901
    """Write overall summary stats for each activity type to summary dict

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics
    :param bool imputation: Impute missing data using data from other days around the same time

    :return: Write dict <summary> keys for each activity type 'overall-<avg/sd>',
        'week<day/end>-avg', '<day..>-avg', 'hourOfDay-<hr..>-avg',
        'hourOfWeek<day/end>-<hr..>-avg'
    :rtype: void
    """

    data = data.copy()
    data['wearTime'] = ~data['missing']
    freq = to_offset(pd.infer_freq(data.index))

    # Hours of activity for each recorded day
    epochInHours = pd.Timedelta(freq).total_seconds() / 3600
    cols = ['wearTime'] + labels
    dailyStats = (
        data[cols].astype('float')
        .groupby(data.index.date)
        .sum()
        * epochInHours
    ).reset_index(drop=True)

    for i, row in dailyStats.iterrows():
        for col in cols:
            summary[f'day{i}-recorded-{col}(hrs)'] = row.loc[col]

    # In the following, we resample, pad and impute the data so that we have a
    # multiple of 24h for the stats calculations
    tStart, tEnd = data.index[0], data.index[-1]
    cols = ['acc', 'wearTime'] + labels
    if 'MET' in data.columns:
        cols.append('MET')
    data = imputeMissing(data[cols].astype('float'))

    # Overall stats (no padding, i.e. only within recording period)
    overallStats = data[tStart:tEnd].apply(['mean', 'std'])
    for col in overallStats:
        summary[f'{col}-overall-avg'] = overallStats[col].loc['mean']
        summary[f'{col}-overall-sd'] = overallStats[col].loc['std']

    dayOfWeekStats = (
        data
        .groupby([data.index.weekday, data.index.hour])
        .mean()
    )
    dayOfWeekStats.index = dayOfWeekStats.index.set_levels(
        dayOfWeekStats
        .index.levels[0].to_series()
        .replace({0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'})
        .to_list(),
        level=0
    )
    dayOfWeekStats.index.set_names(['DayOfWeek', 'Hour'], inplace=True)

    # Week stats
    for col, value in dayOfWeekStats.mean().items():
        summary[f'{col}-week-avg'] = value

    # Stats by day of week (Mon, Tue, ...)
    for col, stats in dayOfWeekStats.groupby(level=0).mean().to_dict().items():
        for dayOfWeek, value in stats.items():
            summary[f'{col}-{dayOfWeek}-avg'] = value

    # Stats by hour of day
    for col, stats in dayOfWeekStats.groupby(level=1).mean().to_dict().items():
        for hour, value in stats.items():
            summary[f'{col}-hourOfDay-{hour}-avg'] = value

    # (not included but could be) Stats by hour of day AND day of week
    # for col, stats in dayOfWeekStats.to_dict().items():
    #     for key, value in stats.items():
    #         dayOfWeek, hour = key
    #         summary[f'{col}-hourOf{dayOfWeek}-{hour}-avg'] = value

    weekdayOrWeekendStats = (
        dayOfWeekStats
        .groupby([
            dayOfWeekStats.index.get_level_values('DayOfWeek').str.contains('sat|sun'),
            dayOfWeekStats.index.get_level_values('Hour')
        ])
        .mean()
    )
    weekdayOrWeekendStats.index = weekdayOrWeekendStats.index.set_levels(
        weekdayOrWeekendStats
        .index.levels[0].to_series()
        .replace({True: 'Weekend', False: 'Weekday'})
        .to_list(),
        level=0
    )
    weekdayOrWeekendStats.index.set_names(['WeekdayOrWeekend', 'Hour'], inplace=True)

    # Weekday/weekend stats
    for col, stats in weekdayOrWeekendStats.groupby(level=0).mean().to_dict().items():
        for weekdayOrWeekend, value in stats.items():
            summary[f'{col}-{weekdayOrWeekend.lower()}-avg'] = value

    # Stats by hour of day AND by weekday/weekend
    for col, stats in weekdayOrWeekendStats.to_dict().items():
        for key, value in stats.items():
            weekdayOrWeekend, hour = key
            summary[f'{col}-hourOf{weekdayOrWeekend}-{hour}-avg'] = value

    return
