"""Module to generate overall activity summary from epoch data."""
from accelerometer import accUtils
from accelerometer import accClassification
from accelerometer import circadianRhythms
import gzip
import numpy as np
import pandas as pd
import pytz
import sys


def getActivitySummary(  # noqa: C901
    epochFile, nonWearFile, summary,
    activityClassification=True, timeZone='Europe/London',
    startTime=None, endTime=None,
    epochPeriod=30, stationaryStd=13, minNonWearDuration=60,
    mgCutPointMVPA=100, mgCutPointVPA=425,
    activityModel="walmsley",
    intensityDistribution=False, imputation=True,
    psd=False, fourierFrequency=False, fourierWithAcc=False, m10l5=False
):
    """Calculate overall activity summary from <epochFile> data

    Get overall activity summary from input <epochFile>. This is achieved by
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
    :param str timeZone: timezone in country/city format to be used for daylight
        savings crossover check
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

    :return: Pandas dataframe of activity epoch data
    :rtype: pandas.DataFrame

    :return: Activity prediction labels (empty if <activityClassification>==False)
    :rtype: list(str)

    :return: Write .csv.gz non wear episodes file to <nonWearFile>
    :rtype: void

    :return: Movement summary values written to dict <summary>
    :rtype: void

    :Example:
    >>> import summariseEpoch
    >>> summary = {}
    >>> epochData, labels = summariseEpoch.getActivitySummary( "epoch.csv.gz",
            "nonWear.csv.gz", summary)
    <nonWear file written to "nonWear.csv.gz" and dict "summary" update with outcomes>
    """

    accUtils.toScreen("=== Summarizing ===")

    if isinstance(epochFile, pd.DataFrame):
        data = epochFile
    else:
        data = pd.read_csv(epochFile, index_col=['time'], parse_dates=['time'], date_parser=accUtils.date_parser)

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
    summary['file-startTime'] = accUtils.date_strftime(startTime)
    summary['file-endTime'] = accUtils.date_strftime(endTime)
    summary['file-firstDay(0=mon,6=sun)'] = startTime.weekday()

    # Quality checks
    checkQuality(data, summary)

    # enmo : Euclidean Norm Minus One
    # Trunc :  negative values truncated to zero (i.e never negative)
    # emmo = 1 - sqrt(x, y, z)
    # enmoTrunc = max(enmo, 0)
    data['acc'] = data['enmoTrunc'] * 1000  # convert enmoTrunc to milli-G units

    # Cut-point based MVPA and VPA
    data['CutPointMVPA'] = data['acc'] >= mgCutPointMVPA
    data['CutPointVPA'] = data['acc'] >= mgCutPointVPA

    # Resolve interrupts
    data = resolveInterrupts(data, epochPeriod, summary)
    # Resolve non-wear
    data = resolveNonWear(data, stationaryStd, minNonWearDuration, summary)

    # Predict activity from features, and add label column
    labels = []
    if activityClassification:
        data, labels = accClassification.activityClassification(data, activityModel)

    # Calculate empirical cumulative distribution function of vector magnitudes
    if intensityDistribution:
        calculateECDF(data['acc'], summary)

    # Calculate circadian metrics
    if psd:
        circadianRhythms.calculatePSD(data, epochPeriod, fourierWithAcc, labels, summary)
    if fourierFrequency:
        circadianRhythms.calculateFourierFreq(data, epochPeriod, fourierWithAcc, labels, summary)
    if m10l5:
        circadianRhythms.calculateM10L5(data, epochPeriod, summary)

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
    summary['errs-interrupt-mins'] = accUtils.formatNum(gaps.sum().total_seconds() / 60, 1)

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
    nonWearLen = stationaryLen[stationaryLen > pd.Timedelta(patience, 'm')]
    nonWear = stationaryGroup.isin(nonWearLen.index)
    missing = nonWear | data['missing']
    data = data.mask(missing)  # set non wear rows to nan
    data['missing'] = missing

    epochInDays = pd.Timedelta(pd.infer_freq(data.index)).total_seconds() / (60 * 60 * 24)
    numMissingRows = missing.sum()
    nonWearTime = numMissingRows * epochInDays
    wearTime = (len(data) - numMissingRows) * epochInDays
    isGoodCoverage = not (missing  # check there's at least some data for each hour pocket
                          .groupby(missing.index.hour)
                          .all().any())
    isGoodWearTime = wearTime >= 3  # check there's at least 3 days of wear time

    summary['wearTime-numNonWearEpisodes(>1hr)'] = int(len(nonWearLen))
    summary['wearTime-overall(days)'] = accUtils.formatNum(wearTime, 2)
    summary['nonWearTime-overall(days)'] = accUtils.formatNum(nonWearTime, 2)
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
                freq=pd.infer_freq(data.index),
                closed='left',
                name='time',
            ),
            method='nearest',
            tolerance=pd.Timedelta('1m'),
            limit=1)

    data = (
        data
        # first attempt imputation using same day of week
        .groupby([data.index.weekday, data.index.hour, data.index.minute])
        .transform(lambda x: x.fillna(x.mean()))
        # then try within weekday/weekend
        .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute])
        .transform(lambda x: x.fillna(x.mean()))
        # finally, use all other days
        .groupby([data.index.hour, data.index.minute])
        .transform(lambda x: x.fillna(x.mean()))
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
        summary[f'{x.name}-ecdf-{level}mg'] = accUtils.formatNum(val, 5)


def writeMovementSummaries(data, labels, summary):
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
    data['wear'] = ~data['missing']
    freq = pd.infer_freq(data.index)

    # Hours of activity for each recorded day
    epochInHours = pd.Timedelta(freq).total_seconds() / 3600
    activityLabels = ['wear', 'CutPointMVPA', 'CutPointVPA'] + labels
    hoursByDay = (
        data[activityLabels].astype('float')
        .groupby(data.index.date)
        .sum()
        * epochInHours
    ).reset_index(drop=True)

    for i, row in hoursByDay.iterrows():
        for label in activityLabels:
            summary[f'day{i}-recorded-{label}(hrs)'] = accUtils.formatNum(row.loc[label], 2)

    allCols = ['acc', 'wear', 'CutPointMVPA', 'CutPointVPA'] + labels
    if 'MET' in data.columns:
        allCols.append('MET')

    # To compute the day-of-week stats and overall stats we do
    # resampling and imputation so that we have a multiple of 24h
    data = imputeMissing(data[allCols].astype('float'))

    # Sumarise each type by: overall, week day/end, day of week, and hour of day
    for col in allCols:

        # Overall / weekday / weekend summaries
        summary[col + '-overall-avg'] = accUtils.formatNum(data[col].mean(), 5)
        summary[col + '-overall-sd'] = accUtils.formatNum(data[col].std(), 5)
        summary[col + '-weekday-avg'] = accUtils.formatNum(
            data[col][data.index.weekday < 5].mean(), 2)
        summary[col + '-weekend-avg'] = accUtils.formatNum(
            data[col][data.index.weekday >= 5].mean(), 2)

        # Day-of-week summary
        for i, day in zip(range(0, 7), accUtils.DAYS):
            summary[col + '-' + day + '-avg'] = accUtils.formatNum(
                data[col][data.index.weekday == i].mean(), 2)

        # Hour-of-day summary
        for i in range(0, 24):
            hourOfDay = accUtils.formatNum(data[col][data.index.hour == i].mean(), 2)
            hourOfWeekday = accUtils.formatNum(
                data[col][(data.index.weekday < 5) & (data.index.hour == i)].mean(), 2)
            hourOfWeekend = accUtils.formatNum(
                data[col][(data.index.weekday >= 5) & (data.index.hour == i)].mean(), 2)
            summary[col + '-hourOfDay-' + str(i) + '-avg'] = hourOfDay
            summary[col + '-hourOfWeekday-' + str(i) + '-avg'] = hourOfWeekday
            summary[col + '-hourOfWeekend-' + str(i) + '-avg'] = hourOfWeekend
