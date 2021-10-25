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
        e = epochFile
    else:
        e = pd.read_csv(epochFile, index_col=['time'], parse_dates=['time'], date_parser=accUtils.date_parser)

    # Remove data before/after user specified start/end times
    rows = e.shape[0]
    tz = pytz.timezone(timeZone)
    if startTime:
        localStartTime = tz.localize(startTime)
        e = e[e.index >= localStartTime]
    if endTime:
        localEndTime = tz.localize(endTime)
        e = e[e.index <= localEndTime]
    # Quit if no data left
    if e.shape[0] == 0:
        print("No rows remaining after start/end time removal")
        print("Previously there were %d rows, now shape: %s" % (rows, str(e.shape)))
        sys.exit(-9)

    # Get start & end times
    startTime = e.index[0]
    endTime = e.index[-1]
    summary['file-startTime'] = accUtils.date_strftime(startTime)
    summary['file-endTime'] = accUtils.date_strftime(endTime)
    summary['file-firstDay(0=mon,6=sun)'] = startTime.weekday()

    # Quality checks
    checkQuality(e, summary)

    # enmo : Euclidean Norm Minus One
    # Trunc :  negative values truncated to zero (i.e never negative)
    # emmo = 1 - sqrt(x, y, z)
    # enmoTrunc = max(enmo, 0)
    e['acc'] = e['enmoTrunc'] * 1000  # convert enmoTrunc to milli-G units

    # Cut-point based MVPA and VPA
    e['CutPointMVPA'] = e['acc'] >= mgCutPointMVPA
    e['CutPointVPA'] = e['acc'] >= mgCutPointVPA

    # Resolve read interrupts
    e = resolveInterrupts(e, epochPeriod, summary)

    # Resolve nonwear segments
    e = resolveNonWear(e, epochPeriod, stationaryStd, minNonWearDuration, nonWearFile, summary)

    # Predict activity from features, and add label column
    labels = []
    if activityClassification:
        e, labels = accClassification.activityClassification(e, activityModel)

    # Calculate empirical cumulative distribution function of vector magnitudes
    if intensityDistribution:
        calculateECDF(e['acc'], summary)

    # Calculate circadian metrics
    if psd:
        circadianRhythms.calculatePSD(e, epochPeriod, fourierWithAcc, labels, summary)
    if fourierFrequency:
        circadianRhythms.calculateFourierFreq(e, epochPeriod, fourierWithAcc, labels, summary)
    if m10l5:
        circadianRhythms.calculateM10L5(e, epochPeriod, summary)

    # Impute missing values
    if imputation:
        e = imputeMissing(e)

    # Main movement summaries
    writeMovementSummaries(e, labels, summary)

    # Return physical activity summary
    return e, labels


def checkQuality(e, summary):
    summary['totalReads'] = e['rawSamples'].sum().item()
    # Check DST
    if e.index[0].dst() < e.index[-1].dst():
        summary['quality-daylightSavingsCrossover'] = 1
    elif e.index[0].dst() > e.index[-1].dst():
        summary['quality-daylightSavingsCrossover'] = -1
    else:
        summary['quality-daylightSavingsCrossover'] = 0
    # Check value clips
    summary['clipsBeforeCalibration'] = e['clipsBeforeCalibr'].sum().item()
    summary['clipsAfterCalibration'] = e['clipsAfterCalibr'].sum().item()


def resolveInterrupts(e, epochPeriod, summary):
    """Fix any read interrupts by resampling and filling with NaNs

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param dict summary: Dictionary containing summary metrics

    :return: Write dict <summary> keys 'err-interrupts-num' & 'errs-interrupt-mins'
    :rtype: void
    """
    epochPeriod = pd.Timedelta(epochPeriod, unit='S')
    gaps = e.index.to_series().diff()
    gaps = gaps[gaps > epochPeriod]
    summary['errs-interrupts-num'] = len(gaps)
    summary['errs-interrupt-mins'] = accUtils.formatNum(gaps.sum().total_seconds() / 60, 1)

    e = e.asfreq(epochPeriod, normalize=False, fill_value=None)  # resample and fill gaps with NaNs

    return e


def resolveNonWear(e, epochPeriod, maxStd, minDuration, nonWearFile, summary):
    """Calculate nonWear time, write episodes to file, and return wear statistics

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param int maxStd: Threshold (in mg units) for stationary vs not
    :param int minDuration: Minimum duration of nonwear events (minutes)
    :param str nonWearFile: Output filename for non wear .csv.gz episodes
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> keys 'wearTime-numNonWearEpisodes(>1hr)',
        'wearTime-overall(days)', 'nonWearTime-overall(days)', 'wearTime-diurnalHrs',
        'wearTime-diurnalMins', 'quality-goodWearTime', 'wearTime-<day...>', and
        'wearTime-hourOfDay-<hr...>'
    :rtype: void

    :return: Write .csv.gz non wear episodes file to <nonWearFile>
    :rtype: void
    """

    maxStd = maxStd / 1000.0  # java uses Gravity units (not mg)
    nw = (e['xStd'] < maxStd) & (e['yStd'] < maxStd) & (e['zStd'] < maxStd)
    starts = e.index[nw.astype('bool') & ~(nw.shift(1).fillna(0).astype('bool'))]
    ends = e.index[nw.astype('bool') & ~(nw.shift(-1).fillna(0).astype('bool'))]
    nonWearEpisodes = [(start, end) for start, end in zip(starts, ends)
                       if end > start + np.timedelta64(minDuration, 'm')]

    # Set nonWear data to nan and record to nonWearBouts file
    f = gzip.open(nonWearFile, 'wb')
    f.write('start,end,xStdMax,yStdMax,zStdMax\n'.encode())
    timeFormat = '%Y-%m-%d %H:%M:%S'
    for episode in nonWearEpisodes:
        tmp = e[['xStd', 'yStd', 'zStd']][episode[0]:episode[1]]
        nonWearBout = episode[0].strftime(timeFormat) + ','
        nonWearBout += episode[1].strftime(timeFormat) + ','
        nonWearBout += str(tmp['xStd'].mean()) + ','
        nonWearBout += str(tmp['yStd'].mean()) + ','
        nonWearBout += str(tmp['zStd'].mean()) + '\n'
        f.write(nonWearBout.encode())
        # Set main dataframe values to nan
        e[episode[0]:episode[1]] = np.nan
    f.close()
    # Write to summary
    summary['wearTime-numNonWearEpisodes(>1hr)'] = int(len(nonWearEpisodes))

    # Calculate wear statistics
    wearSamples = e['enmoTrunc'].count()
    nonWearSamples = len(e[np.isnan(e['enmoTrunc'])].index.values)
    wearTimeMin = wearSamples * epochPeriod / 60.0
    nonWearTimeMin = nonWearSamples * epochPeriod / 60.0
    # Write to summary
    summary['wearTime-overall(days)'] = accUtils.formatNum(wearTimeMin / 1440.0, 2)
    summary['nonWearTime-overall(days)'] = accUtils.formatNum(nonWearTimeMin / 1440.0, 2)

    # Get wear time in each of 24 hours across week
    epochsInMin = 60.0 / epochPeriod
    for i, day in zip(range(0, 7), accUtils.DAYS):
        dayWear = e['enmoTrunc'][e.index.weekday == i].count() / epochsInMin
        # Write to summary
        summary['wearTime-' + day + '(hrs)'] = accUtils.formatNum(dayWear / 60.0, 2)
    for i in range(0, 24):
        hourWear = e['enmoTrunc'][e.index.hour == i].count() / epochsInMin
        # Write to summary
        summary['wearTime-hourOfDay' + str(i) + '-(hrs)'] = \
            accUtils.formatNum(hourWear / 60.0, 2)
    summary['wearTime-diurnalHrs'] = accUtils.formatNum(
        e['enmoTrunc'].groupby(e.index.hour).mean().count(), 2)
    summary['wearTime-diurnalMins'] = accUtils.formatNum(
        e['enmoTrunc'].groupby([e.index.hour, e.index.minute]).mean().count(), 2)

    # Write binary decision on whether weartime was good or not
    minDiurnalHrs = 24
    minWearDays = 3
    summary['quality-goodWearTime'] = 1
    if summary['wearTime-diurnalHrs'] < minDiurnalHrs or \
            summary['wearTime-overall(days)'] < minWearDays:
        summary['quality-goodWearTime'] = 0

    return e


def imputeMissing(e):
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

    e['imputed'] = e.isna().any(1)  # record where the NaNs were
    e = e.groupby([e.index.hour, e.index.minute]).apply(lambda x: x.fillna(x.mean()))  # imputation

    return e


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


def writeMovementSummaries(e, labels, summary):
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

    # Identify activity types to summarise
    activityTypes = ['acc', 'CutPointMVPA', 'CutPointVPA']
    activityTypes += labels
    if 'MET' in e.columns:
        activityTypes.append('MET')

    # Sumarise each type by: overall, week day/end, day, and hour of day
    for col in activityTypes:

        # Overall / weekday / weekend summaries
        summary[col + '-overall-avg'] = accUtils.formatNum(e[col].mean(), 5)
        summary[col + '-overall-sd'] = accUtils.formatNum(e[col].std(), 2)
        summary[col + '-weekday-avg'] = accUtils.formatNum(
            e[col][e.index.weekday <= 4].mean(), 2)
        summary[col + '-weekend-avg'] = accUtils.formatNum(
            e[col][e.index.weekday >= 5].mean(), 2)

        # Daily summary
        for i, day in zip(range(0, 7), accUtils.DAYS):
            summary[col + '-' + day + '-avg'] = accUtils.formatNum(
                e[col][e.index.weekday == i].mean(), 2)

        # Hourly summaries
        for i in range(0, 24):
            hourOfDay = accUtils.formatNum(e[col][e.index.hour == i].mean(), 2)
            hourOfWeekday = accUtils.formatNum(
                e[col][(e.index.weekday <= 4) & (e.index.hour == i)].mean(), 2)
            hourOfWeekend = accUtils.formatNum(
                e[col][(e.index.weekday >= 5) & (e.index.hour == i)].mean(), 2)
            # Write derived hourly values to summary dictionary
            summary[col + '-hourOfDay-' + str(i) + '-avg'] = hourOfDay
            summary[col + '-hourOfWeekday-' + str(i) + '-avg'] = hourOfWeekday
            summary[col + '-hourOfWeekend-' + str(i) + '-avg'] = hourOfWeekend
