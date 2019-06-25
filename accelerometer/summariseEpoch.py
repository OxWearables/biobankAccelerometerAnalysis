"""Module to generate overall activity summary from epoch data."""

from accelerometer import accUtils
from accelerometer import accClassification
import datetime
import gzip
import numpy as np
import pandas as pd
import pytz
import sys


def getActivitySummary(epochFile, nonWearFile, summary,
    activityClassification=True, startTime=None, endTime=None,
    epochPeriod=30, stationaryStd=13, minNonWearDuration=60, mgMVPA=100,
    mgVPA=425, activityModel="activityModels/doherty2018.tar",
    intensityDistribution=False, verbose=False):
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
    :param datetime startTime: Remove data before this time in analysis
    :param datetime endTime: Remove data after this time in analysis
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param int stationaryStd: Threshold (in mg units) for stationary vs not
    :param int minNonWearDuration: Minimum duration of nonwear events (minutes)
    :param int mgMVPA: Milli-gravity threshold for moderate intensity activity
    :param int mgVPA: Milli-gravity threshold for vigorous intensity activity
    :param str activityModel: Input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METS for each activity state
    :param bool intensityDistribution: Add intensity outputs to dict <summary>
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

    if isinstance(epochFile, pd.DataFrame):
        e = epochFile
    else:
        # use python PANDAS framework to read in and store epochs
        e = pd.read_csv(epochFile, parse_dates=['time'], index_col=['time'],
            compression='gzip').sort_index()

    # remove data before/after user specified start/end times
    rows = e.shape[0]
    if startTime:
        e = e[e.index >= startTime]
    if endTime:
        e = e[e.index <= endTime]
    # quit if no data left
    if e.shape[0] == 0:
        print("no rows remaining after start/end time removal")
        print("previously there were %d rows, now shape: %s" % (rows, str(e.shape)))
        sys.exit(-9)

    # get start & end times
    startTime = pd.to_datetime(e.index.values[0])
    endTime = pd.to_datetime(e.index.values[-1])
    summary['file-startTime'] = startTime.strftime('%Y-%m-%d %H:%M:%S')
    summary['file-endTime'] = endTime.strftime('%Y-%m-%d %H:%M:%S')
    summary['file-firstDay(0=mon,6=sun)'] = startTime.weekday()

    # get interrupt and data error summary vals
    e = get_interrupts(e, epochPeriod, summary)

    # check if data occurs at a daylight savings crossover
    e = check_daylight_savings_crossover(e, startTime, endTime, summary)

    # calculate wear-time statistics, and write nonWear episodes to file
    get_wear_time_stats(e, epochPeriod, stationaryStd, minNonWearDuration,
        nonWearFile, summary)

    # predict activity from features, and add label column
    if activityClassification:
        e, labels = accClassification.activityClassification(e, activityModel)
    else:
        labels = []

    # enmo : Euclidean Norm Minus One
    # Trunc :  negative values truncated to zero (i.e never negative)
    # emmo = 1 - sqrt(x, y, z)
    # enmoTrunc = max(enmo, 0)
    e['acc'] = e['enmoTrunc'] * 1000 # convert enmoTrunc to milli-G units

    # calculate imputation values to replace nan PA metric values
    e = perform_wearTime_imputation(e, verbose)
    e['MVPA'] = e['accImputed'] >= mgMVPA
    e['VPA'] = e['accImputed'] >= mgVPA

    # calculate empirical cumulative distribution function of vector magnitudes
    if intensityDistribution:
        calculateECDF(e, 'acc', summary)

    # main movement summaries
    writeMovementSummaries(e, labels, summary)

    # return physical activity summary
    return e, labels



def get_interrupts(e, epochPeriod, summary):
    """Identify if there are interrupts in the data recording

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> keys 'err-interrupts-num' & 'errs-interrupt-mins'
    :rtype: void
    """

    epochNs = epochPeriod * np.timedelta64(1, 's')
    interrupts = np.where(np.diff(np.array(e.index)) > epochNs)[0]
    # get duration of each interrupt in minutes
    interruptMins = []
    for i in interrupts:
        interruptMins.append(np.diff(np.array(e[i:i+2].index)) /
                np.timedelta64(1, 'm'))
    # record to output summary
    summary['errs-interrupts-num'] = len(interruptMins)
    summary['errs-interrupt-mins'] = accUtils.formatNum(np.sum(interruptMins), 1)

    frames = [e]
    for i in interrupts:
        start, end = e[i:i+2].index
        dti = pd.date_range(start=start, end=end, freq=str(epochPeriod)+'s')[1:-1]
        frames.append(dti.to_frame().drop(columns=0))
    e = pd.concat(frames).sort_index()

    return e


def check_daylight_savings_crossover(e, startTime, endTime, summary):
    """Check if data occurs at a daylight savings crossover

    If daylight savings crossover, update times after time-change by +/- 1hr.
    Also, if Autumn crossover time, remove last 1hr chunk before time-change.

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param datetime startTime: Remove data before this time in analysis
    :param datetime endTime: Remove data after this time in analysis
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> key 'quality-daylightSavingsCrossover'
    :rtype: void

    :return: Update DataFrame <e> time column after time-change crossover.
    :rtype: void
    """

    daylightSavingsCrossover = 0
    localTime = pytz.timezone('Europe/London')
    # convert because pytz can error if not using python datetime type
    startTimeZone = localTime.localize(startTime.to_pydatetime())
    endTimeZone = localTime.localize(endTime.to_pydatetime())
    if startTimeZone.dst() != endTimeZone.dst():
        daylightSavingsCrossover = 1
        # find whether clock needs to go forward or back
        if endTimeZone.dst() > startTimeZone.dst():
            offset = 1
        else:
            offset = -1
        print('different timezones, offset = ', str(offset))
        # find actual crossover time
        for t in localTime._utc_transition_times:
            if t>startTime:
                transition = t
                break
        # if Autumn crossover time, adjust transition time plus remove 1hr chunk
        if offset == -1:
            # pytz stores dst crossover at 1am, but clocks change at 2am local
            transition = transition + pd.DateOffset(hours=1)
            # remove last hr before DST cut, which will be subsequently overwritten
            e = e[(e.index < transition - pd.DateOffset(hours=1)) |
                    (e.index >= transition)]
        print('day light savings transition at:', str(transition))
        # now update datetime index to 'fix' values after DST crossover
        e['newTime'] = e.index
        e['newTime'] = np.where(e.index >= transition,
                e.index + np.timedelta64(offset,'h'), e.index)
        e['newTime'] = np.where(e['newTime'].isnull(), e.index, e['newTime'])
        e = e.set_index('newTime')
        # reset startTime and endTime variables
        startTime = pd.to_datetime(e.index.values[0])
        endTime = pd.to_datetime(e.index.values[-1])
        # and record to output summary
        summary['quality-daylightSavingsCrossover'] = daylightSavingsCrossover
    return e



def get_wear_time_stats(e, epochPeriod, maxStd, minDuration, nonWearFile,
    summary):
    """Calculate nonWear time, write episodes to file, and return wear statistics

    If daylight savings crossover, update times after time-change by +/- 1hr.
    Also, if Autumn crossover time, remove last 1hr chunk before time-change.

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

    maxStd = maxStd / 1000.0 # java uses Gravity units (not mg)
    e['nw'] = np.where((e['xStd']<maxStd) & (e['yStd']<maxStd) &
            (e['zStd']<maxStd), 1, 0)
    starts = e.index[(e['nw']==True) & (e['nw'].shift(1).fillna(False)==False)]
    ends = e.index[(e['nw']==True) & (e['nw'].shift(-1).fillna(False)==False)]
    nonWearEpisodes = [(start, end) for start, end in zip(starts, ends)
            if end > start + np.timedelta64(minDuration,'m')]

    # set nonWear data to nan and record to nonWearBouts file
    f = gzip.open(nonWearFile,'wb')
    f.write('start,end,xStdMax,yStdMax,zStdMax\n'.encode())
    timeFormat = '%Y-%m-%d %H:%M:%S'
    for episode in nonWearEpisodes:
        tmp = e[['xStd','yStd','zStd']][episode[0]:episode[1]]
        nonWearBout = episode[0].strftime(timeFormat) + ','
        nonWearBout += episode[1].strftime(timeFormat) + ','
        nonWearBout += str(tmp['xStd'].mean()) + ','
        nonWearBout += str(tmp['yStd'].mean()) + ','
        nonWearBout += str(tmp['zStd'].mean()) + '\n'
        f.write(nonWearBout.encode())
        # set main dataframe values to nan
        e[episode[0]:episode[1]] = np.nan
    f.close()
    # write to summary
    summary['wearTime-numNonWearEpisodes(>1hr)'] = int(len(nonWearEpisodes))

    # calculate wear statistics
    wearSamples = e['enmoTrunc'].count()
    nonWearSamples = len(e[np.isnan(e['enmoTrunc'])].index.values)
    wearTimeMin = wearSamples * epochPeriod / 60.0
    nonWearTimeMin = nonWearSamples * epochPeriod / 60.0
    # write to summary
    summary['wearTime-overall(days)'] = accUtils.formatNum(wearTimeMin/1440.0, 2)
    summary['nonWearTime-overall(days)'] = accUtils.formatNum(nonWearTimeMin/1440.0, 2)

    # get wear time in each of 24 hours across week
    epochsInMin = 60.0 / epochPeriod
    for i, day in zip(range(0, 7), accUtils.DAYS):
        dayWear = e['enmoTrunc'][e.index.weekday == i].count() / epochsInMin
        # write to summary
        summary['wearTime-' + day + '(hrs)'] = accUtils.formatNum(dayWear/60.0, 2)
    for i in range(0, 24):
        hourWear = e['enmoTrunc'][e.index.hour == i].count() / epochsInMin
        # write to summary
        summary['wearTime-hourOfDay' + str(i) + '-(hrs)'] = \
            accUtils.formatNum(hourWear/60.0, 2)
    summary['wearTime-diurnalHrs'] = accUtils.formatNum( \
        e['enmoTrunc'].groupby(e.index.hour).mean().count(), 2)
    summary['wearTime-diurnalMins'] = accUtils.formatNum( \
        e['enmoTrunc'].groupby([e.index.hour, e.index.minute]).mean().count(), 2)

    # write binary decision on whether weartime was good or not
    minDiurnalHrs = 24
    minWearDays = 3
    summary['quality-goodWearTime'] = 1
    if summary['wearTime-diurnalHrs'] < minDiurnalHrs or \
         summary['wearTime-overall(days)'] < minWearDays:
        summary['quality-goodWearTime'] = 0



def perform_wearTime_imputation(e, verbose):
    """Calculate imputation values to replace nan PA metric values

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

    e['hour'] = e.index.hour
    e['minute'] = e.index.minute

    wearTimeWeights = e.groupby(['hour', 'minute']).mean()
    # add the wearTimeWeights column to the other data as e.g. 'enmoTrunc_imputed'
    e = e.join(wearTimeWeights, on=['hour', 'minute'], rsuffix='_imputed')

    # now wearTime weight values
    for col in wearTimeWeights:
        e[col+'Imputed'] = e[col].fillna(e[col+'_imputed'])

    if verbose:
        # features averaged over epochs - use imputed version of features for this.
        # this ignores rows with NaN and infinities
        imputedCols = e.filter(regex='Imputed').columns
        print(e[imputedCols].isnull().any(axis=1).sum(), \
            "NaN rows in imputed features")
        with pd.option_context('mode.use_inf_as_null', True):
            null_rows = e[imputedCols].isnull().any(axis=1)
        print(null_rows.sum(), " NaN or inf rows in imputed features out of ",\
            len(e.index))
    return e



def calculateECDF(e, inputCol, summary):
    """Calculate activity intensity empirical cumulative distribution

    The input data must not be imputed, as ECDF requires different imputation
    where nan/non-wear data segments are IMPUTED FOR EACH INTENSITY LEVEL. Here,
    the average of similar time-of-day values is imputed with one minute
    granularity on different days of the measurement. Following intensity levels
    are calculated
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

    ecdf1, step = np.linspace(1, 20, 20, retstep=True)  # 1mg bins from 1-20mg
    ecdf2, step = np.linspace(25, 100, 16, retstep=True)  # 5mg bins from 25-100mg
    ecdf3, step = np.linspace(125, 500, 16, retstep=True)  # 25mg bins from 125-500mg
    ecdf4, step = np.linspace(600, 2000, 15, retstep=True)  # 100mg bins from 500-2000mg
    ecdfXVals = np.concatenate([ecdf1, ecdf2, ecdf3, ecdf4])

    # remove NaNs (necessary for statsmodels.api)
    ecdfData = e[['hour', 'minute', inputCol]][~np.isnan(e[inputCol])]
    if len(ecdfData) > 0:
        # set column names for actual, imputed, and adjusted intensity dist. vals
        cols = []
        colsImputed = []
        colsAdjusted = []
        for xVal in ecdfXVals:
            col = 'ecdf' + str(xVal)
            cols.append(col)
            colsImputed.append(col + 'Imputed')
            colsAdjusted.append(col + 'Adjusted')
            ecdfData[col] = (ecdfData[inputCol] <= xVal) * 1.0
        # calculate imputation values to replace nan metric values
        wearTimeWeights = ecdfData.groupby(['hour', 'minute'])[cols].mean()
        ecdfData = ecdfData.join(wearTimeWeights, on=['hour', 'minute'],
                                rsuffix='Imputed')
        # for each ecdf xVal column, apply missing data imputation
        for col, imputed, adjusted in zip(cols, colsImputed, colsAdjusted):
            ecdfData[adjusted] = ecdfData[col].fillna(ecdfData[imputed])

        accEcdf = ecdfData[colsAdjusted].mean()
    else:
        accEcdf = pd.Series(data=[0.0 for i in ecdfXVals],
                            index=[str(i)+'Adjusted' for i in ecdfXVals])

    # and write to summary dict
    for x, ecdf in zip(ecdfXVals, accEcdf):
        summary[inputCol + '-ecdf-' + str(accUtils.formatNum(x,0)) + 'mg'] = \
            accUtils.formatNum(ecdf, 5)



def writeMovementSummaries(e, labels, summary):
    """Write overall summary stats for each activity type to summary dict

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics

    :return: Write dict <summary> keys for each activity type 'overall-<avg/sd>',
        'week<day/end>-avg', '<day..>-avg', 'hourOfDay-<hr..>-avg',
        'hourOfWeek<day/end>-<hr..>-avg'
    :rtype: void
    """

    # identify activity types to summarise
    activityTypes = ['acc', 'MVPA', 'VPA']
    activityTypes += labels
    if 'MET' in e.columns:
        activityTypes.append('MET')

    # sumarise each type by: overall, week day/end, day, and hour of day
    for accType in activityTypes:
        col = accType + 'Imputed'
        if accType in ['MVPA', 'VPA']:
            col = accType

        # overall / weekday / weekend summaries
        summary[accType + '-overall-avg'] = accUtils.formatNum(e[col].mean(), 5)
        summary[accType + '-overall-sd'] = accUtils.formatNum(e[col].std(), 2)
        summary[accType + '-weekday-avg'] = accUtils.formatNum( \
            e[col][e.index.weekday<=4].mean(), 2)
        summary[accType + '-weekend-avg'] = accUtils.formatNum( \
            e[col][e.index.weekday>=5].mean(), 2)

        # daily summary
        for i, day in zip(range(0, 7), accUtils.DAYS):
            summary[accType + '-' + day + '-avg'] = accUtils.formatNum( \
                e[col][e.index.weekday == i].mean(), 2)

        # hourly summaries
        for i in range(0, 24):
            hourOfDay = accUtils.formatNum(e[col][e.index.hour == i].mean(), 2)
            hourOfWeekday = accUtils.formatNum( \
                e[col][(e.index.weekday<=4) & (e.index.hour == i)].mean(), 2)
            hourOfWeekend = accUtils.formatNum( \
                e[col][(e.index.weekday>=5) & (e.index.hour == i)].mean(), 2)
            # write derived hourly values to summary dictionary
            summary[accType + '-hourOfDay-' + str(i) + '-avg'] = hourOfDay
            summary[accType + '-hourOfWeekday-' + str(i) + '-avg'] = hourOfWeekday
            summary[accType + '-hourOfWeekend-' + str(i) + '-avg'] = hourOfWeekend
