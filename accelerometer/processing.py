import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import entropy, median_abs_deviation
from scipy.ndimage import median_filter
from tqdm.auto import tqdm
import statsmodels.api as sm


GRAVITY_CUTOFF_HZ = 0.5
NOISE_CUTOFF_HZ = 20
MAXVAL = 6  # Note: GENEActiv has a range of +/-6g whereas Axivity and Actigraph +/-8 or more


class Processing():
    def __init__(self, **kwargs):
        self.sampleRate = kwargs['sampleRate']
        self.timeZone = kwargs['timeZone']
        self.epochLen = kwargs['epochLen']
        self.extractor = None
        if kwargs.get('extractFeatures', True):
            self.extractor = FeatureExtractor(
                sampleRate=self.sampleRate,
                epochLen=self.epochLen
            )


    def run(self, data):
        infoTime = setTimeZone(data, tz=self.timeZone)
        data, infoResample = regulariseSampleRate(data, self.sampleRate)
        infoFilter = filterNoise(data, self.sampleRate)
        infoCalibr = calibrateGravity(data)
        infoNonwear = detectNonwear(data)

        epochFeats = None
        if self.extractor is not None:
            epochFeats = self.extractor.run(data)

            # # Saving...
            # if self.epochFile.endswith('.pkl'):
            #     epochFeats.to_pickle(self.epochFile)

            # else:
            #     # epochFeats.index = epochFeats.index.to_series().apply(date_strftime)
            #     epochFeats.to_csv(
            #         self.epochFile, 
            #         index=True, 
            #         index_label='time', 
            #         compression='gzip',
            #         date_format=date_strftime)

        info = {**infoTime,
                **infoResample,
                **infoFilter,
                **infoCalibr,
                **infoNonwear}

        return epochFeats, info


def loadNpyToFrame(npyFile):
    data = pd.DataFrame(np.load(npyFile))
    data['time'] = data['time'].astype('datetime64[ms]').dt.tz_localize('UTC')
    data.set_index('time', inplace=True)
    return data


def setTimeZone(data, tz='Europe/London'):
    data.index = data.index.tz_convert(tz)
    info = {}
    startTime, endTime = data.index[0], data.index[-1]
    info['file-start'] = date_strftime(startTime)
    info['file-end'] = date_strftime(endTime)
    info['file-first-day(0=mon,6=sun)'] = startTime.weekday()

    if data.index[0].dst() < data.index[-1].dst():
        info['daylight-savings-time'] = 1
    elif data.index[0].dst() > data.index[-1].dst():
        info['daylight-savings-time'] = -1
    else:
        info['daylight-savings-time'] = 0

    return info


def regulariseSampleRate(data, sampleRate, method='nearest'):
    info = {}
    info['num-ticks'] = len(data)

    gap = data.index.to_series().diff()
    interrupt = gap > pd.Timedelta('1s')
    interruptLen = gap[interrupt].sum()
    info['num-interrupts'] = int(interrupt.sum())
    info['interrupts-overall(mins)'] = interruptLen.total_seconds() / 60

    samplePeriodNanos = int(1000_000_000/sampleRate)  # in nanos
    if method == 'nearest':
        dataResampled = data.resample(f'{samplePeriodNanos}N').nearest(limit=1)
    elif method == 'linear':
        raise NotImplementedError
    else:
        raise ValueError

    info['num-ticks-resampled'] = len(dataResampled)

    return dataResampled, info


def filterNoise(data, sampleRate):
    info = {}

    xyzCols = ['x', 'y', 'z']

    # Clip unrealistically high values
    info['num-clips'] = int((data[xyzCols].abs() > MAXVAL).any(1).sum())
    data[xyzCols] = np.clip(data[xyzCols].values, -MAXVAL, MAXVAL)

    # Remove stuck values
    rolling = data[xyzCols].rolling('10s', min_periods=10*sampleRate)
    stationary = (rolling.std()==0).any(1)
    values = rolling.mean()
    stuck = stationary & ((values > 1.5).any(1) | (values < -1.5).any(1))
    info['num-stuck-values'] = int(stuck.sum())
    data[stuck] = np.nan  # replace stuck values with NaNs

    # Temporarily fill nan values
    mask = data[xyzCols].isna().any(1)
    data[xyzCols] = data[xyzCols].fillna(method='ffill').fillna(method='bfill')

    # Noise removal by median filtering -- removes outliers
    data[xyzCols] = median_filter(data[xyzCols].values, (5,1), mode='nearest')
    # Noise removal by lowpass filtering -- removes high freqs
    data[xyzCols] = butterfilt(data[xyzCols].values, NOISE_CUTOFF_HZ, sampleRate, axis=0)

    # Restore NaN values
    data.loc[mask, xyzCols] = np.nan

    return info


def calibrateGravity(data, stdWindow='10s', stdTol=13/1000, calibCritSphere=0.3):
    ''' Autocalibration of accelerometer data for free-living physical
    activity assessment using local gravity and temperature: an
    evaluation on four continents. https://pubmed.ncbi.nlm.nih.gov/25103964/ '''

    info = {}

    # The paper uses window means instead of raw values. This sort of
    # reduces the influence of outliers and computational cost
    grouped = data.groupby(pd.Grouper(freq=stdWindow))
    statioData = grouped.mean()[(grouped[['x','y','z']].std() < stdTol).all(1)]

    xyz, T = statioData[['x','y','z']], statioData['T']
    Tref = T.mean()
    dT = T - Tref

    xyz, dT = xyz.values, dT.values

    intercept = np.array([0.0, 0.0, 0.0])
    slope = np.array([1.0, 1.0, 1.0])
    Tslope = np.array([0.0, 0.0, 0.0])
    bestIntercept = np.copy(intercept)
    bestSlope = np.copy(slope)
    bestTslope = np.copy(Tslope)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1)[:,None]
    initError = np.sqrt(np.mean(np.square(curr-target)))  # root mean square error
    bestError = 1e16

    MAXITER = 1000
    TOL = 0.0001  # 0.1mg

    for _ in range(MAXITER):

        for i in range(3):
            inp = np.column_stack((curr[:,i], dT))
            out = target[:,i]
            inp = sm.add_constant(inp, prepend=True)  # add bias/intercept term
            _intercept, _slope, _Tslope = sm.OLS(out, inp).fit().params
            intercept[i] = _intercept + (intercept[i] * _slope)
            slope[i] = _slope * slope[i]
            Tslope[i] = _Tslope + (Tslope[i] * _slope)

        curr = intercept + (xyz * slope) + (dT[:,None] * Tslope)
        target = curr / np.linalg.norm(curr, axis=1)[:,None]

        rms = np.sqrt(np.mean(np.square(curr-target)))
        improvement = (bestError-rms)/bestError
        if rms < bestError:
            bestIntercept = np.copy(intercept)
            bestSlope = np.copy(slope)
            bestTslope = np.copy(Tslope)
            bestError = rms
        if improvement < TOL:
            break

    # Quality control
    if (np.max(xyz, axis=0) < calibCritSphere).any() \
        or (np.min(xyz, axis=0) > -calibCritSphere).any() \
        or bestError > 0.01:
        info['calibration-OK'] = 0
        # Restore default values as calibration failed
        bestIntercept = np.array([0.0, 0.0, 0.0])
        bestSlope = np.array([1.0, 1.0, 1.0])
        bestTslope = np.array([0.0, 0.0, 0.0])
        bestError = initError

    else:
        info['calibration-OK'] = 1
        # Calibrate
        data[['x','y','z']] = \
            bestIntercept + \
            bestSlope * data[['x','y','z']].values + \
            bestTslope * (data['T'].values[:, None] - Tref)

    info['calibration-numStaticWindows'] = len(statioData)
    info['calibration-errsBefore(mg)'] = initError * 1000
    info['calibration-errsAfter(mg)'] = bestError * 1000
    info['calibration-xIntercept'] = bestIntercept[0]
    info['calibration-yIntercept'] = bestIntercept[1]
    info['calibration-zIntercept'] = bestIntercept[2]
    info['calibration-xSlope'] = bestSlope[0]
    info['calibration-ySlope'] = bestSlope[1]
    info['calibration-zSlope'] = bestSlope[2]
    info['calibration-xTSlope'] = bestTslope[0]
    info['calibration-yTSlope'] = bestTslope[1]
    info['calibration-zTSlope'] = bestTslope[2]
    info['calibration-Tref'] = Tref

    return info


def detectNonwear(data, nonwearPatience='1h', stdWindow='10s', stdTol=13/1000):
    info = {}

    stationary = (data[['x','y','z']].rolling(stdWindow).std() < stdTol).all(1)

    group = (stationary != stationary.shift(1)).cumsum()  # group by consecutive values
    groupLen = group.groupby(group).apply(lambda g: g.index[-1] - g.index[0])
    stationaryLen = groupLen[stationary.groupby(group).any()]
    nonwearLen = stationaryLen[stationaryLen > pd.Timedelta(nonwearPatience)]

    info['num-nonwear-episodes'] = len(nonwearLen)
    info['nonwear-overall(days)'] = nonwearLen.sum().total_seconds() / (60*60*24)
    info['wear-overall(days)'] = (groupLen.sum() - nonwearLen.sum()).total_seconds() / (60*60*24)

    # Fill nonwear with NaNs
    nonwear = group.isin(nonwearLen.index)
    data[nonwear] = np.nan

    return info


class FeatureExtractor():
    def __init__(self, **kwargs) -> None:
        self.sampleRate = kwargs['sampleRate']
        self.epochLen = kwargs['epochLen']


    def run(self, data):
        epochs = data.groupby(pd.Grouper(freq=f'{self.epochLen}S'))

        featsFrame = {}
        for t, epoch in tqdm(epochs):
            xyz = epoch[['x', 'y', 'z']].values

            # Check if good chunk else return NaNs
            if (np.isfinite(xyz).all() and (len(xyz) == self.epochLen * self.sampleRate)):
                feats = FeatureExtractor.extract(xyz, self.sampleRate)
            else:
                feats = {name:np.nan for name in FeatureExtractor.featureNames()}

            featsFrame[t] = feats

        featsFrame = pd.DataFrame.from_dict(featsFrame, orient='index')

        return featsFrame


    @staticmethod
    def extract(xyz, sampleRate):
        return {
            **FeatureExtractor.basicFeatures(xyz, sampleRate),
            **FeatureExtractor.sanDiegoFeatures(xyz, sampleRate),
            **FeatureExtractor.unileverFeatures(xyz, sampleRate),
        }


    @staticmethod
    def featureNames():
        return [
            *FeatureExtractor.basicFeaturesNames(),
            *FeatureExtractor.sanDiegoFeaturesNames(),
            *FeatureExtractor.unileverFeaturesNames()
        ]

    
    @staticmethod
    def basicFeatures(xyz, sampleRate):
        feats = {}
        feats['xMean'], feats['yMean'], feats['zMean'] = np.mean(xyz, axis=0)
        feats['xStd'], feats['yStd'], feats['zStd'] = np.std(xyz, axis=0)
        feats['xRange'], feats['yRange'], feats['zRange'] = np.ptp(xyz, axis=0)

        x, y, z = xyz.T

        with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
            feats['xyCorr'] = np.nan_to_num(np.corrcoef(x, y)[0,1])
            feats['yzCorr'] = np.nan_to_num(np.corrcoef(y, z)[0,1])
            feats['zxCorr'] = np.nan_to_num(np.corrcoef(z, x)[0,1])

        m = np.linalg.norm(xyz, axis=1)

        # "Separating Movement and Gravity Components in an Acceleration Signal
        # and Implications for the Assessment of Human Daily Physical Activity"
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061691
        feats['enmoTrunc'] = np.mean(np.maximum(m - 1.0, 0.0))

        # "A universal, accurate intensity-based classification of different
        # physical activities using raw data of accelerometer"
        # https://pubmed.ncbi.nlm.nih.gov/24393233/
        feats['std'] = np.std(m)
        feats['mad'] = stats.median_abs_deviation(m)
        feats['kurt'] = stats.kurtosis(m)
        feats['skew'] = stats.skew(m)

        return feats


    @staticmethod
    def basicFeaturesNames():
        return ['xMean', 'yMean', 'zMean',
                'xStd', 'yStd', 'zStd',
                'xRange', 'yRange', 'zRange',
                'xyCorr', 'yzCorr', 'zxCorr',
                'enmoTrunc', 
                'std', 'mad', 'kurt', 'skew']


    @staticmethod
    def sanDiegoFeatures(xyz, sampleRate):
        ''' "Hip and Wrist Accelerometer Algorithms for Free-Living Behavior Classification"
        https://pubmed.ncbi.nlm.nih.gov/26673126/ 
        Computation of the FFT features are refactored out -- see fftFeatures() '''

        feats = {}

        # It is not really clear from the paper whether some of the features
        # are computed on the raw stream or the body stream ("gravity removed", bandpassed), 
        # but the legacy code uses body stream so we follow that. 

        # Body stream
        xyzb = butterfilt(xyz, (GRAVITY_CUTOFF_HZ, NOISE_CUTOFF_HZ), sampleRate, axis=0)
        xb, yb, zb = xyzb.T
        mb = np.linalg.norm(xyzb, axis=1)

        # Gravity stream
        xyzg = butterfilt(xyz, GRAVITY_CUTOFF_HZ, sampleRate, axis=0)
        xg, yg, zg = xyzg.T

        feats['bodyMean'], feats['bodyStd'] = np.mean(mb), np.std(mb)
        feats['bodyCoefVar'] = feats['bodyMean'] / (feats['bodyStd'] + 1e-8)
        feats['bodyMin'], feats['bodyMax'] = np.min(mb), np.max(mb)
        feats['body25p'], feats['bodyMedian'], feats['body75p'] = np.percentile(mb, (25, 50, 75))
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
            feats['bodyAutocorr'] = np.nan_to_num(np.corrcoef(mb[sampleRate:], mb[:-sampleRate])[0,1])

        with np.errstate(divide='ignore', invalid='ignore'):  # ignore div by 0 warnings
            feats['bodyxyCorr'] = np.nan_to_num(np.corrcoef(xb,yb)[0,1])
            feats['bodyyzCorr'] = np.nan_to_num(np.corrcoef(yb,zb)[0,1])
            feats['bodyzxCorr'] = np.nan_to_num(np.corrcoef(zb,xb)[0,1])

        # ---------------------
        # Orientation features
        # ---------------------
        # Roll, pitch, yaw
        yaw, roll, pitch = np.arctan2(xb, yb), np.arctan2(yb, zb), np.arctan2(zb, xb)
        feats['yawAvg'], feats['rollAvg'], feats['pitchAvg'] = np.mean(yaw), np.mean(roll), np.mean(pitch)
        feats['yawStd'], feats['rollStd'], feats['pitchStd'] = np.std(yaw), np.std(roll), np.std(pitch)
        # Roll, pitch, yaw using gravity stream
        yawg, rollg, pitchg = np.arctan2(xg, yg), np.arctan2(yg, zg), np.arctan2(zg, xg)
        feats['yawgAvg'], feats['rollgAvg'], feats['pitchgAvg'] = np.mean(yawg), np.mean(rollg), np.mean(pitchg)

        # -----------------------------------------------------------------
        # Spectral features. Legacy code uses the raw stream, which
        # makes sense given the search range of 0.3Hz - 3Hz.
        # -----------------------------------------------------------------
        m = np.linalg.norm(xyz, axis=1)

        # Spectrum using Welch's method with 5s segment length
        freqs, powers = welch(m, fs=sampleRate, nperseg=5*sampleRate, detrend='constant')

        feats['entropy'] = entropy(powers)
        # Dominant frequency
        idxmax = np.argmax(powers)
        feats['dominantFreq'], feats['dominantPower'] = freqs[idxmax], powers[idxmax]
        # Dominant frequency between 0.3-3Hz
        mask = (0.3 <= freqs) & (freqs <= 3)
        idxmax = np.argmax(powers[mask])
        feats['dominantFreq_0.3_3'], feats['dominantPower_0.3_3'] = freqs[mask][idxmax], powers[mask][idxmax]

        return feats


    @staticmethod
    def sanDiegoFeaturesNames():
        return ['bodyMean', 'bodyStd', 'bodyCoefVar',
                'bodyMin', 'bodyMax', 'body25p', 'bodyMedian', 'body75p', 
                'bodyAutocorr',
                'yawAvg', 'rollAvg', 'pitchAvg', 
                'yawStd', 'rollStd', 'pitchStd', 
                'yawgAvg', 'rollgAvg', 'pitchgAvg',
                'entropy', 
                'dominantFreq', 'dominantPower', 
                'dominantFreq_0.3_3', 'dominantPower_0.3_3']


    @staticmethod
    def fftFeatures(xyz, sampleRate):
        """ Powers for 1Hz, 2Hz, ..., 15Hz using Welch's method with 1s
        segment length. Used in https://pubmed.ncbi.nlm.nih.gov/26673126/ """
        feats = {}
        m = np.linalg.norm(xyz, axis=1)
        _, powers15 = welch(m, fs=sampleRate, nperseg=sampleRate, detrend='constant')
        powers15 = powers15[1:16]
        feats = {f'sanFFT{i+1}':val for i, val in enumerate(powers15)}

        return feats


    @staticmethod
    def fftFeaturesNames():
        return [f'sanFFT{i+1}' for i in range(15)]


    @staticmethod
    def unileverFeatures(xyz, sampleRate):
        ''' "Physical Activity Classification Using the GENEA Wrist-Worn Accelerometer"
        https://pubmed.ncbi.nlm.nih.gov/21988935/ '''
        feats = {}
        m = np.linalg.norm(xyz, axis=1)

        # Spectrum using Welch's method with 5s segment length
        freqs, powers = welch(m, fs=sampleRate, nperseg=5*sampleRate, detrend='constant')

        # Dominant and 2nd dominant between 0.3-15Hz
        mask = (0.3 <= freqs) & (freqs <= 15)
        _freqs, _powers = freqs[mask], powers[mask]
        idxs = np.argsort(_powers)
        feats['dominantFreq_0.3_15'], feats['dominantPower_0.3_15'] = _freqs[idxs[-1]], _powers[idxs[-1]]
        feats['secondDominantFreq_0.3_15'], feats['secondDominantPower_0.3_15'] = _freqs[idxs[-2]], _powers[idxs[-2]]

        # Total power between 0.3-15Hz
        feats['totalPower_0.3_15'] = np.sum(_powers)

        # Dominant between 0.6-2.5Hz
        mask = (0.6 <= freqs) & (freqs <= 2.5)
        _freqs, _powers = freqs[mask], powers[mask]
        idx = np.argmax(_powers)
        feats['dominantFreq_0.6_2.5'], feats['dominantPower_0.6_2.5'] = _freqs[idx], _powers[idx]

        return feats


    @staticmethod
    def unileverFeaturesNames():
        return ['dominantFreq_0.3_15', 'dominantPower_0.3_15',
                'secondDominantFreq_0.3_15', 'secondDominantPower_0.3_15',
                'totalPower_0.3_15',
                'dominantFreq_0.6_2.5', 'dominantPower_0.6_2.5']


def butterfilt(x, cutoffs, fs, order=4, axis=0):
    nyq = 0.5 * fs
    if isinstance(cutoffs, tuple):
        btype = 'bandpass'
        low, high = cutoffs
        Wn = (low / nyq, high / nyq)
    else:
        btype = 'low'
        Wn = cutoffs / nyq
    sos = butter(order, Wn, btype=btype, analog=False, output='sos')
    y = sosfiltfilt(sos, x, axis=axis)
    return y


def date_strftime(t):
    ''' Convert to time format of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London] '''
    tz = t.tz
    return t.strftime(f'%Y-%m-%d %H:%M:%S.%f%z [{tz}]')

