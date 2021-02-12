import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import entropy, median_abs_deviation
from scipy.ndimage import median_filter
from tqdm.auto import tqdm


GRAVITY_CUTOFF_HZ = 0.5
NOISE_CUTOFF_HZ = 20
MAXVAL, MINVAL = 6, -6  # Note: GENEActiv has a range of +/-6g whereas Axivity and Actigraph +/-8 or more


class Processing():
    def __init__(self, **kwargs):
        self.sampleRate = kwargs['sampleRate']
        self.epochPeriod = kwargs['epochPeriod']
        self.epochFile = kwargs['epochFile']
        self.timeZone = kwargs['timeZone']
        self.extractor = None
        if kwargs.get('extractFeatures', True):
            self.extractor = FeatureExtractor(
                sampleRate=self.sampleRate,
                epochPeriod=self.epochPeriod
            )


    def run(self, npyFile):
        data = loadNpyToFrame(npyFile)

        #TODO quality control

        data = Processing.regularizeSampleRate(data, self.sampleRate)
        data = Processing.filterNoise(data, self.sampleRate)

        #TODO
        # Used to be done in device.py but would make sense to do it here.
        # data = calibrateGravity(...)

        #TODO
        # Currently done in summariseEpoch but would make more sense to do it
        # here. Also the approach is slightly different to the paper.
        # data = detectNonwear(...)

        if self.extractor is not None:
            epochFeats = self.extractor.run(data)

            #TODO Completely fake news!
            epochFeats['rawSamples'] = 0
            epochFeats['clipsBeforeCalibr'] = 0
            epochFeats['clipsAfterCalibr'] = 0

            # Saving...
            # Make output time format contain timezone
            # e.g. 2020-06-14 19:01:15.123000+0100 [Europe/London]
            epochFeats.index = epochFeats.index.to_series().apply(date_strftime)
            epochFeats.to_csv(
                self.epochFile, 
                index=True, 
                index_label='time', 
                compression='gzip',
                date_format=date_strftime)


    @staticmethod
    def regularizeSampleRate(data, sampleRate, method='nearest'):
        samplePeriodNanos = int(1000_000_000/sampleRate)  # in nanos
        if method == 'nearest':
            return data.resample(f'{samplePeriodNanos}N').nearest(limit=1)
        elif method == 'linear':
            raise NotImplementedError
        else:
            raise ValueError


    @staticmethod
    def filterNoise(data, sampleRate):
        xyzCols = ['x', 'y', 'z']

        # Temporarily fill nan values
        mask = data[xyzCols].isna().any(1)
        data[xyzCols] = data[xyzCols].fillna(method='ffill')

        # Clip unrealistically high values
        data[xyzCols] = np.clip(data[xyzCols].values, MINVAL, MAXVAL)
        # Noise removal by median filtering
        data[xyzCols] = median_filter(data[xyzCols].values, (5,1), mode='nearest')
        # Noise removal by lowpass filtering
        data[xyzCols] = butterfilt(data[xyzCols].values, NOISE_CUTOFF_HZ, sampleRate, axis=0)

        # Restore NaN values
        data.loc[mask, xyzCols] = np.nan

        return data


    #TODO
    @staticmethod
    def calibrateGravity():
        pass


    #TODO
    @staticmethod
    def detectNonwear():
        pass


class FeatureExtractor():
    def __init__(self, **kwargs) -> None:
        self.sampleRate = kwargs['sampleRate']
        self.epochPeriod = kwargs['epochPeriod']


    def run(self, data):
        epochs = data.groupby(pd.Grouper(freq=f'{self.epochPeriod}S'))

        featsFrame = {}
        for t, epoch in tqdm(epochs):
            xyz = epoch[['x', 'y', 'z']].values

            # Check if good chunk else return NaNs
            if (np.isfinite(xyz).all() and (len(xyz) == self.epochPeriod * self.sampleRate)):
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
                'bodyYawAvg', 'bodyRollAvg', 'bodyPitchAvg', 
                'bodyYawStd', 'bodyRollStd', 'bodyPitchStd', 
                'bodyYawgAvg', 'bodyRollgAvg', 'bodyPitchgAvg',
                'bodyEntropy', 
                'bodyDominantFreq', 'bodyDominantPower', 
                'bodyDominantFreq_0.3_3', 'bodyDominantPower_0.3_3']


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


def loadNpyToFrame(npyFile, tz='Europe/London'):
    data = pd.DataFrame(np.load(npyFile))
    data['time'] = data['time'].astype('datetime64[ms]').dt.tz_localize('UTC').dt.tz_convert(tz)
    data.set_index('time', inplace=True)
    return data


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
