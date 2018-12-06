"""Module to support machine learning of activity states from acc data"""

from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import tarfile
import warnings

# must be same feature columns used to train classifier
featureCols = ['enmoTrunc', 'xMean', 'yMean', 'zMean', 'xRange', 'yRange',
           'zRange', 'xStd', 'yStd', 'zStd', 'xyCov', 'xzCov', 'yzCov',
           'temp', 'mean', 'sd', 'coefvariation', 'median',
           'min', 'max', '25thp', '75thp', 'autocorr', 'corrxy', 'corrxz',
           'corryz', 'avgroll', 'avgpitch', 'avgyaw', 'sdroll', 'sdpitch',
           'sdyaw', 'rollg', 'pitchg', 'yawg', 'fmax', 'pmax', 'fmaxband',
           'pmaxband', 'entropy', 'fft0', 'fft1', 'fft2', 'fft3', 'fft4',
           'fft5', 'fft6', 'fft7', 'fft8', 'fft9', 'fft10', 'fft11',
           'fft12', 'fft13', 'fft14', 'MAD', 'MPD', 'skew', 'kurt',
           'f1', 'p1', 'f2', 'p2', 'f625', 'p625', 'total', 'xfft0',
           'xfft1', 'xfft2', 'xfft3', 'xfft4', 'xfft5', 'xfft6', 'xfft7',
           'xfft8', 'xfft9', 'xfft10', 'xfft11', 'xfft12', 'xfft13',
           'xfft14', 'yfft0', 'yfft1', 'yfft2', 'yfft3', 'yfft4',
           'yfft5', 'yfft6', 'yfft7', 'yfft8', 'yfft9', 'yfft10',
           'yfft11', 'yfft12', 'yfft13', 'yfft14', 'zfft0', 'zfft1',
           'zfft2', 'zfft3', 'zfft4', 'zfft5', 'zfft6', 'zfft7', 'zfft8',
           'zfft9', 'zfft10', 'zfft11', 'zfft12', 'zfft13', 'zfft14',
           'mfft0', 'mfft1', 'mfft2', 'mfft3', 'mfft4', 'mfft5',
           'mfft6', 'mfft7', 'mfft8', 'mfft9', 'mfft10', 'mfft11',
           'mfft12', 'mfft13', 'mfft14']


def activityClassification(epochFile, 
    activityModel = "activityModels/doherty2018.tar"):
    """Perform classification of activity states from epoch feature data

    Based on a balanced random forest with a Hidden Markov Model containing 
    transitions between predicted activity states and emissions trained using a 
    free-living groundtruth to identify pre-defined classes of behaviour from 
    accelerometer data.

    :param str epochFile: Input csv file of processed epoch data
    :param str activityModel: Input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METS for each activity state

    :return: Pandas dataframe of activity epoch data with one-hot encoded labels
    :rtype: pandas.DataFrame

    :return: Activity state labels
    :rtype: list(str)
    """
    
    X = epochFile

    print(X[featureCols].isnull().any(axis=1).sum(), "NaN rows")
    with pd.option_context('mode.use_inf_as_null', True):
        null_rows = X[featureCols].isnull().any(axis=1)
    print(null_rows.sum(), " null or inf rows out of ", len(X.index))
    
    X['label'] = 'none'
    X.loc[null_rows, 'label'] = 'inf_or_null'
    #setup RF
    # ignore warnings on deployed model using different version of pandas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rf = joblib.load(getFileFromTar(activityModel, 'rfModel.pkl'))
    labels = rf.classes_.tolist()
    rfPredictions = rf.predict(X.loc[~null_rows,featureCols])
    # free memory
    del rf
    #setup HMM
    priors = np.load(getFileFromTar(activityModel, 'hmmPriors.npy'))
    transitions = np.load(getFileFromTar(activityModel, 'hmmTransitions.npy'))
    emissions = np.load(getFileFromTar(activityModel, 'hmmEmissions.npy'))
    hmmPredictions = viterbi(rfPredictions.tolist(), labels, priors, \
        transitions, emissions)
    #save predictions to pandas dataframe
    X.loc[~null_rows, 'label'] = hmmPredictions
    
    # perform MET prediction...
    met_vals = np.load(getFileFromTar(activityModel, 'METs.npy'))
    met_dict = {}
    for l, m in zip(labels, met_vals):
        met_dict[l] = m
    X.loc[~null_rows, 'MET'] = X.loc[~null_rows, 'label'].replace(met_dict)

    # apply one-hot encoding
    for l in labels:
        X[l] = 0
        X.loc[X['label']==l, l] = 1 
    # null values aren't one-hot encoded, so set such instances to NaN
    for l in labels:
        X.loc[X[labels].sum(axis=1) == 0, l] = np.nan
    return X, labels


def viterbi(observations, states, priors, transitions, emissions,
            probabilistic=False):
    """Perform HMM smoothing over observations via Viteri algorithm

    :param list(str) observations: List/sequence of activity states
    :param numpy.array states: List of unique activity state labels
    :param numpy.array priors: Prior probabilities for each activity state
    :param numpy.array transitions: Probability matrix of transitioning from one
        activity state to another
    :param numpy.array emissions: Probability matrix of RF prediction being true
    :param bool probabilistic: Write probabilistic output for each state, rather
        than writing most likely state for any given prediction.

    :return: Smoothed list/sequence of activity states
    :rtype: list(str)
    """

    tinyNum = 0.000001
    nObservations = len(observations)
    nStates = len(states)
    v = np.zeros((nObservations,nStates)) # initialise viterbi table
    # set prior state values for first observation...
    for state in range(0, len(states)):
        v[0,state] = np.log(priors[state] * emissions[state,states.index(observations[0])]+tinyNum)
    # fill in remaning matrix observations
    # e use log space as multiplying successively smaller p values)
    for k in range(1,nObservations):
        for state in range(0, len(states)):
             v[k,state] = np.log(emissions[state,states.index(observations[k])]+tinyNum) + \
                          np.max(v[k-1,:] + np.log(transitions[:,state]+tinyNum), axis=0)
    
    # now construct viterbiPath (propagating backwards)
    viterbiPath = observations
    # pick most probable state for final observation
    viterbiPath[nObservations-1] = states[np.argmax(v[nObservations-1,:],axis=0)]
    
    # probabilistic method will give probability of each label
    norm = lambda x: x / x.sum()
    if probabilistic==True:
        viterbiProba = np.zeros((nObservations,nStates)) # initialize table
        viterbiProba[nObservations-1,:] = norm(v[nObservations-1,:])

    #and then work backwards to pick most probable state for all other observations
    for k in list(reversed(range(0,nObservations-1))):
        viterbiPath[k] = states[np.argmax(v[k,:] + np.log(transitions[:,states.index(viterbiPath[k+1])]+tinyNum),axis=0)]
        if probabilistic == True:
            viterbiProba[k,:] = norm(v[k,:] + np.log(transitions[:,states.index(viterbiPath[k+1])]+tinyNum))
        
    #output as list...
    return viterbiProba if probabilistic else viterbiPath



def getFileFromTar(tarArchive, targetFile):
    """Read file from tar

    This is currently more tricky than it should be see
    https://github.com/numpy/numpy/issues/7989

    :param str tarArchive: input tarfile object
    :param str targetFile: target individual file within .tar
    
    :return: file object byte stream
    :rtype: object
    """

    t = tarfile.open(tarArchive, 'r')
    array_file = BytesIO()
    array_file.write(t.extractfile(targetFile).read())
    array_file.seek(0)
    return array_file