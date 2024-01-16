"""Module to support machine learning of activity states from acc data"""

from accelerometer import utils
from accelerometer.models import MODELS
from io import BytesIO
import tempfile
import numpy as np
import os
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import joblib
import tarfile
import urllib
import pathlib
import shutil
import warnings
import json


def activityClassification(
    epoch,
    activityModel: str = "walmsley", 
    mgCpLPA: int = 45,
    mgCpMPA: int = 100,
    mgCpVPA: int = 400
):
    """
    Perform classification of activity states from epoch feature data. Based on
    a balanced random forest with a Hidden Markov Model containing transitions
    between predicted activity states and emissions trained using a free-living
    groundtruth to identify pre-defined classes of behaviour from accelerometer
    data.

    :param pandas.DataFrame epoch: Dataframe of processed epoch data
    :param str activityModel: Path to input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METs for each activity state

    :return: Tuple containing a pandas dataframe of activity epoch data with one-hot encoded labels, and a list of activity state labels
    :rtype: tuple(pandas.DataFrame, list(str))
    """

    modelPath = resolveModelPath(activityModel)

    featureCols = joblib.load(getFileFromTar(modelPath, 'featureCols'))
    model = joblib.load(getFileFromTar(modelPath, 'model'))
    hmmParams = joblib.load(getFileFromTar(modelPath, 'hmmParams'))
    labels = joblib.load(getFileFromTar(modelPath, 'labels')).tolist()

    X = epoch[featureCols].to_numpy()
    ok = np.isfinite(X).all(axis=1)
    print(f"{len(epoch) - np.sum(ok)} rows with NaN or Inf values, out of {len(epoch)}")

    Y = pd.Series(index=epoch.index)
    Y.loc[ok] = viterbi(model.predict(X[ok]), hmmParams)

    # TODO: Chan's logic hardcoded here
    if activityModel == 'chan':
        enmo = epoch['enmoTrunc'].to_numpy()
        other = (Y == 'other')
        Y.loc[other & (enmo < 100/1000)] = 'light'
        Y.loc[other & (enmo >= 100/1000)] = 'moderate'
        Y.loc[other & (enmo > 400/1000)] = 'vigorous'
        labels.remove('other')
        labels.append('light')
        labels.append('moderate')
        labels.append('vigorous')
        del enmo
        del other

    Y = removeSpuriousSleep(Y, activityModel=activityModel)

    # One-hot encoding
    epoch.loc[ok, labels] = (Y[ok].to_numpy()[:, None] == labels).astype('float')

    # MET prediction
    METs = joblib.load(getFileFromTar(modelPath, 'METs'))
    if METs is not None:
        epoch.loc[:, "MET"] = Y.replace(METs)

    # Cut-point based classification on non-sleep epochs
    YCpOneHot = cutPointModel(
        epoch['enmoTrunc'],
        cuts={'LPA': mgCpLPA/1000, 'MPA': mgCpMPA/1000, 'VPA': mgCpVPA/1000},
        whr=~(Y == 'sleep')  # Note: ~(Y == 'sleep') != (Y != 'sleep') because of NaNs
    )
    epoch = epoch.join(YCpOneHot)
    labelsCp = list(YCpOneHot.columns)
    labels.extend(labelsCp)

    return epoch, labels


def trainClassificationModel(
    trainingFile,
    labelCol="label", participantCol="participant",
    annotationCol="annotation", metCol="MET",
    featuresTxt="activityModels/features.txt",
    nTrees=1000, maxDepth=None, minSamplesLeaf=1,
    cv=None, testParticipants=None,
    outDir='model/',
    nJobs=1,
):
    """
    Train model to classify activity states from epoch feature data. Based on a
    balanced random forest with a Hidden Markov Model containing transitions
    between predicted activity states and emissions trained using the input
    training file to identify pre-defined classes of behaviour from
    accelerometer data.

    :param str trainingFile: Input csv file of training data, pre-sorted by time
    :param str labelCol: Input label column
    :param str participantCol: Input participant column
    :param str annotationCol: Input text annotation e.g. 'walking with dog' vs. 'walking'
    :param str metCol: Input MET column
    :param str featuresTxt: Input txt file listing feature column names
    :param int cv: Number of CV folds. If None, CV is skipped.
    :param str testParticipants: Input comma separated list of participant IDs to test on.
    :param int nTrees: Random forest n_estimators param.
    :param int maxDepth: Random forest max_depth param.
    :param int minSamplesLeaf: Random forest min_samples_leaf param.
    :param str outDir: Output directory. Output files (trained model, predictions, etc.) will be written to this directory.
    :param int nJobs: Number of jobs to run in parallel.

    """

    report = {
        'params__n_estimators': nTrees,
        'params__max_depth': maxDepth,
        'params__min_samples_leaf': minSamplesLeaf,
    }

    os.makedirs(outDir, exist_ok=True)

    # Load list of features to use for training
    featureCols = np.loadtxt(featuresTxt, dtype='str')

    # Load in participant information, and remove null/messy labels/features
    allCols = [participantCol, labelCol, annotationCol] + featureCols.tolist()
    if metCol:
        allCols.append(metCol)
    data = pd.read_csv(trainingFile, usecols=allCols)
    with pd.option_context('mode.use_inf_as_null', True):
        data = data.dropna(axis=0, how='any')

    # Train/test split if testParticipants provided
    if testParticipants is not None:
        testPIDs = testParticipants.split(',')
        test = data[data[participantCol].isin(testPIDs)].copy()
        train = data[~data[participantCol].isin(testPIDs)].copy()
    else:
        train = data

    X, Y, pid = train[featureCols].to_numpy(), train[labelCol].to_numpy(), train[participantCol].to_numpy()

    def _Model(**kwargs):
        return BalancedRandomForestClassifier(
            n_estimators=nTrees,
            max_depth=maxDepth,
            min_samples_leaf=minSamplesLeaf,
            replacement=True,
            sampling_strategy='not minority',
            random_state=42,
            **kwargs
        )

    print('Training...')
    model = _Model(n_jobs=nJobs, verbose=1)
    model = model.fit(X, Y)
    model.verbose = 0  # silence future calls to .predict()
    labels = model.classes_

    print('Cross-predicting to derive the observations for HMM...')
    NJOBS_PER_CV_MODEL = min(2, nJobs)
    cvp = cross_val_predict(
        _Model(n_jobs=NJOBS_PER_CV_MODEL), X, Y, groups=pid,
        cv=10,
        n_jobs=nJobs // NJOBS_PER_CV_MODEL,
        method="predict_proba",
        verbose=3,
    )
    print('Training HMM...')
    hmmParams = trainHMM(cvp, Y, labels)

    # Estimate METs via per-class averaging
    METs = None
    if metCol:
        METs = {y: train[Y == y].groupby(annotationCol)[metCol].mean().mean()
                for y in model.classes_}

    # Write model to file
    outFile = os.path.join(outDir, 'model.tar')
    saveToTar(outFile,
              model=model,
              labels=labels,
              featureCols=featureCols,
              hmmParams=hmmParams,
              METs=METs)
    print(f'Output trained model written to: {outFile}')

    # Assess model performance on test participants
    if testParticipants is not None:
        print('Testing on participant(s):', testParticipants)
        Xtest, Ytest = test[featureCols].to_numpy(), test[labelCol].to_numpy()
        Ypred = model.predict(Xtest)
        YpredHmm = viterbi(Ypred, hmmParams)
        test['predicted'] = YpredHmm

        # Write predictions to file
        outCols = [participantCol, labelCol, 'predicted']
        outFile = os.path.join(outDir, 'test-predictions.csv')
        test[outCols].to_csv(outFile, index=False)
        print(f'Output test predictions written to: {outFile}')

        print('\nTest performance (no HMM):')
        print(metrics.classification_report(Ytest, Ypred))
        testScore = metrics.f1_score(Ytest, Ypred, average='macro', zero_division=0)
        print(f'Score: {testScore:.2f}')
        report['test_score'] = testScore

        print('\nTest performance (HMM):')
        print(metrics.classification_report(Ytest, YpredHmm))
        testHmmScore = metrics.f1_score(Ytest, YpredHmm, average='macro', zero_division=0)
        print(f'Score: {testHmmScore:.2f}')
        report['test_hmm_score'] = testHmmScore

    if cv:
        print("Cross-validating...")
        cvScores = cross_val_score(
            _Model(n_jobs=NJOBS_PER_CV_MODEL),
            # cv with whole data
            data[featureCols].to_numpy(), data[labelCol].to_numpy(), groups=data[participantCol].to_numpy(),
            scoring=metrics.make_scorer(metrics.f1_score, average='macro', zero_division=0),
            cv=cv,
            n_jobs=nJobs // NJOBS_PER_CV_MODEL,
            verbose=3,
        )
        cvScoresAvg = np.mean(cvScores)
        cvScores25th, cvScores75th = np.quantile(cvScores, (.25, .75))
        print(f"CV score: {cvScoresAvg:.2f} ({cvScores25th:.2f}, {cvScores75th:.2f})")
        report['cv_mean_score'] = cvScoresAvg
        report['cv_25th_score'] = cvScores25th
        report['cv_75th_score'] = cvScores75th

    outFile = os.path.join(outDir, 'report.json')
    with open(outFile, 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nOutput report file written to: {outFile}')


def trainHMM(Y_prob, Y_true, labels=None, uniform_prior=True):
    """
    Implements a Hidden Markov Model as described in https://en.wikipedia.org/wiki/Hidden_Markov_model.

    :param numpy.array Y_prob: Array of predicted probabilities for each class.
    :param numpy.array Y_true: Array of true labels.
    :param list(str) labels: List of class labels.
    :param uniform_prior: If True, all labels have equal probability. If False, label probability equals empirical rate.

    :return: Dictionary containing prior, emission and transition matrices, and corresponding labels.
    :rtype: dict
    """

    if labels is None:
        labels = np.unique(Y_true)

    if uniform_prior:
        # All labels with equal probability
        prior = np.ones(len(labels)) / len(labels)
    else:
        # Label probability equals empirical rate
        prior = np.mean(Y_true.reshape(-1, 1) == labels, axis=0)

    emission = np.vstack(
        [np.mean(Y_prob[Y_true == label], axis=0) for label in labels]
    )
    transition = np.vstack(
        [np.mean(Y_true[1:][(Y_true == label)[:-1]].reshape(-1, 1) == labels, axis=0)
            for label in labels]
    )

    params = {'prior': prior, 'emission': emission, 'transition': transition, 'labels': labels}

    return params


def viterbi(Y_obs, hmm_params):
    """
    Performs Hidden Markov Model (HMM) smoothing over observations using the
    Viterbi algorithm. For more information on the Viterbi algorithm, see:
    https://en.wikipedia.org/wiki/Viterbi_algorithm

    :param dict hmm_params: Dictionary containing prior, emission and transition matrices, and corresponding labels.

    :return: Smoothed sequence of activities.
    :rtype: numpy.array
    """

    def log(x):
        SMALL_NUMBER = 1e-16
        return np.log(x + SMALL_NUMBER)

    prior = hmm_params['prior']
    emission = hmm_params['emission']
    transition = hmm_params['transition']
    labels = hmm_params['labels']

    nobs = len(Y_obs)
    nlabels = len(labels)

    Y_obs = np.where(Y_obs.reshape(-1, 1) == labels)[1]  # to numeric

    probs = np.zeros((nobs, nlabels))
    probs[0, :] = log(prior) + log(emission[:, Y_obs[0]])
    for j in range(1, nobs):
        for i in range(nlabels):
            probs[j, i] = np.max(
                log(emission[i, Y_obs[j]]) +
                log(transition[:, i]) +
                probs[j - 1, :])  # probs already in log scale
    viterbi_path = np.zeros_like(Y_obs)
    viterbi_path[-1] = np.argmax(probs[-1, :])
    for j in reversed(range(nobs - 1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j + 1]]) +
            probs[j, :])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path


def removeSpuriousSleep(Y, activityModel='walmsley', sleepTol='1H'):
    """
    Remove spurious sleep epochs from activity classification.

    :param pandas.Series Y: Model output
    :param str activityModel: Model identifier
    :param str sleepTol: Minimum sleep duration, e.g. '1H'

    :return: Dataframe of revised model output
    :rtype: pandas.DataFrame
    """

    newValue = {
        'willetts': 'sit-stand',
        'doherty': 'sedentary',
        'walmsley': 'sedentary',
        'chan': 'sedentary',
    }[activityModel]

    sleep = Y == 'sleep'
    sleepStreak = (
        sleep.ne(sleep.shift())
        .cumsum()
        .pipe(lambda x: x.groupby(x).transform('count') * sleep)
    )
    sleepTol = pd.Timedelta(sleepTol) / Y.index.freq
    whr = sleep & (sleepStreak < sleepTol)
    Y = Y.copy()  # no modify original
    Y.loc[whr] = newValue

    return Y


def cutPointModel(enmo, cuts=None, whr=None):
    """
    Perform classification of activities based on cutpoints.

    :param pandas.Series enmo: Timeseries of ENMO.
    :param dict cuts: Dictionary of cutpoints for each activity.

    :return: Activity labels.
    :rtype: pandas.Series
    """

    if cuts is None:
        # default cutpoints
        cuts = {'LPA': 45/1000, 'MPA': 100/1000, 'VPA': 400/1000}

    if whr is None:
        whr = pd.Series(True, index=enmo.index)

    Y = pd.DataFrame(index=enmo.index, columns=['CpSB', 'CpLPA', 'CpMPA', 'CpVPA', 'CpMVPA'])

    Y.loc[:, 'CpSB'] = (enmo <= cuts['LPA']) & whr
    Y.loc[:, 'CpLPA'] = (enmo > cuts['LPA']) & (enmo <= cuts['MPA']) & whr
    Y.loc[:, 'CpMPA'] = (enmo > cuts['MPA']) & (enmo <= cuts['VPA']) & whr
    Y.loc[:, 'CpVPA'] = (enmo > cuts['VPA']) & whr
    Y.loc[:, 'CpMVPA'] = (enmo > cuts['MPA']) & whr

    Y.loc[enmo.isna()] = np.nan
    Y = Y.astype('float')

    return Y


def perParticipantSummaryHTML(dfParam, yTrueCol, yPredCol, pidCol, outHTML):
    """
    Provide HTML summary of how well activity classification model works at the per-participant level.

    :param pandas.DataFrame dfParam: Input pandas dataframe
    :param str yTrueCol: Input for y_true column label
    :param str yPredCol: Input for y_pred column label
    :param str pidCol: Input for participant ID column label
    :param str outHTML: Output file to print HTML summary to

    """
    # get kappa & accuracy on a per-participant basis
    pIDs = dfParam[pidCol].unique()
    pIDKappa = []
    pIDAccuracy = []
    for pID in pIDs:
        d_tmp = dfParam[dfParam[pidCol] == pID]
        pIDKappa += [metrics.cohen_kappa_score(d_tmp[yTrueCol], d_tmp[yPredCol])]
        pIDAccuracy += [metrics.accuracy_score(d_tmp[yTrueCol], d_tmp[yPredCol])]
    d_summary = pd.DataFrame()
    d_summary['pid'] = pIDs
    d_summary['kappa'] = pIDKappa
    d_summary['accuracy'] = pIDAccuracy
    # print out values to html string
    kappaSDHTML = "Mean Kappa (SD) = "
    kappaSDHTML += utils.meanSDstr(d_summary['kappa'].mean(),
                                   d_summary['kappa'].std(), 2)
    accuracySDHTML = "Mean accuracy (SD) = "
    accuracySDHTML += utils.meanSDstr(d_summary['accuracy'].mean() * 100,
                                      d_summary['accuracy'].std() * 100, 1) + ' %'
    kappaCIHTML = "Mean Kappa (95% CI) = "
    kappaCIHTML += utils.meanCIstr(d_summary['kappa'].mean(),
                                   d_summary['kappa'].std(), len(d_summary), 2)
    accuracyCIHTML = "Mean accuracy (95% CI) = "
    accuracyCIHTML += utils.meanCIstr(d_summary['accuracy'].mean() * 100,
                                      d_summary['accuracy'].std() * 100, len(d_summary), 1) + ' %'

    # get confusion matrix to pandas dataframe
    y_true = dfParam[yTrueCol]
    y_pred = dfParam[yPredCol]
    labels = sorted(list(set(y_true) | set(y_pred)))
    cnf_matrix = confusion_matrix(y_true, y_pred, labels)
    df_confusion = pd.DataFrame(data=cnf_matrix, columns=labels, index=labels)
    confusionHTML = df_confusion.to_html()

    # construct final output string
    htmlStr = '<html><head><title>Classification summary</title></head><body>'
    htmlStr += kappaSDHTML + '<br>\n' + accuracySDHTML + '<br><br>\n'
    htmlStr += kappaCIHTML + '<br>\n' + accuracyCIHTML + '<br>\n'
    htmlStr += confusionHTML + '<br>\n'
    htmlStr += '</body></html>'

    # write HTML file
    w = open(outHTML, 'w')
    w.write(htmlStr)
    w.close()


def saveToTar(tarOut, **kwargs):
    """
    Save objects to tar file. Objects must be passed as keyworded arguments, then the key is used for the object name in the tar file.

    :param kwargs: Objects to be saved passed as keyworded arguments.

    """
    try:

        tmpdir = tempfile.mkdtemp()

        with tarfile.open(tarOut, mode='w') as tf:

            for key, val in kwargs.items():
                pth = os.path.join(tmpdir, key)
                joblib.dump(val, pth, compress=True)
                tf.add(pth, arcname=key)

        print('Models saved to', tarOut)

    finally:

        try:
            shutil.rmtree(tmpdir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def getFileFromTar(tarArchive, targetFile):
    """
    Read file from tar. This is currently more tricky than it should be. See https://github.com/numpy/numpy/issues/7989

    :param str tarArchive: Input tarfile object
    :param str targetFile: Target individual file within .tar

    :return: file object byte stream
    :rtype: io.BytesIO

    """
    with tarfile.open(tarArchive, 'r') as t:
        b = BytesIO()
        try:
            b.write(t.extractfile(targetFile).read())
        except KeyError:
            return None
        b.seek(0)

    return b


def addReferenceLabelsToNewFeatures(
        featuresFile,
        referenceLabelsFile,
        outputFile,
        featuresTxt="activityModels/features.txt",
        labelCol="label", participantCol="participant",
        annotationCol="annotation", metCol="MET"
):
    """
    Append reference annotations to newly extracted feature data. This method
    helps add existing curated labels (from referenceLabelsFile) to a file with
    newly extracted features (both pre-sorted by participant and time).

    :param str featuresFile: Input csv file of new features data, pre-sorted by time
    :param str referenceLabelsFile: Input csv file of reference labelled data, pre-sorted by time
    :param str outputFile: Output csv file of new features data with reference labels
    :param str featuresTxt: Input txt file listing feature column names
    :param str labelCol: Input label column
    :param str participantCol: Input participant column
    :param str annotationCol: Input text annotation e.g. 'walking with dog' vs. 'walking'
    :param str metCol: Input MET column

    :return: None. Writes a new csv file to <outputFile>.

    .. code-block:: python

        from accelerometer import accClassification
        accClassification.addReferenceLabelsToNewFeatures("newFeats.csv", "refLabels.csv", "newFeatsPlusLabels.csv")
    """

    # load new features file
    featureCols = np.loadtxt(featuresTxt, dtype='str')
    dFeat = pd.read_csv(featuresFile, usecols=featureCols + [participantCol, 'time'])

    # load in reference annotations file
    refCols = [participantCol, 'age', 'sex', 'time', annotationCol, labelCol,
               'code', metCol, 'MET_label']
    dRef = pd.read_csv(referenceLabelsFile, usecols=refCols)

    # join dataframes
    indexCols = [participantCol, 'time']
    dOut = dRef.set_index(indexCols).join(dFeat.set_index(indexCols), how='left')

    # write out new labelled features file
    dOut.to_csv(outputFile, index=True)
    print('New file written to: ', outputFile)


def resolveModelPath(pathOrModelName):

    if pathlib.Path(pathOrModelName).exists():
        return pathOrModelName

    else:

        # versions before January 2022 no longer supported
        if pathOrModelName in (
            'walmsley-jan21', 'doherty-jan21', 'willetts-jan21',
            'walmsley-may20', 'doherty-may20', 'willetts-may20'
        ):
            pathOrModelName = pathOrModelName.split("-")[0]
            warnings.warn(
                f"Activity model versions before January 2022 are no longer supported. "
                f"Defaulting to --activityModel {pathOrModelName}"
            )

        model = MODELS.get(pathOrModelName, None)
        if model is None:
            raise FileNotFoundError(f"Model file {pathOrModelName} not found")
        if model["pth"].exists():
            return model["pth"]
        else:
            return downloadModel(model)


def downloadModel(model):
    url = model["url"]
    pth = model["pth"]

    os.makedirs(os.path.dirname(pth), exist_ok=True)

    print(f"Downloading {url}...")

    with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
        shutil.copyfileobj(f_src, f_dst)

    return pth
