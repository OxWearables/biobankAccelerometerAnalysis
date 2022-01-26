"""Module to support machine learning of activity states from acc data"""

from accelerometer import utils
from accelerometer.models import MODELS
from io import BytesIO
import tempfile
import numpy as np
import os
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import joblib
import tarfile
import urllib
import pathlib
import shutil
import warnings


def activityClassification(epoch, activityModel="walmsley"):
    """Perform classification of activity states from epoch feature data

    Based on a balanced random forest with a Hidden Markov Model containing
    transitions between predicted activity states and emissions trained using a
    free-living groundtruth to identify pre-defined classes of behaviour from
    accelerometer data.

    :param str epoch: Dataframe of processed epoch data
    :param str activityModel: Input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METs for each activity state

    :return: Pandas dataframe of activity epoch data with one-hot encoded labels
    :rtype: pandas.DataFrame

    :return: Activity state labels
    :rtype: list(str)
    """

    activityModel = resolveModelPath(activityModel)

    featureCols = joblib.load(getFileFromTar(activityModel, 'featureCols'))

    X = epoch[featureCols].to_numpy()
    mask = np.isfinite(X).any(axis=1)
    X = X[mask]
    print(f"{len(epoch) - np.sum(mask)} rows with NaN or Inf values, out of {len(epoch)}")

    model = joblib.load(getFileFromTar(activityModel, 'model'))
    hmmParams = joblib.load(getFileFromTar(activityModel, 'hmmParams'))
    Y = viterbi(model.predict(X), hmmParams)

    # Append predicted activities to epoch dataframe
    epoch["label"] = np.nan
    epoch.loc[mask, "label"] = Y

    # MET prediction
    METs = joblib.load(getFileFromTar(activityModel, 'METs'))
    if METs is not None:
        epoch["MET"] = epoch["label"].replace(METs)

    labels = joblib.load(getFileFromTar(activityModel, 'labels')).tolist()

    # One-hot encoding
    for lab in labels:
        epoch[lab] = 0
        epoch.loc[epoch['label'] == lab, lab] = 1
    # Null values aren't one-hot encoded, so set such instances to NaN
    for lab in labels:
        epoch.loc[epoch[labels].sum(axis=1) == 0, lab] = np.nan

    return epoch, labels


def trainClassificationModel(
        trainingFile,
        labelCol="label", participantCol="participant",
        annotationCol="annotation", metCol="MET",
        featuresTxt="activityModels/features.txt",
        nTrees=1000, nJobs=1,
        trainParticipants=None, testParticipants=None,
        outputPredict="predictions.csv", outputModel=None
):
    """Train model to classify activity states from epoch feature data

    Based on a balanced random forest with a Hidden Markov Model containing
    transitions between predicted activity states and emissions trained using
    the input training file to identify pre-defined classes of behaviour from
    accelerometer data.

    :param str trainingFile: Input csv file of training data, pre-sorted by time
    :param str labelCol: Input label column
    :param str participantCol: Input participant column
    :param str annotationCol: Input text annotation e.g. 'walking with dog'
        vs. 'walking'
    :param str metCol: Input MET column
    :param str featuresTxt: Input txt file listing feature column names
    :param str trainParticipants: Input comma separated list of participant IDs
        to train on.
    :param str testParticipants: Input comma separated list of participant IDs
        to test on.
    :param int nJobs: Input num threads to use when training random forest
    :param int nTrees: Input num decision trees to include in random forest
    :param str outputPredict: Output CSV of person, label, predicted
    :param str outputModel: Output tarfile object which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METs for each activity state. Will only output trained model if this
        is not null e.g. "activityModels/sample-model.tar"

    :return: New model written to <outputModel> OR csv of test predictions
        written to <outputPredict>
    :rtype: void
    """

    # Load list of features to use in analysis
    featureCols = np.loadtxt(featuresTxt, dtype='str')

    # Load in participant information, and remove null/messy labels/features
    allCols = [participantCol, labelCol, annotationCol] + featureCols.tolist()
    if metCol:
        allCols.append(metCol)
    train = pd.read_csv(trainingFile, usecols=allCols)
    with pd.option_context('mode.use_inf_as_null', True):
        train = train.dropna(axis=0, how='any')

    # Reduce size of train/test sets if we are training/testing on some people
    if testParticipants is not None:
        testPIDs = testParticipants.split(',')
        test = train[train[participantCol].isin(testPIDs)]
        train = train[~train[participantCol].isin(testPIDs)]
    if trainParticipants is not None:
        trainPIDs = trainParticipants.split(',')
        train = train[train[participantCol].isin(trainPIDs)]

    X, Y = train[featureCols].to_numpy(), train[labelCol].to_numpy()

    model = BalancedRandomForestClassifier(
        n_estimators=nTrees,
        replacement=True,
        sampling_strategy='not minority',
        n_jobs=nJobs,
        oob_score=True,
        verbose=True,
        random_state=42)

    model = model.fit(X, Y)
    labels = model.classes_

    # Train Hidden Markov Model
    hmmParams = trainHMM(model.oob_decision_function_, Y, labels)

    model.oob_decision_function_ = None  # out-of-bag predictions no longer needed
    model.verbose = False  # silence future calls to .predict()

    # Estimate METs via per-class averaging
    METs = None
    if metCol:
        METs = {y: train[Y == y].groupby(annotationCol)[metCol].mean().mean()
                for y in model.classes_}

    # Now write out model
    if outputModel is not None:
        saveToTar(outputModel,
                  model=model,
                  labels=labels,
                  featureCols=featureCols,
                  hmmParams=hmmParams,
                  METs=METs)

    # Assess model performance on test participants
    if testParticipants is not None:
        print('Test on participant(s):', testParticipants)
        X_test, Y_test = test[featureCols].to_numpy(), test[labelCol].to_numpy()
        Y_pred = model.predict(X_test)
        Y_pred_hmm = viterbi(Y_pred, hmmParams)
        test['predicted'] = Y_pred_hmm
        # And write out to file
        outCols = [participantCol, labelCol, 'predicted']
        test[outCols].to_csv(outputPredict, index=False)
        print('Output predictions written to: ', outputPredict)
        print(metrics.classification_report(Y_test, Y_pred_hmm))


def trainHMM(Y_prob, Y_true, labels=None, uniform_prior=True):
    """ https://en.wikipedia.org/wiki/Hidden_Markov_model

    :return: Dictionary containing prior, emission and transition
        matrices, and corresponding labels.
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
    """ Perform HMM smoothing over observations via Viteri algorithm

    https://en.wikipedia.org/wiki/Viterbi_algorithm

    :param dict hmm_params: Dictionary containing prior, emission and transition
        matrices, and corresponding labels

    :return: Smoothed sequence of activities
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


def perParticipantSummaryHTML(dfParam, yTrueCol, yPredCol, pidCol, outHTML):
    """Provide HTML summary of how well activity classification model works
    at the per-participant level

    :param dataframe dfParam: Input pandas dataframe
    :param str yTrueCol: Input for y_true column label
    :param str yPregCol: Input for y_pred column label
    :param str pidCol: Input for participant ID column label
    :param str outHTML: Output file to print HTML summary to

    :return: HTML file reporting kappa, accuracy, and confusion matrix
    :rtype: void
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
    """Save objects to tar file. Objects must be passed as keyworded arguments,
    then the key is used for the object name in the tar file.

    :param **kwargs: Objects to be saved passed as keyworded arguments.

    :return: tar file written to <tarOut>
    :rtype: void
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
    """Read file from tar

    This is currently more tricky than it should be see
    https://github.com/numpy/numpy/issues/7989

    :param str tarArchive: Input tarfile object
    :param str targetFile: Target individual file within .tar

    :return: file object byte stream
    :rtype: object
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
        annotationCol="annotation", metCol="MET"):
    """Append reference annotations to newly extracted feature data

    This method helps add existing curated labels (from referenceLabelsFile)
    to a file with newly extracted features (both pre-sorted by participant
    and time).

    :param str featuresFile: Input csv file of new features data, pre-sorted by time
    :param str referenceLabelsFile: Input csv file of reference labelled data,
        pre-sorted by time
    :param str outputFile: Output csv file of new features data with refernce labels
    :param str featuresTxt: Input txt file listing feature column names
    :param str labelCol: Input label column
    :param str participantCol: Input participant column
    :param str annotationCol: Input text annotation e.g. 'walking with dog'
        vs. 'walking'
    :param str metCol: Input MET column

    :return: New csv file written to <outputFile>
    :rtype: void

    :Example:
    >>> from accelerometer import accClassification
    >>> accClassification.addReferenceLabelsToNewFeatures("newFeats.csv",
            "refLabels.csv", "newFeatsPlusLabels.csv")
    <file written to newFeatsPlusLabels.csv>
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

    print(f"Downloading {url}...")

    with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
        shutil.copyfileobj(f_src, f_dst)

    return pth
