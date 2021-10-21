"""Module to support machine learning of activity states from acc data"""

from accelerometer import accUtils
from accelerometer.models import MODELS
from io import BytesIO
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble._forest as forest
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import joblib
import tarfile
import warnings
import urllib
import pathlib
import shutil


def activityClassification(epochFile, activityModel="walmsley"):
    """Perform classification of activity states from epoch feature data

    Based on a balanced random forest with a Hidden Markov Model containing
    transitions between predicted activity states and emissions trained using a
    free-living groundtruth to identify pre-defined classes of behaviour from
    accelerometer data.

    :param str epochFile: Input csv file of processed epoch data
    :param str activityModel: Input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METs for each activity state

    :return: Pandas dataframe of activity epoch data with one-hot encoded labels
    :rtype: pandas.DataFrame

    :return: Activity state labels
    :rtype: list(str)
    """

    activityModel = resolveModelPath(activityModel)

    X = epochFile
    featureColsFile = getFileFromTar(activityModel, 'featureCols.txt').getvalue()
    featureColsList = featureColsFile.decode().split('\n')
    featureCols = list(filter(None, featureColsList))

    with pd.option_context('mode.use_inf_as_null', True):
        null_rows = X[featureCols].isnull().any(axis=1)
    print(null_rows.sum(), "rows with missing (NaN, None, or NaT) or Inf values, out of", len(X))

    X['label'] = 'none'
    X.loc[null_rows, 'label'] = 'inf_or_null'
    # Setup RF
    # Ignore warnings on deployed model using different version of pandas
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rf = joblib.load(getFileFromTar(activityModel, 'rfModel.pkl'))
    labels = rf.classes_.tolist()
    rfPredictions = rf.predict(X.loc[~null_rows, featureCols].to_numpy())
    # Free memory
    del rf
    # Setup HMM
    priors = np.load(getFileFromTar(activityModel, 'hmmPriors.npy'))
    transitions = np.load(getFileFromTar(activityModel, 'hmmTransitions.npy'))
    emissions = np.load(getFileFromTar(activityModel, 'hmmEmissions.npy'))
    hmmPredictions = viterbi(rfPredictions.tolist(), labels, priors,
                             transitions, emissions)
    # Save predictions to pandas dataframe
    X.loc[~null_rows, 'label'] = hmmPredictions

    # Perform MET prediction...
    # Pandas .replace method has a small bug
    # See https://github.com/pandas-dev/pandas/issues/23305
    # We need to force type
    met_vals = np.load(getFileFromTar(activityModel, 'METs.npy'))
    met_dict = dict(zip(labels, met_vals))
    X.loc[~null_rows, 'MET'] = X.loc[~null_rows, 'label'].replace(met_dict).astype('float')

    # Apply one-hot encoding
    for l in labels:
        X[l] = 0
        X.loc[X['label'] == l, l] = 1
    # Null values aren't one-hot encoded, so set such instances to NaN
    for l in labels:
        X.loc[X[labels].sum(axis=1) == 0, l] = np.nan
    return X, labels


MIN_TRAIN_CLASS_COUNT = 100


def trainClassificationModel(
        trainingFile,
        labelCol="label", participantCol="participant",
        atomicLabelCol="annotation", metCol="MET",
        featuresTxt="activityModels/features.txt",
        trainParticipants=None, testParticipants=None,
        rfThreads=1, rfTrees=1000, rfFeats=None, rfDepth=None,
        outputPredict="activityModels/test-predictions.csv",
        outputModel=None
):
    """Train model to classify activity states from epoch feature data

    Based on a balanced random forest with a Hidden Markov Model containing
    transitions between predicted activity states and emissions trained using
    the input training file to identify pre-defined classes of behaviour from
    accelerometer data.

    :param str trainingFile: Input csv file of training data, pre-sorted by time
    :param str labelCol: Input label column
    :param str participantCol: Input participant column
    :param str atomicLabelCol: Input 'atomic' annotation e.g. 'walking with dog'
        vs. 'walking'
    :param str metCol: Input MET column
    :param str featuresTxt: Input txt file listing feature column names
    :param str trainParticipants: Input comma separated list of participant IDs
        to train on.
    :param str testParticipants: Input comma separated list of participant IDs
        to test on.
    :param int rfThreads: Input num threads to use when training random forest
    :param int rfTrees: Input num decision trees to include in random forest
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
    featureCols = getListFromTxtFile(featuresTxt)

    # Load in participant information, and remove null/messy labels/features
    train = pd.read_csv(trainingFile)
    train = train[~pd.isnull(train[labelCol])]
    allCols = [participantCol, labelCol, atomicLabelCol, metCol] + featureCols
    with pd.option_context('mode.use_inf_as_null', True):
        train = train[allCols].dropna(axis=0, how='any')

    # Reduce size of train/test sets if we are training/testing on some people
    if testParticipants is not None:
        testPIDs = testParticipants.split(',')
        test = train[train[participantCol].isin(testPIDs)]
        train = train[~train[participantCol].isin(testPIDs)]
    if trainParticipants is not None:
        trainPIDs = trainParticipants.split(',')
        train = train[train[participantCol].isin(trainPIDs)]

    # Train Random Forest model
    # First "monkeypatch" RF function to perform per-class balancing
    global MIN_TRAIN_CLASS_COUNT
    MIN_TRAIN_CLASS_COUNT = train[labelCol].value_counts().min()
    forest._parallel_build_trees = _parallel_build_trees
    # Then train RF model (which include per-class balancing)
    rfClassifier = RandomForestClassifier(n_estimators=rfTrees,
                                          n_jobs=rfThreads,
                                          max_features=rfFeats,
                                          max_depth=rfDepth,
                                          oob_score=True)

    rfModel = rfClassifier.fit(train[featureCols], train[labelCol].tolist())

    # Train Hidden Markov Model
    states, priors, emissions, transitions = train_HMM(rfModel, train[labelCol], labelCol)
    rfModel.oob_decision_function_ = None  # out of bound errors are no longer needed

    # Estimate usual METs-per-class
    METs = []
    for s in states:
        MET = train[train[labelCol] == s].groupby(atomicLabelCol)[metCol].mean().mean()
        METs += [MET]

    # Now write out model
    if outputModel is not None:
        saveModelsToTar(outputModel, featureCols, rfModel, priors, transitions, emissions, METs)

    # Assess model performance on test participants
    if testParticipants is not None:
        print('test on participant(s):, ', testParticipants)
        labels = rfModel.classes_.tolist()
        rfPredictions = rfModel.predict(test[featureCols])
        hmmPredictions = viterbi(rfPredictions.tolist(), labels, priors,
                                 transitions, emissions)
        test['predicted'] = hmmPredictions
        # And write out to file
        outCols = [participantCol, labelCol, 'predicted']
        test[outCols].to_csv(outputPredict, index=False)
        print('Output predictions written to: ', outputPredict)


def train_HMM(rfModel, y_trainF, labelCol):
    """Train Hidden Markov Model

    Use data not considered in construction of random forest to estimate
    probabilities of: i) starting in a given state; ii) transitioning from
    one state to another; and iii) probabilitiy of the random forest being
    correct when predicting a given class (emission probability)

    :param sklearn.RandomForestClassifier rfModel: Input random forest object
    :param dataframe.Column y_trainF: Input groundtruth for each intance
    :param str labelCol: Input label column

    :return: states - List of unique activity state labels
    rtype: numpy.array

    :return: priors - Prior probabilities for each activity state
    rtype: numpy.array

    :return: transitions - Probability matrix of transitioning from one activity
        state to another
    rtype: numpy.array

    :return: emissions - Probability matrix of RF prediction being true
    rtype: numpy.array
    """

    states = rfModel.classes_

    # Get out of bag (OOB) predictions from Random Forest
    predOOB = pd.DataFrame(rfModel.oob_decision_function_)
    predOOB.columns = states
    predOOB['labelOOB'] = predOOB.idxmax(axis=1)
    predOOB['groundTruth'] = y_trainF.values

    # Initial state probabilities
    prior = []
    for s in states:
        sProb = len(y_trainF[y_trainF == s]) / (len(y_trainF) * 1.0)
        prior += [sProb]

    # Emission probabilities
    emissions = np.zeros((len(states), len(states)))
    j = 0
    for predictedState in states:
        k = 0
        for actualState in states:
            emissions[j, k] = predOOB[actualState][predOOB['groundTruth'] == predictedState].sum()
            emissions[j, k] /= len(predOOB[predOOB['groundTruth'] == predictedState])
            k += 1
        j += 1

    # Transition probabilities
    train = y_trainF.to_frame()
    train['nextLabel'] = train[labelCol].shift(-1)
    transitions = np.zeros((len(states), len(states)))
    j = 0
    for s1 in states:
        k = 0
        for s2 in states:
            transitions[j, k] = len(train[(train[labelCol] == s1) & (train['nextLabel'] == s2)]
                                    ) / (len(train[train[labelCol] == s1]) * 1.0)
            k += 1
        j += 1

    # Return HMM matrices
    return states, prior, emissions, transitions


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

    def norm(x):
        return x / x.sum()

    tinyNum = 0.000001
    nObservations = len(observations)
    nStates = len(states)
    v = np.zeros((nObservations, nStates))  # initialise viterbi table
    # Set prior state values for first observation...
    for state in range(0, len(states)):
        v[0, state] = np.log(priors[state] * emissions[state, states.index(observations[0])] + tinyNum)
    # Fill in remaning matrix observations
    # Use log space as multiplying successively smaller p values)
    for k in range(1, nObservations):
        for state in range(0, len(states)):
            v[k, state] = np.log(emissions[state, states.index(observations[k])] + tinyNum) + \
                np.max(v[k - 1, :] + np.log(transitions[:, state] + tinyNum), axis=0)

    # Now construct viterbiPath (propagating backwards)
    viterbiPath = observations
    # Pick most probable state for final observation
    viterbiPath[nObservations - 1] = states[np.argmax(v[nObservations - 1, :], axis=0)]

    # Probabilistic method will give probability of each label
    if probabilistic:
        viterbiProba = np.zeros((nObservations, nStates))  # initialize table
        viterbiProba[nObservations - 1, :] = norm(v[nObservations - 1, :])

    # And then work backwards to pick most probable state for all other observations
    for k in list(reversed(range(0, nObservations - 1))):
        viterbiPath[k] = states[np.argmax(
            v[k, :] + np.log(transitions[:, states.index(viterbiPath[k + 1])] + tinyNum), axis=0)]
        if probabilistic:
            viterbiProba[k, :] = norm(v[k, :] + np.log(transitions[:, states.index(viterbiPath[k + 1])] + tinyNum))

    # Output as list...
    return viterbiProba if probabilistic else viterbiPath


GLOBAL_INDICES = []


def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None, n_samples_bootstrap=None):
    """Monkeypatch scikit learn to use per-class balancing

    Private function used to fit a single tree in parallel.
    """

    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    indices = np.empty(shape=0, dtype='int64')
    for y_class in np.unique(y):
        sample_indices, selected = np.where(y == y_class)
        # SELECT min_count FROM CLASS WITH REPLACEMENT
        sample_indices = np.random.choice(sample_indices,
                                          size=MIN_TRAIN_CLASS_COUNT, replace=True)
        indices = np.concatenate((indices, sample_indices))
    # IGNORE sample_weight AND SIMPLY PASS SELECTED DATA
    tree.fit(X[indices, :], y[indices], check_input=True)
    GLOBAL_INDICES.append(indices)
    return tree


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
    kappaSDHTML += accUtils.meanSDstr(d_summary['kappa'].mean(),
                                      d_summary['kappa'].std(), 2)
    accuracySDHTML = "Mean accuracy (SD) = "
    accuracySDHTML += accUtils.meanSDstr(d_summary['accuracy'].mean() * 100,
                                         d_summary['accuracy'].std() * 100, 1) + ' %'
    kappaCIHTML = "Mean Kappa (95% CI) = "
    kappaCIHTML += accUtils.meanCIstr(d_summary['kappa'].mean(),
                                      d_summary['kappa'].std(), len(d_summary), 2)
    accuracyCIHTML = "Mean accuracy (95% CI) = "
    accuracyCIHTML += accUtils.meanCIstr(d_summary['accuracy'].mean() * 100,
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


def saveModelsToTar(tarArchive, featureCols, rfModel, priors, transitions,
                    emissions, METs, featuresTxt="featureCols.txt", rfModelFile="rfModel.pkl",
                    hmmPriors="hmmPriors.npy", hmmEmissions="hmmEmissions.npy",
                    hmmTransitions="hmmTransitions.npy", hmmMETs="METs.npy"):
    """Save random forest and hidden markov models to tarArchive file

    Note we must use the same version of python and scikit learn as in the
    intended deployment environment

    :param str tarArchive: Output tarfile
    :param list featureCols: Input list of feature columns
    :param sklearn.RandomForestClassifier rfModel: Input random forest model
    :param numpy.array priors: Input prior probabilities for each activity state
    :param numpy.array transitions: Input probability matrix of transitioning
        from one activity state to another
    :param numpy.array emissions: Input probability matrix of RF prediction
        being true
    :param numpy.array METs: Input array of average METs per activity state
    :param str featuresTxt: Intermediate output txt file of features
    :param str rfModelFile: Intermediate output random forest pickle model
    :param str hmmPriors: Intermediate output HMM priors npy
    :param str hmmEmissions: Intermediate output HMM emissions npy
    :param str hmmTransitions: Intermediate output HMM transitions npy
    :param str hmmMETs: Intermediate output HMM METs npy

    :return: tar file of RF + HMM written to tarArchive
    :rtype: void
    """

    wristListToTxtFile(featureCols, featuresTxt)
    np.save(hmmPriors, priors)
    np.save(hmmEmissions, emissions)
    np.save(hmmTransitions, transitions)
    np.save(hmmMETs, METs)
    joblib.dump(rfModel, rfModelFile, compress=9)

    # Create single .tar file...
    tarOut = tarfile.open(tarArchive, mode='w')
    tarOut.add(featuresTxt)
    tarOut.add(hmmPriors)
    tarOut.add(hmmEmissions)
    tarOut.add(hmmTransitions)
    tarOut.add(hmmMETs)
    tarOut.add(rfModelFile)
    tarOut.close()

    # Remove intermediate files
    os.remove(featuresTxt)
    os.remove(hmmPriors)
    os.remove(hmmEmissions)
    os.remove(hmmTransitions)
    os.remove(hmmMETs)
    os.remove(rfModelFile)
    print('Models saved to', tarArchive)


def getFileFromTar(tarArchive, targetFile):
    """Read file from tar

    This is currently more tricky than it should be see
    https://github.com/numpy/numpy/issues/7989

    :param str tarArchive: Input tarfile object
    :param str targetFile: Target individual file within .tar

    :return: file object byte stream
    :rtype: object
    """

    t = tarfile.open(tarArchive, 'r')
    array_file = BytesIO()
    array_file.write(t.extractfile(targetFile).read())
    array_file.seek(0)
    return array_file


def addReferenceLabelsToNewFeatures(
        featuresFile,
        referenceLabelsFile,
        outputFile,
        featuresTxt="activityModels/features.txt",
        labelCol="label", participantCol="participant",
        atomicLabelCol="annotation", metCol="MET"):
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
    :param str atomicLabelCol: Input 'atomic' annotation e.g. 'walking with dog'
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
    featureCols = getListFromTxtFile(featuresTxt)
    dFeat = pd.read_csv(featuresFile, usecols=featureCols + [participantCol, 'time'])

    # load in reference annotations file
    refCols = [participantCol, 'age', 'sex', 'time', atomicLabelCol, labelCol,
               'code', metCol, 'MET_label']
    dRef = pd.read_csv(referenceLabelsFile, usecols=refCols)

    # join dataframes
    indexCols = [participantCol, 'time']
    dOut = dRef.set_index(indexCols).join(dFeat.set_index(indexCols), how='left')

    # write out new labelled features file
    dOut.to_csv(outputFile, index=True)
    print('New file written to: ', outputFile)


def wristListToTxtFile(inputList, outputFile):
    """Write list of items to txt file

    :param list inputList: input list
    :param str outputFile: Output txt file

    :return: list of feature columns
    :rtype: void
    """

    f = open(outputFile, 'w')
    for item in inputList:
        f.write(item + '\n')
    f.close()


def getListFromTxtFile(inputFile):
    """Read list of items from txt file and return as list

    :param str inputFile: Input file listing items

    :return: list of items
    :rtype: list
    """

    items = []
    f = open(inputFile, 'r')
    for l in f:
        items.append(l.strip())
    f.close()
    return items


def resolveModelPath(pathOrModelName):

    if pathlib.Path(pathOrModelName).exists():
        return pathOrModelName

    else:
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
