"""Module to support machine learning of activity states from acc data"""

from io import BytesIO
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.ensemble.forest as forest
from sklearn.externals import joblib
import tarfile
from time import sleep
import warnings


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
    featureCols = getListFromTxtFile(getFileFromTar(activityModel, 'featureCols.txt'))

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



MIN_TRAIN_CLASS_COUNT = 100
def trainClassificationModel(trainingFile, tarArchive, labelCol="label", 
    participantCol="participant", atomicLabelCol="annotation", metCol="MET",
    testParticipants=None, testMatrix="activityModels/confusionMatrix.txt", 
    rfThreads=1, rfTrees = 1000):
    """Train model to classify activity states from epoch feature data

    Based on a balanced random forest with a Hidden Markov Model containing 
    transitions between predicted activity states and emissions trained using 
    the input training file to identify pre-defined classes of behaviour from 
    accelerometer data.

    :param str trainingFile: Input csv file of training data
    :param str tarArchive: Output tarfile object which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METS for each activity state
    :param str labelCol: Input label column
    :param str participantCol: Input participant column
    :param str atomicLabelCol: Input 'atomic' annotation e.g. 'walking with dog'
        vs. 'walking'
    :param str metCol: Input MET column
    :param str testParticipants: Input comma separated list of participant IDs 
        to test on. Will only output trained model if this is null (i.e. train 
        on all possible data)
    :param int rfThreads: Input num threads to use when training random forest
    :param int rfTrees: Input num decision trees to include in random forest
    :param str testMatrix: Output confusion matrix (on test participants)

    :return: New model written to <tarArchive> OR confusion matrix in test
        dataset written to <testMatrix>
    :rtype: void
    """

    # load list of features to use in analysis
    featureCols = getListFromTxtFile("activityModels/features.txt")

    #load in participant information, and remove null/messy labels/features
    train = pd.read_csv(trainingFile)
    train = train[~pd.isnull(train[labelCol])]
    allCols = [participantCol, labelCol, atomicLabelCol, metCol] + featureCols
    train = train[allCols].dropna(axis=0, how='any')

    # reduce size of training set if we are testing on some participants
    if testParticipants is not None:
        testPIDs = testParticipants.split(',')
        test = train[train[participantCol].isin(testPIDs)]
        train = train[~train[participantCol].isin(testPIDs)]

    #train Random Forest model
    # first "monkeypatch" RF function to perform per-class balancing
    MIN_TRAIN_CLASS_COUNT = train[labelCol].value_counts().min()
    forest._parallel_build_trees = _parallel_build_trees
    # then train RF model (which include per-class balancing)
    rfClassifier = RandomForestClassifier(n_estimators = rfTrees,
                                            n_jobs = rfThreads, oob_score=True)
    rfModel = rfClassifier.fit(train[featureCols], train[labelCol].tolist())
    
    # train Hidden Markov Model
    states, priors, emissions, transitions = train_HMM(rfModel, train[labelCol],
                                                        labelCol)
    rfModel.oob_decision_function_ = None # out of bound errors are no longer needed

    # estimate usual METs-per-class
    mets = []
    for s in states:
        met = train[train[labelCol]==s].groupby(atomicLabelCol)[metCol].mean().mean()
        mets += [met]

    # now write out model (or it's performance on test participants)
    if testParticipants is None:
        saveModelsToTar(tarArchive, featureCols, rfModel, priors, transitions, 
            emissions, mets)
    else:
        print('test on participant(s):, ', testParticipants)
        labels = rfModel.classes_.tolist()
        rfPredictions = rfModel.predict(test[featureCols])
        hmmPredictions = viterbi(rfPredictions.tolist(), labels, priors, 
            transitions, emissions)
        test['prediction'] = hmmPredictions
        w = open(testMatrix, 'w')
        w.write("y_true," + ",".join(states))
        for i in states:
            w.write('\n' + i)
            for j in states:
                n = len(test[(test[labelCol]==i) & (test['prediction']==j)])
                w.write(',' + str(n))
        w.close()
        print('confusion matrix written to: ', testMatrix)



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
    
    # get out of bag (OOB) predictions from Random Forest
    predOOB = pd.DataFrame(rfModel.oob_decision_function_)
    predOOB.columns = states
    predOOB['labelOOB'] = predOOB.idxmax(axis=1)
    predOOB['groundTruth'] = y_trainF.values
    
    # initial state probabilities
    prior = []
    for s in states:
        sProb = len(y_trainF[y_trainF == s]) / (len(y_trainF) * 1.0)
        prior += [sProb]
        
    # emission probabilities
    emissions = np.zeros((len(states), len(states)))
    j = 0
    for predictedState in states:
        k=0
        for actualState in states:
            emissions[j,k] = predOOB[actualState][predOOB['groundTruth'] == predictedState].sum()
            emissions[j,k] /= len(predOOB[predOOB['groundTruth'] == predictedState])
            k+=1
        j+=1
        
    # transition probabilities
    train = y_trainF.to_frame()
    train['nextLabel'] = train[labelCol].shift(-1)
    transitions = np.zeros((len(states), len(states)))
    j = 0
    for s1 in states:
        k = 0
        for s2 in states:
            transitions[j, k] = len(train[(train[labelCol]==s1) & (train['nextLabel']==s2)]) / ( len(train[train[labelCol]==s1]) * 1.0)
            k += 1
        j += 1
    
    # return HMM matrices
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



GLOBAL_INDICES = []
def _parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                          verbose=0, class_weight=None):
    """Monkeypatch scikit learn to use per-class balancing

    Private function used to fit a single tree in parallel.
    """

    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    indices = np.empty(shape=0, dtype='int64')
    for y_class in np.unique(y):
        sample_indices, selected = np.where(y==y_class)
        ## SELECT min_count FROM CLASS WITH REPLACEMENT
        sample_indices = np.random.choice(sample_indices, 
            size=MIN_TRAIN_CLASS_COUNT, replace=True)
        indices = np.concatenate((indices,sample_indices))
    ## IGNORE sample_weight AND SIMPLY PASS SELECTED DATA
    tree.fit(X[indices,:], y[indices], check_input=True)
    GLOBAL_INDICES.append(indices)
    return tree



def summary(y_true, y_pred):
    """Provide summary of how well activity classification model works

    :param list y_true: Input list of groundtruth labels
    :param list y_pred: Input list of predicted labels
    
    :return: Print out of kappa + accuracy statistics
    :rtype: void
    """

    print('kappa = ', metrics.cohen_kappa_score(y_true, y_pred))
    print('accuracy = ', metrics.accuracy_score(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred))



def saveModelsToTar(tarArchive, featureCols, rfModel, priors, transitions, 
    emissions, mets, featuresTxt="featureCols.txt", rfModelFile="rfModel.pkl",
    hmmPriors="hmmPriors.npy", hmmEmissions="hmmEmissions.npy", 
    hmmTransitions="hmmTransitions.npy", hmmMETs="METS.npy"):
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
    :param numpy.array mets: Input array of average mets per activity state
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
    np.save(hmmMETs, mets)
    joblib.dump(rfModel, rfModelFile, compress=9)

    # create single .tar file...
    tarOut = tarfile.open(tarArchive, mode='w')
    tarOut.add(featuresTxt)
    tarOut.add(hmmPriors)
    tarOut.add(hmmEmissions)
    tarOut.add(hmmTransitions)
    tarOut.add(hmmMETs)
    tarOut.add(rfModelFile)
    tarOut.close()

    # remove intermediate files
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
    f = open(inputFile,'rU')
    for l in f:
        items.append(l.strip())
    f.close()
    return items