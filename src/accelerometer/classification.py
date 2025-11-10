"""Module to support machine learning of activity states from acc data"""

from accelerometer import utils
from accelerometer.exceptions import ClassificationError
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


def activity_classification(
    epoch,
    activity_model: str = "walmsley",
    mg_cp_lpa: int = 45,
    mg_cp_mpa: int = 100,
    mg_cp_vpa: int = 400,
    spurious_sleep_removal: bool = True,
    spurious_sleep_tol: int = 60,  # in minutes
):
    """
    Perform classification of activity states from epoch feature data. Based on
    a balanced random forest with a Hidden Markov Model containing transitions
    between predicted activity states and emissions trained using a free-living
    groundtruth to identify pre-defined classes of behaviour from accelerometer
    data.

    :param pandas.DataFrame epoch: Dataframe of processed epoch data
    :param str activity_model: Path to input tar model file which contains random forest
        pickle model, HMM priors/transitions/emissions npy files, and npy file
        of METs for each activity state

    :return: Tuple containing a pandas dataframe of activity epoch data with one-hot encoded labels, and a list of activity state labels
    :rtype: tuple(pandas.DataFrame, list(str))
    """

    model_path = resolve_model_path(activity_model)

    feature_cols = joblib.load(get_file_from_tar(model_path, 'featureCols'))
    model = joblib.load(get_file_from_tar(model_path, 'model'))
    hmm_params = joblib.load(get_file_from_tar(model_path, 'hmmParams'))
    labels = joblib.load(get_file_from_tar(model_path, 'labels')).tolist()

    features = epoch[feature_cols].to_numpy()
    ok = np.isfinite(features).all(axis=1)
    print(f"{len(epoch) - np.sum(ok)} rows with NaN or Inf values, out of {len(epoch)}")

    predicted_labels = pd.Series(index=epoch.index)
    if ok.any():
        predicted_labels.loc[ok] = viterbi(model.predict(features[ok]), hmm_params)

    # TODO: Chan's logic hardcoded here
    if activity_model == 'chan':
        enmo = epoch['enmoTrunc'].to_numpy()
        other = (predicted_labels == 'other')
        predicted_labels.loc[other & (enmo < 100 / 1000)] = 'light'
        predicted_labels.loc[other & (enmo >= 100 / 1000)] = 'moderate'
        predicted_labels.loc[other & (enmo > 400 / 1000)] = 'vigorous'
        labels.remove('other')
        labels.append('light')
        labels.append('moderate')
        labels.append('vigorous')
        del enmo
        del other

    if spurious_sleep_removal:
        predicted_labels = remove_spurious_sleep(predicted_labels, activity_model=activity_model, sleep_tol=spurious_sleep_tol)

    # One-hot encoding
    epoch.loc[ok, labels] = (predicted_labels[ok].to_numpy()[:, None] == labels).astype('float')

    # MET prediction
    mets = joblib.load(get_file_from_tar(model_path, 'METs'))
    if mets is not None:
        epoch.loc[:, "MET"] = predicted_labels.replace(mets)

    # Cut-point based classification on non-sleep epochs
    y_cp_one_hot = cut_point_model(
        epoch['enmoTrunc'],
        cuts={'LPA': mg_cp_lpa / 1000, 'MPA': mg_cp_mpa / 1000, 'VPA': mg_cp_vpa / 1000},
        whr=~(predicted_labels == 'sleep')  # Note: ~(predicted_labels == 'sleep') != (predicted_labels != 'sleep') because of NaNs
    )
    epoch = epoch.join(y_cp_one_hot)
    labels_cp = list(y_cp_one_hot.columns)
    labels.extend(labels_cp)

    return epoch, labels


def train_classification_model(
    training_file,
    label_col="label", participant_col="participant",
    annotation_col="annotation", met_col="MET",
    features_txt="activity_models/features.txt",
    n_trees=1000, max_depth=None, min_samples_leaf=1,
    cv=None, test_participants=None,
    out_dir='model/',
    n_jobs=1,
):
    """
    Train model to classify activity states from epoch feature data. Based on a
    balanced random forest with a Hidden Markov Model containing transitions
    between predicted activity states and emissions trained using the input
    training file to identify pre-defined classes of behaviour from
    accelerometer data.

    :param str training_file: Input csv file of training data, pre-sorted by time
    :param str label_col: Input label column
    :param str participant_col: Input participant column
    :param str annotation_col: Input text annotation e.g. 'walking with dog' vs. 'walking'
    :param str met_col: Input MET column
    :param str features_txt: Input txt file listing feature column names
    :param int cv: Number of CV folds. If None, CV is skipped.
    :param str test_participants: Input comma separated list of participant IDs to test on.
    :param int n_trees: Random forest n_estimators param.
    :param int max_depth: Random forest max_depth param.
    :param int min_samples_leaf: Random forest min_samples_leaf param.
    :param str out_dir: Output directory. Output files (trained model, predictions, etc.) will be written to this directory.
    :param int n_jobs: Number of jobs to run in parallel.

    """

    report = {
        'params__n_estimators': n_trees,
        'params__max_depth': max_depth,
        'params__min_samples_leaf': min_samples_leaf,
    }

    os.makedirs(out_dir, exist_ok=True)

    # Load list of features to use for training
    feature_cols = np.loadtxt(features_txt, dtype='str')

    # Load in participant information, and remove null/messy labels/features
    all_cols = [participant_col, label_col, annotation_col] + feature_cols.tolist()
    if met_col:
        all_cols.append(met_col)
    data = pd.read_csv(training_file, usecols=all_cols)
    with pd.option_context('mode.use_inf_as_null', True):
        data = data.dropna(axis=0, how='any')

    # Train/test split if test_participants provided
    if test_participants is not None:
        test_pids = test_participants.split(',')
        test = data[data[participant_col].isin(test_pids)].copy()
        train = data[~data[participant_col].isin(test_pids)].copy()
    else:
        train = data

    features, labels_true, participant_ids = train[feature_cols].to_numpy(), train[label_col].to_numpy(), train[participant_col].to_numpy()

    def _Model(**kwargs):
        return BalancedRandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            replacement=True,
            sampling_strategy='not minority',
            random_state=42,
            **kwargs
        )

    print('Training...')
    model = _Model(n_jobs=n_jobs, verbose=1)
    model = model.fit(features, labels_true)
    model.verbose = 0  # silence future calls to .predict()
    labels = model.classes_

    print('Cross-predicting to derive the observations for HMM...')
    n_jobs_per_cv_model = min(2, n_jobs)
    cvp = cross_val_predict(
        _Model(n_jobs=n_jobs_per_cv_model), features, labels_true, groups=participant_ids,
        cv=10,
        n_jobs=n_jobs // n_jobs_per_cv_model,
        method="predict_proba",
        verbose=3,
    )
    print('Training HMM...')
    hmm_params = train_hmm(cvp, labels_true, labels)

    # Estimate METs via per-class averaging
    mets = None
    if met_col:
        mets = {y: train[labels_true == y].groupby(annotation_col)[met_col].mean().mean()
                for y in model.classes_}

    # Write model to file
    out_file = os.path.join(out_dir, 'model.tar')
    save_to_tar(out_file,
                model=model,
                labels=labels,
                featureCols=feature_cols,
                hmmParams=hmm_params,
                METs=mets)
    print(f'Output trained model written to: {out_file}')

    # Assess model performance on test participants
    if test_participants is not None:
        print('Testing on participant(s):', test_participants)
        x_test, y_test = test[feature_cols].to_numpy(), test[label_col].to_numpy()
        y_pred = model.predict(x_test)
        y_pred_hmm = viterbi(y_pred, hmm_params)
        test['predicted'] = y_pred_hmm

        # Write predictions to file
        out_cols = [participant_col, label_col, 'predicted']
        out_file = os.path.join(out_dir, 'test-predictions.csv')
        test[out_cols].to_csv(out_file, index=False)
        print(f'Output test predictions written to: {out_file}')

        print('\nTest performance (no HMM):')
        print(metrics.classification_report(y_test, y_pred))
        test_score = metrics.f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f'Score: {test_score:.2f}')
        report['test_score'] = test_score

        print('\nTest performance (HMM):')
        print(metrics.classification_report(y_test, y_pred_hmm))
        test_hmm_score = metrics.f1_score(y_test, y_pred_hmm, average='macro', zero_division=0)
        print(f'Score: {test_hmm_score:.2f}')
        report['test_hmm_score'] = test_hmm_score

    if cv:
        print("Cross-validating...")
        cv_scores = cross_val_score(
            _Model(n_jobs=n_jobs_per_cv_model),
            # cv with whole data
            data[feature_cols].to_numpy(), data[label_col].to_numpy(), groups=data[participant_col].to_numpy(),
            scoring=metrics.make_scorer(metrics.f1_score, average='macro', zero_division=0),
            cv=cv,
            n_jobs=n_jobs // n_jobs_per_cv_model,
            verbose=3,
        )
        cv_scores_avg = np.mean(cv_scores)
        cv_scores_25th, cv_scores_75th = np.quantile(cv_scores, (.25, .75))
        print(f"CV score: {cv_scores_avg:.2f} ({cv_scores_25th:.2f}, {cv_scores_75th:.2f})")
        report['cv_mean_score'] = cv_scores_avg
        report['cv_25th_score'] = cv_scores_25th
        report['cv_75th_score'] = cv_scores_75th

    out_file = os.path.join(out_dir, 'report.json')
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=4)
    print(f'\nOutput report file written to: {out_file}')


def train_hmm(y_prob, y_true, labels=None, uniform_prior=True):
    """
    Implements a Hidden Markov Model as described in https://en.wikipedia.org/wiki/Hidden_Markov_model.

    :param numpy.array y_prob: Array of predicted probabilities for each class.
    :param numpy.array y_true: Array of true labels.
    :param list(str) labels: List of class labels.
    :param uniform_prior: If True, all labels have equal probability. If False, label probability equals empirical rate.

    :return: Dictionary containing prior, emission and transition matrices, and corresponding labels.
    :rtype: dict
    """

    if labels is None:
        labels = np.unique(y_true)

    if uniform_prior:
        # All labels with equal probability
        prior = np.ones(len(labels)) / len(labels)
    else:
        # Label probability equals empirical rate
        prior = np.mean(y_true.reshape(-1, 1) == labels, axis=0)

    emission = np.vstack(
        [np.mean(y_prob[y_true == label], axis=0) for label in labels]
    )
    transition = np.vstack(
        [np.mean(y_true[1:][(y_true == label)[:-1]].reshape(-1, 1) == labels, axis=0)
            for label in labels]
    )

    params = {'prior': prior, 'emission': emission, 'transition': transition, 'labels': labels}

    return params


def viterbi(y_obs, hmm_params):
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

    n_obs = len(y_obs)
    n_labels = len(labels)

    y_obs = np.where(y_obs.reshape(-1, 1) == labels)[1]  # to numeric

    probs = np.zeros((n_obs, n_labels))
    probs[0, :] = log(prior) + log(emission[:, y_obs[0]])
    for j in range(1, n_obs):
        for i in range(n_labels):
            probs[j, i] = np.max(
                log(emission[i, y_obs[j]]) +
                log(transition[:, i]) +
                probs[j - 1, :])  # probs already in log scale
    viterbi_path = np.zeros_like(y_obs)
    viterbi_path[-1] = np.argmax(probs[-1, :])
    for j in reversed(range(n_obs - 1)):
        viterbi_path[j] = np.argmax(
            log(transition[:, viterbi_path[j + 1]]) +
            probs[j, :])  # probs already in log scale

    viterbi_path = labels[viterbi_path]  # to labels

    return viterbi_path


def remove_spurious_sleep(y, activity_model='walmsley', sleep_tol=60):
    """
    Remove spurious sleep epochs from activity classification.

    :param pandas.Series y: Model output
    :param str activity_model: Model identifier
    :param str sleep_tol: Sleep tolerance in minutes. If a sleep streak is shorter
        than this, it will be replaced with sedentary activity.

    :return: Dataframe of revised model output
    :rtype: pandas.DataFrame
    """

    new_value = {
        'willetts': 'sit-stand',
        'doherty': 'sedentary',
        'walmsley': 'sedentary',
        'chan': 'sedentary',
    }[activity_model]

    sleep = y == 'sleep'
    sleep_streak = (
        sleep.ne(sleep.shift())
        .cumsum()
        .pipe(lambda x: x.groupby(x).transform('count') * sleep)
    )
    sleep_tol = pd.Timedelta(f"{sleep_tol}min") / y.index.freq
    whr = sleep & (sleep_streak < sleep_tol)
    y = y.copy()  # no modify original
    y.loc[whr] = new_value

    return y


def cut_point_model(enmo, cuts=None, whr=None):
    """
    Perform classification of activities based on cutpoints.

    :param pandas.Series enmo: Timeseries of ENMO.
    :param dict cuts: Dictionary of cutpoints for each activity.

    :return: Activity labels.
    :rtype: pandas.Series
    """

    if cuts is None:
        # default cutpoints
        cuts = {'LPA': 45 / 1000, 'MPA': 100 / 1000, 'VPA': 400 / 1000}

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


def per_participant_summary_html(df_param, y_true_col, y_pred_col, pid_col, out_html):
    """
    Provide HTML summary of how well activity classification model works at the per-participant level.

    :param pandas.DataFrame df_param: Input pandas dataframe
    :param str y_true_col: Input for y_true column label
    :param str y_pred_col: Input for y_pred column label
    :param str pid_col: Input for participant ID column label
    :param str out_html: Output file to print HTML summary to

    """
    # get kappa & accuracy on a per-participant basis
    pids = df_param[pid_col].unique()
    pid_kappa = []
    pid_accuracy = []
    for pid in pids:
        d_tmp = df_param[df_param[pid_col] == pid]
        pid_kappa += [metrics.cohen_kappa_score(d_tmp[y_true_col], d_tmp[y_pred_col])]
        pid_accuracy += [metrics.accuracy_score(d_tmp[y_true_col], d_tmp[y_pred_col])]
    d_summary = pd.DataFrame()
    d_summary['pid'] = pids
    d_summary['kappa'] = pid_kappa
    d_summary['accuracy'] = pid_accuracy
    # print out values to html string
    kappa_sd_html = "Mean Kappa (SD) = "
    kappa_sd_html += utils.mean_sd_str(d_summary['kappa'].mean(),
                                       d_summary['kappa'].std(), 2)
    accuracy_sd_html = "Mean accuracy (SD) = "
    accuracy_sd_html += utils.mean_sd_str(d_summary['accuracy'].mean() * 100,
                                          d_summary['accuracy'].std() * 100, 1) + ' %'
    kappa_ci_html = "Mean Kappa (95% CI) = "
    kappa_ci_html += utils.mean_ci_str(d_summary['kappa'].mean(),
                                       d_summary['kappa'].std(), len(d_summary), 2)
    accuracy_ci_html = "Mean accuracy (95% CI) = "
    accuracy_ci_html += utils.mean_ci_str(d_summary['accuracy'].mean() * 100,
                                          d_summary['accuracy'].std() * 100, len(d_summary), 1) + ' %'

    # get confusion matrix to pandas dataframe
    y_true = df_param[y_true_col]
    y_pred = df_param[y_pred_col]
    labels = sorted(list(set(y_true) | set(y_pred)))
    cnf_matrix = confusion_matrix(y_true, y_pred, labels)
    df_confusion = pd.DataFrame(data=cnf_matrix, columns=labels, index=labels)
    confusion_html = df_confusion.to_html()

    # construct final output string
    html_str = '<html><head><title>Classification summary</title></head><body>'
    html_str += kappa_sd_html + '<br>\n' + accuracy_sd_html + '<br><br>\n'
    html_str += kappa_ci_html + '<br>\n' + accuracy_ci_html + '<br>\n'
    html_str += confusion_html + '<br>\n'
    html_str += '</body></html>'

    # write HTML file
    with open(out_html, 'w') as html_file:
        html_file.write(html_str)


def save_to_tar(tar_out, **kwargs):
    """
    Save objects to tar file. Objects must be passed as keyworded arguments, then the key is used for the object name in the tar file.

    :param kwargs: Objects to be saved passed as keyworded arguments.

    """
    try:

        tmpdir = tempfile.mkdtemp()

        with tarfile.open(tar_out, mode='w') as tf:

            for key, val in kwargs.items():
                pth = os.path.join(tmpdir, key)
                joblib.dump(val, pth, compress=True)
                tf.add(pth, arcname=key)

        print('Models saved to', tar_out)

    finally:

        try:
            shutil.rmtree(tmpdir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def get_file_from_tar(tar_archive, target_file):
    """
    Read file from tar. This is currently more tricky than it should be. See https://github.com/numpy/numpy/issues/7989

    :param str tar_archive: Input tarfile object
    :param str target_file: Target individual file within .tar

    :return: file object byte stream
    :rtype: io.BytesIO

    """
    with tarfile.open(tar_archive, 'r') as t:
        byte_buffer = BytesIO()
        try:
            byte_buffer.write(t.extractfile(target_file).read())
        except KeyError:
            raise ClassificationError(
                f"Required model file '{target_file}' not found in archive '{tar_archive}'. "
                f"The model archive may be corrupted or from an incompatible version."
            )
        byte_buffer.seek(0)

    return byte_buffer


def add_reference_labels_to_new_features(
        features_file,
        reference_labels_file,
        output_file,
        features_txt="activity_models/features.txt",
        label_col="label", participant_col="participant",
        annotation_col="annotation", met_col="MET"
):
    """
    Append reference annotations to newly extracted feature data. This method
    helps add existing curated labels (from reference_labels_file) to a file with
    newly extracted features (both pre-sorted by participant and time).

    :param str features_file: Input csv file of new features data, pre-sorted by time
    :param str reference_labels_file: Input csv file of reference labelled data, pre-sorted by time
    :param str output_file: Output csv file of new features data with reference labels
    :param str features_txt: Input txt file listing feature column names
    :param str label_col: Input label column
    :param str participant_col: Input participant column
    :param str annotation_col: Input text annotation e.g. 'walking with dog' vs. 'walking'
    :param str met_col: Input MET column

    :return: None. Writes a new csv file to <output_file>.

    .. code-block:: python

        from accelerometer import classification
        classification.add_reference_labels_to_new_features("newFeats.csv", "refLabels.csv", "newFeatsPlusLabels.csv")
    """

    # load new features file
    feature_cols = np.loadtxt(features_txt, dtype='str')
    d_feat = pd.read_csv(features_file, usecols=feature_cols + [participant_col, 'time'])

    # load in reference annotations file
    ref_cols = [participant_col, 'age', 'sex', 'time', annotation_col, label_col,
                'code', met_col, 'MET_label']
    d_ref = pd.read_csv(reference_labels_file, usecols=ref_cols)

    # join dataframes
    index_cols = [participant_col, 'time']
    d_out = d_ref.set_index(index_cols).join(d_feat.set_index(index_cols), how='left')

    # write out new labelled features file
    d_out.to_csv(output_file, index=True)
    print('New file written to: ', output_file)


def resolve_model_path(path_or_model_name):

    if pathlib.Path(path_or_model_name).exists():
        return path_or_model_name

    else:

        # versions before January 2022 no longer supported
        if path_or_model_name in (
            'walmsley-jan21', 'doherty-jan21', 'willetts-jan21',
            'walmsley-may20', 'doherty-may20', 'willetts-may20'
        ):
            path_or_model_name = path_or_model_name.split("-")[0]
            warnings.warn(
                f"Activity model versions before January 2022 are no longer supported. "
                f"Defaulting to --activityModel {path_or_model_name}"
            )

        model = MODELS.get(path_or_model_name, None)
        if model is None:
            raise FileNotFoundError(f"Model file {path_or_model_name} not found")
        if model["pth"].exists():
            return model["pth"]
        else:
            return download_model(model)


def download_model(model):
    url = model["url"]
    pth = model["pth"]

    os.makedirs(os.path.dirname(pth), exist_ok=True)

    print(f"Downloading {url}...")

    with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
        shutil.copyfileobj(f_src, f_dst)

    return pth
