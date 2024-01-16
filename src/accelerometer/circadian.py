"""Module to support calculation of metrics of circadian rhythm from acc data"""

import numpy as np
import scipy as sp
from scipy import fftpack
from datetime import timedelta


def calculatePSD(e, epochPeriod, fourierWithAcc, labels, summary):
    """
    Calculate the power spectral density from fourier analysis of a 1 day frequency.

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param bool fourierWithAcc: True calculates fourier done with acceleration data instead of sleep data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics. This dictionary will be modified in-place: a new key 'PSD-<W/Hz>' will be added with the calculated frequency as its value.

    """
    if fourierWithAcc:
        y = e['accImputed'].values
    else:
        cols = [label + 'Imputed' for label in labels]  # 'sleepImputed', 'sedentaryImputed', etc...
        y = e[cols].idxmax(axis=1) == 'sleepImputed'  # is sleepImputed highest?
        y = y.values.astype('int')  # e.g. [0,0,0,1,1,0,0,1,...]
        y = 2 * y - 1  # center the signal, [0,1] -> [-1,1]

    n = len(y)
    k = len(y) * epochPeriod / (60 * 60 * 24)
    a = -2.j * np.pi * k * np.arange(n) / n
    # Find the power spectral density for a one day cycle using fourier analysis
    res = np.sum(np.exp(a) * y) / n
    PSD = np.abs(res)**2
    summary['PSD'] = PSD


def calculateFourierFreq(e, epochPeriod, fourierWithAcc, labels, summary):
    """
    Calculate the most prevalent frequency in a fourier analysis.

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param bool fourierWithAcc: True calculates fourier done with acceleration data instead of sleep data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics. This dictionary will be modified in-place: a new key 'fourier frequency-<1/days>' will be added with the calculated frequency as its value.

    """
    if fourierWithAcc:
        y = e['accImputed'].values
    else:
        cols = [label + 'Imputed' for label in labels]  # 'sleepImputed', 'sedentaryImputed', etc...
        y = e[cols].idxmax(axis=1) == 'sleepImputed'  # is sleepImputed highest?
        y = y.values.astype('int')  # e.g. [0,0,0,1,1,0,0,1,...]
        y = 2 * y - 1  # center the signal, [0,1] -> [-1,1]

    # Fast fourier transform of the sleep column
    fft_y = np.abs(fftpack.fft(y))

    i = np.arange(1, len(fft_y))
    k_max = np.argmax(fft_y[i]) + 1
    n = len(y)

    def func(k):
        """ Maximise the fourier transform function (func) using the fft_y as a first esitmate """
        return -np.abs(np.sum(np.exp(-2.j * np.pi * k * np.arange(n) / n) * y) / n)

    res = sp.optimize.minimize_scalar(func, bracket=(k_max - 1, k_max + 1))
    # Adjust the frequency to have the units 1/days
    freq_mx = float(res.x) / (len(y) * epochPeriod / (60 * 60 * 24))
    summary['fourier-frequency'] = freq_mx


def calculateM10L5(e, epochPeriod, summary):
    """
    Calculates the M10 L5 relative amplitude from the average acceleration from
    the ten most active hours and 5 least most active hours.

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param int epochPeriod: Size of epoch time window (in seconds)
    :param dict summary: Output dictionary containing all summary metrics. This dictionary will be modified in-place: a new key 'M10 L5-<rel amp>' will be added with the calculated frequency as its value.

    """
    TEN_HOURS = int(10 * 60 * 60 / epochPeriod)
    FIVE_HOURS = int(5 * 60 * 60 / epochPeriod)
    num_days = (e.index[-1] - e.index[0]).days

    days_split = []
    for n in range(num_days):
        # creates a new list which is used to identify the 24 hour periods in the data frame
        days_split += [n for x in e.index if e.index[0] + timedelta(days=n) <= x < e.index[0] + timedelta(days=n + 1)]
    dct = {}
    for i in range(num_days):
        # create new lists with the acceleration data from each 24 hour period
        dct['day_%s' % i] = [e.loc[:, 'accImputed'][n] for n in range(len(days_split)) if days_split[n] == i]
    dct_10 = {}
    dct_5 = {}
    for i in dct:
        #  sums each 10 or 5 hour window with steps of 30s for each day
        dct_10['%s' % i] = [sum(dct['%s' % i][j:j + TEN_HOURS]) for j in range(len(dct['%s' % i]) - TEN_HOURS)]
        dct_5['%s' % i] = [sum(dct['%s' % i][j:j + FIVE_HOURS]) for j in range(len(dct['%s' % i]) - FIVE_HOURS)]
    avg_10 = {}
    avg_5 = {}
    #   average acceleration (for each 30s) for the max and min windows
    for i in dct:
        avg_10['%s' % i] = (np.max(dct_10['%s' % i])) / TEN_HOURS
    for i in dct:
        avg_5['%s' % i] = (np.min(dct_5['%s' % i])) / FIVE_HOURS

    if num_days > 0:
        M10 = sum(avg_10.values()) / num_days
        L5 = sum(avg_5.values()) / num_days
        rel_amp = (M10 - L5) / (M10 + L5)
    if num_days < 1:
        rel_amp = 'NA_too_few_days'
    summary['M10L5'] = rel_amp
