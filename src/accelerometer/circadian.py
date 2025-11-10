"""Module to support calculation of metrics of circadian rhythm from acc data"""

import numpy as np
import scipy as sp
from scipy import fftpack
from datetime import timedelta


def calculate_psd(epoch_data, epoch_period, fourier_with_acc, labels, summary):
    """
    Calculate the power spectral density from fourier analysis of a 1 day frequency.

    :param pandas.DataFrame epoch_data: Pandas dataframe of epoch data
    :param int epoch_period: Size of epoch time window (in seconds)
    :param bool fourier_with_acc: True calculates fourier done with acceleration data instead of sleep data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics. This dictionary will be modified in-place: a new key 'PSD-<W/Hz>' will be added with the calculated frequency as its value.

    """
    if fourier_with_acc:
        signal = epoch_data['acc'].values
    else:
        signal = epoch_data[labels].idxmax(axis=1) == 'sleep'  # is sleep highest?
        signal = signal.values.astype('int')  # e.g. [0,0,0,1,1,0,0,1,...]
        signal = 2 * signal - 1  # center the signal, [0,1] -> [-1,1]

    num_samples = len(signal)
    cycles_per_day = len(signal) * epoch_period / (60 * 60 * 24)
    fourier_coefficients = -2.j * np.pi * cycles_per_day * np.arange(num_samples) / num_samples
    # Find the power spectral density for a one day cycle using fourier analysis
    res = np.sum(np.exp(fourier_coefficients) * signal) / num_samples
    psd = np.abs(res)**2
    summary['PSD'] = psd


def calculate_fourier_freq(epoch_data, epoch_period, fourier_with_acc, labels, summary):
    """
    Calculate the most prevalent frequency in a fourier analysis.

    :param pandas.DataFrame epoch_data: Pandas dataframe of epoch data
    :param int epoch_period: Size of epoch time window (in seconds)
    :param bool fourier_with_acc: True calculates fourier done with acceleration data instead of sleep data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics. This dictionary will be modified in-place: a new key 'fourier frequency-<1/days>' will be added with the calculated frequency as its value.

    """
    if fourier_with_acc:
        signal = epoch_data['acc'].values
    else:
        signal = epoch_data[labels].idxmax(axis=1) == 'sleep'  # is sleep highest?
        signal = signal.values.astype('int')  # e.g. [0,0,0,1,1,0,0,1,...]
        signal = 2 * signal - 1  # center the signal, [0,1] -> [-1,1]

    # Fast fourier transform of the sleep column
    fft_magnitudes = np.abs(fftpack.fft(signal))

    i = np.arange(1, len(fft_magnitudes))
    k_max = np.argmax(fft_magnitudes[i]) + 1
    num_samples = len(signal)

    def fourier_objective(k):
        """ Maximise the fourier transform function (fourier_objective) using the fft_magnitudes as a first esitmate """
        return -np.abs(np.sum(np.exp(-2.j * np.pi * k * np.arange(num_samples) / num_samples) * signal) / num_samples)

    res = sp.optimize.minimize_scalar(fourier_objective, bracket=(k_max - 1, k_max + 1))
    # Adjust the frequency to have the units 1/days
    freq_mx = float(res.x) / (len(signal) * epoch_period / (60 * 60 * 24))
    summary['fourier-frequency'] = freq_mx


def calculate_m10l5(epoch_data, epoch_period, summary):
    """
    Calculates the M10 L5 relative amplitude from the average acceleration from
    the ten most active hours and 5 least most active hours.

    :param pandas.DataFrame epoch_data: Pandas dataframe of epoch data
    :param int epoch_period: Size of epoch time window (in seconds)
    :param dict summary: Output dictionary containing all summary metrics. This dictionary will be modified in-place: a new key 'M10 L5-<rel amp>' will be added with the calculated frequency as its value.

    """
    TEN_HOURS = int(10 * 60 * 60 / epoch_period)
    FIVE_HOURS = int(5 * 60 * 60 / epoch_period)
    num_days = (epoch_data.index[-1] - epoch_data.index[0]).days

    days_split = []
    for n in range(num_days):
        # creates a new list which is used to identify the 24 hour periods in the data frame
        days_split += [n for x in epoch_data.index if epoch_data.index[0] + timedelta(days=n) <= x < epoch_data.index[0] + timedelta(days=n + 1)]
    day_data = {}
    for i in range(num_days):
        # create new lists with the acceleration data from each 24 hour period
        day_data['day_%s' % i] = [epoch_data.loc[:, 'acc'][n] for n in range(len(days_split)) if days_split[n] == i]
    dct_10 = {}
    dct_5 = {}
    for i in day_data:
        #  sums each 10 or 5 hour window with steps of 30s for each day
        dct_10['%s' % i] = [sum(day_data['%s' % i][j:j + TEN_HOURS]) for j in range(len(day_data['%s' % i]) - TEN_HOURS)]
        dct_5['%s' % i] = [sum(day_data['%s' % i][j:j + FIVE_HOURS]) for j in range(len(day_data['%s' % i]) - FIVE_HOURS)]
    avg_10 = {}
    avg_5 = {}
    #   average acceleration (for each 30s) for the max and min windows
    for i in day_data:
        avg_10['%s' % i] = (np.max(dct_10['%s' % i])) / TEN_HOURS
    for i in day_data:
        avg_5['%s' % i] = (np.min(dct_5['%s' % i])) / FIVE_HOURS

    if num_days > 0:
        M10 = sum(avg_10.values()) / num_days
        L5 = sum(avg_5.values()) / num_days
        rel_amp = (M10 - L5) / (M10 + L5)
    if num_days < 1:
        rel_amp = 'NA_too_few_days'
    summary['M10L5'] = rel_amp
