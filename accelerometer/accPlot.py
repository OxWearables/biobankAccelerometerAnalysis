"""Script to plot accelerometer traces."""

from pandas.plotting import register_matplotlib_converters
import sys
import pandas as pd
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import argparse
from accelerometer import accUtils
import matplotlib
matplotlib.use('Agg')

# http://pandas-docs.github.io/pandas-docs-travis/whatsnew/v0.21.1.html#restore-matplotlib-datetime-converter-registration
register_matplotlib_converters()

DOHERTY_NatComms_COLOURS = {'sleep': 'blue', 'sedentary': 'red',
                            'tasks-light': 'darkorange', 'walking': 'lightgreen', 'moderate': 'green'}

WILLETS_SciReports_COLOURS = {'sleep': 'blue', 'sit.stand': 'red',
                              'vehicle': 'darkorange', 'walking': 'lightgreen', 'mixed': 'green',
                              'bicycling': 'purple'}

WALMSLEY_Nov2020_COLOURS = {'sleep': 'blue', 'sedentary': 'red',
                            'light': 'darkorange', 'MVPA': 'green'}


def main():  # noqa: C901
    """
    Application entry point responsible for parsing command line requests
    """

    parser = argparse.ArgumentParser(
        description="A script to plot acc time series data.", add_help=True)
    # required
    parser.add_argument('timeSeriesFile', metavar='input file', type=str,
                        help="input .csv.gz time series file to plot")
    parser.add_argument('--plotFile', metavar='output file', type=str,
                        help="output .png file to plot to")
    parser.add_argument('--activityModel', type=str,
                        default="walmsley",
                        help="""trained activity model .tar file""")
    parser.add_argument('--useRecommendedImputation',
                        metavar='True/False', default=True, type=str2bool,
                        help="""Highly recommended method to show imputed
                            missing data (default : %(default)s)""")
    parser.add_argument('--imputedLabels',
                        metavar='True/False', default=False, type=str2bool,
                        help="""If activity classification during imputed
                            period will be displayed (default : %(default)s)""")
    parser.add_argument('--imputedLabelsHeight',
                        metavar='Proportion i.e. 0-1.0', default=0.9,
                        type=float, help="""Proportion of plot labels take
                            if activity classification during imputed
                            period will be displayed (default : %(default)s)""")
    parser.add_argument('--showFileName',
                        metavar='True/False', default=False, type=str2bool,
                        help="""Toggle showing filename as title in output
                            image (default : %(default)s)""")
    parser.add_argument('--showFirstNDays',
                        metavar='days', default=None,
                        type=int, help="Show just first n days")

    # check input is ok
    if len(sys.argv) < 2:
        msg = "\nInvalid input, please enter at least 1 parameter, e.g."
        msg += "\npython accPlot.py timeSeries.csv.gz \n"
        accUtils.toScreen(msg)
        parser.print_help()
        sys.exit(-1)
    args = parser.parse_args()

    # determine output file name
    if args.plotFile is None:
        inputFileFolder, inputFileName = os.path.split(args.timeSeriesFile)
        inputFileName = inputFileName.split('.')[0]  # remove any extension
        args.plotFile = os.path.join(inputFileFolder, inputFileName + "-plot.png")

    # and then call plot function
    plotTimeSeries(args.timeSeriesFile, args.plotFile,
                   showFirstNDays=args.showFirstNDays,
                   activityModel=args.activityModel,
                   useRecommendedImputation=args.useRecommendedImputation,
                   imputedLabels=args.imputedLabels,
                   imputedLabelsHeight=args.imputedLabelsHeight,
                   showFileName=args.showFileName)


def plotTimeSeries(  # noqa: C901
        tsFile,
        plotFile,
        showFirstNDays=None,
        activityModel="walmsley",
        useRecommendedImputation=True,
        imputedLabels=False,
        showFileName=False,
        imputedLabelsHeight=0.9):
    """Plot overall activity and classified activity types

    :param str tsFile: Input filename with .csv.gz time series data
    :param str tsFile: Output filename for .png image
    :param int showFirstNDays: Only show first n days of time series (if specified)
    :param str activityModel: Input tar model file used for activity classification
    :param bool useRecommendedImputation: Highly recommended method to show
        imputed values for missing data
    :param bool imputedLabels: If activity classification during imputed period
        will be displayed
    :param float imputedLabelsHeight: Proportion of plot labels take up if
        <imputedLabels> is True
    :param float showFileName: Toggle showing filename as title in output image

    :return: Writes plot to <plotFile>
    :rtype: void

    :Example:
    >>> import accPlot
    >>> accPlot.plotTimeSeries("sample-timeSeries.csv.gz", "sample-plot.png")
    <plot file written to sample-plot.png>
    """

    # read time series file to pandas DataFrame
    d = pd.read_csv(
        tsFile, index_col='time',
        parse_dates=['time'], date_parser=accUtils.date_parser
    )
    if showFirstNDays is not None:
        d = d.first(str(showFirstNDays) + 'D')

    d['acc'] = d['acc'].rolling(window=12, min_periods=1).mean()  # smoothing
    d['time'] = d.index.time
    ymin = d['acc'].min()
    ymax = d['acc'].max()

    # infer labels
    labels = []
    for col in d.columns.tolist():
        if col not in [accUtils.TIME_SERIES_COL, 'imputed', 'acc', 'MET']:
            labels += [col]
    print(labels)
    if 'walmsley' in activityModel:
        labels_as_col = WALMSLEY_Nov2020_COLOURS
    elif 'doherty' in activityModel:
        labels_as_col = DOHERTY_NatComms_COLOURS
    elif 'willetts' in activityModel:
        labels_as_col = WILLETS_SciReports_COLOURS
    # add imputation label colour
    labels_as_col['imputed'] = '#fafc6f'

    convert_date = np.vectorize(lambda day, x: matplotlib.dates.date2num(datetime.combine(day, x)))

    # number of rows to display in figure (all days + legend)
    d['date'] = d.index.date
    if not useRecommendedImputation:
        d = d[d['imputed'] == 0]  # if requested, do not show imputed values

    if imputedLabels:
        labelsPosition = imputedLabelsHeight
    else:
        labelsPosition = 1

    groupedDays = d[['acc', 'time', 'imputed'] + labels].groupby(by=d['date'])
    nrows = len(groupedDays) + 1

    # create overall figure
    fig = plt.figure(1, figsize=(10, nrows), dpi=100)
    if showFileName:
        plt.suptitle(tsFile)

    # create individual plot for each day
    i = 0
    ax_list = []
    for day, group in groupedDays:
        # set start and end to zero to avoid diagonal fill boxes
        group['imputed'].values[0] = 0
        group['imputed'].values[-1] = 0

        # retrieve time series data for this day
        timeSeries = convert_date(day, group['time'])
        # and then plot time series data for this day
        plt.subplot(nrows, 1, i + 1)
        plt.plot(timeSeries, group['acc'], c='k')
        if imputedLabels:
            plt.fill_between(timeSeries,
                             y1=np.multiply(group['imputed'], ymax),
                             y2=np.multiply(group['imputed'], ymax * labelsPosition),
                             color=labels_as_col['imputed'], alpha=1.0,
                             where=group['imputed'] == 1)
        else:
            plt.fill(timeSeries, np.multiply(group['imputed'], ymax),
                     labels_as_col['imputed'], alpha=1.0)

        # change display properties of this subplot
        ax = plt.gca()
        if len(labels) > 0:
            ax.stackplot(timeSeries,
                         [np.multiply(group[l], ymax * labelsPosition) for l in labels],
                         colors=[labels_as_col[l] for l in labels],
                         alpha=0.5, edgecolor="none")
        # add date label to left hand side of each day's activity plot
        plt.title(
            day.strftime("%A,\n%d %B"), weight='bold',
            x=-.2, y=0.5,
            horizontalalignment='left',
            verticalalignment='center',
            rotation='horizontal',
            transform=ax.transAxes,
            fontsize='medium',
            color='k'
        )
        # run gridlines for each hour bar
        ax.get_xaxis().grid(True, which='major', color='grey', alpha=0.5)
        ax.get_xaxis().grid(True, which='minor', color='grey', alpha=0.25)
        # set x and y-axes
        ax.set_xlim((datetime.combine(day, time(0, 0, 0, 0)),
                     datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0))))
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq='4H'))
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq='1H'), minor=True)
        ax.set_ylim((ymin, ymax))
        ax.get_yaxis().set_ticks([])  # hide y-axis lables
        # make border less harsh between subplots
        ax.spines['top'].set_color('#d3d3d3')  # lightgray
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # set background colour to lightgray
        ax.set_facecolor('#d3d3d3')

        # append to list and incrament list counter
        ax_list.append(ax)
        i += 1

    # create new subplot to display legend
    plt.subplot(nrows, 1, i + 1)
    ax = plt.gca()
    ax.axis('off')  # don't display axis information
    # create a 'patch' for each legend entry
    legend_patches = [mpatches.Patch(color=labels_as_col['imputed'],
                                     label='imputed', alpha=1.0),
                      mlines.Line2D([], [], color='k', label='acceleration')]
    # create lengend entry for each label
    for label in labels:
        col = labels_as_col[label]
        legend_patches.append(mpatches.Patch(color=col, label=label, alpha=0.5))
    # create overall legend
    plt.legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.),
               loc='center', ncol=min(4, len(legend_patches)), mode="best",
               borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)

    # remove legend border
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax_list.append(ax)

    # format x-axis to show hours
    plt.gcf().autofmt_xdate()
    # add hour labels to top of plot
    hrLabels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    ax_list[0].set_xticklabels(hrLabels)
    ax_list[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    plt.savefig(plotFile, dpi=200, bbox_inches='tight')
    print('Plot file written to:', plotFile)


def str2bool(v):
    """
    Used to parse true/false values from the command line. E.g. "True" -> True
    """

    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    main()  # Standard boilerplate to call the main() function to begin the program.
