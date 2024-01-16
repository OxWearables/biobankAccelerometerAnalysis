"""Script to plot accelerometer traces."""

import sys
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import argparse
from accelerometer import utils
import matplotlib

# http://pandas-docs.github.io/pandas-docs-travis/whatsnew/v0.21.1.html#restore-matplotlib-datetime-converter-registration
register_matplotlib_converters()

LABELS_AND_COLORS = {
    'imputed': '#fafc6f',
    'sleep': 'midnightblue',
    'sit-stand': 'red',
    'sedentary': 'red',
    'vehicle': 'saddlebrown',
    'light': 'darkorange',
    'mixed': 'seagreen',
    'walking': 'limegreen',
    'moderate-vigorous': 'green',
    'moderate': 'green',
    'vigorous': 'springgreen',
    'bicycling': 'springgreen',
    'tasks-light': 'darkorange',
    'SB': 'red',  # sedentary behaviour
    'LPA': 'darkorange',  # light physical activity
    'MVPA': 'green',  # moderate-vigorous physical activity
    'MPA': 'green',  # moderate physical activity
    'VPA': 'springgreen',  # vigorous physical activity
}


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
        utils.toScreen(msg)
        parser.print_help()
        sys.exit(-1)
    args = parser.parse_args()

    # determine output file name
    if args.plotFile is None:
        inputFileFolder, inputFileName = os.path.split(args.timeSeriesFile)
        inputFileName = inputFileName.split('.')[0]  # remove any extension
        args.plotFile = os.path.join(inputFileFolder, inputFileName + "-plot.png")

    # read time series file to pandas DataFrame
    data = pd.read_csv(
        args.timeSeriesFile,
        index_col='time',
        parse_dates=['time'],
        date_parser=utils.date_parser
    )

    # set backend if run from main
    matplotlib.use('Agg')

    # set plot title
    title = args.timeSeriesFile if args.showFileName else None

    # call plot function and save figure
    fig = plotTimeSeries(data, title=title, showFirstNDays=args.showFirstNDays)
    fig.savefig(args.plotFile, dpi=200, bbox_inches='tight')
    print('Plot file written to:', args.plotFile)


def plotTimeSeries(  # noqa: C901
    data,
    title=None,
    showFirstNDays=None
):
    """
    Plot acceleration traces and classified activities.

    :param data: Input time-series of acceleration and activity classes. Index: DatetimeIndex. Columns (4-class example):
                 - Name: acc, dtype=float (optional)
                 - Name: light, dtype=Any numeric, value=0 or 1
                 - Name: moderate-vigorous, dtype=Any numeric, value=0 or 1
                 - Name: sedentary, dtype=Any numeric, value=0 or 1
                 - Name: sleep, dtype=Any numeric, value=0 or 1
    :type data: pd.DataFrame
    :param title: Optional plot title
    :type title: str, optional
    :param showFirstNDays: Only show first n days of time series (if specified)
    :type showFirstNDays: int, optional
    :return: pyplot Figure
    :rtype: plt.Figure

    :Example:

    .. code-block:: python

        from accelerometer.accPlot import plotTimeSeries
        df = pd.DataFrame(...)
        fig = plotTimeSeries(df)
        fig.show()
    """

    # check index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    # use tz-naive local time
    if data.index.tz is not None:
        data.index = (
            # convoluted way to mask the ambiguous DST pushback hour, if any
            data.index
            .tz_localize(None)
            .tz_localize(data.index.tz, ambiguous='NaT', nonexistent='NaT')
            .tz_localize(None)
        )
        data = data[data.index.notnull()]

    # fix gaps or irregular sampling
    if pd.infer_freq(data) is None:
        freq = pd.infer_freq(data.head()) or '30s'  # try to infer from first few rows, else default to 30s
        new_index = pd.date_range(data.index[0], data.index[-1], freq=freq)
        data = data.reindex(new_index, method='nearest', tolerance=freq, fill_value=np.NaN)

    if showFirstNDays is not None:
        data = data.first(str(showFirstNDays) + 'D')

    labels = [label for label in LABELS_AND_COLORS.keys() if label in data.columns]
    colors = [LABELS_AND_COLORS[label] for label in labels]

    if 'imputed' in data.columns:
        mask = data['imputed'].astype('bool')
        labels_excl_imputed = [label for label in labels if label != 'imputed']
        data.loc[mask, labels_excl_imputed] = 0
        data.loc[mask, "acc"] = np.nan

    # setup plotting range
    MAXRANGE = 2 * 1000  # 2g (above this is very rare)
    if 'acc' in data:
        data['acc'] = data['acc'].rolling('1T').mean()  # minute average
        data['acc'] = data['acc'].clip(0, MAXRANGE)
    data[labels] = data[labels].astype('f4') * MAXRANGE

    # number of rows to display in figure (all days + legend)
    groupedDays = data.groupby(data.index.date)
    nrows = len(groupedDays) + 1

    # create overall figure
    fig = plt.figure(None, figsize=(10, nrows), dpi=100)
    if title is not None:
        fig.suptitle(title)

    # create individual plot for each day
    i = 0
    axs = []
    for day, group in groupedDays:

        ax = fig.add_subplot(nrows, 1, i + 1)

        if 'acc' in group:
            ax.plot(group.index, group['acc'].to_numpy(), c='k')

        if len(labels) > 0:
            ax.stackplot(group.index,
                         group[labels].to_numpy().T,
                         colors=colors,
                         edgecolor="none")

        # add date label to left hand side of each day's activity plot
        ax.set_ylabel(day.strftime("%A\n%d %B"),
                      weight='bold',
                      horizontalalignment='right',
                      verticalalignment='center',
                      rotation='horizontal',
                      fontsize='medium',
                      color='k',
                      labelpad=5)
        # run gridlines for each hour bar
        ax.get_xaxis().grid(True, which='major', color='grey', alpha=0.5)
        ax.get_xaxis().grid(True, which='minor', color='grey', alpha=0.25)
        # set x and y-axes
        ax.set_xlim(group.index[0], group.index[-1])
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq='4H'))
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq='1H'), minor=True)
        ax.set_ylim(0, MAXRANGE)
        ax.get_yaxis().set_ticks([])  # hide y-axis lables
        # make border less harsh between subplots
        ax.spines['top'].set_color('#d3d3d3')  # lightgray
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # set background colour to lightgray
        ax.set_facecolor('#d3d3d3')

        # append to list and incrament list counter
        axs.append(ax)
        i += 1

    # create new subplot to display legends
    ax = fig.add_subplot(nrows, 1, i + 1)
    ax.axis('off')
    legend_patches = [mlines.Line2D([], [], color='k', label='acceleration')]
    for label, color in zip(labels, colors):
        legend_patches.append(mpatches.Patch(color=color, label=label))
    # create overall legend
    plt.legend(handles=legend_patches, bbox_to_anchor=(0., 0., 1., 1.),
               loc='center', ncol=4, mode="best",
               borderaxespad=0, framealpha=0.6, frameon=True, fancybox=True)

    # remove legend border
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    axs.append(ax)

    # format x-axis to show hours
    fig.autofmt_xdate()
    # add hour labels to top of plot
    hrLabels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    axs[0].set_xticklabels(hrLabels)
    axs[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    # auto trim borders
    fig.tight_layout()

    return fig


def str2bool(v):
    """
    Used to parse true/false values from the command line. E.g. "True" -> True
    """

    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    main()  # Standard boilerplate to call the main() function to begin the program.
