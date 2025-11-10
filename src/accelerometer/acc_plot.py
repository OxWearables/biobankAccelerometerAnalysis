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
from accelerometer.utils import str2bool
import matplotlib

# http://pandas-docs.github.io/pandas-docs-travis/whatsnew/v0.21.1.html#restore-matplotlib-datetime-converter-registration
register_matplotlib_converters()

# ============================================================================
# PLOTTING CONFIGURATION CONSTANTS
# ============================================================================

# Acceleration range (mg)
# Maximum acceleration display range - values above 2g are physiologically rare
PLOT_MAX_ACCELERATION_MG = 2000  # 2g (2000 mg)

# Figure dimensions
PLOT_FIGURE_WIDTH = 10  # inches
PLOT_FIGURE_DPI = 100   # dots per inch for screen display
PLOT_SAVE_DPI = 200     # dots per inch for saved files (higher quality)

# Time grid intervals
PLOT_HOUR_GRID_MAJOR = '4H'  # Major gridlines every 4 hours
PLOT_HOUR_GRID_MINOR = '1H'  # Minor gridlines every 1 hour

# Rolling average window for acceleration trace
PLOT_ACC_ROLLING_WINDOW = '1T'  # 1 minute average for smoother display

# Colors
PLOT_GRID_COLOR_MAJOR = 'grey'
PLOT_GRID_COLOR_MINOR = 'grey'
PLOT_GRID_ALPHA_MAJOR = 0.5
PLOT_GRID_ALPHA_MINOR = 0.25
PLOT_BORDER_COLOR = '#d3d3d3'  # lightgray

# Activity labels and their corresponding colors for visualization
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
        utils.to_screen(msg)
        parser.print_help()
        sys.exit(-1)
    args = parser.parse_args()

    # determine output file name
    if args.plotFile is None:
        input_file_folder, input_file_name = os.path.split(args.timeSeriesFile)
        input_file_name = input_file_name.split('.')[0]  # remove any extension
        args.plotFile = os.path.join(input_file_folder, input_file_name + "-plot.png")

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
    fig = plot_time_series(data, title=title, show_first_n_days=args.showFirstNDays)
    fig.savefig(args.plotFile, dpi=PLOT_SAVE_DPI, bbox_inches='tight')
    print('Plot file written to:', args.plotFile)


def plot_time_series(  # noqa: C901
    data,
    title=None,
    show_first_n_days=None
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
    :param show_first_n_days: Only show first n days of time series (if specified)
    :type show_first_n_days: int, optional
    :return: pyplot Figure
    :rtype: plt.Figure

    :Example:

    .. code-block:: python

        from accelerometer.accPlot import plot_time_series
        df = pd.DataFrame(...)
        fig = plot_time_series(df)
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

    if show_first_n_days is not None:
        data = data.first(str(show_first_n_days) + 'D')

    labels = [label for label in LABELS_AND_COLORS.keys() if label in data.columns]
    colors = [LABELS_AND_COLORS[label] for label in labels]

    if 'imputed' in data.columns:
        mask = data['imputed'].astype('bool')
        labels_excl_imputed = [label for label in labels if label != 'imputed']
        data.loc[mask, labels_excl_imputed] = 0
        data.loc[mask, "acc"] = np.nan

    # setup plotting range
    if 'acc' in data:
        data['acc'] = data['acc'].rolling(PLOT_ACC_ROLLING_WINDOW).mean()  # minute average
        data['acc'] = data['acc'].clip(0, PLOT_MAX_ACCELERATION_MG)
    data[labels] = data[labels].astype('f4') * PLOT_MAX_ACCELERATION_MG

    # number of rows to display in figure (all days + legend)
    grouped_days = data.groupby(data.index.date)
    nrows = len(grouped_days) + 1

    # create overall figure
    fig = plt.figure(None, figsize=(PLOT_FIGURE_WIDTH, nrows), dpi=PLOT_FIGURE_DPI)
    if title is not None:
        fig.suptitle(title)

    # create individual plot for each day
    i = 0
    axs = []
    for day, group in grouped_days:

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
        ax.get_xaxis().grid(True, which='major', color=PLOT_GRID_COLOR_MAJOR, alpha=PLOT_GRID_ALPHA_MAJOR)
        ax.get_xaxis().grid(True, which='minor', color=PLOT_GRID_COLOR_MINOR, alpha=PLOT_GRID_ALPHA_MINOR)
        # set x and y-axes
        ax.set_xlim(group.index[0], group.index[-1])
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq=PLOT_HOUR_GRID_MAJOR))
        ax.set_xticks(pd.date_range(start=datetime.combine(day, time(0, 0, 0, 0)),
                                    end=datetime.combine(day + timedelta(days=1), time(0, 0, 0, 0)),
                                    freq=PLOT_HOUR_GRID_MINOR), minor=True)
        ax.set_ylim(0, PLOT_MAX_ACCELERATION_MG)
        ax.get_yaxis().set_ticks([])  # hide y-axis lables
        # make border less harsh between subplots
        ax.spines['top'].set_color(PLOT_BORDER_COLOR)
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
    hr_labels = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00']
    axs[0].set_xticklabels(hr_labels)
    axs[0].tick_params(labelbottom=False, labeltop=True, labelleft=False)

    # auto trim borders
    fig.tight_layout()

    return fig


if __name__ == '__main__':
    main()  # Standard boilerplate to call the main() function to begin the program.
