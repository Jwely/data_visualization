from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import math
from datetime import datetime, timedelta


def _freq_to_radial(x: pd.Series, cycle_len: timedelta):
    """
    converts a series of timedelta objects to 360 degree angular positions
    based on the cycle length (theta).
    """

    shifted = x - x.min()
    shifted = shifted.apply(lambda x: x.total_seconds())
    factor = 2 * math.pi / cycle_len.total_seconds()    # degrees

    result = shifted * factor
    return result


# data parsing, manipulation
def load_data():
    """
    Loads all the subreddit data in the data directory and flattens it into
    a single data frame.
    :return:
    """

    # list all csv files in the data directory
    path = Path("data")
    data_files = [x for x in path.iterdir() if x.suffix == ".csv"]

    # load as dataframes and concatenate them
    dataframes = [pd.read_csv(path) for path in data_files]
    df = pd.concat(dataframes, axis=0)

    # coerce numeric columns that have a few errors or format odities
    offset = timedelta(hours=-7)  # TODO: i see no reason to add an offset, UTC is good!
    df["pull_timestamp"] = pd.to_datetime(df["pull_timestamp"]) + offset
    df["users_here"] = pd.to_numeric(df["users_here"], errors="coerce")
    df["subscribers"] = pd.to_numeric(df["subscribers"], errors="coerce")
    df["subreddit"] = df["subreddit"].str.lower()

    # add a few more useful columns
    df.loc[:, "users_here_per_1M"] = 1e6 * df["users_here"] / df["subscribers"]
    return df.set_index("pull_timestamp")


# semi-general cyclic operations
def cyclify(
        df: pd.DataFrame,
        freq: timedelta,
        cycle: timedelta,
        freq_key: str,
        cycle_key: str,
        theta_key: str) -> pd.DataFrame:
    """
    Cyclifies a pandas series (not necessarily discreet, regular, or sorted)
    into a reference cyclic time.

    :param df: dataframe with data, index should be time type
    :param freq: a string tag to use for frequency grouping. *
    :param cycle: a string tag to use for grouping data in cycles.
    :param freq_key: the name of column to add frequency tag data (type timedelta)
    :param cycle_key: the name of column to add cycle tag labels (type datetime)
    :param theta_key: the name of the column to add a cyclical "theta" value.
    :param cycle_len: timedelta representing length of a cycle

    * See table of offset aliases at:
        http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    """

    time_key = df.index.name    # assume time key is index of dataframe

    groups = df.groupby(
        [pd.Grouper(freq=cycle),
         pd.Grouper(freq=freq),
         "subreddit"]).mean()

    # cycle length index then rename the column to reflect cycle length
    groups = groups.reset_index(level=0).rename(columns={time_key: cycle_key})

    # frequency length index then subreddit
    groups = groups.reset_index().rename(columns={time_key: freq_key})

    # strip the cycle length related info from the frequency timestamp
    groups[time_key] = groups[freq_key]
    groups[freq_key] = groups[freq_key] - groups[cycle_key]

    groups[theta_key] = _freq_to_radial(groups[freq_key], cycle)
    groups.sort_values(by="theta", inplace=True)

    return groups


def format_polar_axes(
        df: pd.DataFrame,
        x_key: str,
        theta_key: str,
        cycle_key: str,
        theta_tick_dist: timedelta = None,
        theta_tick_fmt: str = None,
        axes: plt.PolarAxes = None) -> plt.PolarAxes:
    """
    :param df: cyclified data frame
    :param x_key: key of the time data, probably the same as freq_key
    :param theta_key: key produced by cyclification
    :param cycle_key: key produced by cyclification
    :param theta_tick_dist: time between theta tick marks
    :param theta_tick_fmt: format of theta tick marks (as datetime.strftime())*
    :param axes: a matplotlib PolarAxes instance to modifiy if desired.

    * See format help here:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    :return:
    """

    if axes is None:
        axes = plt.subplot(projection="polar")

    # calulate the number of labels given the theta tick distance
    x_range = df[x_key].max() - df[x_key].min()
    n_labels = int(math.ceil(x_range / theta_tick_dist))

    theta_tick_targets = [i * theta_tick_dist for i in range(n_labels)]
    theta_tick_ids = [abs(tt - df[x_key]).idxmin() for tt in theta_tick_targets]
    theta_ticks = df.loc[theta_tick_ids, theta_key] * 180 / math.pi
    theta_labels = df.loc[theta_tick_ids, x_key] + df[cycle_key].min()

    # format them (using datetime.strftime) if format was given
    if theta_tick_fmt is not None:
        theta_labels = [tl.strftime(theta_tick_fmt) for tl in theta_labels]

    axes.set_thetagrids(angles=theta_ticks, labels=theta_labels)
    axes.set_theta_direction(-1)
    return axes


def cyclic_polar_plot(
        df: pd.DataFrame,
        theta_key: str,
        r_key: str,
        axes: plt.PolarAxes,
        **kwargs) -> plt.PolarAxes:
    """
    Basic function to plot
    :param df: cyclified (and probably grouped and aggregated) dataframe
    :param theta_key: key produced by cyclification, theta position
    :param r_key: key for radial data to plot
    :param axes: a matplotlib PolarAxes instance to modify
    :param kwargs: passed  to matplotlib.pyplot.plot()

    :return: modified axes
    """
    # group by the key and sort
    df = df.groupby(by=theta_key).agg({r_key: "mean"}).reset_index()

    # close the loop by duplicating first entry to end position
    df = df.append(df.iloc[0, :])

    # plot the data
    axes.plot(df["theta"], df[r_key], **kwargs)

    return axes


# do the actual things
if __name__ == "__main__":

    df = load_data()
    print(df["subreddit"].unique())

    df = cyclify(
        df,                                 # all the data flattened
        freq=timedelta(hours=1),            # one hour sampling frequency
        cycle=timedelta(days=7),            # one week cycle length
        freq_key="hour",                    # frequency key
        cycle_key="cycle",                  # period key
        theta_key="theta",                  # radially transformed angle key
    )

    ax = format_polar_axes(
        df=df,                              # cyclified dataframe
        x_key="hour",                       # use sampling frequency key as x
        theta_key="theta",                  # same as specified for output
        cycle_key="cycle",                  # same as specified for output
        theta_tick_dist=timedelta(hours=12), # Tick marks a couple times a day
        theta_tick_fmt="%a-%I%p"            # string format for those ticks
    )

    # # subset our data
    subs = df.groupby(by=["hour", "subreddit"]).mean().reset_index(level=1)

    # define shared kwargs for calls to cyclic_polar_plot
    cyclic_polar_kwargs = dict(r_key="users_here", theta_key="theta", axes=ax)

    # plots!
    plotpairs = [
        ("dataisbeautiful", "blue"),
        ("art", "red"),
        ("space", "green")
    ]

    for sub, color in plotpairs:
        ax = cyclic_polar_plot(
            **cyclic_polar_kwargs,
            df=subs[subs["subreddit"] == sub],
            color=color,
            linewidth=2)

    ax.legend([p[0] for p in plotpairs], loc=(1.1,1))
    plt.show(ax)
