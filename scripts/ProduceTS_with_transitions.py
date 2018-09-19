""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted. """

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from ChannelAnalysis2.Initialize import Project
import itertools
from collections import defaultdict
from ChannelAnalysis2.Transitions import state_transitions_per_trajres
from ChannelAnalysis2.Plot import plot_ts_by_resid_with_trans
import matplotlib.pyplot as plt

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def prune_time(df, start, end):
    return df[(df["Time"] >= start) & (df["Time"] <= end)]

def prune_column(df, col, start, end):
    return df[(df[col] >= start) & (df[col] <= end)]

if __name__ == '__main__':
    parser = ArgumentParser(
    description='This script extracts basic transition statistics of \
    coordination with attention to statistics across multiple timeseries.')
    parser.add_argument(
    '-c', dest='cfg_path', type=str, required=True,
    help='a configfile describing data paths and parameters for a project')
    parser.add_argument(
    '-d', dest='dwell_cut', type=int, required=True,
    help='minimum number of timesteps in state to count as transition')
    parser.add_argument(
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    fig = {}
    axes = {}

    for series_num, series_name in enumerate(args.series_name):

        # Extract the mean number of ions in the range -2.0 to 2.0
        coord_df2_trim = prune_column(TestProject.coord_ts[series_name], "Z", -14, 14)

        # Plot the rates extracted from transcount
        trans = state_transitions_per_trajres(coord_df2_trim,
                                           TestProject.state_dividers,
                                           state_names=TestProject.state_names,
                                           dwell_cut=args.dwell_cut,
                                           return_stats=False)

        # Kind of a hack, we should have a Time_Start, etc.
        trans["Frame_Start"] *= TestProject.dt
        trans["Dwell_Start"] *= TestProject.dt
        trans["Frame_End"] *= TestProject.dt
        trans["Dwell_End"] *= TestProject.dt

        for traj in coord_df2_trim["TrajNum"].unique():

            if traj not in fig:
                f1, ax = plt.subplots(len(args.series_name))
                fig[traj] = f1
                axes[traj] = ax

            # Probably a better way to do this...
            if len(args.series_name) > 1:
                target_ax = axes[traj][series_num]
            else:
                target_ax = axes[traj]

            trans_subset = trans[trans["TrajNum"]==traj]
            coord_subset = coord_df2_trim[coord_df2_trim["TrajNum"]==traj]


            plot_ts_by_resid_with_trans(target_ax, coord_subset, trans=trans_subset, xlim=[0, 1000],
                                   ylim=[-10,14],
                                   x_label="time (ns)", y_label="axial position (nm)",
                                   ts_hline=TestProject.state_dividers)

    for traj,f1 in fig.iteritems():
        f1.gca().invert_yaxis()
        f1.set_size_inches(18.5, 5.5*len(args.series_name))
        f1.savefig(TestProject.output_name+"_ts_n"+str(traj)+"B.pdf", dpi=200)

