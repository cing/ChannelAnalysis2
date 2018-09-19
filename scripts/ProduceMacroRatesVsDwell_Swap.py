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
#from Transitions import mean_unique_macrostate_transition_counts
from ChannelAnalysis2.Transitions import macrostate_transitions_per_traj
from ChannelAnalysis2.Coordination import macrostate_labels, ordering_labels
import matplotlib.pyplot as plt
import ast

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
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', type=str, required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    #f1, ax1 = plt.subplots()
    TestProject = Project(args.cfg_path, coord_ts=[args.series_name])
    print("Successfully Loaded Project: %s" % TestProject.name)

    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name],
                                  TestProject.start_time,
                                  TestProject.end_time)
    coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
    coord_df2_orders = ordering_labels(coord_df2_trim)

    coord_df2_occprime = macrostate_labels(coord_df2_orders)
    coord_df2_occprime_low = coord_df2_occprime[coord_df2_occprime["Order"] == 0]

    states_per_transition=2
    transition_column="Ordering"

    trans = macrostate_transitions_per_traj(coord_df2_occprime_low[["Frame","Ordering","TrajNum"]],
                                    transition_column=transition_column,
                                    #dwell_cut=5, return_stats=True,
                                    dwell_cut=10, return_stats=True,
                                    states_per_transition=states_per_transition)
    #import pdb
    #pdb.set_trace()

    trans['Before'], trans['After'] = trans['Transition'].str.split('-', 1).str

    def subfinder(mylist, pattern):
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
                return True
        return False

    def newion_bool(x):
        b = ast.literal_eval(x["Before"])
        a = ast.literal_eval(x["After"])
        if subfinder(b,a) or subfinder(a,b):
            return True
        return False

    novel_trans = trans.iloc[:, 1:].apply(newion_bool, axis=1)
    print(trans[~novel_trans])
    print(trans[~novel_trans]["Count"].sum())

    #import pdb
    #pdb.set_trace()
    compute_rates = True
    # Compute total number of frames using the last recorded frame per trajectory
    if compute_rates:
        total_steps = coord_df2_occprime_low.groupby(["TrajNum"])["Time"].apply(lambda x: max(x.unique()))
        trans_normalized = trans.set_index("TrajNum")
        # Factor of 1000 puts things in units of microsecond.
        trans_normalized["Rate"] = 1000*trans_normalized["Count"].div(total_steps,
                                                                 axis="index").dropna()
        trans = trans_normalized.reset_index()
        gb_column="Rate"
    else:
        gb_column="Count"

    gg = trans[~novel_trans].groupby(["TrajNum"])[gb_column].sum().reset_index()
    trajs = coord_df2_occprime_low["TrajNum"].unique()
    df_fill = pd.DataFrame({i:[0.0,] for i in set(trajs) - set(gg["TrajNum"])}).T.reset_index()
    df_fill.columns=["TrajNum",gb_column]
    gg = pd.concat([gg, df_fill])
    print("Mean, SEM: ")
    print(gg[gb_column].mean(), gg[gb_column].sem())
