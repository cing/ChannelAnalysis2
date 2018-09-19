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
from ChannelAnalysis2.Coordination import macrostate_labels
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
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', type=str, required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-oc', dest='outer_cut', type=float, default=-10.0,
    help='where to move the outer-ion boundary for 2-3 occ')
    args = parser.parse_args()

    #f1, ax1 = plt.subplots()
    TestProject = Project(args.cfg_path, coord_ts=[args.series_name])
    print("Successfully Loaded Project: %s" % TestProject.name)

    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name],
                                  TestProject.start_time,
                                  TestProject.end_time)
    #coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
    coord_df2_trim = prune_column(coord_df_noequil, "Z", args.outer_cut, 14)

    #coord_df2_occprime = macrostate_labels(coord_df2_trim)
    coord_df2_occprime = macrostate_labels(coord_df2_trim, drop_prime=True)
    coord_df2_occprime_low = coord_df2_occprime[coord_df2_occprime["Order"] == 0]

    states_per_transition=2
    transition_column="OccMacrostate"

    trans = macrostate_transitions_per_traj(coord_df2_occprime_low[["Frame","OccMacrostate","TrajNum"]],
                                    transition_column=transition_column,
                                    #dwell_cut=10, return_stats=True, 
                                    #dwell_cut=25, return_stats=True, 
                                    dwell_cut=250, return_stats=True, 
                                    states_per_transition=states_per_transition)
    trans2 = macrostate_transitions_per_traj(coord_df2_occprime_low[["Frame","OccMacrostate","TrajNum"]],
                                    transition_column=transition_column,
                                    #dwell_cut=10, return_stats=True, 
                                    #dwell_cut=25, return_stats=False,
                                    dwell_cut=250, return_stats=False,
                                    states_per_transition=states_per_transition)

    #trans = macrostate_transitions_per_traj(coord_df2_occprime_low[["Frame","OccMacrostate","TrajNum"]],
    #                                transition_column="OccMacrostate",
    #                                dwell_cut=5, return_stats=True)

    compute_rates = True
    print(trans)
    print(trans2)
    #import pdb
    #pdb.set_trace()


    # Compute total number of frames using the last recorded frame per trajectory
    if compute_rates:
        total_steps = coord_df2_trim.groupby(["TrajNum"])["Time"].apply(lambda x: max(x.unique()))
        trans_normalized = trans.set_index("TrajNum")
        # Factor of 1000 puts things in units of microsecond.
        trans_normalized["Rate"] = 1000*trans_normalized["Count"].div(total_steps,
                                                                 axis="index").dropna()
        trans = trans_normalized.reset_index()
        gb_column="Rate"
    else:
        gb_column="Count"

    gg = trans.groupby(["TrajNum","Transition"])[gb_column].sum().reset_index()
    ggpivot = gg.pivot(index="Transition",columns="TrajNum",values=gb_column).fillna(0)
    ggstats = pd.concat([ggpivot.mean(axis=1),
                         ggpivot.std(axis=1),
                         ggpivot.sem(axis=1)], axis=1)

    ggstats.columns = ["Mean","STD","SEM"]

    print(ggstats)
    macrostate_size=coord_df2_occprime_low.groupby(["OccMacrostate"]).size()
    macrostate_percent=macrostate_size/sum(macrostate_size)
    print(macrostate_size)
    print(macrostate_percent)

    #beforeafter=pd.Series(ggstats.index).str.split("-", return_type="frame") #Deprecated in Pandas 0.16
    beforeafter=pd.Series(ggstats.index).str.split("-", expand=True) #Supported in Pandas 0.16
    beforeafter.columns=["OccMacrostate","NextOccMacrostate"]
    ggstats_beforeafter=ggstats.reset_index().join(beforeafter)
    ggstats_macro=ggstats_beforeafter[["Transition","Mean","SEM","OccMacrostate"]].set_index("OccMacrostate")

    ggstats_normed=ggstats_macro.join(pd.DataFrame(macrostate_percent, columns=["Percent"]))
    ggstats_normed["MeanPercent"]=ggstats_normed["Mean"]/ggstats_normed["Percent"]
    ggstats_normed["SEMPercent"]=ggstats_normed["SEM"]/ggstats_normed["Percent"]
    print(ggstats_normed[ggstats_normed["Percent"]>0.05])

